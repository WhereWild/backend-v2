"""Materialize per-taxon relative position parquets from rank index parquets.

This script processes each ``<rank>_index.parquet`` file exactly once as a
producer pass:

1. Read one ancestor index parquet.
2. Build in-memory position rows keyed by descendant taxon.
3. Flush those rows to each descendant's ``relative_ranks_positions.parquet``
   with upsert semantics (no blind overwrite, no blind append).

The written parquet schema matches ``util.indexing.load_relative_ranks``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

import util.taxa_navigation as taxa_navigation
from util.config import load_config
from util.storage import ParquetStorageProxy


CONFIG = load_config("global")
PARQUET = ParquetStorageProxy(CONFIG.data_root, CONFIG.project_root)
FLUSH_ROWS = max(1, int(getattr(CONFIG, "process_positions_flush_rows", 1_000_000)))

POSITION_FILENAME = "relative_ranks_positions.parquet"
POSITION_COLUMNS = (
    "variable",
    "metric",
    "position",
    "count",
    "sampleCount",
    "contextTaxonId",
    "contextLabel",
)
POSITION_SCHEMA = pa.schema(
    [
        pa.field("variable", pa.string()),
        pa.field("metric", pa.string()),
        pa.field("position", pa.int64()),
        pa.field("count", pa.int64()),
        pa.field("sampleCount", pa.int64()),
        pa.field("contextTaxonId", pa.string()),
        pa.field("contextLabel", pa.string()),
    ]
)


def _iter_taxa(root_taxon_id: str) -> Iterable[taxa_navigation.TaxonRecord]:
    root = taxa_navigation.get_taxon_by_id(str(root_taxon_id))
    if root is None:
        raise ValueError(f"Unknown taxon id {root_taxon_id}")
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = taxa_navigation.get_children(node["taxon_key"])
        if children:
            stack.extend(children)


def _rank_index_paths(node_path: Path) -> list[Path]:
    if PARQUET.is_remote or not node_path.exists():
        return []
    paths = [
        child
        for child in node_path.iterdir()
        if child.is_file()
        and child.name.endswith("_index.parquet")
        and child.name != "occurrence_index.parquet"
    ]
    return sorted(paths, key=lambda path: path.name)


def _load_column_lengths(index_path: Path) -> dict[str, int]:
    try:
        schema = PARQUET.read_schema(index_path)
    except (OSError, ValueError):
        return {}
    metadata = schema.metadata or {}
    raw = metadata.get(b"column_lengths")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    lengths: dict[str, int] = {}
    for key, value in decoded.items():
        try:
            length = int(value)
        except (TypeError, ValueError):
            continue
        if length > 0:
            lengths[str(key)] = length
    return lengths


def _normalize_context_label(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("_", " ").split())


def _resolve_context_label(taxon: taxa_navigation.TaxonRecord) -> str:
    scientific = taxon.get("scientific_name")
    scientific_label = _normalize_context_label(scientific)
    if scientific_label:
        return scientific_label
    common = taxon.get("common_name")
    common_label = _normalize_context_label(common)
    if common_label:
        return common_label
    return str(taxon["taxon_key"])


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["variable"]),
        str(row["metric"]),
        str(row["contextTaxonId"]),
    )


def _normalize_position_row(row: dict[str, Any]) -> dict[str, Any] | None:
    variable = row.get("variable")
    metric = row.get("metric")
    context_taxon_id = row.get("contextTaxonId")
    context_label = row.get("contextLabel")
    if variable is None or metric is None or context_taxon_id is None:
        return None
    try:
        position = int(row.get("position"))
        count = int(row.get("count"))
    except (TypeError, ValueError):
        return None
    sample_count_raw = row.get("sampleCount")
    sample_count: int | None
    if sample_count_raw is None:
        sample_count = None
    else:
        try:
            sample_count = int(sample_count_raw)
        except (TypeError, ValueError):
            sample_count = None
    return {
        "variable": str(variable),
        "metric": str(metric),
        "position": position,
        "count": count,
        "sampleCount": sample_count,
        "contextTaxonId": str(context_taxon_id),
        "contextLabel": _normalize_context_label(context_label),
    }


def _load_existing_rows(positions_path: Path) -> list[dict[str, Any]]:
    local_exists = positions_path.exists()
    if not local_exists and not PARQUET.exists(positions_path):
        return []
    try:
        table = PARQUET.read_table(positions_path, columns=list(POSITION_COLUMNS))
    except Exception:
        try:
            table = pq.read_table(positions_path, columns=list(POSITION_COLUMNS))
        except Exception:
            return []
    rows: list[dict[str, Any]] = []
    for raw in table.to_pylist():
        normalized = _normalize_position_row(raw)
        if normalized is not None:
            rows.append(normalized)
    return rows


def _write_rows(positions_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        positions_path.unlink(missing_ok=True)
        return
    arrays = {
        "variable": pa.array([row["variable"] for row in rows], type=pa.string()),
        "metric": pa.array([row["metric"] for row in rows], type=pa.string()),
        "position": pa.array([row["position"] for row in rows], type=pa.int64()),
        "count": pa.array([row["count"] for row in rows], type=pa.int64()),
        "sampleCount": pa.array([row["sampleCount"] for row in rows], type=pa.int64()),
        "contextTaxonId": pa.array(
            [row["contextTaxonId"] for row in rows],
            type=pa.string(),
        ),
        "contextLabel": pa.array(
            [_normalize_context_label(row["contextLabel"]) for row in rows],
            type=pa.string(),
        ),
    }
    table = pa.Table.from_arrays(
        [arrays[name] for name in POSITION_COLUMNS],
        schema=POSITION_SCHEMA,
    )
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=positions_path.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, positions_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _upsert_rows_for_taxon(
    taxon_key: str,
    new_rows: list[dict[str, Any]],
) -> tuple[bool, int]:
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
    if taxon is None or not new_rows:
        return False, 0
    positions_path = Path(taxon["path"]) / POSITION_FILENAME
    existing_rows = _load_existing_rows(positions_path)
    existing_keys = {_row_key(row) for row in existing_rows}

    unique_new: list[dict[str, Any]] = []
    seen_new: set[tuple[str, str, str]] = set()
    for row in new_rows:
        key = _row_key(row)
        if key in existing_keys or key in seen_new:
            continue
        seen_new.add(key)
        unique_new.append(row)

    if not unique_new:
        return False, 0

    merged_rows = existing_rows + unique_new
    _write_rows(positions_path, merged_rows)
    return True, len(unique_new)


def _existing_row_keys_for_taxon(taxon_key: str) -> set[tuple[str, str, str]]:
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
    if taxon is None:
        return set()
    positions_path = Path(taxon["path"]) / POSITION_FILENAME
    rows = _load_existing_rows(positions_path)
    return {_row_key(row) for row in rows}


def _metric_columns_by_variable(schema_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in schema_names:
        if "::" not in name:
            continue
        variable, _metric = name.split("::", 1)
        grouped.setdefault(variable, []).append(name)
    for variable in grouped:
        grouped[variable] = sorted(grouped[variable], key=lambda column: column.lower())
    return grouped


def _collect_positions_for_columns(
    ancestor: taxa_navigation.TaxonRecord,
    table: pa.Table,
    column_names: list[str],
    column_lengths: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    context_taxon_id = str(ancestor["taxon_key"])
    context_label = _resolve_context_label(ancestor)

    by_taxon: dict[str, list[dict[str, Any]]] = {}
    if not column_names:
        return by_taxon
    for column_name in column_names:
        variable, metric = column_name.split("::", 1)
        column = table.column(column_name).combine_chunks()

        column_length = int(column_lengths.get(column_name) or 0)
        if column_length <= 0:
            derived = len(column)
            while derived > 0 and column[derived - 1].as_py() is None:
                derived -= 1
            column_length = derived
        if column_length <= 0:
            continue
        if column_length < len(column):
            column = column.slice(0, column_length)

        taxon_keys = column.field("taxonKey").to_pylist()
        sample_counts = column.field("sampleCount").to_pylist()
        count = len(taxon_keys)

        for position, (taxon_key, sample_count) in enumerate(
            zip(taxon_keys, sample_counts)
        ):
            if taxon_key is None:
                continue
            row = {
                "variable": variable,
                "metric": metric,
                "position": position,
                "count": count,
                "sampleCount": int(sample_count) if sample_count is not None else None,
                "contextTaxonId": context_taxon_id,
                "contextLabel": context_label,
            }
            by_taxon.setdefault(str(taxon_key), []).append(row)
    return by_taxon


def _variable_already_processed(
    by_taxon: dict[str, list[dict[str, Any]]],
) -> bool:
    if not by_taxon:
        return False
    representative = sorted(
        by_taxon.keys(),
        key=lambda value: (
            0 if taxa_navigation.taxon_id_as_int(str(value)) is not None else 1,
            taxa_navigation.taxon_id_as_int(str(value)) or str(value),
        ),
    )[0]
    candidate_rows = by_taxon.get(representative) or []
    if not candidate_rows:
        return False
    expected = {_row_key(row) for row in candidate_rows}
    if not expected:
        return False
    existing = _existing_row_keys_for_taxon(representative)
    return expected.issubset(existing)


def _add_pending_rows(
    pending_by_taxon: dict[str, list[dict[str, Any]]],
    by_taxon: dict[str, list[dict[str, Any]]],
) -> int:
    added = 0
    for taxon_key, rows in by_taxon.items():
        if not rows:
            continue
        bucket = pending_by_taxon.setdefault(taxon_key, [])
        bucket.extend(rows)
        added += len(rows)
    return added


def _flush_pending_rows(
    pending_by_taxon: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    touched = 0
    added = 0
    for taxon_key, rows in pending_by_taxon.items():
        wrote, inserted = _upsert_rows_for_taxon(taxon_key, rows)
        if wrote:
            touched += 1
            added += inserted
    pending_by_taxon.clear()
    return touched, added


def process_positions(
    root_taxon_id: str = CONFIG.root_taxon_id,
) -> None:
    """Build per-taxon relative_ranks_positions.parquet from rank index parquets."""
    if PARQUET.is_remote:
        raise RuntimeError("process_positions currently requires local parquet storage.")

    files_seen = 0
    taxa_touched = 0
    rows_added = 0

    for node in _iter_taxa(root_taxon_id):
        node_path = Path(node["path"])
        index_paths = _rank_index_paths(node_path)
        if not index_paths:
            continue
        for index_path in index_paths:
            files_seen += 1
            print(
                f"[positions] read ancestor={node['taxon_key']} file={index_path.name}"
            )
            try:
                schema = PARQUET.read_schema(index_path)
            except (OSError, ValueError):
                print(
                    f"[positions] skip ancestor={node['taxon_key']} "
                    f"file={index_path.name} (schema read failed)"
                )
                continue

            grouped_columns = _metric_columns_by_variable(list(schema.names))
            if not grouped_columns:
                print(
                    f"[positions] skip ancestor={node['taxon_key']} "
                    f"file={index_path.name} (no metric columns)"
                )
                continue

            column_lengths = _load_column_lengths(index_path)
            all_metric_columns = sorted(
                [name for columns in grouped_columns.values() for name in columns],
                key=str.lower,
            )
            try:
                table = PARQUET.read_table(index_path, columns=all_metric_columns)
            except (OSError, ValueError):
                print(
                    f"[positions] skip ancestor={node['taxon_key']} "
                    f"file={index_path.name} (table read failed)"
                )
                continue
            touched_in_file = 0
            added_in_file = 0
            pending_by_taxon: dict[str, list[dict[str, Any]]] = {}
            pending_rows = 0
            for variable in sorted(grouped_columns.keys(), key=str.lower):
                variable_columns = grouped_columns[variable]
                print(
                    f"[positions] variable ancestor={node['taxon_key']} "
                    f"file={index_path.name} variable={variable} metrics={len(variable_columns)}"
                )
                by_taxon = _collect_positions_for_columns(
                    node,
                    table,
                    variable_columns,
                    column_lengths,
                )
                if not by_taxon:
                    print(
                        f"[positions] skip ancestor={node['taxon_key']} file={index_path.name} "
                        f"variable={variable} (no rows)"
                    )
                    continue
                if _variable_already_processed(by_taxon):
                    print(
                        f"[positions] skip ancestor={node['taxon_key']} file={index_path.name} "
                        f"variable={variable} (flag present)"
                    )
                    continue
                buffered_rows = _add_pending_rows(pending_by_taxon, by_taxon)
                pending_rows += buffered_rows
                print(
                    f"[positions] buffered ancestor={node['taxon_key']} file={index_path.name} "
                    f"variable={variable} rows={buffered_rows} pending_rows={pending_rows}"
                )
                if pending_rows >= FLUSH_ROWS:
                    print(
                        f"[positions] flush ancestor={node['taxon_key']} file={index_path.name} "
                        f"reason=threshold pending_taxa={len(pending_by_taxon)} pending_rows={pending_rows}"
                    )
                    touched_now, added_now = _flush_pending_rows(pending_by_taxon)
                    pending_rows = 0
                    touched_in_file += touched_now
                    added_in_file += added_now
                    taxa_touched += touched_now
                    rows_added += added_now
                    print(
                        f"[positions] flushed ancestor={node['taxon_key']} file={index_path.name} "
                        f"reason=threshold touched={touched_now} added={added_now}"
                    )

            if pending_rows > 0:
                print(
                    f"[positions] flush ancestor={node['taxon_key']} file={index_path.name} "
                    f"reason=file-end pending_taxa={len(pending_by_taxon)} pending_rows={pending_rows}"
                )
                touched_now, added_now = _flush_pending_rows(pending_by_taxon)
                pending_rows = 0
                touched_in_file += touched_now
                added_in_file += added_now
                taxa_touched += touched_now
                rows_added += added_now
                print(
                    f"[positions] flushed ancestor={node['taxon_key']} file={index_path.name} "
                    f"reason=file-end touched={touched_now} added={added_now}"
                )

            print(
                f"[positions] file-done ancestor={node['taxon_key']} "
                f"file={index_path.name} touched={touched_in_file} added={added_in_file}"
            )

    print(
        f"[positions] done files={files_seen} taxa_touched={taxa_touched} rows_added={rows_added}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read each <rank>_index.parquet once and materialize per-taxon "
            "relative_ranks_positions.parquet files."
        )
    )
    parser.add_argument(
        "--root-taxon-id",
        default=CONFIG.root_taxon_id,
        help="Taxon id to treat as the subtree root.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    process_positions(root_taxon_id=str(args.root_taxon_id))


if __name__ == "__main__":
    main()
