"""
Run iNat->GBIF mapping (offline, observations, API) and merge into the taxon catalog.

Steps are skipped if their output CSV already exists.
"""

from __future__ import annotations

import csv
import json
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from util.config import load_config
from util.taxa_navigation import _normalize_taxon_path


CONFIG = load_config("global")

inat_api_progress_every = 250

inat_api_rate_limit_per_second = 4.0

inat_api_timeout_seconds = 20

inat_mapping_ranks = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")

inat_obs_batch_size = 200

inat_obs_request_limit = 0

inat_taxa_filename = "taxa.csv"


INAT_OBS_ENDPOINT = "https://api.inaturalist.org/v1/observations"

DEFAULT_RANKS = {"SPECIES", "SUBSPECIES", "VARIETY", "FORM"}
INFRA_MARKERS = {"var.", "subsp.", "f.", "nothosubsp.", "nothovar."}
INFRA_RANKS = {"SUBSPECIES", "VARIETY", "FORM"}


def canonical_rank(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().upper()


def normalize_name(value: str) -> str:
    if not value:
        return ""
    cleaned = value.replace("_", " ").replace("×", "x").lower()
    return " ".join(cleaned.split())


def strip_infra_markers(value: str) -> str:
    if not value:
        return ""
    tokens = [token for token in value.split() if token not in INFRA_MARKERS]
    return " ".join(tokens)


def load_catalog() -> dict[str, Any]:
    catalog_path = CONFIG.taxon_catalog_path
    print(f"Loading catalog from {catalog_path}...")
    with open(catalog_path, "rb") as f:
        payload = pickle.load(f)
    return payload


def save_catalog(payload: dict[str, Any]) -> None:
    catalog_path = CONFIG.taxon_catalog_path
    with open(catalog_path, "wb") as f:
        pickle.dump(payload, f)


def build_gbif_indexes(
    catalog: dict[str, Any],
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]]]:
    exact_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    stripped_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for taxon_key, taxon in catalog.items():
        rank = canonical_rank(taxon.get("rank"))
        sci_raw = taxon.get("scientific_name") or ""
        name = normalize_name(sci_raw)
        if not name or not rank:
            continue
        exact_index[(rank, name)].append(taxon_key)
        stripped = strip_infra_markers(name)
        if stripped:
            stripped_index[(rank, stripped)].append(taxon_key)

    return exact_index, stripped_index


def build_offline_mapping() -> None:
    output_path = CONFIG.inat_mapping_offline_path
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping offline mapping: {output_path.name} exists.")
        return

    inat_taxa_path = CONFIG.species_dir / CONFIG.inat_dwca_dirname / inat_taxa_filename
    if not inat_taxa_path.exists():
        raise FileNotFoundError(f"iNat taxa.csv not found at {inat_taxa_path}")

    payload = load_catalog()
    catalog = payload["catalog"]
    print(f"  Catalog taxa: {len(catalog):,}")

    ranks = {canonical_rank(rank) for rank in inat_mapping_ranks} or set(DEFAULT_RANKS)

    print("Building GBIF indexes...")
    exact_index, stripped_index = build_gbif_indexes(catalog)
    print(f"  Exact keys: {len(exact_index):,}")
    print(f"  Stripped keys: {len(stripped_index):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    matched = Counter()
    ambiguous = Counter()
    unmatched = Counter()
    conflicts = Counter()
    used_gbif: dict[str, str] = {}
    ambiguous_samples: dict[str, list[str]] = defaultdict(list)
    conflict_samples: dict[str, list[str]] = defaultdict(list)

    with open(inat_taxa_path, "r", encoding="utf-8", newline="") as infile, \
         open(output_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = [
            "gbif_taxon_key",
            "inat_id",
            "inat_taxon_url",
            "rank",
            "scientific_name",
            "match_type",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        scanned = 0
        for row in reader:
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  Scanned {scanned:,} rows...")

            rank = canonical_rank(row.get("taxonRank"))
            if rank not in ranks:
                continue

            name = normalize_name(row.get("scientificName") or "")
            if not name:
                continue

            totals[rank] += 1
            gbif_keys = exact_index.get((rank, name), [])
            match_type = "exact"

            if not gbif_keys and rank in INFRA_RANKS:
                stripped = strip_infra_markers(name)
                if stripped:
                    gbif_keys = stripped_index.get((rank, stripped), [])
                    match_type = "stripped"

            if len(gbif_keys) == 1:
                gbif_key = gbif_keys[0]
                inat_id = str(row.get("id") or "").strip()
                inat_taxon_url = str(row.get("taxonID") or "").strip()
                if not inat_id:
                    continue
                existing = used_gbif.get(gbif_key)
                if existing and existing != inat_id:
                    conflicts[rank] += 1
                    if len(conflict_samples[rank]) < 5:
                        conflict_samples[rank].append(name)
                    continue
                used_gbif[gbif_key] = inat_id
                matched[rank] += 1
                writer.writerow(
                    {
                        "gbif_taxon_key": gbif_key,
                        "inat_id": inat_id,
                        "inat_taxon_url": inat_taxon_url,
                        "rank": rank,
                        "scientific_name": name,
                        "match_type": match_type,
                    }
                )
            elif len(gbif_keys) > 1:
                ambiguous[rank] += 1
                if len(ambiguous_samples[rank]) < 5:
                    ambiguous_samples[rank].append(name)
            else:
                unmatched[rank] += 1

    print(f"\nSaved mapping to {output_path}")
    print("\nMatch summary (iNat -> GBIF):")
    for rank in sorted(totals.keys()):
        total = totals[rank]
        match_count = matched[rank]
        amb = ambiguous[rank]
        miss = unmatched[rank]
        conflict = conflicts[rank]
        rate = (match_count / total * 100) if total else 0
        print(
            f"  {rank:<12} total={total:>9,}  matched={match_count:>9,} "
            f"ambiguous={amb:>9,} conflicts={conflict:>9,} "
            f"missing={miss:>9,}  rate={rate:5.1f}%"
        )

    if ambiguous_samples:
        print("\nAmbiguous samples:")
        for rank, names in ambiguous_samples.items():
            print(f"  {rank}: {', '.join(names)}")

    if conflict_samples:
        print("\nConflict samples (multiple iNat taxa mapped to same GBIF key):")
        for rank, names in conflict_samples.items():
            print(f"  {rank}: {', '.join(names)}")


def extract_observation_id(raw_values: list[Any]) -> str | None:
    for value in raw_values:
        if value is None:
            continue
        if isinstance(value, int):
            if value >= 0:
                return str(value)
            continue
        if isinstance(value, float):
            if value.is_integer() and value >= 0:
                return str(int(value))
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            return text
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        matches = re.findall(r"\d+", text)
        if len(matches) == 1:
            return matches[0]
    return None


def load_existing_mappings(paths: list[Path]) -> set[str]:
    mapped: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(row.get("gbif_taxon_key") or "").strip()
                if key:
                    mapped.add(key)
    return mapped


def fetch_observations(obs_ids: list[str], timeout: int) -> list[dict[str, Any]]:
    params = {
        "id": ",".join(obs_ids),
        "per_page": str(len(obs_ids)),
    }
    url = f"{INAT_OBS_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "wherewild-inat-obs-mapper/1.0"})
    with urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results") or []
    return results if isinstance(results, list) else []


def build_observation_mapping() -> None:
    output_path = CONFIG.inat_mapping_obs_path
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping observation mapping: {output_path.name} exists.")
        return

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for observation mapping.") from exc

    payload = load_catalog()
    catalog = payload["catalog"]
    print(f"  Catalog taxa: {len(catalog):,}")

    mapped = load_existing_mappings(
        [
            CONFIG.inat_mapping_offline_path,
            output_path,
        ]
    )
    print(f"  Existing mappings: {len(mapped):,}")

    ranks = {canonical_rank(rank) for rank in inat_mapping_ranks} or set(DEFAULT_RANKS)
    request_limit = inat_obs_request_limit
    batch_size = max(1, min(inat_obs_batch_size, 200))
    rate_limit = max(inat_api_rate_limit_per_second, 0.1)
    timeout = inat_api_timeout_seconds
    progress_every = inat_api_progress_every

    needed = {
        key
        for key, taxon in catalog.items()
        if canonical_rank(taxon.get("rank")) in ranks and key not in mapped
    }
    if request_limit:
        needed = set(list(needed)[:request_limit])
    print(f"  Taxa needing observation lookup: {len(needed):,}")

    if not needed:
        print("Nothing to do.")
        return

    obs_by_taxon: dict[str, str] = {}

    print("Scanning occurrence parquets for observation IDs...")
    for taxon_key in needed:
        taxon = catalog.get(taxon_key)
        if not taxon:
            continue
        parquet_path = _normalize_taxon_path(taxon["path"]) / CONFIG.occurrence_parquet_filename
        if not parquet_path.exists():
            continue
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            if parquet_file.num_row_groups == 0:
                continue
            table = parquet_file.read_row_group(0, columns=["catalogNumber"])
        except Exception:
            continue
        if table.num_rows == 0:
            continue
        catalog_numbers = table.column("catalogNumber").to_pylist()
        if not catalog_numbers:
            continue
        observation_id = extract_observation_id(catalog_numbers[:50])
        if not observation_id:
            continue
        obs_by_taxon[taxon_key] = observation_id

    print(f"  Collected observation IDs for {len(obs_by_taxon):,} taxa")

    obs_to_taxon = {obs_id: taxon_key for taxon_key, obs_id in obs_by_taxon.items()}
    obs_ids = list(obs_to_taxon.keys())

    counters = Counter()
    last_print = time.time()

    print(
        f"Querying iNat observations in batches of {batch_size} @ "
        f"{rate_limit:.1f} req/s..."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        fieldnames = [
            "gbif_taxon_key",
            "inat_id",
            "inat_taxon_url",
            "rank",
            "scientific_name",
            "match_type",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(0, len(obs_ids), batch_size):
            batch = obs_ids[idx: idx + batch_size]
            try:
                results = fetch_observations(batch, timeout=timeout)
            except Exception:
                counters["errors"] += 1
                time.sleep(1.0 / rate_limit)
                continue

            counters["batches"] += 1
            counters["attempted"] += len(batch)

            for obs in results:
                obs_id = str(obs.get("id") or "")
                taxon = obs.get("taxon") or {}
                taxon_id = str(taxon.get("id") or "").strip()
                taxon_rank = canonical_rank(taxon.get("rank"))
                taxon_name = str(taxon.get("name") or "").strip()
                if not obs_id or not taxon_id:
                    continue
                gbif_key = obs_to_taxon.get(obs_id)
                if not gbif_key:
                    continue
                writer.writerow(
                    {
                        "gbif_taxon_key": gbif_key,
                        "inat_id": taxon_id,
                        "inat_taxon_url": f"https://www.inaturalist.org/taxa/{taxon_id}",
                        "rank": taxon_rank,
                        "scientific_name": taxon_name,
                        "match_type": "obs",
                    }
                )
                counters["matched"] += 1

            if progress_every and counters["batches"] % progress_every == 0:
                elapsed = time.time() - last_print
                last_print = time.time()
                print(
                    f"  batches={counters['batches']:,} matched={counters['matched']:,} "
                    f"errors={counters['errors']:,} elapsed={elapsed:0.1f}s",
                    flush=True,
                )

            time.sleep(1.0 / rate_limit)

    print("\nObservation mapping summary:")
    for key in ("batches", "attempted", "matched", "errors"):
        if key in counters:
            print(f"  {key}: {counters[key]:,}")
    print(f"\nSaved observation mappings to {output_path}")


def load_mapping(path: Path, source: str) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gbif_key = str(row.get("gbif_taxon_key") or "").strip()
            inat_id = str(row.get("inat_id") or "").strip()
            if not gbif_key or not inat_id:
                continue
            mapping[gbif_key] = {
                "inat_id": inat_id,
                "inat_taxon_url": str(row.get("inat_taxon_url") or "").strip(),
                "inat_match_type": str(row.get("match_type") or "").strip() or source,
                "inat_scientific_name": str(row.get("scientific_name") or "").strip(),
                "inat_rank": str(row.get("rank") or "").strip(),
                "source": source,
            }
    return mapping


def merge_mappings(
    obs: dict[str, dict[str, str]],
    offline: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for source_map in (offline, obs):
        for key, payload in source_map.items():
            merged[key] = payload
    return merged


def apply_mapping_to_catalog(mapping: dict[str, dict[str, str]]) -> int:
    payload = load_catalog()
    catalog = payload["catalog"]

    updated = 0
    for gbif_key, payload_map in mapping.items():
        taxon = catalog.get(gbif_key)
        if not taxon:
            continue
        taxon["inat_id"] = payload_map.get("inat_id")
        if payload_map.get("inat_taxon_url"):
            taxon["inat_taxon_url"] = payload_map.get("inat_taxon_url")
        if payload_map.get("inat_match_type"):
            taxon["inat_match_type"] = payload_map.get("inat_match_type")
        if payload_map.get("inat_scientific_name"):
            taxon["inat_scientific_name"] = payload_map.get("inat_scientific_name")
        if payload_map.get("inat_rank"):
            taxon["inat_rank"] = payload_map.get("inat_rank")
        updated += 1

    save_catalog(payload)
    print(f"Updated {updated:,} catalog entries.")
    return updated


def main() -> None:
    offline_path = CONFIG.inat_mapping_offline_path
    obs_path = CONFIG.inat_mapping_obs_path

    build_offline_mapping()
    build_observation_mapping()

    offline_map = load_mapping(offline_path, "offline")
    obs_map = load_mapping(obs_path, "obs")

    merged = merge_mappings(obs_map, offline_map)
    print(
        f"Merged mappings: offline={len(offline_map):,} "
        f"obs={len(obs_map):,} total={len(merged):,}"
    )

    apply_mapping_to_catalog(merged)


if __name__ == "__main__":
    main()
