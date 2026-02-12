from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fsspec

from util.config import load_config

CONFIG = load_config("global")

TEMPORAL_ROOT = CONFIG.gis_root / "temporal"
DEFAULT_MAX_WORKERS = 4
# Configurable defaults (no CLI): tweak here if needed
START_YEAR = 2010          # download years back to 2010
END_YEAR = None
ALL_MODELS = False         # False => only preferred model per variable
OVERWRITE = False


@dataclass
class FileEntry:
    name: str
    size: int
    uri: str


def _iter_variables(selected_vars: Iterable[str] | None, selected_models: Iterable[str] | None, *, all_models: bool):
    """Yield (variable, models_for_var) based on config and user filters."""
    models_by_var = CONFIG.temporal_models_by_variable
    model_preference = CONFIG.temporal_model_preference
    if selected_vars:
        vars_to_use = [v for v in selected_vars if v in models_by_var]
    else:
        vars_to_use = list(models_by_var.keys())

    for var in vars_to_use:
        models = list(models_by_var[var])
        if selected_models:
            models = [m for m in models if m in selected_models]
        if not models:
            continue
        if not all_models:
            # pick preferred model
            chosen = next((m for m in model_preference if m in models), models[0])
            models = [chosen]
        yield var, models


def _list_remote_files(model: str, variable: str, start_year: int | None, end_year: int | None) -> List[FileEntry]:
    # Derived variable has no source files
    if variable == "weather_code_simple":
        return []
    base = f"s3://openmeteo/data/{model}/{variable}"
    fs = fsspec.filesystem("s3", anon=True)
    names: list[str] = [Path(item.get("name") if isinstance(item, dict) else item).name for item in fs.ls(base)]
    entries: list[FileEntry] = []
    for name in names:
        if not (name.startswith("chunk_") or name.startswith("year_")):
            continue
        if name.startswith("year_"):
            try:
                yr = int(name.replace("year_", "").replace(".om", ""))
            except ValueError:
                continue
            if start_year and yr < start_year:
                continue
            if end_year and yr > end_year:
                continue
        uri = f"{base}/{name}"
        try:
            info = fs.info(uri)
            size = int(info.get("size") or 0)
        except Exception:
            size = 0
        entries.append(FileEntry(name=name, size=size, uri=uri))
    return entries


def _download(entry: FileEntry, dest: Path, overwrite: bool) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        try:
            if dest.stat().st_size == entry.size:
                return False
        except OSError:
            pass
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with fsspec.open(entry.uri, mode="rb", s3={"anon": True}) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        tmp.replace(dest)
        return True
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _write_manifest(dest_dir: Path, entries: List[FileEntry]) -> None:
    manifest = {
        "version": 1,
        "files": [{"name": e.name, "size": e.size} for e in sorted(entries, key=lambda e: e.name)],
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)


def main():
    selected_vars = None
    selected_models = None

    to_download: list[tuple[FileEntry, Path]] = []
    skipped_existing = 0

    for variable, models in _iter_variables(selected_vars, selected_models, all_models=ALL_MODELS):
        for model in models:
            remote_files = _list_remote_files(model, variable, START_YEAR, END_YEAR)
            dest_dir = TEMPORAL_ROOT / model / variable
            for entry in remote_files:
                dest = dest_dir / entry.name
                if dest.exists() and not OVERWRITE:
                    try:
                        if dest.stat().st_size == entry.size:
                            skipped_existing += 1
                            continue
                    except OSError:
                        pass
                to_download.append((entry, dest))
            _write_manifest(dest_dir, remote_files)

    if not to_download:
        print("Nothing to download.")
        return

    total_bytes = sum(entry.size for entry, _ in to_download)
    print(f"Queued {len(to_download)} files (skipped {skipped_existing} existing); total size ~{total_bytes/1e9:.2f} GB; downloading with {DEFAULT_MAX_WORKERS} workers…")
    updated = 0
    with ThreadPoolExecutor(max_workers=max(1, DEFAULT_MAX_WORKERS)) as ex:
        futures = {ex.submit(_download, entry, dest, OVERWRITE): (entry, dest) for entry, dest in to_download}
        for fut in as_completed(futures):
            entry, dest = futures[fut]
            try:
                changed = fut.result()
                if changed:
                    updated += 1
                    mb = entry.size / 1e6 if entry.size else 0
                    print(f"[done] {entry.name} -> {dest} ({mb:.1f} MB)")
            except Exception as exc:
                print(f"[error] {entry.name} -> {dest}: {exc}")
    print(f"Completed. Downloaded/overwrote {updated} files; cached at {TEMPORAL_ROOT}")
    if updated:
        print("Finished files:")
        for entry, dest in to_download:
            if dest.exists():
                print(f"  {dest}")


if __name__ == "__main__":
    main()
