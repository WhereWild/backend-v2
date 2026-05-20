"""
Add English common names to taxon_catalog.pkl from iNat DWC-A and GBIF backbone,
then fetch iNat preferred common names and default photo metadata via the iNat API.

Sources:
- iNat DWC-A VernacularNames-*.csv  (matched via inat_id)
- GBIF backbone VernacularName.tsv  (matched via GBIF taxon key, extracted via range requests)
- iNat API /v1/taxa                 (preferred_common_name + default_photo per taxon)

Both file sources are ETag-cached in data/taxonomy/cache/ via data/sync_state.json.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CATALOG_PATH = Path("data/taxonomy/catalog/taxon_catalog.pkl")
CACHE_DIR = Path("data/taxonomy/cache")
OCCURRENCE_PATH = Path("data/occurrences/occurrence.txt")
MULTIMEDIA_PATH = Path("data/occurrences/multimedia.txt")
INAT_DWCA_CACHE = CACHE_DIR / "inat_dwca.zip"
BACKBONE_VERNACULAR_CACHE = CACHE_DIR / "gbif_vernacular.tsv"
SYNC_STATE_PATH = Path("data/sync_state.json")

INAT_DWCA_URL = "https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip"
BACKBONE_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
BACKBONE_VERNACULAR_FILENAME = "VernacularName.tsv"

_UA = "wherewild-add-names/1.0"

csv.field_size_limit(sys.maxsize)


# --- sync state ---

def _load_state() -> dict:
    return json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}


def _save_state(state: dict) -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))


# --- iNat DWC-A ---

def fetch_inat_dwca() -> bytes:
    state = _load_state()
    cached_etag = state.get("inat_taxonomy", {}).get("etag", "")

    print(f"Checking {INAT_DWCA_URL} ...")
    req = Request(INAT_DWCA_URL, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        remote_etag = r.headers.get("ETag", "")

    if remote_etag and remote_etag == cached_etag and INAT_DWCA_CACHE.exists():
        print("  iNat DWC-A: cache up to date")
        return INAT_DWCA_CACHE.read_bytes()

    print(f"Downloading {INAT_DWCA_URL} ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={INAT_DWCA_CACHE.parent}",
            f"--out={INAT_DWCA_CACHE.name}",
            INAT_DWCA_URL,
        ],
        check=True,
    )
    data = INAT_DWCA_CACHE.read_bytes()
    print(f"  Downloaded {len(data) / 1_048_576:.1f} MB")

    if remote_etag:
        state.setdefault("inat_taxonomy", {})["etag"] = remote_etag
        _save_state(state)

    return data


# --- GBIF backbone VernacularName.tsv via range requests ---

def _range_get(url: str, start: int, end: int) -> bytes:
    req = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": _UA})
    with urlopen(req, timeout=60) as r:
        return r.read()


def _extract_file_from_remote_zip(url: str, filename: str) -> bytes:
    """Extract a single file from a remote ZIP using HTTP range requests."""
    req = Request(url, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        content_length = int(r.headers.get("Content-Length", 0))

    tail_size = min(65536, content_length)
    tail = _range_get(url, content_length - tail_size, content_length - 1)

    eocd_rel = tail.rfind(b"PK\x05\x06")
    if eocd_rel == -1:
        raise ValueError("EOCD not found in remote zip")

    cd_size = struct.unpack_from("<I", tail, eocd_rel + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, eocd_rel + 16)[0]

    cd_data = _range_get(url, cd_offset, cd_offset + cd_size - 1)

    pos = 0
    while pos + 46 <= len(cd_data):
        comp_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
        fname_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
        lh_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]
        fname = cd_data[pos + 46:pos + 46 + fname_len].decode("utf-8", errors="replace")

        if fname == filename:
            lh = _range_get(url, lh_offset, lh_offset + 29)
            lh_fname_len = struct.unpack_from("<H", lh, 26)[0]
            lh_extra_len = struct.unpack_from("<H", lh, 28)[0]
            data_start = lh_offset + 30 + lh_fname_len + lh_extra_len
            compressed = _range_get(url, data_start, data_start + comp_size - 1)
            return zlib.decompress(compressed, -15)

        pos += 46 + fname_len + extra_len + comment_len

    raise FileNotFoundError(f"{filename!r} not found in {url}")


def fetch_backbone_vernacular() -> bytes:
    state = _load_state()
    cached_etag = state.get("gbif_backbone", {}).get("etag", "")

    print(f"Checking {BACKBONE_URL} ...")
    req = Request(BACKBONE_URL, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        remote_etag = r.headers.get("ETag", "")

    if remote_etag and remote_etag == cached_etag and BACKBONE_VERNACULAR_CACHE.exists():
        print("  GBIF backbone VernacularName.tsv: cache up to date")
        return BACKBONE_VERNACULAR_CACHE.read_bytes()

    print(f"Fetching {BACKBONE_VERNACULAR_FILENAME} from GBIF backbone...")
    data = _extract_file_from_remote_zip(BACKBONE_URL, BACKBONE_VERNACULAR_FILENAME)
    print(f"  {len(data) / 1_048_576:.1f} MB")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_VERNACULAR_CACHE.write_bytes(data)
    if remote_etag:
        state.setdefault("gbif_backbone", {})["etag"] = remote_etag
        _save_state(state)

    return data


# --- name loading ---

def load_inat_vernacular(dwca_bytes: bytes) -> dict[str, list[str]]:
    """Return inat_id -> all English vernacular names found."""
    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    for entry in sorted(zf.namelist()):
        if not (entry.startswith("VernacularNames") and entry.endswith(".csv")):
            continue
        with io.TextIOWrapper(zf.open(entry), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                taxon_id = (row.get("id") or "").strip()
                name = (row.get("vernacularName") or "").strip()
                lang = (row.get("language") or "").strip().lower()
                if not taxon_id or not name or lang not in ("en", "eng"):
                    continue
                if name not in seen.setdefault(taxon_id, set()):
                    result.setdefault(taxon_id, []).append(name)
                    seen[taxon_id].add(name)
    return result


def load_gbif_vernacular(tsv_bytes: bytes) -> dict[str, list[str]]:
    """Return gbif_taxon_key -> all English vernacular names (preferred names first)."""
    preferred: dict[str, list[str]] = {}
    others: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    reader = csv.DictReader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t")
    for row in reader:
        taxon_id = (row.get("taxonID") or "").strip()
        name = (row.get("vernacularName") or "").strip()
        lang = (row.get("language") or "").strip().lower()
        is_preferred = (row.get("isPreferredName") or "").strip() in ("1", "true")
        if not taxon_id or not name or lang not in ("en", "eng"):
            continue
        if "/" in taxon_id:
            taxon_id = taxon_id.rsplit("/", 1)[-1]
        if name in seen.setdefault(taxon_id, set()):
            continue
        seen[taxon_id].add(name)
        if is_preferred:
            preferred.setdefault(taxon_id, []).append(name)
        else:
            others.setdefault(taxon_id, []).append(name)
    result = {**others}
    for tid, names in preferred.items():
        result[tid] = names + result.get(tid, [])
    return result


# --- catalog update ---

def apply_names(
    catalog: dict,
    inat_map: dict[str, list[str]],
    gbif_map: dict[str, list[str]],
) -> int:
    updated = 0
    for taxon_key, taxon in catalog.items():
        inat_id = str(taxon.get("inat_id") or "").strip()
        inat_names = (inat_map.get(inat_id) if inat_id else None) or []
        gbif_names = gbif_map.get(taxon_key) or []
        # iNat names first (preferred for display), then any GBIF names not already present
        seen = set(inat_names)
        merged = list(inat_names) + [n for n in gbif_names if n not in seen]
        if merged:
            taxon["common_name"] = merged[0]
            taxon["vernacular_names"] = merged
            updated += 1
    return updated


# --- iNat preferred names and images ---

INAT_TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa"
INAT_PHOTO_BASE_URL = "https://www.inaturalist.org/photos"
INAT_BATCH_SIZE = 200
INAT_RATE_LIMIT = 1.0  # requests per second


def fetch_taxa_batch(ids: list[str], timeout: int = 30) -> list[dict]:
    params = {"id": ",".join(ids), "locale": "en", "per_page": str(len(ids))}
    url = f"{INAT_TAXA_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    results = payload.get("results") or []
    return results if isinstance(results, list) else []


def _clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null"} else text


def extract_preferred_image_metadata(taxon_payload: dict) -> dict[str, str]:
    default_photo = taxon_payload.get("default_photo")
    if not isinstance(default_photo, dict):
        return {}
    image_url = ""
    for field in ("original_url", "large_url", "medium_url", "url", "square_url"):
        v = _clean(default_photo.get(field))
        if v:
            image_url = v
            break
    if not image_url:
        return {}
    photo_id = _clean(default_photo.get("id"))
    return {
        "inat_preferred_image": image_url,
        "inat_preferred_image_license": _clean(default_photo.get("license_code")),
        "inat_preferred_image_creator": _clean(default_photo.get("attribution_name")),
        "inat_preferred_image_attribution": _clean(default_photo.get("attribution")),
        "inat_preferred_image_references": f"{INAT_PHOTO_BASE_URL}/{photo_id}" if photo_id else "",
    }


def apply_inat_preferred(
    catalog: dict,
    inat_to_taxa: dict[str, list[str]],
    results: list[dict],
) -> tuple[int, int]:
    names_updated = 0
    images_updated = 0
    for taxon in results:
        inat_id = _clean(taxon.get("id"))
        preferred_name = _clean(taxon.get("preferred_common_name"))
        image_meta = extract_preferred_image_metadata(taxon)
        if not inat_id:
            continue
        for taxon_key in inat_to_taxa.get(inat_id, []):
            entry = catalog.get(taxon_key)
            if not entry:
                continue
            if preferred_name and not _clean(entry.get("inat_preferred_common_name")):
                entry["inat_preferred_common_name"] = preferred_name
                names_updated += 1
            if image_meta and not _clean(entry.get("inat_preferred_image")):
                entry.update(image_meta)
                images_updated += 1
    return names_updated, images_updated


def run_inat_preferred(catalog: dict) -> tuple[int, int]:
    inat_to_taxa: dict[str, list[str]] = {}
    for taxon_key, taxon in catalog.items():
        inat_id = _clean(taxon.get("inat_id"))
        if not inat_id:
            continue
        has_name = bool(_clean(taxon.get("inat_preferred_common_name")))
        has_image = bool(_clean(taxon.get("inat_preferred_image")))
        if has_name and has_image:
            continue
        inat_to_taxa.setdefault(inat_id, []).append(taxon_key)

    inat_ids = list(inat_to_taxa.keys())
    total_batches = (len(inat_ids) + INAT_BATCH_SIZE - 1) // INAT_BATCH_SIZE
    eta_min = total_batches / INAT_RATE_LIMIT / 60
    print(
        f"  Taxa needing iNat preferred metadata: {len(inat_ids):,} "
        f"({total_batches} batches, ~{eta_min:.0f} min)",
        flush=True,
    )
    if not inat_ids:
        print("  Nothing to do.")
        return 0, 0

    names_updated = 0
    images_updated = 0
    errors = 0

    for i in range(0, len(inat_ids), INAT_BATCH_SIZE):
        batch = inat_ids[i : i + INAT_BATCH_SIZE]
        try:
            results = fetch_taxa_batch(batch)
        except Exception as exc:
            print(f"  Batch error: {exc}", flush=True)
            errors += 1
            time.sleep(1.0 / INAT_RATE_LIMIT)
            continue
        n, im = apply_inat_preferred(catalog, inat_to_taxa, results)
        names_updated += n
        images_updated += im
        request_num = i // INAT_BATCH_SIZE + 1
        if request_num % 10 == 0:
            remaining = max(total_batches - request_num, 0)
            print(
                f"  [{request_num:,}/{total_batches:,}] names={names_updated:,} "
                f"images={images_updated:,} errors={errors:,} remaining={remaining:,}",
                flush=True,
            )
        time.sleep(1.0 / INAT_RATE_LIMIT)

    return names_updated, images_updated


def _normalize_index_key(value: str) -> str:
    return " ".join(value.replace("_", " ").lower().split())


def update_name_index(payload: dict) -> int:
    """Add all vernacular/preferred name entries missing from the index."""
    catalog = payload["catalog"]
    index = payload["combined_name_index"]
    added = 0
    for taxon_key, taxon in catalog.items():
        candidates: list[str] = []
        for field in ("common_name", "inat_preferred_common_name"):
            raw = str(taxon.get(field) or "").strip()
            if raw:
                candidates.append(raw)
        for name in taxon.get("vernacular_names") or []:
            raw = str(name).strip()
            if raw:
                candidates.append(raw)
        for raw in candidates:
            key = _normalize_index_key(raw)
            if not key:
                continue
            existing = set(index.get(key, []))
            if taxon_key not in existing:
                existing.add(taxon_key)
                index[key] = sorted(existing)
                added += 1
    return added


# --- GBIF backup images from occurrence DWCA ---

_LICENSE_PRIORITY = [
    ("publicdomain", 0), ("cc0", 0),
    ("/by/4", 1), ("/by/3", 1), ("/by/2", 1), ("cc by ", 1),
    ("/by-sa/", 2), ("cc by-sa", 2),
    ("/by-nc/", 3), ("cc by-nc ", 3),
    ("/by-nc-sa/", 4), ("cc by-nc-sa", 4),
]
_USABLE_LICENSES = {
    "cc0", "cc by", "cc-by", "/by/", "/by-sa/", "/by-nc/", "/by-nc-sa/",
    "publicdomain", "public domain",
}
_BAD_EVIDENCE = {"track", "scat", "feather", "bone", "molt", "hair"}
_OKAY_EVIDENCE = {"gall", "egg", "construction", "leafmine"}
_SUBSPECIES_RANKS = {"SUBSPECIES", "VARIETY", "FORM"}


def _license_score(s: str) -> int:
    n = s.strip().lower()
    for pattern, score in _LICENSE_PRIORITY:
        if pattern in n:
            return score
    return 99


def _is_usable_license(s: str) -> bool:
    n = s.strip().lower()
    return any(p in n for p in _USABLE_LICENSES)


def _image_quality(license_str: str, vitality: str, evidence: str, rcs: str) -> tuple:
    v = 0 if vitality == "alive" else (2 if vitality == "dead" else 1)
    if evidence == "organism":
        e = 0
    elif not evidence:
        e = 1
    elif any(b in evidence for b in _BAD_EVIDENCE):
        e = 3
    elif any(o in evidence for o in _OKAY_EVIDENCE):
        e = 2
    else:
        e = 1
    r = (0 if rcs == "flowers" else
         1 if rcs in {"fruits or seeds", "fruits", "seeds"} else
         2 if rcs == "flower buds" else 3)
    return (v, e, r, _license_score(license_str))


def _build_gbif_to_taxon(catalog_keys: set[str]) -> dict[str, tuple]:
    """Stream occurrence.txt → gbifID: (taxon_key, vitality, evidence, rcs)."""
    mapping: dict[str, tuple] = {}
    rows = 0
    with open(OCCURRENCE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows += 1
            if rows % 1_000_000 == 0:  # pragma: no cover
                print(f"  {rows:,} rows, {len(mapping):,} matched...", flush=True)
            gbif_id = (row.get("gbifID") or "").strip()
            if not gbif_id:
                continue
            rank = (row.get("taxonRank") or "").upper()
            taxon_key = (row.get("taxonKey") or "").strip()
            species_key = (row.get("speciesKey") or "").strip()
            key = taxon_key if rank in _SUBSPECIES_RANKS else (species_key or taxon_key)
            if not key or key not in catalog_keys:
                continue
            vitality = (row.get("vitality") or "").strip().lower()
            rcs = (row.get("reproductiveCondition") or "").strip().lower()
            evidence = ""
            dp = row.get("dynamicProperties") or ""
            if dp:
                try:
                    obj = json.loads(dp)
                    ev = obj.get("evidenceOfPresence", "")
                    evidence = (",".join(ev) if isinstance(ev, list) else (ev or "")).lower()
                except (json.JSONDecodeError, TypeError):
                    pass
            mapping[gbif_id] = (key, vitality, evidence, rcs)
    print(f"  {rows:,} rows scanned, {len(mapping):,} gbifIDs mapped", flush=True)
    return mapping


def _build_gbif_images(gbif_to_taxon: dict[str, tuple]) -> dict[str, dict]:
    """Stream multimedia.txt → taxon_key: best image record."""
    best: dict[str, tuple] = {}
    rows = 0
    with open(MULTIMEDIA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows += 1
            if rows % 5_000_000 == 0:  # pragma: no cover
                print(f"  {rows:,} rows, {len(best):,} taxa covered...", flush=True)
            gbif_id = (row.get("gbifID") or "").strip()
            info = gbif_to_taxon.get(gbif_id)
            if not info:
                continue
            taxon_key, vitality, evidence, rcs = info
            license_str = row.get("license") or ""
            if not _is_usable_license(license_str):
                continue
            media_type = (row.get("type") or "").strip().lower()
            media_format = (row.get("format") or "").strip().lower()
            if not (media_type in {"stillimage", "image"} or media_format.startswith("image/")):
                continue
            url = (row.get("identifier") or "").strip()
            if not url:
                continue
            score = _image_quality(license_str, vitality, evidence, rcs)
            if taxon_key in best and score >= best[taxon_key][0]:
                continue
            best[taxon_key] = (score, {
                "gbif_backup_image": url,
                "gbif_backup_image_license": license_str,
                "gbif_backup_image_creator": _clean(row.get("creator")),
                "gbif_backup_image_attribution": _clean(row.get("rightsHolder")),
                "gbif_backup_image_references": _clean(row.get("references")),
            })
    print(f"  {rows:,} rows scanned, {len(best):,} taxa with images", flush=True)
    return {k: v[1] for k, v in best.items()}


def run_gbif_backup(catalog: dict) -> int:
    """Apply best GBIF occurrence image to every catalog taxon that has one."""
    if not OCCURRENCE_PATH.exists() or not MULTIMEDIA_PATH.exists():
        print("  Occurrence data not yet downloaded, skipping.", flush=True)
        return 0
    gbif_to_taxon = _build_gbif_to_taxon(set(catalog.keys()))
    taxon_images = _build_gbif_images(gbif_to_taxon)
    updated = 0
    for taxon_key, fields in taxon_images.items():
        entry = catalog.get(taxon_key)
        if entry is not None:
            entry.update(fields)
            updated += 1
    return updated


def main() -> None:
    dwca_bytes = fetch_inat_dwca()
    vernacular_bytes = fetch_backbone_vernacular()

    print("Loading vernacular names...")
    inat_map = load_inat_vernacular(dwca_bytes)
    print(f"  iNat: {len(inat_map):,} English names")
    gbif_map = load_gbif_vernacular(vernacular_bytes)
    print(f"  GBIF: {len(gbif_map):,} English names")

    with open(CATALOG_PATH, "rb") as f:
        payload = pickle.load(f)
    catalog = payload["catalog"]
    print(f"  Catalog: {len(catalog):,} taxa")

    updated = apply_names(catalog, inat_map, gbif_map)

    print("Fetching iNat preferred names and images...")
    names_n, images_n = run_inat_preferred(catalog)

    print("Fetching GBIF backup images from occurrence data...")
    backup_n = run_gbif_backup(catalog)

    index_added = update_name_index(payload)
    print(f"Added {index_added:,} new entries to name search index.")

    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated {updated:,} catalog entries with common names.")
    print(f"Updated {names_n:,} preferred common names, {images_n:,} preferred images, {backup_n:,} GBIF backup images.")


def rebuild_index() -> None:
    with open(CATALOG_PATH, "rb") as f:
        payload = pickle.load(f)
    added = update_name_index(payload)
    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Added {added:,} new entries to name search index.")


if __name__ == "__main__":  # pragma: no cover
    main()
