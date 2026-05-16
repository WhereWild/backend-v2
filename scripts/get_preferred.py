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
import sys
import time
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CATALOG_PATH = Path("data/taxonomy/catalog/taxon_catalog.pkl")
CACHE_DIR = Path("data/taxonomy/cache")
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
    req = Request(INAT_DWCA_URL, headers={"User-Agent": _UA})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data) / 1_048_576:.1f} MB")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INAT_DWCA_CACHE.write_bytes(data)
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

def load_inat_vernacular(dwca_bytes: bytes) -> dict[str, str]:
    """Return inat_id -> first English vernacular name found."""
    result: dict[str, str] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    for entry in sorted(zf.namelist()):
        if not (entry.startswith("VernacularNames") and entry.endswith(".csv")):
            continue
        with io.TextIOWrapper(zf.open(entry), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                taxon_id = (row.get("id") or "").strip()
                name = (row.get("vernacularName") or "").strip()
                lang = (row.get("language") or "").strip().lower()
                if taxon_id and name and lang in ("en", "eng") and taxon_id not in result:
                    result[taxon_id] = name
    return result


def load_gbif_vernacular(tsv_bytes: bytes) -> dict[str, str]:
    """Return gbif_taxon_key -> best English vernacular name (preferred names win)."""
    preferred: dict[str, str] = {}
    any_english: dict[str, str] = {}
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
        if is_preferred and taxon_id not in preferred:
            preferred[taxon_id] = name
        elif taxon_id not in any_english:
            any_english[taxon_id] = name
    return {**any_english, **preferred}


# --- catalog update ---

def apply_names(
    catalog: dict,
    inat_map: dict[str, str],
    gbif_map: dict[str, str],
) -> int:
    updated = 0
    for taxon_key, taxon in catalog.items():
        inat_id = str(taxon.get("inat_id") or "").strip()
        name = (inat_map.get(inat_id) if inat_id else None) or gbif_map.get(taxon_key, "")
        if name:
            taxon["common_name"] = name
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
    """Add common_name and inat_preferred_common_name entries missing from the index."""
    catalog = payload["catalog"]
    index = payload["combined_name_index"]
    added = 0
    for taxon_key, taxon in catalog.items():
        for field in ("common_name", "inat_preferred_common_name"):
            raw = str(taxon.get(field) or "").strip()
            if not raw:
                continue
            key = _normalize_index_key(raw)
            if not key:
                continue
            existing = set(index.get(key, []))
            if taxon_key not in existing:
                existing.add(taxon_key)
                index[key] = sorted(existing)
                added += 1
    return added


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

    index_added = update_name_index(payload)
    print(f"Added {index_added:,} new entries to name search index.")

    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated {updated:,} catalog entries with common names.")
    print(f"Updated {names_n:,} preferred common names, {images_n:,} preferred images.")


def rebuild_index() -> None:
    with open(CATALOG_PATH, "rb") as f:
        payload = pickle.load(f)
    added = update_name_index(payload)
    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Added {added:,} new entries to name search index.")


if __name__ == "__main__":  # pragma: no cover
    main()
