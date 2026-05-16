"""
Add English common names to taxon_catalog.pkl from iNat DWC-A and GBIF backbone.

Sources:
- iNat DWC-A VernacularNames-*.csv  (matched via inat_id)
- GBIF backbone VernacularName.tsv  (matched via GBIF taxon key, extracted via range requests)

Both sources are ETag-cached in data/taxonomy/cache/ via data/sync_state.json.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import struct
import sys
import zipfile
import zlib
from pathlib import Path
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

    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated {updated:,} catalog entries with common names.")


if __name__ == "__main__":  # pragma: no cover
    main()
