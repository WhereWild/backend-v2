import csv
import io
import json
import pickle
import zipfile
from unittest.mock import MagicMock

import scripts.add_names as an

CATALOG = {
    "2923970": {"taxon_key": "2923970", "scientific_name": "Opuntia_humifusa",
                "rank": "SPECIES", "common_name": "", "inat_id": "55555"},
    "2923968": {"taxon_key": "2923968", "scientific_name": "Opuntia",
                "rank": "GENUS", "common_name": "", "inat_id": ""},
}
PAYLOAD = {"catalog": CATALOG, "combined_name_index": {}}


def _make_dwca(vernacular_rows: list[dict]) -> bytes:
    """Build a fake iNat DWC-A zip with VernacularNames-1.csv."""
    fields = ["id", "vernacularName", "language", "lexicon"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in vernacular_rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, "w") as zf:
        zf.writestr("VernacularNames-1.csv", buf.getvalue())
    return zf_buf.getvalue()


def _make_tsv(rows: list[dict]) -> bytes:
    fields = ["taxonID", "vernacularName", "language", "isPreferredName"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue().encode("utf-8")


def _make_zip_with_file(filename: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def _urlopen_mock_for_zip(zip_bytes: bytes, etag: str = ""):
    """urlopen mock that serves HEAD (content-length) and range GETs from zip_bytes."""
    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        range_hdr = req.get_header("Range") or ""
        if range_hdr:
            start, end = map(int, range_hdr.replace("bytes=", "").split("-"))
            resp.headers = {}
            resp.read.return_value = zip_bytes[start:end + 1]
        else:
            resp.headers = {"Content-Length": str(len(zip_bytes)), "ETag": etag}
            resp.read.return_value = b""
        return resp
    return mock_urlopen


# --- load_inat_vernacular ---

def test_load_inat_vernacular_english():
    dwca = _make_dwca([{"id": "55555", "vernacularName": "Eastern Prickly Pear", "language": "en"}])
    result = an.load_inat_vernacular(dwca)
    assert result == {"55555": "Eastern Prickly Pear"}


def test_load_inat_vernacular_skips_non_english():
    dwca = _make_dwca([{"id": "55555", "vernacularName": "Nopal", "language": "es"}])
    assert an.load_inat_vernacular(dwca) == {}


def test_load_inat_vernacular_accepts_eng():
    dwca = _make_dwca([{"id": "55555", "vernacularName": "Cactus", "language": "eng"}])
    assert an.load_inat_vernacular(dwca) == {"55555": "Cactus"}


def test_load_inat_vernacular_first_wins():
    dwca = _make_dwca([
        {"id": "55555", "vernacularName": "First", "language": "en"},
        {"id": "55555", "vernacularName": "Second", "language": "en"},
    ])
    assert an.load_inat_vernacular(dwca)["55555"] == "First"


def test_load_inat_vernacular_skips_empty():
    dwca = _make_dwca([{"id": "", "vernacularName": "Cactus", "language": "en"}])
    assert an.load_inat_vernacular(dwca) == {}


def test_load_inat_vernacular_skips_non_vernacular_files():
    # Zip contains taxa.csv (should be ignored) + VernacularNames-1.csv
    fields = ["id", "vernacularName", "language", "lexicon"]
    vern_buf = io.StringIO()
    writer = csv.DictWriter(vern_buf, fieldnames=fields)
    writer.writeheader()
    writer.writerow({"id": "55555", "vernacularName": "Cactus", "language": "en", "lexicon": ""})
    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, "w") as zf:
        zf.writestr("taxa.csv", "id,scientificName\n55555,Opuntia humifusa\n")
        zf.writestr("VernacularNames-1.csv", vern_buf.getvalue())
    result = an.load_inat_vernacular(zf_buf.getvalue())
    assert result == {"55555": "Cactus"}


# --- load_gbif_vernacular ---

def test_load_gbif_vernacular_preferred():
    tsv = _make_tsv([{"taxonID": "2923970", "vernacularName": "Prickly Pear",
                      "language": "eng", "isPreferredName": "1"}])
    assert an.load_gbif_vernacular(tsv) == {"2923970": "Prickly Pear"}


def test_load_gbif_vernacular_non_preferred_fallback():
    tsv = _make_tsv([{"taxonID": "2923970", "vernacularName": "Cactus",
                      "language": "eng", "isPreferredName": "0"}])
    assert an.load_gbif_vernacular(tsv) == {"2923970": "Cactus"}


def test_load_gbif_vernacular_preferred_wins():
    tsv = _make_tsv([
        {"taxonID": "2923970", "vernacularName": "Fallback",
         "language": "eng", "isPreferredName": "0"},
        {"taxonID": "2923970", "vernacularName": "Preferred",
         "language": "eng", "isPreferredName": "1"},
    ])
    assert an.load_gbif_vernacular(tsv)["2923970"] == "Preferred"


def test_load_gbif_vernacular_strips_uri():
    tsv = _make_tsv([{"taxonID": "http://www.gbif.org/species/2923970",
                      "vernacularName": "Cactus", "language": "en", "isPreferredName": "1"}])
    assert "2923970" in an.load_gbif_vernacular(tsv)


def test_load_gbif_vernacular_skips_non_english():
    tsv = _make_tsv([{"taxonID": "2923970", "vernacularName": "Nopal",
                      "language": "spa", "isPreferredName": "1"}])
    assert an.load_gbif_vernacular(tsv) == {}


# --- apply_names ---

def test_apply_names_inat_source():
    catalog = {k: dict(v) for k, v in CATALOG.items()}
    updated = an.apply_names(catalog, {"55555": "Prickly Pear"}, {})
    assert updated == 1
    assert catalog["2923970"]["common_name"] == "Prickly Pear"


def test_apply_names_gbif_fallback():
    catalog = {"2923970": {"inat_id": "", "common_name": ""}}
    updated = an.apply_names(catalog, {}, {"2923970": "Prickly Pear"})
    assert updated == 1
    assert catalog["2923970"]["common_name"] == "Prickly Pear"


def test_apply_names_inat_over_gbif():
    catalog = {"2923970": {"inat_id": "55555", "common_name": ""}}
    an.apply_names(catalog, {"55555": "iNat Name"}, {"2923970": "GBIF Name"})
    assert catalog["2923970"]["common_name"] == "iNat Name"


def test_apply_names_no_match():
    catalog = {"2923970": {"inat_id": "", "common_name": ""}}
    updated = an.apply_names(catalog, {}, {})
    assert updated == 0


# --- _extract_file_from_remote_zip ---

def test_extract_file_from_remote_zip(monkeypatch):
    content = b"taxonID\tvernacularName\n1\tfoo\n"
    zip_bytes = _make_zip_with_file("VernacularName.tsv", content)
    monkeypatch.setattr(an, "urlopen", _urlopen_mock_for_zip(zip_bytes))
    result = an._extract_file_from_remote_zip("http://fake/backbone.zip", "VernacularName.tsv")
    assert result == content


def test_extract_file_from_remote_zip_not_found(monkeypatch):
    import pytest
    zip_bytes = _make_zip_with_file("other.tsv", b"data")
    monkeypatch.setattr(an, "urlopen", _urlopen_mock_for_zip(zip_bytes))
    with pytest.raises(FileNotFoundError):
        an._extract_file_from_remote_zip("http://fake/backbone.zip", "VernacularName.tsv")


def test_extract_file_from_remote_zip_no_eocd(monkeypatch):
    import pytest
    # Bytes with no EOCD signature
    monkeypatch.setattr(an, "urlopen", _urlopen_mock_for_zip(b"\x00" * 1024))
    with pytest.raises(ValueError, match="EOCD"):
        an._extract_file_from_remote_zip("http://fake/backbone.zip", "VernacularName.tsv")


# --- fetch_inat_dwca ---

def test_fetch_inat_dwca_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(an, "INAT_DWCA_CACHE", tmp_path / "inat_dwca.zip")
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    cached = b"cached zip"
    (tmp_path / "inat_dwca.zip").write_bytes(cached)
    (tmp_path / "sync_state.json").write_text(json.dumps({"inat_taxonomy": {"etag": '"same"'}}))

    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.headers = {"ETag": '"same"'}
        resp.read.return_value = b""
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    assert an.fetch_inat_dwca() == cached


def test_fetch_inat_dwca_download(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(an, "INAT_DWCA_CACHE", tmp_path / "inat_dwca.zip")
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    new_data = b"new zip data"
    call_count = 0

    def mock_urlopen(req, timeout=30):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.headers = {"ETag": '"new-etag"'}
        resp.read.return_value = b"" if req.get_method() == "HEAD" else new_data
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    result = an.fetch_inat_dwca()
    assert result == new_data
    assert (tmp_path / "inat_dwca.zip").read_bytes() == new_data
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert state["inat_taxonomy"]["etag"] == '"new-etag"'


# --- fetch_backbone_vernacular ---

def test_fetch_backbone_vernacular_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(an, "BACKBONE_VERNACULAR_CACHE", tmp_path / "gbif_vernacular.tsv")
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    cached = b"taxonID\tvernacularName\n"
    (tmp_path / "gbif_vernacular.tsv").write_bytes(cached)
    (tmp_path / "sync_state.json").write_text(json.dumps({"gbif_backbone": {"etag": '"same"'}}))

    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.headers = {"ETag": '"same"'}
        resp.read.return_value = b""
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    assert an.fetch_backbone_vernacular() == cached


def test_fetch_backbone_vernacular_download(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(an, "BACKBONE_VERNACULAR_CACHE", tmp_path / "gbif_vernacular.tsv")
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")

    content = b"taxonID\tvernacularName\n1\tfoo\n"
    zip_bytes = _make_zip_with_file("VernacularName.tsv", content)

    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        range_hdr = req.get_header("Range") or ""
        if range_hdr:
            start, end = map(int, range_hdr.replace("bytes=", "").split("-"))
            resp.headers = {}
            resp.read.return_value = zip_bytes[start:end + 1]
        else:
            resp.headers = {"Content-Length": str(len(zip_bytes)), "ETag": '"new-etag"'}
            resp.read.return_value = b""
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    result = an.fetch_backbone_vernacular()
    assert result == content
    assert (tmp_path / "gbif_vernacular.tsv").read_bytes() == content
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert state["gbif_backbone"]["etag"] == '"new-etag"'


# --- main ---

def test_main(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CATALOG_PATH", tmp_path / "taxon_catalog.pkl")
    monkeypatch.setattr(an, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(an, "INAT_DWCA_CACHE", tmp_path / "inat_dwca.zip")
    monkeypatch.setattr(an, "BACKBONE_VERNACULAR_CACHE", tmp_path / "gbif_vernacular.tsv")
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")

    with open(tmp_path / "taxon_catalog.pkl", "wb") as f:
        pickle.dump(PAYLOAD, f)

    dwca = _make_dwca([{"id": "55555", "vernacularName": "Prickly Pear", "language": "en"}])
    tsv = _make_tsv([])
    monkeypatch.setattr(an, "fetch_inat_dwca", lambda: dwca)
    monkeypatch.setattr(an, "fetch_backbone_vernacular", lambda: tsv)

    an.main()

    with open(tmp_path / "taxon_catalog.pkl", "rb") as f:
        result = pickle.load(f)
    assert result["catalog"]["2923970"]["common_name"] == "Prickly Pear"
