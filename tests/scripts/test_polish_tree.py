import csv
import io
import json
import pickle
import zipfile
from unittest.mock import MagicMock

import scripts.polish_tree as an

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
    assert result == {"55555": ["Eastern Prickly Pear"]}


def test_load_inat_vernacular_skips_non_english():
    dwca = _make_dwca([{"id": "55555", "vernacularName": "Nopal", "language": "es"}])
    assert an.load_inat_vernacular(dwca) == {}


def test_load_inat_vernacular_accepts_eng():
    dwca = _make_dwca([{"id": "55555", "vernacularName": "Cactus", "language": "eng"}])
    assert an.load_inat_vernacular(dwca) == {"55555": ["Cactus"]}


def test_load_inat_vernacular_collects_all():
    dwca = _make_dwca([
        {"id": "55555", "vernacularName": "First", "language": "en"},
        {"id": "55555", "vernacularName": "Second", "language": "en"},
    ])
    assert an.load_inat_vernacular(dwca)["55555"] == ["First", "Second"]


def test_load_inat_vernacular_deduplicates():
    dwca = _make_dwca([
        {"id": "55555", "vernacularName": "Cactus", "language": "en"},
        {"id": "55555", "vernacularName": "Cactus", "language": "en"},
    ])
    assert an.load_inat_vernacular(dwca)["55555"] == ["Cactus"]


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
    assert result == {"55555": ["Cactus"]}


# --- load_gbif_vernacular ---

def test_load_gbif_vernacular_preferred():
    tsv = _make_tsv([{"taxonID": "2923970", "vernacularName": "Prickly Pear",
                      "language": "eng", "isPreferredName": "1"}])
    assert an.load_gbif_vernacular(tsv) == {"2923970": ["Prickly Pear"]}


def test_load_gbif_vernacular_non_preferred_fallback():
    tsv = _make_tsv([{"taxonID": "2923970", "vernacularName": "Cactus",
                      "language": "eng", "isPreferredName": "0"}])
    assert an.load_gbif_vernacular(tsv) == {"2923970": ["Cactus"]}


def test_load_gbif_vernacular_preferred_first():
    tsv = _make_tsv([
        {"taxonID": "2923970", "vernacularName": "Fallback",
         "language": "eng", "isPreferredName": "0"},
        {"taxonID": "2923970", "vernacularName": "Preferred",
         "language": "eng", "isPreferredName": "1"},
    ])
    names = an.load_gbif_vernacular(tsv)["2923970"]
    assert names[0] == "Preferred"
    assert "Fallback" in names


def test_load_gbif_vernacular_collects_all():
    tsv = _make_tsv([
        {"taxonID": "2923970", "vernacularName": "Brittle Pricklypear",
         "language": "en", "isPreferredName": "0"},
        {"taxonID": "2923970", "vernacularName": "Fragile Cactus",
         "language": "en", "isPreferredName": "0"},
    ])
    names = an.load_gbif_vernacular(tsv)["2923970"]
    assert "Brittle Pricklypear" in names
    assert "Fragile Cactus" in names


def test_load_gbif_vernacular_deduplicates():
    tsv = _make_tsv([
        {"taxonID": "2923970", "vernacularName": "Cactus", "language": "en", "isPreferredName": "0"},
        {"taxonID": "2923970", "vernacularName": "Cactus", "language": "en", "isPreferredName": "0"},
    ])
    assert an.load_gbif_vernacular(tsv)["2923970"] == ["Cactus"]


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
    updated = an.apply_names(catalog, {"55555": ["Prickly Pear"]}, {})
    assert updated == 1
    assert catalog["2923970"]["common_name"] == "Prickly Pear"
    assert catalog["2923970"]["vernacular_names"] == ["Prickly Pear"]


def test_apply_names_gbif_fallback():
    catalog = {"2923970": {"inat_id": "", "common_name": ""}}
    updated = an.apply_names(catalog, {}, {"2923970": ["Prickly Pear", "Fragile Cactus"]})
    assert updated == 1
    assert catalog["2923970"]["common_name"] == "Prickly Pear"
    assert catalog["2923970"]["vernacular_names"] == ["Prickly Pear", "Fragile Cactus"]


def test_apply_names_inat_over_gbif():
    catalog = {"2923970": {"inat_id": "55555", "common_name": ""}}
    an.apply_names(catalog, {"55555": ["iNat Name"]}, {"2923970": ["GBIF Name"]})
    assert catalog["2923970"]["common_name"] == "iNat Name"
    assert catalog["2923970"]["vernacular_names"] == ["iNat Name", "GBIF Name"]


def test_apply_names_merges_inat_and_gbif():
    catalog = {"2923970": {"inat_id": "55555", "common_name": ""}}
    an.apply_names(
        catalog,
        {"55555": ["Brittle Pricklypear"]},
        {"2923970": ["Brittle Pricklypear", "Fragile Cactus"]},
    )
    assert catalog["2923970"]["common_name"] == "Brittle Pricklypear"
    assert catalog["2923970"]["vernacular_names"] == ["Brittle Pricklypear", "Fragile Cactus"]


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
    cache_path = tmp_path / "inat_dwca.zip"
    monkeypatch.setattr(an, "INAT_DWCA_CACHE", cache_path)
    monkeypatch.setattr(an, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    new_data = b"new zip data"

    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.headers = {"ETag": '"new-etag"'}
        resp.read.return_value = b""
        return resp

    def fake_aria2c(args, **kwargs):
        cache_path.write_bytes(new_data)

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    monkeypatch.setattr(an.subprocess, "run", fake_aria2c)
    result = an.fetch_inat_dwca()
    assert result == new_data
    assert cache_path.read_bytes() == new_data
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


# --- fetch_taxa_batch ---

def test_fetch_taxa_batch(monkeypatch):
    payload = {"results": [{"id": "55555", "preferred_common_name": "Prickly Pear"}]}

    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    result = an.fetch_taxa_batch(["55555"])
    assert result == [{"id": "55555", "preferred_common_name": "Prickly Pear"}]


def test_fetch_taxa_batch_empty_results(monkeypatch):
    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = json.dumps({"results": []}).encode("utf-8")
        return resp

    monkeypatch.setattr(an, "urlopen", mock_urlopen)
    assert an.fetch_taxa_batch(["99999"]) == []


# --- extract_preferred_image_metadata ---

_PHOTO = {
    "id": "123",
    "medium_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/123/medium.jpg",
    "license_code": "cc-by-nc",
    "attribution_name": "Alice",
    "attribution": "(c) Alice, some rights reserved",
}


def test_extract_preferred_image_metadata_full():
    result = an.extract_preferred_image_metadata({"default_photo": _PHOTO})
    assert result["inat_preferred_image"] == _PHOTO["medium_url"]
    assert result["inat_preferred_image_license"] == "cc-by-nc"
    assert result["inat_preferred_image_creator"] == "Alice"
    assert result["inat_preferred_image_references"] == "https://www.inaturalist.org/photos/123"


def test_extract_preferred_image_metadata_no_photo():
    assert an.extract_preferred_image_metadata({}) == {}


def test_extract_preferred_image_metadata_no_url():
    assert an.extract_preferred_image_metadata({"default_photo": {"id": "1"}}) == {}


def test_extract_preferred_image_metadata_prefers_original():
    photo = {**_PHOTO, "original_url": "https://example.com/original.jpg"}
    result = an.extract_preferred_image_metadata({"default_photo": photo})
    assert result["inat_preferred_image"] == "https://example.com/original.jpg"


def test_extract_preferred_image_metadata_no_photo_id():
    photo = {k: v for k, v in _PHOTO.items() if k != "id"}
    result = an.extract_preferred_image_metadata({"default_photo": photo})
    assert result["inat_preferred_image_references"] == ""


# --- apply_inat_preferred ---

_EMPTY_ENTRY = {"inat_id": "55555", "inat_preferred_common_name": "", "inat_preferred_image": ""}


def test_apply_inat_preferred_name():
    catalog = {"2923970": dict(_EMPTY_ENTRY)}
    inat_to_taxa = {"55555": ["2923970"]}
    results = [{"id": "55555", "preferred_common_name": "Prickly Pear"}]
    n, im = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert n == 1
    assert catalog["2923970"]["inat_preferred_common_name"] == "Prickly Pear"
    assert im == 0


def test_apply_inat_preferred_image():
    catalog = {"2923970": {"inat_preferred_common_name": "", "inat_preferred_image": ""}}
    inat_to_taxa = {"55555": ["2923970"]}
    results = [{"id": "55555", "preferred_common_name": "", "default_photo": _PHOTO}]
    n, im = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert im == 1
    assert catalog["2923970"]["inat_preferred_image"] == _PHOTO["medium_url"]


def test_apply_inat_preferred_skips_existing_name():
    catalog = {"2923970": {"inat_preferred_common_name": "Existing", "inat_preferred_image": ""}}
    inat_to_taxa = {"55555": ["2923970"]}
    results = [{"id": "55555", "preferred_common_name": "New Name"}]
    n, _ = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert n == 0
    assert catalog["2923970"]["inat_preferred_common_name"] == "Existing"


def test_apply_inat_preferred_skips_existing_image():
    catalog = {"2923970": {
        "inat_preferred_common_name": "",
        "inat_preferred_image": "https://example.com/img.jpg",
    }}
    inat_to_taxa = {"55555": ["2923970"]}
    results = [{"id": "55555", "preferred_common_name": "", "default_photo": _PHOTO}]
    _, im = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert im == 0


def test_apply_inat_preferred_no_inat_id():
    catalog = {}
    inat_to_taxa = {}
    results = [{"preferred_common_name": "Name"}]  # no "id"
    n, im = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert n == 0
    assert im == 0


def test_apply_inat_preferred_missing_catalog_entry():
    catalog = {}  # taxon_key not in catalog
    inat_to_taxa = {"55555": ["2923970"]}
    results = [{"id": "55555", "preferred_common_name": "Name"}]
    n, im = an.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert n == 0
    assert im == 0


# --- run_inat_preferred ---

def test_run_inat_preferred_nothing_to_do():
    catalog = {"1": {"inat_id": "", "inat_preferred_common_name": "", "inat_preferred_image": ""}}
    n, im = an.run_inat_preferred(catalog)
    assert n == 0
    assert im == 0


def test_run_inat_preferred_skips_complete():
    catalog = {"1": {
        "inat_id": "55555",
        "inat_preferred_common_name": "Prickly Pear",
        "inat_preferred_image": "https://example.com/img.jpg",
    }}
    n, im = an.run_inat_preferred(catalog)
    assert n == 0
    assert im == 0


def test_run_inat_preferred_fetches_and_applies(monkeypatch):
    catalog = {"2923970": dict(_EMPTY_ENTRY)}
    monkeypatch.setattr(an, "fetch_taxa_batch", lambda ids, timeout=30: [
        {"id": "55555", "preferred_common_name": "Prickly Pear", "default_photo": _PHOTO}
    ])
    monkeypatch.setattr(an.time, "sleep", lambda _: None)
    n, im = an.run_inat_preferred(catalog)
    assert n == 1
    assert im == 1
    assert catalog["2923970"]["inat_preferred_common_name"] == "Prickly Pear"


def test_run_inat_preferred_progress_print(monkeypatch, capsys):
    catalog = {
        str(i): {"inat_id": str(i), "inat_preferred_common_name": "", "inat_preferred_image": ""}
        for i in range(1, 2002)
    }
    monkeypatch.setattr(an, "INAT_BATCH_SIZE", 200)
    monkeypatch.setattr(an, "fetch_taxa_batch", lambda ids, timeout=30: [])
    monkeypatch.setattr(an.time, "sleep", lambda _: None)
    an.run_inat_preferred(catalog)
    out = capsys.readouterr().out
    assert "[10/" in out


def test_run_inat_preferred_error_continues(monkeypatch):
    catalog = {"2923970": dict(_EMPTY_ENTRY)}
    call_count = 0

    def bad_fetch(ids, timeout=30):
        nonlocal call_count
        call_count += 1
        raise OSError("network error")

    monkeypatch.setattr(an, "fetch_taxa_batch", bad_fetch)
    monkeypatch.setattr(an.time, "sleep", lambda _: None)
    n, im = an.run_inat_preferred(catalog)
    assert call_count == 1
    assert n == 0
    assert im == 0


# --- rebuild_index ---

def test_rebuild_index(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "CATALOG_PATH", tmp_path / "taxon_catalog.pkl")
    payload = {
        "catalog": {"2923970": {"common_name": "Prickly Pear", "inat_preferred_common_name": ""}},
        "combined_name_index": {},
    }
    with open(tmp_path / "taxon_catalog.pkl", "wb") as f:
        pickle.dump(payload, f)
    an.rebuild_index()
    with open(tmp_path / "taxon_catalog.pkl", "rb") as f:
        result = pickle.load(f)
    assert "2923970" in result["combined_name_index"]["prickly pear"]


# --- update_name_index ---

def test_update_name_index_adds_common_name():
    payload = {
        "catalog": {"2923970": {"common_name": "Prickly Pear", "inat_preferred_common_name": ""}},
        "combined_name_index": {},
    }
    added = an.update_name_index(payload)
    assert added == 1
    assert "2923970" in payload["combined_name_index"]["prickly pear"]


def test_update_name_index_adds_preferred_name():
    payload = {
        "catalog": {
            "2923970": {"common_name": "", "inat_preferred_common_name": "Eastern Prickly Pear"},
        },
        "combined_name_index": {},
    }
    added = an.update_name_index(payload)
    assert added == 1
    assert "2923970" in payload["combined_name_index"]["eastern prickly pear"]


def test_update_name_index_adds_vernacular_names():
    payload = {
        "catalog": {"2923970": {
            "common_name": "Brittle Pricklypear",
            "inat_preferred_common_name": "",
            "vernacular_names": ["Brittle Pricklypear", "Fragile Cactus"],
        }},
        "combined_name_index": {},
    }
    added = an.update_name_index(payload)
    assert "2923970" in payload["combined_name_index"]["fragile cactus"]
    assert "2923970" in payload["combined_name_index"]["brittle pricklypear"]
    # "Brittle Pricklypear" appears in both common_name and vernacular_names — counted once
    assert added == 2


def test_update_name_index_skips_existing():
    payload = {
        "catalog": {"2923970": {"common_name": "Prickly Pear", "inat_preferred_common_name": ""}},
        "combined_name_index": {"prickly pear": ["2923970"]},
    }
    added = an.update_name_index(payload)
    assert added == 0


def test_update_name_index_skips_empty():
    payload = {
        "catalog": {"2923970": {"common_name": "", "inat_preferred_common_name": ""}},
        "combined_name_index": {},
    }
    assert an.update_name_index(payload) == 0


def test_update_name_index_skips_normalizes_to_empty():
    # "_" passes the raw strip check but normalizes to "" via replace("_", " ").split()
    payload = {
        "catalog": {"2923970": {"common_name": "_", "inat_preferred_common_name": ""}},
        "combined_name_index": {},
    }
    assert an.update_name_index(payload) == 0


# --- GBIF backup image helpers ---

def test_license_score_public_domain():
    assert an._license_score("publicdomain") == 0

def test_license_score_cc_by():
    assert an._license_score("https://creativecommons.org/licenses/by/4.0/") == 1

def test_license_score_cc_by_nc():
    assert an._license_score("CC BY-NC 4.0") == 3

def test_license_score_unknown():
    assert an._license_score("all rights reserved") == 99

def test_license_score_empty():
    assert an._license_score("") == 99


def test_is_usable_license_cc_by():
    assert an._is_usable_license("cc by 4.0")

def test_is_usable_license_rejects_unknown():
    assert not an._is_usable_license("all rights reserved")

def test_is_usable_license_empty():
    assert not an._is_usable_license("")


def test_image_quality_alive_organism_flowers():
    score = an._image_quality("cc0", "alive", "organism", "flowers")
    assert score == (0, 0, 0, 0)

def test_image_quality_dead_penalised():
    dead = an._image_quality("cc0", "dead", "organism", "flowers")
    alive = an._image_quality("cc0", "alive", "organism", "flowers")
    assert dead > alive

def test_image_quality_bad_evidence_penalised():
    bad = an._image_quality("cc0", "alive", "track", "flowers")
    good = an._image_quality("cc0", "alive", "organism", "flowers")
    assert bad > good

def test_image_quality_license_tiebreaker():
    pd_score = an._image_quality("cc0", "alive", "organism", "")
    nc_score = an._image_quality("cc by-nc 4.0", "alive", "organism", "")
    assert pd_score < nc_score

def test_image_quality_okay_evidence():
    okay = an._image_quality("cc0", "alive", "gall", "")
    bad = an._image_quality("cc0", "alive", "track", "")
    good = an._image_quality("cc0", "alive", "organism", "")
    assert good < okay < bad

def test_image_quality_unknown_evidence_fallthrough():
    unknown = an._image_quality("cc0", "alive", "someunknownthing", "")
    missing = an._image_quality("cc0", "alive", "", "")
    assert unknown[1] == missing[1] == 1


def _make_occurrence_tsv(rows: list[dict]) -> str:
    fields = ["gbifID", "taxonRank", "taxonKey", "speciesKey",
              "vitality", "reproductiveCondition", "dynamicProperties"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue()


def _make_multimedia_tsv(rows: list[dict]) -> str:
    fields = ["gbifID", "type", "format", "identifier",
              "license", "creator", "rightsHolder", "references"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue()


def test_build_gbif_to_taxon(tmp_path):
    occ = _make_occurrence_tsv([
        {"gbifID": "1", "taxonRank": "SPECIES", "taxonKey": "999", "speciesKey": "2923970",
         "vitality": "alive", "reproductiveCondition": "flowers", "dynamicProperties": ""},
        {"gbifID": "2", "taxonRank": "SPECIES", "taxonKey": "888", "speciesKey": "UNKNOWN",
         "vitality": "", "reproductiveCondition": "", "dynamicProperties": ""},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    an.OCCURRENCE_PATH = tmp_path / "occurrence.txt"
    mapping = an._build_gbif_to_taxon({"2923970"})
    assert "1" in mapping
    assert mapping["1"] == ("2923970", "alive", "", "flowers")
    assert "2" not in mapping  # speciesKey not in catalog


def test_build_gbif_to_taxon_subspecies_uses_taxon_key(tmp_path):
    occ = _make_occurrence_tsv([
        {"gbifID": "1", "taxonRank": "SUBSPECIES", "taxonKey": "2923970", "speciesKey": "111",
         "vitality": "", "reproductiveCondition": "", "dynamicProperties": ""},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    an.OCCURRENCE_PATH = tmp_path / "occurrence.txt"
    mapping = an._build_gbif_to_taxon({"2923970"})
    assert mapping["1"][0] == "2923970"


def test_build_gbif_to_taxon_skips_empty_gbif_id(tmp_path):
    occ = _make_occurrence_tsv([
        {"gbifID": "", "taxonRank": "SPECIES", "taxonKey": "9", "speciesKey": "2923970"},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    an.OCCURRENCE_PATH = tmp_path / "occurrence.txt"
    assert an._build_gbif_to_taxon({"2923970"}) == {}


def test_build_gbif_to_taxon_bad_dynamic_properties_json(tmp_path):
    occ = _make_occurrence_tsv([
        {"gbifID": "1", "taxonRank": "SPECIES", "taxonKey": "9", "speciesKey": "2923970",
         "vitality": "", "reproductiveCondition": "", "dynamicProperties": "{notjson}"},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    an.OCCURRENCE_PATH = tmp_path / "occurrence.txt"
    mapping = an._build_gbif_to_taxon({"2923970"})
    assert mapping["1"][2] == ""  # evidence falls back to empty string


def test_build_gbif_to_taxon_evidence_from_dynamic_properties(tmp_path):
    dp = json.dumps({"evidenceOfPresence": "organism"})
    occ = _make_occurrence_tsv([
        {"gbifID": "1", "taxonRank": "SPECIES", "taxonKey": "9", "speciesKey": "2923970",
         "vitality": "", "reproductiveCondition": "", "dynamicProperties": dp},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    an.OCCURRENCE_PATH = tmp_path / "occurrence.txt"
    mapping = an._build_gbif_to_taxon({"2923970"})
    assert mapping["1"][2] == "organism"


def test_build_gbif_images_picks_best(tmp_path):
    gbif_to_taxon = {
        "1": ("2923970", "alive", "organism", "flowers"),
        "2": ("2923970", "dead", "", ""),
    }
    mm = _make_multimedia_tsv([
        {"gbifID": "2", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://bad.com/img.jpg", "license": "cc by 4.0",
         "creator": "Bob", "rightsHolder": "Bob", "references": ""},
        {"gbifID": "1", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://good.com/img.jpg", "license": "cc0",
         "creator": "Alice", "rightsHolder": "Alice", "references": "https://example.com"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    result = an._build_gbif_images(gbif_to_taxon)
    assert result["2923970"]["gbif_backup_image"] == "https://good.com/img.jpg"
    assert result["2923970"]["gbif_backup_image_license"] == "cc0"
    assert result["2923970"]["gbif_backup_image_creator"] == "Alice"
    assert result["2923970"]["gbif_backup_image_attribution"] == "Alice"
    assert result["2923970"]["gbif_backup_image_references"] == "https://example.com"


def test_build_gbif_images_skips_bad_license(tmp_path):
    gbif_to_taxon = {"1": ("2923970", "alive", "organism", "")}
    mm = _make_multimedia_tsv([
        {"gbifID": "1", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://example.com/img.jpg", "license": "all rights reserved"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    assert an._build_gbif_images(gbif_to_taxon) == {}


def test_build_gbif_images_skips_unknown_gbif_id(tmp_path):
    mm = _make_multimedia_tsv([
        {"gbifID": "999", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://example.com/img.jpg", "license": "cc0"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    assert an._build_gbif_images({"1": ("2923970", "alive", "organism", "")}) == {}


def test_build_gbif_images_skips_empty_url(tmp_path):
    gbif_to_taxon = {"1": ("2923970", "alive", "organism", "")}
    mm = _make_multimedia_tsv([
        {"gbifID": "1", "type": "StillImage", "format": "image/jpeg",
         "identifier": "", "license": "cc0"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    assert an._build_gbif_images(gbif_to_taxon) == {}


def test_build_gbif_images_keeps_better_skips_worse(tmp_path):
    gbif_to_taxon = {
        "1": ("2923970", "alive", "organism", "flowers"),
        "2": ("2923970", "dead", "", ""),
    }
    mm = _make_multimedia_tsv([
        {"gbifID": "1", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://good.com/img.jpg", "license": "cc0"},
        {"gbifID": "2", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://worse.com/img.jpg", "license": "cc by-nc 4.0"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    result = an._build_gbif_images(gbif_to_taxon)
    assert result["2923970"]["gbif_backup_image"] == "https://good.com/img.jpg"


def test_build_gbif_images_skips_non_image(tmp_path):
    gbif_to_taxon = {"1": ("2923970", "alive", "organism", "")}
    mm = _make_multimedia_tsv([
        {"gbifID": "1", "type": "Sound", "format": "audio/mp3",
         "identifier": "https://example.com/sound.mp3", "license": "cc0"},
    ])
    (tmp_path / "multimedia.txt").write_text(mm)
    an.MULTIMEDIA_PATH = tmp_path / "multimedia.txt"
    assert an._build_gbif_images(gbif_to_taxon) == {}


def test_run_gbif_backup_skips_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "OCCURRENCE_PATH", tmp_path / "occurrence.txt")
    monkeypatch.setattr(an, "MULTIMEDIA_PATH", tmp_path / "multimedia.txt")
    catalog = {"2923970": {}}
    assert an.run_gbif_backup(catalog) == 0
    assert "gbif_backup_image" not in catalog["2923970"]


def test_run_gbif_backup_applies_images(monkeypatch, tmp_path):
    occ = _make_occurrence_tsv([
        {"gbifID": "1", "taxonRank": "SPECIES", "taxonKey": "9", "speciesKey": "2923970",
         "vitality": "alive", "reproductiveCondition": "flowers", "dynamicProperties": ""},
    ])
    mm = _make_multimedia_tsv([
        {"gbifID": "1", "type": "StillImage", "format": "image/jpeg",
         "identifier": "https://example.com/img.jpg", "license": "cc0",
         "creator": "Alice", "rightsHolder": "Alice", "references": ""},
    ])
    (tmp_path / "occurrence.txt").write_text(occ)
    (tmp_path / "multimedia.txt").write_text(mm)
    monkeypatch.setattr(an, "OCCURRENCE_PATH", tmp_path / "occurrence.txt")
    monkeypatch.setattr(an, "MULTIMEDIA_PATH", tmp_path / "multimedia.txt")
    catalog = {"2923970": {}}
    n = an.run_gbif_backup(catalog)
    assert n == 1
    assert catalog["2923970"]["gbif_backup_image"] == "https://example.com/img.jpg"


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
    monkeypatch.setattr(an, "run_inat_preferred", lambda catalog: (0, 0))
    monkeypatch.setattr(an, "run_gbif_backup", lambda catalog: 0)

    an.main()

    with open(tmp_path / "taxon_catalog.pkl", "rb") as f:
        result = pickle.load(f)
    assert result["catalog"]["2923970"]["common_name"] == "Prickly Pear"
    assert "2923970" in result["combined_name_index"]["prickly pear"]
