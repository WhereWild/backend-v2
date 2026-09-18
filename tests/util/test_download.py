# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import util.download as dl

# ---------------------------------------------------------------------------
# _license_label / _image_fields
# ---------------------------------------------------------------------------


def test_license_label_cc0():
    assert dl._license_label("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0 1.0"


def test_license_label_cc_by():
    assert dl._license_label("https://creativecommons.org/licenses/by-nc/4.0/") == "CC BY-NC 4.0"


def test_license_label_none():
    assert dl._license_label(None) is None


def test_image_fields_prefers_inat_when_present():
    taxon = {
        "inat_preferred_image": "https://inat.example/img.jpg",
        "inat_preferred_image_license": "https://creativecommons.org/licenses/by/4.0/",
        "inat_preferred_image_creator": "Jane",
        "inat_preferred_image_attribution": "Jane Doe",
        "gbif_backup_image": "https://gbif.example/img.jpg",
    }
    fields = dl._image_fields(taxon)
    assert fields["image_url"] == "https://inat.example/img.jpg"
    assert fields["image_license"] == "CC BY 4.0"
    assert fields["image_creator"] == "Jane"
    assert fields["image_rights_holder"] == "Jane Doe"


def test_image_fields_falls_back_to_gbif_when_no_inat_image():
    taxon = {"gbif_backup_image": "https://gbif.example/img.jpg"}
    fields = dl._image_fields(taxon)
    assert fields["image_url"] == "https://gbif.example/img.jpg"
    assert fields["image_license"] is None


# ---------------------------------------------------------------------------
# build_species_archive -- always embeds description_profile + real image
# fields into the archive, unconditionally (unlike the custom-upload path,
# which gates the same thing behind user-chosen options).
# ---------------------------------------------------------------------------


def _taxon(**overrides):
    base = {
        "taxon_key": "123",
        "scientific_name": "Testus taxus",
        "common_name": "Test Taxon",
        "inat_preferred_image": "https://inat.example/img.jpg",
        "inat_preferred_image_license": None,
        "inat_preferred_image_creator": None,
        "inat_preferred_image_attribution": None,
    }
    base.update(overrides)
    return base


def test_build_species_archive_returns_none_when_no_occurrences():
    with patch("util.download.collect_taxon_df", return_value=None):
        result = dl.build_species_archive(_taxon(), storage=MagicMock())
    assert result is None


def test_build_species_archive_embeds_description_and_image():
    df = pd.DataFrame({
        "catalogNumber": ["OBS1"],
        "decimalLatitude": [45.0],
        "decimalLongitude": [-120.0],
    })
    fake_profile = {"sections": [{"id": "climate", "title": "Climates", "lines": []}]}

    with patch("util.download.collect_taxon_df", return_value=df), \
         patch("util.download._build_layer_meta", return_value={}), \
         patch("util.download._build_temporal_var_meta", return_value=[]), \
         patch("util.download._copy_taxon_stats"), \
         patch("util.download._add_ternary_classification_overlay"), \
         patch("util.download._package_archive", return_value=Path("/tmp/fake.zip")), \
         patch("util.download.build_description_profile_for_df", return_value=fake_profile), \
         patch("util.download._add_metadata_to_archive") as mock_add_metadata:
        result = dl.build_species_archive(_taxon(), storage=MagicMock())

    assert result is not None
    archive_path, archive_name, work_dir = result
    try:
        assert archive_path == Path("/tmp/fake.zip")
        mock_add_metadata.assert_called_once()
        _, kwargs = mock_add_metadata.call_args
        assert kwargs["description_profile"] == fake_profile
        assert kwargs["image_url"] == "https://inat.example/img.jpg"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_species_archive_cleans_up_on_failure():
    df = pd.DataFrame({
        "catalogNumber": ["OBS1"],
        "decimalLatitude": [45.0],
        "decimalLongitude": [-120.0],
    })
    with patch("util.download.collect_taxon_df", return_value=df), \
         patch("util.download._build_layer_meta", return_value={}), \
         patch("util.download._build_temporal_var_meta", return_value=[]), \
         patch("util.download._copy_taxon_stats", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            dl.build_species_archive(_taxon(), storage=MagicMock())
