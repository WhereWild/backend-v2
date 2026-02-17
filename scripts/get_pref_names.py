"""
Fetch iNaturalist preferred metadata for taxa with inat_id and store in the catalog.

Currently stores:
- inat_preferred_common_name
- inat_preferred_image (URL from default_photo)
- inat_preferred_image_license
- inat_preferred_image_creator
- inat_preferred_image_attribution
- inat_preferred_image_references
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from util.config import load_config

CONFIG = load_config("global")

INAT_TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa"
INAT_PHOTO_BASE_URL = "https://www.inaturalist.org/photos"


def fetch_taxa_batch(ids: list[str], locale: str, timeout: int) -> list[dict[str, Any]]:
    params = {
        "id": ",".join(ids),
        "locale": locale,
        "per_page": str(len(ids)),
    }
    url = f"{INAT_TAXA_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "wherewild-inat-preferred/1.0"})
    with urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results") or []
    return results if isinstance(results, list) else []


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def extract_preferred_image_metadata(taxon_payload: dict[str, Any]) -> dict[str, str]:
    """Extract best-available image metadata from iNat taxon payload."""
    default_photo = taxon_payload.get("default_photo")
    if not isinstance(default_photo, dict):
        return {}
    image_url = ""
    for field in ("original_url", "large_url", "medium_url", "url", "square_url"):
        value = _clean_text(default_photo.get(field))
        if value:
            image_url = value
            break
    if not image_url:
        return {}
    photo_id = _clean_text(default_photo.get("id"))
    references = f"{INAT_PHOTO_BASE_URL}/{photo_id}" if photo_id else ""
    return {
        "inat_preferred_image": image_url,
        "inat_preferred_image_license": _clean_text(default_photo.get("license_code")),
        "inat_preferred_image_creator": _clean_text(default_photo.get("attribution_name")),
        "inat_preferred_image_attribution": _clean_text(default_photo.get("attribution")),
        "inat_preferred_image_references": references,
    }


def main() -> None:
    catalog_path = CONFIG.taxon_catalog_path
    locale = CONFIG.inat_preferred_common_name_locale or "en"
    batch_size = max(1, min(CONFIG.inat_preferred_common_name_batch_size, 200))
    request_limit = CONFIG.inat_preferred_common_name_request_limit or 0
    progress_every = max(1, CONFIG.inat_preferred_common_name_progress_every)
    overwrite = CONFIG.inat_preferred_common_name_overwrite
    rate_limit = max(getattr(CONFIG, "inat_preferred_common_name_rate_limit_per_second", 1.0), 0.1)
    max_requests = getattr(CONFIG, "inat_preferred_common_name_max_requests", 10_000)
    timeout = getattr(CONFIG, "inat_api_timeout_seconds", 20)

    print(f"Loading catalog from {catalog_path}...")
    with open(catalog_path, "rb") as f:
        import pickle
        payload = pickle.load(f)
    catalog = payload["catalog"]
    print(f"  Catalog taxa: {len(catalog):,}")

    targets: list[tuple[str, str]] = []
    for taxon_key, taxon in catalog.items():
        inat_id = str(taxon.get("inat_id") or "").strip()
        if not inat_id:
            continue
        has_preferred_name = bool(str(taxon.get("inat_preferred_common_name") or "").strip())
        has_preferred_image = bool(str(taxon.get("inat_preferred_image") or "").strip())
        has_image_license = bool(str(taxon.get("inat_preferred_image_license") or "").strip())
        has_image_creator = bool(
            str(taxon.get("inat_preferred_image_creator") or "").strip()
            or str(taxon.get("inat_preferred_image_attribution") or "").strip()
        )
        has_image_reference = bool(str(taxon.get("inat_preferred_image_references") or "").strip())
        if (
            not overwrite
            and has_preferred_name
            and has_preferred_image
            and has_image_license
            and has_image_creator
            and has_image_reference
        ):
            continue
        targets.append((taxon_key, inat_id))
        if request_limit and len(targets) >= request_limit:
            break

    print(f"  Taxa needing preferred iNat metadata: {len(targets):,}")
    if not targets:
        print("Nothing to do.")
        return

    inat_to_taxa: dict[str, list[str]] = {}
    for taxon_key, inat_id in targets:
        inat_to_taxa.setdefault(inat_id, []).append(taxon_key)
    inat_ids = list(inat_to_taxa.keys())

    names_updated = 0
    images_updated = 0
    image_metadata_updated = 0
    errors = 0
    requests = 0

    for idx in range(0, len(inat_ids), batch_size):
        if max_requests and requests >= max_requests:
            print(f"Reached max requests cap: {max_requests}")
            break
        batch = inat_ids[idx : idx + batch_size]
        try:
            results = fetch_taxa_batch(batch, locale=locale, timeout=timeout)
        except Exception:
            errors += 1
            time.sleep(1.0 / rate_limit)
            continue

        requests += 1
        for taxon in results:
            inat_id = str(taxon.get("id") or "").strip()
            preferred_name = str(taxon.get("preferred_common_name") or "").strip()
            image_metadata = extract_preferred_image_metadata(taxon)
            if not inat_id:
                continue
            for taxon_key in inat_to_taxa.get(inat_id, []):
                catalog_taxon = catalog.get(taxon_key)
                if not catalog_taxon:
                    continue
                if preferred_name and (
                    overwrite
                    or not str(catalog_taxon.get("inat_preferred_common_name") or "").strip()
                ):
                    catalog_taxon["inat_preferred_common_name"] = preferred_name
                    names_updated += 1
                if image_metadata:
                    if overwrite or not str(catalog_taxon.get("inat_preferred_image") or "").strip():
                        catalog_taxon["inat_preferred_image"] = image_metadata["inat_preferred_image"]
                        images_updated += 1
                    metadata_changed = False
                    for field in (
                        "inat_preferred_image_license",
                        "inat_preferred_image_creator",
                        "inat_preferred_image_attribution",
                        "inat_preferred_image_references",
                    ):
                        value = image_metadata.get(field, "")
                        if not value:
                            continue
                        if overwrite or not str(catalog_taxon.get(field) or "").strip():
                            catalog_taxon[field] = value
                            metadata_changed = True
                    if metadata_changed:
                        image_metadata_updated += 1

        if requests % progress_every == 0:
            print(
                f"  requests={requests:,} names_updated={names_updated:,} "
                f"images_updated={images_updated:,} "
                f"image_metadata_updated={image_metadata_updated:,} "
                f"errors={errors:,} remaining={max(len(inat_ids) - idx - batch_size, 0):,}",
                flush=True,
            )

        time.sleep(1.0 / rate_limit)

    print(f"\nUpdated preferred common names: {names_updated:,}")
    print(f"Updated preferred images: {images_updated:,}")
    print(f"Updated preferred image metadata: {image_metadata_updated:,}")
    if errors:
        print(f"Errors: {errors:,}")

    with open(catalog_path, "wb") as f:
        import pickle
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved catalog.")


if __name__ == "__main__":
    main()
