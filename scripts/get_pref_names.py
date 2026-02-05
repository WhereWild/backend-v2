"""
Fetch iNaturalist preferred_common_name for taxa with inat_id and store in the catalog.

Uses the iNat API /v1/taxa endpoint with batched ids.
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
        if not overwrite and taxon.get("inat_preferred_common_name"):
            continue
        targets.append((taxon_key, inat_id))
        if request_limit and len(targets) >= request_limit:
            break

    print(f"  Taxa needing preferred common names: {len(targets):,}")
    if not targets:
        print("Nothing to do.")
        return

    inat_to_taxa: dict[str, list[str]] = {}
    for taxon_key, inat_id in targets:
        inat_to_taxa.setdefault(inat_id, []).append(taxon_key)
    inat_ids = list(inat_to_taxa.keys())

    updated = 0
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
            preferred = taxon.get("preferred_common_name")
            if not inat_id or not preferred:
                continue
            for taxon_key in inat_to_taxa.get(inat_id, []):
                catalog_taxon = catalog.get(taxon_key)
                if not catalog_taxon:
                    continue
                if not overwrite and catalog_taxon.get("inat_preferred_common_name"):
                    continue
                catalog_taxon["inat_preferred_common_name"] = preferred
                updated += 1

        if requests % progress_every == 0:
            print(
                f"  requests={requests:,} updated={updated:,} "
                f"errors={errors:,} remaining={max(len(inat_ids) - idx - batch_size, 0):,}",
                flush=True,
            )

        time.sleep(1.0 / rate_limit)

    print(f"\nUpdated {updated:,} taxa with inat_preferred_common_name")
    if errors:
        print(f"Errors: {errors:,}")

    with open(catalog_path, "wb") as f:
        import pickle
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved catalog.")


if __name__ == "__main__":
    main()
