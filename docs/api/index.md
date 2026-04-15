# API

This section documents the backend HTTP API. The public interface is defined in
`main.py`, which exposes FastAPI routes that wrap the core `util/` libraries.

This page is a curated overview so the API documentation lives alongside the library and script docs. The default FastAPI page may still contain useful information not present here.

## Feature Docs

- [Upload Processing](upload-processing.md): processed upload ZIP behavior,
  artifact contracts, and frontend integration guidance.

## Canonical Search Endpoint

Use `GET /api/taxa/query` for taxon search.

Use `GET /api/taxa/ranking-options` when the client needs the valid `(variable, metric)` ranking combinations for a scoped ancestor and descendant rank before issuing a ranked query.

Frontend clients should prefer passing taxon ids for `within_taxon`. Scientific-name slugs are a convenience path only, and are valid only when they resolve unambiguously to a single taxon.

- Scoped ranking options by id: `GET /api/taxa/ranking-options?within_taxon=2519&descendant_rank=SPECIES`
- Scoped ranking options by slug: `GET /api/taxa/ranking-options?within_taxon=quercus&descendant_rank=SPECIES`

- Plain text search: `GET /api/taxa/query?q=oak`
- Plain text search with filters: `GET /api/taxa/query?q=oak&location=ETH&min_samples=10`
- Plain text search scoped to a subtree: `GET /api/taxa/query?q=oak&within_taxon=2519&descendant_rank=SPECIES`
- Plain text search scoped by slug: `GET /api/taxa/query?q=oak&within_taxon=quercus&descendant_rank=SPECIES`
- Text search with direct sorting: `GET /api/taxa/query?q=oak&sort_variable=bio_1&sort_metric=mean`
- Scoped ranked search: `GET /api/taxa/query?q=oak&within_taxon=2519&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean`
- Scoped sort-only query: `GET /api/taxa/query?within_taxon=2519&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean`

Supported query parameters:

- `q`: Optional scientific or common-name search term.
- `within_taxon`: Optional ancestor taxon reference. Prefer a taxon id. Scientific-name slugs are a convenience path only when they resolve unambiguously.
- `descendant_rank`: Optional descendant rank used with `within_taxon`.
- `sort_variable`: Optional environmental variable id used for sorting.
- `sort_metric`: Optional metric name used with `sort_variable`.
- `sort_order`: Optional sort direction, `asc` or `desc`.
- `location`: Optional location GID used to filter results by descendant presence in that location. Higher-rank taxa remain eligible when descendant taxa occur there.
- `min_samples`: Optional sample-count threshold.
- `include_species_like`: When `descendant_rank=SPECIES`, includes subspecies-like ranks.
- `limit`: Page size.
- `offset`: Pagination offset.
- `unit_system`: Optional display unit system for sorted values.

Ranking-option responses include:

- `ancestor_taxon_id`: Resolved ancestor taxon id for the scope.
- `rank`: Canonical descendant rank for the scope.
- `options`: Available ranking combinations for that scope only.
- `options[].variable`: Variable id.
- `options[].metric`: Metric name.
- `options[].count`: Indexed row count for the option when available.
- `options[].column`: Backing parquet column name when available.

Supported combinations:

- `q` by itself returns plain text matches.
- `q` with `location`, `min_samples`, `within_taxon`, or `descendant_rank` filters the text-match set. Location filtering uses descendant presence for the requested taxon rank rather than requiring a direct row for that higher-rank taxon.
- `q` with `sort_variable` and `sort_metric` but without `within_taxon` and `descendant_rank` ranks the full text-match set directly.
- `q` with `sort_variable`, `sort_metric`, `within_taxon`, and `descendant_rank` applies the text query inside the scoped leaderboard and ranks the matched taxa in that scope.
- `sort_variable` and `sort_metric` without `q` require both `within_taxon` and `descendant_rank`.
- `sort_variable` and `sort_metric` must be provided together.

The response is always an object with top-level metadata and a `results` array.
Search responses also include explicit outcome metadata so clients can distinguish
between different empty states without making a second probe request.

Top-level response metadata includes:

- `query`: Normalized query string or `null`.
- `scope`: The normalized scope object, including resolved `within_taxon`, `descendant_rank`, `location`, `min_samples`, and `include_species_like`.
- `sort`: Sort metadata including `variable`, `metric`, `order`, and display `units`.
- `total`: Total number of result rows after all ranking and filter rules are applied.
- `matched_total`: Number of taxa that matched the text query before scope, location, and sample-count eligibility filtering.
- `eligible_total`: Number of taxa still eligible before pagination. For scoped ranked queries, this is the post-ranking eligible count from the leaderboard index rather than a separate text-prefilter count.
- `empty_reason`: `null` when results are present. Empty text responses use `no_query`, `no_text_matches`, or `filtered_out`. Empty ranked responses use `ranking_ineligible` whenever the request included sorting and no rows survived.
- `limit`: Page size.
- `offset`: Pagination offset.

Scoped ranked queries use the leaderboard index as the source of truth for eligibility and apply `q` inside that scoped ranking pass.

Unscoped ranked queries evaluate the full text-match set before ranking those candidates.

Each result item may include:

- `taxon_id`
- `scientific_name`
- `common_name`
- `common_names`
- `rank`
- `slug`
- `description`
- `image_*`
- `match_score`
- `sample_count`
- `sort_value`
- `sort_variable`
- `sort_metric`
- `position`
- `percentile`

Result field names are normalized to a single snake_case schema. For example,
taxon search responses use `image_url`, not mixed `imageUrl` or `image_file`
variants.

## Canonical Location Search Endpoints

Use `GET /api/locations/search` for free-text location search.

Use `GET /api/locations/search_hierarchy` when the client needs parent-aware or
level-aware hierarchy search.

Examples:

- `GET /api/locations/search?q=utah`
- `GET /api/locations/search_hierarchy?q=utah&level=state`
- `GET /api/locations/search_hierarchy?parent=United+States&limit=10`

Legacy unprefixed routes under `/locations/...` remain available for
compatibility, but `/api/locations/...` is the canonical namespace.

## Canonical Location Hierarchy Lookup

Use `GET /api/locations/{gid}` when the client needs the canonical hierarchy for
an opaque location gid.

Example:

- `GET /api/locations/USA.45.1_1`

The response includes:

- `gid`: Requested location gid.
- `name`: Canonical location name.
- `level`: Numeric hierarchy level.
- `parent_gid`: Immediate parent gid or `null` for root-like nodes.
- `hierarchy`: Ancestor names from highest level to immediate parent.
- `ancestors`: Structured ancestor objects, each containing `gid`, `name`, and `level`.

This endpoint is intended for route hydration and cached location hierarchy lookup,
so clients do not need to infer parent selections from gid formatting.

## Removed Legacy Search Route

`GET /api/species?q=...` is no longer supported.

`GET /api/species/{taxon_id}` still exists for taxon detail pages. Only the old
search route was removed.
