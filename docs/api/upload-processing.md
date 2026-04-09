# Upload Processing

This page documents the processed upload ZIP returned by the raw observations
upload endpoint.

## Endpoint

The processed archive is returned by the raw observations upload flow exposed in
`main.py`.

The endpoint accepts an uploaded CSV, TSV, or Parquet file, enriches each row
with derived GIS data, and returns a ZIP file named `processed_observations.zip`.

## Input Normalization

Before enrichment, the backend normalizes a few required concepts:

- `decimalLatitude` and `decimalLongitude` are inferred from common aliases when
  possible.
- `catalogNumber` is inferred from explicit id-style aliases when possible.
- `observationName` is inferred from `observationName`, `observation_name`,
  `name`, `title`, or `label`.
- If no observation name column exists, fallback names are synthesized as
  `Observation #1`, `Observation #2`, and so on.

## Reserved Derived Columns

The upload preserves user-provided columns, but some column names are reserved
for backend-derived enrichment data.

If the uploaded file already contains any of these reserved columns, the upload
fails with HTTP 422 instead of silently overwriting data:

- `tileId`
- Any GIS layer id present in the catalog, such as `bio_1`, `landcover`, or
  `wrb`

This keeps the original data intact and avoids ambiguous merges between user
data and backend-derived values.

## ZIP Contents

The ZIP currently contains these Parquet files and CSV mirrors when writing the
CSV succeeds:

- `occurrence.parquet` and `occurrence.csv`
- `variable_metadata.parquet` and `variable_metadata.csv`
- `summary_stats.parquet` and `summary_stats.csv`
- `density_graph.parquet` and `density_graph.csv`
- `categorical_stats.parquet` and `categorical_stats.csv`
- `categorical_value_lookup.parquet` and `categorical_value_lookup.csv`
- `occurrence_index.parquet` and `occurrence_index.csv`

Some internal artifacts may be omitted when there is no data to write for that
artifact.

## Artifact Contracts

### occurrence.parquet / occurrence.csv

This is the user-facing enriched upload.

It preserves the original upload columns and adds derived backend columns such
as `tileId` and GIS variables.

GIS variable columns in this file use human-friendly exported names rather than
stable backend ids. For example, a variable with stable id `bio_1` may be
exported as `Annual Mean Temperature`.

### variable_metadata.parquet / variable_metadata.csv

This file is the stable lookup table for GIS variables.

Columns:

- `id`: stable backend variable id
- `name`: display label
- `exported_name`: column name used in `occurrence.parquet`
- `category`
- `units`
- `value_type`

This schema is stable even when metadata loading falls back internally.

### summary_stats.parquet

This file stores numeric summary statistics for GIS variables.

Important fields:

- `variable`: stable backend variable id
- `variableName`: display label
- `variableCategory`: display category

The `variable` column should be treated as the machine key.

### density_graph.parquet

This file stores precomputed numeric density curves.

Important fields:

- `variable`: stable backend variable id
- `variableName`: display label
- `variableCategory`: display category

### categorical_stats.parquet

This file stores categorical distributions.

Important fields:

- `variable`: stable backend variable id
- `metric`: stable category key, typically `class_<id>`
- `metricLabel`: display label for class metrics when a legend is available
- `variableName`: display label
- `variableCategory`: display category

For categorical variables, `metric` is the machine key. Use `metricLabel` or
the categorical lookup file for display text.

### occurrence_index.parquet

This file is an internal/query-oriented index structure.

- Column names remain stable variable ids.
- Numeric variables store numeric values.
- Categorical variables store raw class ids or codes.

This file should not be treated as a display dataset.

### categorical_value_lookup.parquet / categorical_value_lookup.csv

This file bridges categorical codes in `occurrence_index.parquet` to the exact
category keys used by `categorical_stats.parquet`.

Columns:

- `variable`: stable backend variable id
- `variableName`: display label
- `variableCategory`: display category
- `code`: raw categorical code stored in `occurrence_index`
- `metric`: exact category key used in `categorical_stats.metric`
- `label`: display label from the legend
- `description`
- `group`
- `groupLabel`

## Frontend Usage

Use these rules when consuming a processed upload ZIP:

1. Use `occurrence.parquet` or `occurrence.csv` for user-visible table/export
   views.
2. Use `variable_metadata` to map stable GIS ids to exported occurrence column
   names and display labels.
3. Treat `summary_stats.variable`, `density_graph.variable`,
   `categorical_stats.variable`, and `occurrence_index` column names as stable
   ids.
4. For categorical highlighting, do not compare `occurrence_index` categorical
   codes directly to `categorical_stats.metric`.
5. Instead, resolve categorical highlighting through:

   `occurrence_index raw code -> categorical_value_lookup -> categorical_stats.metric`

This is the supported way to match a selected observation's categorical value to
the corresponding bar or distribution entry in the categorical stats artifact.

For display text, prefer `categorical_stats.metricLabel` when present or
`categorical_value_lookup.label`.
