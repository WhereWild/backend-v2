<!--
SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# WhereWild Backend v2

![CI](https://github.com/WhereWild/backend-v2/actions/workflows/ci.yml/badge.svg)

## Getting started

The repository relies on Docker for easy access to GDAL as installing it can be tricky based on the device; thus, having Docker installed and enabled is a prerequisite for getting set up for local development.

> [!NOTE]
> Development has primarily been done on Linux, and things are not guaranteed to work on different OSes. Things seem to work fine if using WSL2 on Window for example but there may be some quirks, especially around the filesystem that need to be configured to work.

## Entering the container

A small bash script is provided to make it quick and easy to enter the container. It is rebuilt if necessary, but will usually be able to drop you in without a full rebuild.

```bash
./gt.sh
```

You may need a `ghcr` token to pull the GDAL image, but it can and should be scoped to read-only.

## Inside the container

There are many aliases built into the container for convenience.

| Command | Description |
|---|---|
| `api` | Start API in foreground with reload |
| `api-bg` | Start API in background (logs to `logs/api.log`) |
| `api-stop` | Stop background API |
| `pt` | Run tests with coverage |
| `pt --temporal` | Run live S3 end-to-end tests for temporal data |
| `pl` | Lint (ruff) |
| `pp` | Lint + test (pipeline approximation) |
| `pd <script>` | Run a script in foreground (`scripts/` prefix assumed) |
| `pdb <script>` | Run a script in background with logging |
| `pdbs <script>` | Stop a background script |
| `pdbc <s1> <s2> ...` | Run scripts sequentially in background (chain) |
| `ww-help` | Show this command list |

The `p` prefix on short commands stands for Python (all run via `uv`). The suffixes: `t` = test, `l` = lint, `p` (in `pp`) = pipeline, `d` = run in docker, `db` = background, `dbs` = stop a background process, `dbc` = chain several background processes. Logs are sent to `logs/`, and the API runs on `http://localhost:8000`.

## Getting set up with local data

The local API is not very useful without any local data to serve, so it is best to get some.
That being said, the full extent of the local data is quite large, and can take a very long amount of time to process.
Luckily, only a subset of the data is needed for most use cases, and it is easy to configure this.

### Setting up your .env

Copy the .env.example to .env so you have the template:

```bash
cp .env.example .env
```

It will already have most values filled out in a manner that should make it very quick to get a small snapshot of local data, but you will need to fill out the top 3 values with your GBIF account information (creating one if not yet) so you can pull occurrence data. `PLANTAE_KEY` denotes the root of the tree you will be dealing with, and `2519` its the key for Cactaceae, which is a much smaller subset of plants with decent geographical diversity. `VARS_TO_DOWNLOAD` denotes GIS variables to download, and the example env has 4 small CHELSA variables that should only take a few minutes to download. `SKIPPABLE_REBUILD_STAGES` skips stages of the data "rebuild", in this case, temporal data enrichment, as it has lots of overhead on small datasets and isn't necessary. But of course, any of these values can be tweaked if desired, keeping in mind certain layers like `landcover` or `elevation` are at very fine resolutions and will take a lot of time and disk space to process. The full catalog of supported variables can be viewed in `config/gis/catalog.json`.

> [!NOTE]
> The `.env` file is a great way to override many variables you might want changed for whatever reason in a convenient way. Any module-level constant the scripts wire up via `os.environ.get` can be overridden this way.

> [!WARNING]
> Rebuild scripts were developed and ran on a machine with 64 GB of RAM. Scripts are not guaranteed to work perfectly on machines with less RAM, and thus certain parameters may have to be tuned. Many of these parameters, such as `GDAL_CACHEMAX`, can be configured in the `.env` file as just previously discussed.

## Actually getting local data

Once the `.env` is set up (the `.env.example` default should be more than fine for most cases), all that remains is to simply run the data fetch/build script at `scripts/rebuild.py`. Within the container, this is simply done as

```bash
pd rebuild.py
```

It may take a while for the GBIF export to process, but once that is done the rest should complete within 10-20 minutes. Once it completes, you should be able to start the api and see data for the scope taxon populated - the frontend repository will be necessary for exploring it via the webapp.

## Temporal Rebuild Pipeline

Adjacent to the taxonomy data pipeline, there is the pipeline for temporal rasters being built and updated in `build_temporal.py`. The current setup is that the script is ran every 30 minutes via a cronjob that targets `bash/run_temporal.sh`. Initial builds will take ~15 minutes but incrimental changes afterwards should complete within only a few seconds.

## Tile Build Pipeline

There is also the tile building pipeline as there is an option for self-hosted vector basemaps. An offline fallback should automatically work without any of this being touched, but there is also a `bash/run_basemap_tiles.sh` script that can generate them for local use, however this takes a few hours, may need up to 200 GB of disk headroom, and a decent amount of RAM.