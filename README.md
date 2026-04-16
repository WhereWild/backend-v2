# WhereWild

## Overview

This is the backend repository for WhereWild. Currently, all of the code is written in Python. There are 3 different types of code files right now:

1. Libraries that live in `/util`. These expose many functions that are useful for operating on and fetching data. These files contain the bulk of the backend logic.
2. Scripts that live in `/scripts`. These scripts are designed to be wrappers on many library functions and provide an entry point for actually *calling* said functions. These are "data processing" scripts that only need to be run once, or once in a while (e.g. when adding a new GIS layer or metric or something).
3. The api file `main.py`. This file is how frontend clients or developers get relevant data from the backend. Many API calls are basic wrappers of library functions already written in `/util`.

There is also data that the backend needs to operate on and serve via API calls. The `/data` folder is on the `.gitignore` as it is quite large. The shared source of truth is in Backblaze B2, and you can mount or selectively download data as needed.

## Requirements and Setup

After pulling the repo, the first step is to make sure your data is set up. You can mount the B2 bucket or selectively download files into `/data`.

## Docker

The backend uses Docker when running Python, as many GIS Python libraries require to be connected to an installation of GDAL to function. However, this makes version management, especially across different OSes, to be much more difficult. A better approach we found is to use a provided Docker image with GDAL baked in and call Python within the container.

If you are inside the `wherewild` repo, you can run:

```sh
./gt.sh
```

The script is portable across macOS/Linux/WSL and uses its own location to find the
repo's Docker Compose project, so it does not depend on the repo folder being named
`wherewild`.
If you get a permission error, run `chmod +x ./gt.sh` once.
It also runs `b2-mount` automatically before opening the shell, so you usually do not need to call it manually.

If you keep multiple local clones of the back-end repo, set per-copy Docker values in a local `.env` file next to `docker-compose.yml`. Docker Compose loads this automatically for both `./gt.sh` and direct `docker compose` commands, which avoids collisions in image tags, host ports, host data paths, and Compose project names. Start from `.env.example` and adjust the values for each clone.

The downside to this is that using Docker can require lots of typing to use simple commands. A great way around this is to use bash aliases. Inside `gt`, these helpers are already available via the container image, so there is nothing to copy into your `~/.bashrc`.

`gt` stands for "GDAL Terminal" and simply opens a terminal within the GDAL docker. `pd` stands for "Python Docker" and can simply be run as `pd build_locations` for example; it automatically looks for Python files within the `/scripts` directory and runs them through Docker. `pdb` runs the same way in the background and writes logs to `logs/scripts/<script_name>`. `pdbs` stops a background `pdb` script by name. `pdbc` chains multiple scripts in the background, running each after the previous completes, and writes per-script logs to the same folder.

### Build Image

#### For Local Testing

```sh
docker compose up -d gdal
docker compose exec -it gdal /bin/bash
```

The local `gdal` service now supports these optional Compose variables:

- `WHEREWILD_COMPOSE_NAME` for the Compose project name
- `WHEREWILD_IMAGE` for the local image tag
- `WHEREWILD_HOST_DATA_DIR` for the host path mounted at `/workspace/data`
- `WHEREWILD_API_PORT` for the host port mapped to container port `8000`
- `WHEREWILD_DOCS_PORT` for the host port mapped to container port `9101`

If these are unset, Compose falls back to the historical defaults.

#### For Deployment

```sh
docker build --platform linux/amd64 -t wherewild-backend:latest .
```

Then export it using

```sh
docker save wherewild-backend:latest | gzip > wherewild-backend_latest.tar.gz
```

You might want to change the `latest` tag to something else, such as a timestamp like `2026-01-29T0600`, if you are building a lot of containers and need to differentiate them.

You can test run the exported image with

```sh
docker run --rm -it \
  -e WHEREWILD_MODE=api \
  -e RCLONE_CONFIG=/workspace/docker/rclone.conf \
  -p 8000:8000 \
  -v <HOST_DATA_DIR>:/workspace/data \
  -v <HOST_LOG_DIR>:/workspace/logs \
  -v <HOST_RCLONE_CONF_FILE>:/workspace/docker/rclone.conf:ro \
  wherewild-backend:latest
```

- This image expects a mounted `/workspace/data` folder. You will need to mount a **host directory**. It works like this: `-v /absolute/host/path:/container/path` mounts the absolute directory path on your machine (the host) to the directory path within the container. Remove the angle brackets and replace them with your actual host paths.
- If you want API log files to survive container replacement, also mount a host directory at `/workspace/logs` as shown above. Without that mount, `/workspace/logs/api.log` and `/workspace/logs/api.previous.log` exist only in the container filesystem.
- If you want the container to use a different local data directory than `/workspace/data`, set `WHEREWILD_LOCAL_DATA_ROOT`.
- When `WHEREWILD_MODE=api`, the entrypoint runs the API as the foreground container process for deployment. On startup it rotates `/workspace/logs/api.log` to `/workspace/logs/api.previous.log`, then mirrors fresh API logs to both container stdout/stderr and `/workspace/logs/api.log` using `tee`, so `docker logs` and the file stay in sync.
- When `WHEREWILD_MODE=api` **and** the image includes `/etc/wherewild_aliases.sh` **and** `ww_data_root` resolves to the local data root, the entrypoint also runs `b2-pull-all` in the background on startup and starts an in-container scheduler that triggers `b2-pull-all` again at `03:00`, `09:00`, `15:00`, and `21:00` using the container timezone. If those conditions are not met (for example, aliases are not baked into the image), those automatic pulls will not be invoked, and you must run them manually inside the container if you want data to sync from B2. For any of these uses to work, you must provide an rclone config file and point `RCLONE_CONFIG` at it (as shown above). The second `-v` flag in the example is a **file-to-file bind mount**: the left-hand side must be the path to a single rclone config file on the host (for example `/home/me/.config/rclone/rclone.conf`), and the right-hand side is the file path `/workspace/docker/rclone.conf` inside the container. Do not mount a directory there, or rclone will not read the config.
- Set `TZ` if you need a specific scheduler timezone. Override `WHEREWILD_AUTO_PULL_HOURS` if you need different trigger hours, or set `WHEREWILD_AUTO_PULL_ENABLED=false` to disable the recurring scheduler while keeping the API container behavior otherwise unchanged.

Example minimal `rclone.conf` for B2:

```conf
[wherewild-localdev-reader]
type = b2
account = <B2 keyId>
key = <B2 applicationKey>
```

#### Image Hosting

Currently, the image is hosted on Docker Hub at `kellynyanbinary/wherewild-backend`.

### Removing Containers

List containers

```sh
docker ps     # running containers
docker ps -a  # all containers, running and stopped
```

You can remove containers with

```sh
docker image rm <container-id-or-name>     # stopped containers
docker image rm -f <container-id-or-name>  # running containers
docker container prune                     # remove all stopped containers
```

### Removing Images

List images

```sh
docker image list
```

Remove images

```sh
docker rmi <image name>
docker image prune       # remove images not referened by any container
docker image prune -a    # remove all unused images, including untagged build layers
```

## Inside the Running Container

Once you open up the GDAL terminal with `./gt.sh`, there should already be helper commands in the Docker terminal that are added when building.

- `api` starts the backend api in hybrid mode: it uses the B2 mount if it is mounted, otherwise it falls back to `/workspace/data`. You can view a nice looking documentation of the API endpoints at `localhost:8000/docs`. Use `api --remote` to force the mount or `api --local` to force `/workspace/data`.
- `docs` starts a nice looking webpage at `localhost:9101` that can be used to view an organized documentation of the Python libraries.

The aliases automatically run each service in the background so you do not have to keep a terminal open for each one. If you want to stop one of them, you can run `api-stop` or `docs-stop` within a gdal terminal.
You can also run `api-fg` to have the api run in the foreground so you can get logs in the terminal which can be useful when working on the api. However, logs are still present in /logs even when running in the background (for the most recent run), they are just a bit less convenient when debugging in real time.
In the gdal terminal, you also have:

- `pd <script>` to run a Python script/module (defaults to `/scripts`).
- `pt [options] [pytest args...]` to run tests with repo defaults (supports changed-only mode, coverage toggles, and local/remote data roots).
- `pdb <script>` to run a script in the background and log to `logs/scripts/<script_name>.log`.
- `pdbs <script>` to stop a background `pdb` script by name.
- `pdbc <script ...>` to run multiple scripts in the background, one after another, logging each to `logs/scripts/<script_name>.log`.
- `b2-mount` to mount the B2 data at `/workspace/.b2-mount` (background, logs to `logs/rclone/mount.log`). `gt` auto-mounts on shell start, so you usually do not need to call this manually.
- `b2-umount` to unmount the B2 mount.
- `b2-pull-all` to copy the entire remote data tree into `/workspace/data` (background, logs to `logs/rclone/clone.log`).
- `b2-pull-sync [--force|--dry-run]` to sync remote data into `/workspace/data` and delete local extras not present remotely (logs to `logs/rclone/pull-sync.log`).
- `b2-push-all [--force|--dry-run]` to copy local data to B2 without deletions (logs to `logs/rclone/copy.log`). Requires `--force` for real writes.
- `b2-overwrite-remote [--force|--dry-run]` to sync local data to B2 (makes remote EXACTLY match local and deletes remote extras, logs to `logs/rclone/sync.log`). Requires `--force` for real writes.
- `b2-stop` to stop any running `b2-*` jobs using pid files.
- `b2-env` to print a `WHEREWILD_DATA_ROOT` export for the mount path.
- `b2-help` to print a quick help menu of B2 helper commands.
- `b2-pull <path> [dest] [--dry-run] [--force]` to download a single file from B2 (defaults to `/workspace/data`). Example: `b2-pull gis/catalog.json`.
- `b2-push <path> [dest] [--dry-run] [--force]` to upload a single local file (relative to `/workspace/data`) to B2. Example: `b2-push gis/catalog.json`.
- When the API container is running in local-data mode, the same container keeps local data refreshed automatically four times daily at `03:00`, `09:00`, `15:00`, and `21:00`. Scheduler logs go to `logs/auto-pull/service.log`; each scheduled pull still logs its rclone output to `logs/rclone/clone.log`.

### Running Tests (`pt`)

Use `pt` inside the GDAL container (`./gt.sh`) or via one-off exec:

```sh
docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt'
```

Common examples:

```sh
pt                         # default run (remote mode + changed-mode via testmon)
pt --no-cache              # full run, clears pytest cache, disables changed-only mode
pt --no-cov                # run without coverage
pt --local                 # force local data root (/workspace/data)
pt --remote                # force B2 mount data root (/workspace/.b2-mount), requires b2-mount
pt tests/api/test_health.py -q
```

Notes:

- `pt` defaults to remote mode (`--remote`) and requires `b2-mount` to be active.
- `pt` defaults to `--testmon` changed-mode behavior.
- In changed-mode, coverage is disabled unless explicitly requested with pytest `--cov...` args.
- Coverage-enabled runs include `main`, `util`, and `docs`.

## Backblaze B2 Setup

We use Backblaze B2 for shared data. The container creates `/workspace/docker/rclone.conf` from the template on first run; fill it in with the keys you create.

1. The B2 bucket is `wherewild-data`.
2. Create two Application Keys in the B2 console:
   - Read Only key for mounts (`read`).
   - *Only if you are going to be uploading data to the remote bucket after running local processing scripts*: Read/Write key for uploads (`read` + `write`).
   - Only give them access to the `wherewild-data` bucket.
3. Inside gdal, run `rclone config`. Edit the relevant existing remotes to use keys for application keys you need to create on B2. The `keyID` on B2 is the first field rclone asks for, the second field should be the other one it gives you, which is only available right after creation. *Of course, keep this secret!*
4. Use `b2-mount` for read-only access and (again, only if you created a read/write key that you only need if you are going to be uploading data!) `b2-push-all`/`b2-overwrite-remote` for uploads. Use `b2-pull-all` or `b2-pull` to download data into `/workspace/data`.

### Increase IOPS for `b2-pull-all`

`b2-pull-all` exposes rclone concurrency knobs through environment variables:

- `WW_RCLONE_CHECKERS` (metadata/object-operation concurrency)
- `WW_RCLONE_TRANSFERS` (file transfer concurrency)
- `WW_RCLONE_MULTI_THREAD_STREAMS` (parallel streams per large file)
- `WW_RCLONE_BUFFER_SIZE` (memory buffer per transfer)

A good high-IOPS starting profile (especially for many small files) is:

```sh
export WW_RCLONE_CHECKERS=128
export WW_RCLONE_TRANSFERS=64
export WW_RCLONE_MULTI_THREAD_STREAMS=1
export WW_RCLONE_BUFFER_SIZE=16M
b2-stop && b2-pull-all
```

Then tail `logs/rclone/clone.log` and tune:

- If you see throttling/retries (for example HTTP `429`), back off to `CHECKERS=64` and `TRANSFERS=32`.
- If there is no throttling and your disk/network still has headroom, increase gradually (`CHECKERS` by `+16`, `TRANSFERS` by `+8`).

Finally, it's a great idea to install the [parquet viewer](https://marketplace.visualstudio.com/items?itemName=dvirtz.parquet-viewer) extension on VSCode which allows the viewing of parquet files as simple csvs which really helps quick manual inspection. It will likely require you install pyarrow or fastparquet OUTSIDE of Docker or something similar so it can convert the parquets to CSVs. [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) is also a great addition with this that makes it easy to tell which values are part of which columns.

## More

If there is anything missing or anything new you discover please add it to the README. I will be trying to add more information on infrastructural stuff and setup to individual files or some Lucid boards.
