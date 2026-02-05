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

The script is portable across macOS/Linux/WSL and uses the repo root to locate the
`wherewild` Docker Compose project.
If you get a permission error, run `chmod +x ./gt.sh` once.
It also runs `b2-mount` automatically before opening the shell, so you usually do not need to call it manually.

The downside to this is that using Docker can require lots of typing to use simple commands. A great way around this is to use bash aliases. Inside `gt`, these helpers are already available via the container image, so there is nothing to copy into your `~/.bashrc`.

`gt` stands for "GDAL Terminal" and simply opens a terminal within the GDAL docker. `pd` stands for "Python Docker" and can simply be run as `pd build_locations` for example; it automatically looks for Python files within the `/scripts` directory and runs them through Docker. `pdb` runs the same way in the background and writes logs to `logs/scripts/<script_name>`. `pdbs` stops a background `pdb` script by name. `pdbc` chains multiple scripts in the background, running each after the previous completes, and writes per-script logs to the same folder.

### Build Image

#### For Local Testing

```sh
docker compose up -d gdal
docker compose exec -it gdal /bin/bash
```

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
  -p 8000:8000 \
  -v <CHANGE ME>:/data \
  wherewild-backend:latest
```

**This image expects a mounted `/data` folder**. For using B2, I think `entrypoint.sh` might need modifitcation.

You may need to change the mount. It works like this: `-v a: b` mounts absolute directory path `a` on your machine (the host) to `b` within the container. Remove the angle brackets.

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
- `pdb <script>` to run a script in the background and log to `logs/scripts/<script_name>.log`.
- `pdbs <script>` to stop a background `pdb` script by name.
- `pdbc <script ...>` to run multiple scripts in the background, one after another, logging each to `logs/scripts/<script_name>.log`.
- `b2-mount` to mount the B2 data at `/workspace/.b2-mount` (background, logs to `logs/rclone/mount.log`). `gt` auto-mounts on shell start, so you usually do not need to call this manually.
- `b2-umount` to unmount the B2 mount.
- `b2-pull-all` to copy the entire remote data tree into `/workspace/data` (background, logs to `logs/rclone/clone.log`).
- `b2-push-all [--force|--dry-run]` to copy local data to B2 without deletions (logs to `logs/rclone/copy.log`). Requires `--force` for real writes.
- `b2-overwrite-remote [--force|--dry-run]` to sync local data to B2 (makes remote EXACTLY match local and deletes remote extras, logs to `logs/rclone/sync.log`). Requires `--force` for real writes.
- `b2-stop` to stop any running `b2-*` jobs using pid files.
- `b2-env` to print a `WHEREWILD_DATA_ROOT` export for the mount path.
- `b2-help` to print a quick help menu of B2 helper commands.
- `b2-pull <path> [dest] [--dry-run] [--force]` to download a single file from B2 (defaults to `/workspace/data`). Example: `b2-pull gis/catalog.json`.
- `b2-push <path> [dest] [--dry-run] [--force]` to upload a single local file (relative to `/workspace/data`) to B2. Example: `b2-push gis/catalog.json`.

## Backblaze B2 Setup

We use Backblaze B2 for shared data. The container creates `/workspace/docker/rclone.conf` from the template on first run; fill it in with the keys you create.

1. The B2 bucket is `wherewild-data`.
2. Create two Application Keys in the B2 console:
   - Read Only key for mounts (`read`).
   - *Only if you are going to be uploading data to the remote bucket after running local processing scripts*: Read/Write key for uploads (`read` + `write`).
   - Only give them access to the `wherewild-data` bucket.
3. Inside gdal, run `rclone config`. Edit the relevant existing remotes to use keys for application keys you need to create on B2. The `keyID` on B2 is the first field rclone asks for, the second field should be the other one it gives you, which is only available right after creation. *Of course, keep this secret!*
4. Use `b2-mount` for read-only access and (again, only if you created a read/write key that you only need if you are going to be uploading data!) `b2-push-all`/`b2-overwrite-remote` for uploads. Use `b2-pull-all` or `b2-pull` to download data into `/workspace/data`.

Finally, it's a great idea to install the [parquet viewer](https://marketplace.visualstudio.com/items?itemName=dvirtz.parquet-viewer) extension on VSCode which allows the viewing of parquet files as simple csvs which really helps quick manual inspection. It will likely require you install pyarrow or fastparquet OUTSIDE of Docker or something similar so it can convert the parquets to CSVs. [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) is also a great addition with this that makes it easy to tell which values are part of which columns.

## More

If there is anything missing or anything new you discover please add it to the README. I will be trying to add more information on infrastructural stuff and setup to individual files or some Lucid boards.
