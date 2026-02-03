# WhereWild

## Overview

This is the backend repository for WhereWild. Currently, all of the code is written in Python. There are 3 different types of code files right now:

1. Libraries that live in `/util`. These expose many functions that are useful for operating on and fetching data. These files contain the bulk of the backend logic.
2. Scripts that live in `/scripts`. These scripts are designed to be wrappers on many library functions and provide an entry point for actually *calling* said functions. These are "data processing" scripts that only need to be run once, or once in a while (e.g. when adding a new GIS layer or metric or something).
3. The api file `main.py`. This file is how frontend clients or developers get relevant data from the backend. Many API calls are basic wrappers of library functions already written in `/util`.

There is also data that the backend needs to operate on and serve via API calls. This `/data` folder is on the `.gitignore` as it is quite large and contains a lot of files. The data will soon be moved to object storage as this will allow collaborators to avoid having to download the current state of the `/data` folder and keep it on their disk.

## Requirements and setup

After pulling the repo, the first step is to make sure your data is set up. Currently we will probably sync the `/data` folder with Syncthing until it is shortly added to object storage.

# Docker and relevant tooling

The backend uses Docker when running Python, as many GIS Python libraries require to be connected to an installation of GDAL to function. However, this makes version management, especially across different OSes, to be much more difficult. A better approach we found is to use a provided Docker image with GDAL baked in and call Python within the container.

The downside to this is that using Docker can require lots of typing to use simple commands. A great way around this is to use bash aliases. Inside `gt`, these helpers are already available via the container image, so there is nothing to copy into your `~/.bashrc`.

`gt` stands for "GDAL Terminal" and simply opens a terminal within the GDAL docker. `pd` stands for "Python Docker" and can simply be run as `pd build_locations` for example; it automatically looks for Python files within the `/scripts` directory and runs them through Docker. `pdb` runs the same way in the background and writes logs to `logs/scripts/<script_name>`. `pdbs` stops a background `pdb` script by name. `pdbc` chains multiple scripts in the background, running each after the previous completes, and writes per-script logs to the same folder.

Once you open up the GDAL terminal with `gt`, there should already be helper commands in the Docker terminal that are added when building.
- `api` starts the backend api. You can view a nice looking documentation of the API endpoints at `localhost:8000/docs`
-  `docs` starts a nice looking webpage at `localhost:9101` that can be used to view an organized documentation of the Python libraries.
The aliases automatically run each service in the background so you do not have to keep a terminal open for each one. If you want to stop one of them, you can run `api-stop` or `docs-stop` within a `gt` terminal.
You can also run `api-fg` to have the api run in the foreground so you can get logs in the terminal which can be useful when working on the api. However, logs are still present in /logs even when running in the background (for the most recent run), they are just a bit less convenient when debugging in real time.
In the `gt` terminal, you also have:
- `pd <script>` to run a Python script/module (defaults to `/scripts`).
- `pdb <script>` to run a script in the background and log to `logs/scripts/<script_name>.log`.
- `pdbs <script>` to stop a background `pdb` script by name.
- `pdbc <script ...>` to run multiple scripts in the background, one after another, logging each to `logs/scripts/<script_name>.log`.

Finally, it's a great idea to install the [parquet viewer](https://marketplace.visualstudio.com/items?itemName=dvirtz.parquet-viewer) extension on VSCode which allows the viewing of parquet files as simple csvs which really helps quick manual inspection. It will likely require you install pyarrow or fastparquet OUTSIDE of Docker or something similar so it can convert the parquets to CSVs. [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) is also a great addition with this that makes it easy to tell which values are part of which columns.

# More

If there is anything missing or anything new you discover please add it to the README. I will be trying to add more information on infrastructural stuff and setup to individual files or some Lucid boards.
