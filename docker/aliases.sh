api() {
  local data_root
  local log_dir="/workspace/logs"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/api.pid"
  local storage_mode="${WHEREWILD_PARQUET_STORAGE:-b2}"
  local raster_mode="${WHEREWILD_RASTER_STORAGE:-auto}"
  ww_load_b2_env
  data_root="$(ww_data_root "$@")"
  if [[ "${1:-}" == "--local" || "${1:-}" == "--remote" ]]; then
    if [[ "${1:-}" == "--local" ]]; then
      storage_mode="local"
      raster_mode="local"
    else
      storage_mode="b2"
      raster_mode="b2"
    fi
    shift
  fi
  export WHEREWILD_PARQUET_STORAGE="$storage_mode"
  export WHEREWILD_RASTER_STORAGE="$raster_mode"
  mkdir -p "$log_dir" "$pid_dir"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "api: already running (pid $(cat "$pid_file"))"
    return 0
  fi
  WHEREWILD_DATA_ROOT="$data_root" \
    setsid uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info \
    --reload --reload-dir /workspace/main.py --reload-dir /workspace/util \
    > /workspace/logs/api.log 2>&1 &
  echo "$!" > "$pid_file"
  echo "api started: http://localhost:8000/docs (data: $data_root)"
}

api-fg() {
  local data_root
  local storage_mode="${WHEREWILD_PARQUET_STORAGE:-b2}"
  local raster_mode="${WHEREWILD_RASTER_STORAGE:-auto}"
  ww_load_b2_env
  data_root="$(ww_data_root "$@")"
  if [[ "${1:-}" == "--local" || "${1:-}" == "--remote" ]]; then
    if [[ "${1:-}" == "--local" ]]; then
      storage_mode="local"
      raster_mode="local"
    else
      storage_mode="b2"
      raster_mode="b2"
    fi
    shift
  fi
  export WHEREWILD_PARQUET_STORAGE="$storage_mode"
  export WHEREWILD_RASTER_STORAGE="$raster_mode"
  WHEREWILD_DATA_ROOT="$data_root" \
    uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info \
    --reload --reload-dir /workspace/main.py --reload-dir /workspace/util
}
alias docs='mkdir -p /workspace/logs && cd /workspace && setsid -f mkdocs serve --dev-addr 0.0.0.0:9101 > /workspace/logs/docs.log 2>&1 && echo "docs started: http://localhost:9101/"'
api-stop() {
  local pid_file="/workspace/logs/pids/api.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      rm -f "$pid_file"
      echo "api stopped"
      return 0
    fi
    rm -f "$pid_file"
  fi
  if pkill -f "uvicorn main:app" >/dev/null 2>&1; then
    echo "api stopped"
  else
    echo "api-stop: no running api process found"
  fi
}
alias docs-stop='pkill -f "mkdocs serve" && echo "docs stopped"'

b2-mount() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  local cache_size="${WW_B2_CACHE_SIZE:-5G}"
  local cache_age="${WW_B2_CACHE_AGE:-24h}"
  local read_chunk="${WW_B2_READ_CHUNK:-128M}"
  local read_chunk_limit="${WW_B2_READ_CHUNK_LIMIT:-2G}"
  local buffer_size="${WW_B2_BUFFER_SIZE:-64M}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/mount.log"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/rclone-mount.pid"
  local remote_path="$bucket"

  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}"
  fi

  mkdir -p "$mount_point"
  mkdir -p "$log_dir"
  mkdir -p "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-mount: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  : > "$log_file"
  LD_PRELOAD= rclone mount "${remote}:${remote_path}" "$mount_point" \
    --read-only \
    --fast-list \
    --vfs-cache-mode=full \
    --vfs-cache-max-size "$cache_size" \
    --vfs-cache-max-age "$cache_age" \
    --vfs-read-chunk-size "$read_chunk" \
    --vfs-read-chunk-size-limit "$read_chunk_limit" \
    --buffer-size "$buffer_size" \
    --timeout 5m \
    --retries 10 \
    --low-level-retries 20 \
    --dir-cache-time=1h \
    --log-file "$log_file" \
    --log-level INFO > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-mount started (pid ${pid}); log: ${log_file}"
}

b2-pull-all() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local data_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local checkers="${WW_RCLONE_CHECKERS:-32}"
  local mt_streams="${WW_RCLONE_MULTI_THREAD_STREAMS:-4}"
  local buffer_size="${WW_RCLONE_BUFFER_SIZE:-64M}"
  local stats_interval="${WW_RCLONE_STATS_INTERVAL:-1m}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/clone.log"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/rclone-clone.pid"
  local remote_path="$bucket"

  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}"
  fi

  mkdir -p "$data_root"
  mkdir -p "$log_dir"
  mkdir -p "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-pull-all: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  : > "$log_file"
  rclone copy "${remote}:${remote_path}" "$data_root" \
    --fast-list \
    --transfers "$transfers" \
    --checkers "$checkers" \
    --multi-thread-streams "$mt_streams" \
    --buffer-size "$buffer_size" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-pull-all started (pid ${pid}); log: ${log_file}"
}

b2-umount() {
  local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  fusermount -u "$mount_point" 2>/dev/null || umount "$mount_point"
}

b2-push-all() {
  local remote="${WW_B2_WRITER_REMOTE:-wherewild-localdev-writer}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local data_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local stats_interval="${WW_RCLONE_STATS_INTERVAL:-1m}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/copy.log"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/rclone-copy.pid"
  local force=0
  local dry_run=0
  local args=()
  local arg
  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      --*) args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done
  if [[ "${#args[@]}" -ne 0 ]]; then
    echo "b2-push-all: unexpected arguments: ${args[*]}"
    return 1
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-copy: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi
  mkdir -p "$log_dir"
  mkdir -p "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-copy: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  local dry_flag="--dry-run=false"
  if [[ "$dry_run" -eq 1 ]]; then
    dry_flag="--dry-run"
  fi
  : > "$log_file"
  rclone copy "$data_root" "${remote}:${bucket}/${prefix}" \
    --exclude "species/occurrence.txt" \
    --exclude "species/taxa.csv" \
    --exclude "species/multimedia.txt" \
    --exclude "species/VernacularName.tsv" \
    --exclude "species/inaturalist-taxonomy.dwca/**" \
    --exclude "species/taxonomy/gbif_taxon_lookup.txt" \
    --exclude "species/taxonomy/inat_gbif_mapping_api.csv" \
    --exclude "species/taxonomy/inat_gbif_mapping_obs.csv" \
    --exclude "species/taxonomy/inat_gbif_mapping.csv" \
    --exclude "gbif_occurrence.zip" \
    --exclude "gis/temporal/**" \
    --fast-list \
    --transfers "$transfers" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO \
    "$dry_flag" > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-copy started (pid ${pid}); log: ${log_file}"
}

b2-overwrite-remote() {
  local remote="${WW_B2_WRITER_REMOTE:-wherewild-localdev-writer}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local data_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local stats_interval="${WW_RCLONE_STATS_INTERVAL:-1m}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/sync.log"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/rclone-sync.pid"
  local force=0
  local dry_run=0
  local args=()
  local arg
  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      --*) args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done
  if [[ "${#args[@]}" -ne 0 ]]; then
    echo "b2-overwrite-remote: unexpected arguments: ${args[*]}"
    return 1
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-overwrite-remote: refuses to overwrite/delete remote without --force (use --dry-run to preview)"
    return 1
  fi
  if [[ "$dry_run" -ne 1 ]]; then
    echo "b2-overwrite-remote WARNING:"
    echo "This will make the remote EXACTLY match your local data."
    echo "Remote files not present locally will be deleted."
    echo "Type 'destroy' to proceed:"
    local confirm
    read -r confirm
    if [[ "$confirm" != "destroy" ]]; then
      echo "b2-overwrite-remote: aborted"
      return 1
    fi
  fi
  mkdir -p "$log_dir"
  mkdir -p "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-overwrite-remote: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  local dry_flag="--dry-run=false"
  if [[ "$dry_run" -eq 1 ]]; then
    dry_flag="--dry-run"
  fi
  : > "$log_file"
  rclone sync "$data_root" "${remote}:${bucket}/${prefix}" \
    --exclude "species/occurrence.txt" \
    --exclude "species/taxa.csv" \
    --exclude "species/multimedia.txt" \
    --exclude "species/VernacularName.tsv" \
    --exclude "species/inaturalist-taxonomy.dwca/**" \
    --exclude "species/taxonomy/gbif_taxon_lookup.txt" \
    --exclude "species/taxonomy/inat_gbif_mapping_api.csv" \
    --exclude "species/taxonomy/inat_gbif_mapping_obs.csv" \
    --exclude "species/taxonomy/inat_gbif_mapping.csv" \
    --exclude "gbif_occurrence.zip" \
    --exclude "gis/temporal/**" \
    --fast-list \
    --transfers "$transfers" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO \
    "$dry_flag" > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-overwrite-remote started (pid ${pid}); log: ${log_file}"
}


b2-stop() {
  local pid_dir="/workspace/logs/pids"
  local stopped=0

  for name in rclone-mount rclone-clone rclone-pull-sync rclone-copy rclone-sync; do
    local pid_file="${pid_dir}/${name}.pid"
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "b2-stop: stopped ${name} (pid ${pid})"
        stopped=1
      fi
      rm -f "$pid_file"
    fi
  done

  if [[ "$stopped" -eq 0 ]]; then
    echo "b2-stop: no running rclone jobs"
  fi
}

b2-env() {
  local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  local prefix="${WW_B2_PREFIX:-data}"
  if [[ -n "$prefix" ]]; then
    echo "export WHEREWILD_DATA_ROOT=${mount_point}"
  else
    echo "export WHEREWILD_DATA_ROOT=${mount_point}/data"
  fi
}

b2-help() {
  cat <<'EOF'
B2 helpers (inside gt):

- b2-mount
  Mount remote data read-only at /workspace/.b2-mount (gt auto-mounts on shell start).

- b2-umount
  Unmount the B2 mount.

- b2-pull-all
  Copy the entire remote data tree into /workspace/data (background, logs to /workspace/logs/rclone/clone.log).

- b2-pull-sync [--force|--dry-run]
  Sync remote data to /workspace/data; makes local EXACTLY match remote and deletes local extras.

- b2-pull <path> [dest] [--dry-run] [--force]
  Download a single file from B2. Default dest: /workspace/data/<path>.
  Example: b2-pull gis/catalog.json

- b2-push <path> [dest] [--dry-run] [--force]
  Upload a single local file from /workspace/data to B2.
  Example: b2-push gis/catalog.json --dry-run

- b2-push-all [--force|--dry-run]
  Copy local data to B2 without deletions.

- b2-overwrite-remote [--force|--dry-run]
  Sync local data to B2; makes remote EXACTLY match local and deletes remote extras.

- b2-stop
  Stop any running b2 jobs (mount/copy/sync).

- b2-ls <path>
  List files on the remote at the given path (relative to /data).
  Example: b2-ls gis/temporal/homepage

- b2-pull-dir <path> [--dry-run]
  Download a directory from B2 into /workspace/data/<path>.
  Example: b2-pull-dir gis/temporal/homepage

- b2-push-dir <path> [--dry-run] [--force]
  Upload a local directory from /workspace/data/<path> to B2.
  Example: b2-push-dir gis/temporal/homepage --force

- b2-env
  Print WHEREWILD_DATA_ROOT for the mount path.
EOF
}

b2-pull-sync() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local data_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local checkers="${WW_RCLONE_CHECKERS:-32}"
  local mt_streams="${WW_RCLONE_MULTI_THREAD_STREAMS:-4}"
  local buffer_size="${WW_RCLONE_BUFFER_SIZE:-64M}"
  local stats_interval="${WW_RCLONE_STATS_INTERVAL:-1m}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/pull-sync.log"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/rclone-pull-sync.pid"
  local remote_path="$bucket"
  local force=0
  local dry_run=0
  local args=()
  local arg

  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      --*) args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done

  if [[ "${#args[@]}" -ne 0 ]]; then
    echo "b2-pull-sync: unexpected arguments: ${args[*]}"
    return 1
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-pull-sync: refuses to delete local files without --force (use --dry-run to preview)"
    return 1
  fi

  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}"
  fi

  if [[ "$dry_run" -ne 1 ]]; then
    echo "b2-pull-sync WARNING:"
    echo "This will make local data EXACTLY match the remote data tree."
    echo "Local files not present on remote will be deleted."
    echo "Type 'destroy' to proceed:"
    local confirm
    read -r confirm
    if [[ "$confirm" != "destroy" ]]; then
      echo "b2-pull-sync: aborted"
      return 1
    fi
  fi

  mkdir -p "$data_root"
  mkdir -p "$log_dir"
  mkdir -p "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-pull-sync: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  : > "$log_file"
  rclone sync "${remote}:${remote_path}" "$data_root" \
    --fast-list \
    --transfers "$transfers" \
    --checkers "$checkers" \
    --multi-thread-streams "$mt_streams" \
    --buffer-size "$buffer_size" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO \
    ${dry_run:+--dry-run} > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-pull-sync started (pid ${pid}); log: ${log_file}"
}

ww_data_root() {
  local mode="${1:-}"
  local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  local local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"

  if [[ "$mode" == "--local" ]]; then
    echo "$local_root"
    return 0
  fi

  if [[ "$mode" == "--remote" ]]; then
    echo "$mount_point"
    return 0
  fi

  if mountpoint -q "$mount_point" 2>/dev/null; then
    echo "$mount_point"
  else
    echo "$local_root"
  fi
}

ww_load_b2_env() {
  local config="${RCLONE_CONFIG:-/workspace/docker/rclone.conf}"
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local account key endpoint fallback_endpoint

  if [[ ! -f "$config" ]]; then
    return 0
  fi

  account="$(
    awk -v r="[$remote]" -F'=' '
      $0==r {in_section=1; next}
      /^\[/ {in_section=0}
      in_section && $1 ~ /account/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}
    ' "$config"
  )"
  key="$(
    awk -v r="[$remote]" -F'=' '
      $0==r {in_section=1; next}
      /^\[/ {in_section=0}
      in_section && $1 ~ /^key/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}
    ' "$config"
  )"
  endpoint="$(
    awk -v r="[$remote]" -F'=' '
      $0==r {in_section=1; next}
      /^\[/ {in_section=0}
      in_section && $1 ~ /endpoint/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}
    ' "$config"
  )"

  if [[ -z "$endpoint" ]]; then
    fallback_endpoint="${WW_B2_S3_ENDPOINT_DEFAULT:-https://s3.us-west-004.backblazeb2.com}"
    endpoint="$fallback_endpoint"
  fi

  if [[ -z "${WW_B2_KEY_ID:-}" && -n "$account" ]]; then
    export WW_B2_KEY_ID="$account"
  fi
  if [[ -z "${WW_B2_APP_KEY:-}" && -n "$key" ]]; then
    export WW_B2_APP_KEY="$key"
  fi
  if [[ -z "${WW_B2_S3_ENDPOINT:-}" && -n "$endpoint" ]]; then
    export WW_B2_S3_ENDPOINT="$endpoint"
  fi
}

b2-pull() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local get_root="${WW_B2_GET_ROOT:-/workspace/data}"
  local force=0
  local dry_run=0
  local remote_path
  local dest_path
  local args=()
  local arg

  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      --*) args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done

  remote_path="${args[0]:-}"
  dest_path="${args[1]:-}"

  if [[ -z "$remote_path" ]]; then
    echo "b2-pull: provide a remote-relative path (relative to /data, e.g. gis/catalog.json)"
    return 1
  fi

  remote_path="${remote_path#./}"
  remote_path="${remote_path#/}"
  remote_path="${remote_path#workspace/}"
  remote_path="${remote_path#data/}"

  if [[ -z "$dest_path" ]]; then
    dest_path="${get_root}/${remote_path}"
  fi

  if [[ -d "$dest_path" ]]; then
    dest_path="${dest_path%/}/$(basename "$remote_path")"
  fi

  if [[ -f "$dest_path" && "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-pull: destination exists (use --force to overwrite): $dest_path"
    return 1
  fi

  mkdir -p "$(dirname "$dest_path")"

  if [[ -n "$prefix" ]]; then
    rclone copyto "${remote}:${bucket}/${prefix}/${remote_path}" "$dest_path" ${dry_run:+--dry-run}
  else
    rclone copyto "${remote}:${bucket}/${remote_path}" "$dest_path" ${dry_run:+--dry-run}
  fi
}

b2-push() {
  local remote="${WW_B2_WRITER_REMOTE:-wherewild-localdev-writer}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local force=0
  local dry_run=0
  local local_path=""
  local remote_path=""
  local args=()
  local arg

  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      --*) args+=("$arg") ;;
      *) args+=("$arg") ;;
    esac
  done

  local_path="${args[0]:-}"
  remote_path="${args[1]:-}"

  if [[ -z "$local_path" ]]; then
    echo "b2-push: provide a local path (relative to /data, e.g. gis/catalog.json)"
    return 1
  fi

  local_path="${local_path#./}"
  local_path="${local_path#/}"
  local_path="${local_path#workspace/}"
  local_path="${local_path#data/}"

  local source_path="${local_root}/${local_path}"
  if [[ ! -f "$source_path" ]]; then
    echo "b2-push: local file not found: $source_path"
    return 1
  fi

  if [[ -z "$remote_path" ]]; then
    remote_path="${local_path}"
  else
    remote_path="${remote_path#./}"
    remote_path="${remote_path#/}"
    remote_path="${remote_path#data/}"
    remote_path="${remote_path#workspace/}"
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-push: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi

  local dry_flag="--dry-run=false"
  if [[ "$dry_run" -eq 1 ]]; then
    dry_flag="--dry-run"
  fi
  local rclone_bin="/usr/bin/rclone"
  if [[ ! -x "$rclone_bin" ]]; then
    rclone_bin="$(command -v rclone)"
  fi
  if [[ -n "$prefix" ]]; then
    "$rclone_bin" copyto "$source_path" "${remote}:${bucket}/${prefix}/${remote_path}" "$dry_flag"
  else
    "$rclone_bin" copyto "$source_path" "${remote}:${bucket}/${remote_path}" "$dry_flag"
  fi
}

b2-ls() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local path="${1:-}"

  path="${path#./}"
  path="${path#/}"
  path="${path#workspace/}"
  path="${path#data/}"

  local remote_path="${bucket}"
  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}"
  fi
  if [[ -n "$path" ]]; then
    remote_path="${remote_path}/${path}"
  fi

  rclone lsf "${remote}:${remote_path}"
}

b2-pull-dir() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local dry_run=0
  local args=()
  local arg

  for arg in "$@"; do
    case "$arg" in
      --dry-run) dry_run=1 ;;
      *) args+=("$arg") ;;
    esac
  done

  local path="${args[0]:-}"
  if [[ -z "$path" ]]; then
    echo "b2-pull-dir: provide a path (relative to /data, e.g. gis/temporal/homepage)"
    return 1
  fi

  path="${path#./}"
  path="${path#/}"
  path="${path#workspace/}"
  path="${path#data/}"

  local remote_path="${bucket}"
  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}/${path}"
  else
    remote_path="${bucket}/${path}"
  fi

  local dest="${local_root}/${path}"
  mkdir -p "$dest"
  echo "b2-pull-dir: ${remote}:${remote_path} → ${dest}"
  rclone copy "${remote}:${remote_path}" "$dest" \
    --transfers "$transfers" \
    ${dry_run:+--dry-run}
}

b2-push-dir() {
  local remote="${WW_B2_WRITER_REMOTE:-wherewild-localdev-writer}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
  local transfers="${WW_RCLONE_TRANSFERS:-4}"
  local force=0
  local dry_run=0
  local args=()
  local arg

  for arg in "$@"; do
    case "$arg" in
      --force) force=1 ;;
      --dry-run) dry_run=1 ;;
      *) args+=("$arg") ;;
    esac
  done

  local path="${args[0]:-}"
  if [[ -z "$path" ]]; then
    echo "b2-push-dir: provide a path (relative to /data, e.g. gis/temporal/homepage)"
    return 1
  fi

  path="${path#./}"
  path="${path#/}"
  path="${path#workspace/}"
  path="${path#data/}"

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-push-dir: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi

  local source="${local_root}/${path}"
  if [[ ! -d "$source" ]]; then
    echo "b2-push-dir: local directory not found: $source"
    return 1
  fi

  local remote_path="${bucket}"
  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}/${path}"
  else
    remote_path="${bucket}/${path}"
  fi

  echo "b2-push-dir: ${source} → ${remote}:${remote_path}"
  rclone copy "$source" "${remote}:${remote_path}" \
    --transfers "$transfers" \
    ${dry_run:+--dry-run}
}

pd() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pd: provide a script name"
    return 1
  fi

  if [[ "$module" != */* && ( "$module" != *.* || "$module" == *.py ) ]]; then
    module="scripts/$module"
  fi

  module="${module%.py}"
  module="${module#/}"
  module="${module#./}"
  module="${module//\//.}"

  export WHEREWILD_PARQUET_STORAGE="${WHEREWILD_PARQUET_STORAGE:-local}"
  python -m "$module" "$@"
}

pt() {
  # Default to remote-mounted data unless caller explicitly requests local.
  local mode="remote"
  local -a args=()
  local -a raw_args=()
  local pt_verbose=0
  local pt_quiet=0
  local pt_timings=0
  local pt_cov=1
  local pt_changed=1
  local pt_no_cache=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local)
        if [[ "$mode" == "remote" ]]; then
          echo "pt: choose only one of --local or --remote"
          return 1
        fi
        mode="local"
        ;;
      --remote)
        if [[ "$mode" == "local" ]]; then
          echo "pt: choose only one of --local or --remote"
          return 1
        fi
        mode="remote"
        ;;
      --verbose)
        pt_verbose=1
        ;;
      --quiet)
        pt_quiet=1
        ;;
      --timings|--timing)
        pt_timings=1
        ;;
      --no-cov)
        pt_cov=0
        ;;
      --changed)
        pt_changed=1
        ;;
      --no-cache)
        pt_no_cache=1
        pt_changed=0
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          args+=("$1")
          shift
        done
        break
        ;;
      *)
        args+=("$1")
        raw_args+=("$1")
        ;;
    esac
    shift
  done

  local explicit_cov=0
  for arg in "${raw_args[@]}"; do
    case "$arg" in
      --cov|--cov=*|--cov-report|--cov-report=*|--no-cov)
        explicit_cov=1
        break
        ;;
    esac
  done

  # Coverage with testmon can be misleading when no tests are selected.
  # Default changed-mode runs to no coverage unless caller explicitly asks for it.
  if [[ "$pt_changed" -eq 1 && "$pt_no_cache" -eq 0 && "$explicit_cov" -eq 0 ]]; then
    pt_cov=0
  fi

  local -a base_args=()
  if [[ "$pt_verbose" -eq 1 ]]; then
    base_args+=("-vv")
  elif [[ "$pt_quiet" -eq 1 ]]; then
    base_args+=("-q")
  fi
  if [[ "$pt_timings" -eq 1 ]]; then
    base_args+=("--durations=0" "--durations-min=0")
  fi
  if [[ "$pt_no_cache" -eq 1 ]]; then
    base_args+=("--cache-clear")
  fi
  if [[ "$pt_changed" -eq 1 ]]; then
    base_args+=("--testmon")
  fi
  base_args+=("${args[@]}")

  local -a cov_args=()
  if [[ "$pt_cov" -eq 1 ]]; then
    cov_args+=(
      "--cov=main"
      "--cov=util"
      "--cov=docs"
      "--cov-report=term-missing"
    )
  fi

  if [[ "$mode" == "local" ]]; then
    WHEREWILD_DATA_ROOT="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}" \
    WHEREWILD_PARQUET_STORAGE=local \
      pytest "${cov_args[@]}" "${base_args[@]}"
    return $?
  fi

  if [[ "$mode" == "remote" ]]; then
    local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
    if ! mountpoint -q "$mount_point" 2>/dev/null; then
      echo "pt: B2 mount not active — run b2-mount first"
      return 1
    fi
    WHEREWILD_DATA_ROOT="$mount_point" \
    WHEREWILD_PARQUET_STORAGE=local \
      pytest "${cov_args[@]}" "${base_args[@]}"
    return $?
  fi

  pytest "${cov_args[@]}" "${base_args[@]}"
}

pdb() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pdb: provide a script name"
    return 1
  fi

  if [[ "$module" != */* && ( "$module" != *.* || "$module" == *.py ) ]]; then
    module="scripts/$module"
  fi

  local log_dir="/workspace/logs/scripts"
  local pid_dir="/workspace/logs/pids"
  local log_name="${module##*/}"
  log_name="${log_name%.py}"
  local pid_file="$pid_dir/$log_name.pid"
  local log_file="$log_dir/$log_name.log"

  module="${module%.py}"
  module="${module#/}"
  module="${module#./}"
  module="${module//\//.}"

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "pdb: $log_name already running (use pdbs $log_name to stop)"
    return 1
  fi

  export WHEREWILD_PARQUET_STORAGE="${WHEREWILD_PARQUET_STORAGE:-local}"
  PYTHONUNBUFFERED=1 setsid python -u -m "$module" "$@" > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "pdb started: $log_file"
}

pdbs() {
  local module="$1"

  if [[ -z "$module" ]]; then
    echo "pdbs: provide a script name"
    return 1
  fi

  local module_path="$module"
  if [[ "$module_path" != */* && ( "$module_path" != *.* || "$module_path" == *.py ) ]]; then
    module_path="scripts/$module_path"
  fi

  local log_name="${module_path##*/}"
  log_name="${log_name%.py}"
  local pid_file="/workspace/logs/pids/$log_name.pid"
  local module_dotted="${module_path%.py}"
  module_dotted="${module_dotted#/}"
  module_dotted="${module_dotted#./}"
  module_dotted="${module_dotted//\//.}"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "pdbs stopped: $log_name"
    else
      local fallback_pids
      fallback_pids="$(pgrep -f "python -m $module_dotted")"
      if [[ -n "$fallback_pids" ]]; then
        kill $fallback_pids
        echo "pdbs stopped: $log_name"
      else
        echo "pdbs: no running process for $log_name"
      fi
    fi
    rm -f "$pid_file"
  else
    local fallback_pids
    fallback_pids="$(pgrep -f "python -m $module_dotted")"
    if [[ -n "$fallback_pids" ]]; then
      kill $fallback_pids
      echo "pdbs stopped: $log_name"
    else
      echo "pdbs: no running process for $log_name"
    fi
  fi
}

pdbc() {
  local log_dir="/workspace/logs/scripts"
  local pid_dir="/workspace/logs/pids"

  if [[ "$#" -eq 0 ]]; then
    echo "pdbc: provide one or more script names"
    return 1
  fi

  mkdir -p "$log_dir" "$pid_dir"

  (
    for module in "$@"; do
      if [[ "$module" != */* && ( "$module" != *.* || "$module" == *.py ) ]]; then
        module="scripts/$module"
      fi

      log_name="${module##*/}"
      log_name="${log_name%.py}"
      pid_file="$pid_dir/$log_name.pid"
      log_file="$log_dir/$log_name.log"

      module="${module%.py}"
      module="${module#/}"
      module="${module#./}"
      module="${module//\//.}"

      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "pdbc: $log_name already running (use pdbs $log_name to stop)"
        break
      fi

      PYTHONUNBUFFERED=1 setsid python -u -m "$module" > "$log_file" 2>&1 &
      pid="$!"
      echo "$pid" > "$pid_file"
      wait "$pid"
      rm -f "$pid_file"
    done
  ) &

  echo "pdbc started: $log_dir"
}

pdbca() {
  local log_dir="/workspace/logs/scripts"
  local pid_dir="/workspace/logs/pids"
  local roots_csv="${WW_PDBCA_ROOTS:-0,1,2,3,4,5,6,7,8}"
  local modules=("$@")

  if [[ "${#modules[@]}" -eq 0 ]]; then
    modules=(enrich_tree process_tree process_positions)
  fi

  mkdir -p "$log_dir" "$pid_dir"

  local run_id
  run_id="$(date +%Y%m%d-%H%M%S)"
  local queue_log="$log_dir/pdbca-$run_id.log"

  (
    IFS=',' read -r -a roots <<< "$roots_csv"
    for raw_root in "${roots[@]}"; do
      root="${raw_root//[[:space:]]/}"
      if [[ -z "$root" ]]; then
        continue
      fi

      echo "[pdbca] root=$root begin ts=$(date -Iseconds)" >> "$queue_log"

      for module in "${modules[@]}"; do
        module_path="$module"
        if [[ "$module_path" != */* && ( "$module_path" != *.* || "$module_path" == *.py ) ]]; then
          module_path="scripts/$module_path"
        fi

        log_name="${module_path##*/}"
        log_name="${log_name%.py}"
        pid_file="$pid_dir/${log_name}.root-${root}.pid"
        log_file="$log_dir/${log_name}.root-${root}.log"

        module_dotted="$module_path"
        module_dotted="${module_dotted%.py}"
        module_dotted="${module_dotted#/}"
        module_dotted="${module_dotted#./}"
        module_dotted="${module_dotted//\//.}"

        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
          echo "[pdbca] skip root=$root module=$module_dotted reason=already-running pid=$(cat "$pid_file")" >> "$queue_log"
          exit 1
        fi

        echo "[pdbca] start root=$root module=$module_dotted log=$log_file ts=$(date -Iseconds)" >> "$queue_log"
        WHEREWILD_ROOT_TAXON_ID="$root" \
          WHEREWILD_PARQUET_STORAGE="${WHEREWILD_PARQUET_STORAGE:-local}" \
          PYTHONUNBUFFERED=1 \
          setsid python -u -m "$module_dotted" > "$log_file" 2>&1 &
        pid="$!"
        echo "$pid" > "$pid_file"

        wait "$pid"
        rc="$?"
        rm -f "$pid_file"

        if [[ "$rc" -ne 0 ]]; then
          echo "[pdbca] fail root=$root module=$module_dotted rc=$rc log=$log_file ts=$(date -Iseconds)" >> "$queue_log"
          exit "$rc"
        fi

        echo "[pdbca] done root=$root module=$module_dotted rc=$rc ts=$(date -Iseconds)" >> "$queue_log"
      done

      echo "[pdbca] root=$root done ts=$(date -Iseconds)" >> "$queue_log"
    done

    echo "[pdbca] complete ts=$(date -Iseconds)" >> "$queue_log"
  ) &

  echo "pdbca started: $queue_log"
  echo "pdbca roots: $roots_csv"
  echo "pdbca modules: ${modules[*]}"
}
