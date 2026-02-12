api() {
  local data_root
  local log_dir="/workspace/logs"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/api.pid"
  data_root="$(ww_data_root "$@")"
  if [[ "${1:-}" == "--local" || "${1:-}" == "--remote" ]]; then
    shift
  fi
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
  data_root="$(ww_data_root "$@")"
  if [[ "${1:-}" == "--local" || "${1:-}" == "--remote" ]]; then
    shift
  fi
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
  rclone mount "${remote}:${remote_path}" "$mount_point" \
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
    ${dry_run:+--dry-run} > /dev/null 2>&1 &
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
    ${dry_run:+--dry-run} > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-overwrite-remote started (pid ${pid}); log: ${log_file}"
}


b2-stop() {
  local pid_dir="/workspace/logs/pids"
  local stopped=0

  for name in rclone-mount rclone-copy rclone-sync; do
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

- b2-env
  Print WHEREWILD_DATA_ROOT for the mount path.
EOF
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

  if [[ -n "$prefix" ]]; then
    rclone copyto "$source_path" "${remote}:${bucket}/${prefix}/${remote_path}" ${dry_run:+--dry-run}
  else
    rclone copyto "$source_path" "${remote}:${bucket}/${remote_path}" ${dry_run:+--dry-run}
  fi
}

pd() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pd: provide a script name"
    return 1
  fi

  if [[ "$module" != */* ]]; then
    module="scripts/$module"
  fi

  if [[ "$module" == */*.py ]]; then
    module="${module%.py}"
    module="${module#/}"
    module="${module#./}"
    module="${module//\//.}"
  fi

  python -m "$module" "$@"
}

pt() {
  pytest -q "$@"
}

pdb() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pdb: provide a script name"
    return 1
  fi

  if [[ "$module" != */* ]]; then
    module="scripts/$module"
  fi

  local log_dir="/workspace/logs/scripts"
  local pid_dir="/workspace/logs/pids"
  local log_name="${module##*/}"
  log_name="${log_name%.py}"
  local pid_file="$pid_dir/$log_name.pid"
  local log_file="$log_dir/$log_name.log"

  if [[ "$module" == */*.py ]]; then
    module="${module%.py}"
    module="${module#/}"
    module="${module#./}"
    module="${module//\//.}"
  fi

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "pdb: $log_name already running (use pdbs $log_name to stop)"
    return 1
  fi

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
  if [[ "$module_path" != */* ]]; then
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
      if [[ "$module" != */* ]]; then
        module="scripts/$module"
      fi

      log_name="${module##*/}"
      log_name="${log_name%.py}"
      pid_file="$pid_dir/$log_name.pid"
      log_file="$log_dir/$log_name.log"

      if [[ "$module" == */*.py ]]; then
        module="${module%.py}"
        module="${module#/}"
        module="${module#./}"
        module="${module//\//.}"
      fi

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
