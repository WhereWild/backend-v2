_uv() {
    uv run --env-file /workspace/.env "$@"
}

# ---------------------------------------------------------------------------
# B2 helpers
# ---------------------------------------------------------------------------

ww_load_b2_env() {
  local config="${RCLONE_CONFIG:-/workspace/docker/rclone.conf}"
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local account key endpoint

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
    endpoint="${WW_B2_S3_ENDPOINT_DEFAULT:-https://s3.us-west-004.backblazeb2.com}"
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

  echo "$local_root"
}

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

  mkdir -p "$mount_point" "$log_dir" "$pid_dir"

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

b2-umount() {
  local mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  fusermount -u "$mount_point" 2>/dev/null || umount "$mount_point"
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

  mkdir -p "$data_root" "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "b2-pull-all: already running (pid ${existing_pid})"
      return 0
    fi
    rm -f "$pid_file"
  fi

  : > "$log_file"
  (
    trap 'if [[ -f "$pid_file" ]] && [[ "$(cat "$pid_file" 2>/dev/null || true)" == "$BASHPID" ]]; then rm -f "$pid_file"; fi' EXIT
    rclone copy "${remote}:${remote_path}" "$data_root" \
      --fast-list \
      --transfers "$transfers" \
      --checkers "$checkers" \
      --multi-thread-streams "$mt_streams" \
      --buffer-size "$buffer_size" \
      --stats "$stats_interval" \
      --stats-log-level INFO \
      --log-file "$log_file" \
      --log-level INFO > /dev/null 2>&1
  ) &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-pull-all started (pid ${pid}); log: ${log_file}"
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

  mkdir -p "$data_root" "$log_dir" "$pid_dir"

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
      *) args+=("$arg") ;;
    esac
  done

  if [[ "${#args[@]}" -ne 0 ]]; then
    echo "b2-push-all: unexpected arguments: ${args[*]}"
    return 1
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-push-all: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-push-all: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  local dry_flag="--dry-run=false"
  if [[ "$dry_run" -eq 1 ]]; then
    dry_flag="--dry-run"
  fi
  : > "$log_file"
  rclone copy "$data_root" "${remote}:${bucket}/${prefix}" \
    --exclude "cache/**" \
    --exclude "gis/temporal/chunks/**" \
    --fast-list \
    --transfers "$transfers" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO \
    "$dry_flag" > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-push-all started (pid ${pid}); log: ${log_file}"
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

  mkdir -p "$log_dir" "$pid_dir"

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
    --exclude "cache/**" \
    --exclude "gis/temporal/chunks/**" \
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
  echo "export WHEREWILD_DATA_ROOT=${mount_point}"
}

b2-pull() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
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
      *) args+=("$arg") ;;
    esac
  done

  remote_path="${args[0]:-}"
  dest_path="${args[1]:-}"

  if [[ -z "$remote_path" ]]; then
    echo "b2-pull: provide a remote-relative path (relative to /data, e.g. gis/catalog.json)"
    return 1
  fi

  remote_path="${remote_path#./}"; remote_path="${remote_path#/}"
  remote_path="${remote_path#workspace/}"; remote_path="${remote_path#data/}"

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
      *) args+=("$arg") ;;
    esac
  done

  local_path="${args[0]:-}"
  remote_path="${args[1]:-}"

  if [[ -z "$local_path" ]]; then
    echo "b2-push: provide a local path (relative to /data, e.g. gis/catalog.json)"
    return 1
  fi

  local_path="${local_path#./}"; local_path="${local_path#/}"
  local_path="${local_path#workspace/}"; local_path="${local_path#data/}"

  local source_path="${local_root}/${local_path}"
  if [[ ! -f "$source_path" ]]; then
    echo "b2-push: local file not found: $source_path"
    return 1
  fi

  if [[ -z "$remote_path" ]]; then
    remote_path="${local_path}"
  else
    remote_path="${remote_path#./}"; remote_path="${remote_path#/}"
    remote_path="${remote_path#data/}"; remote_path="${remote_path#workspace/}"
  fi

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-push: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi

  local dry_flag="--dry-run=false"
  if [[ "$dry_run" -eq 1 ]]; then
    dry_flag="--dry-run"
  fi
  if [[ -n "$prefix" ]]; then
    rclone copyto "$source_path" "${remote}:${bucket}/${prefix}/${remote_path}" "$dry_flag"
  else
    rclone copyto "$source_path" "${remote}:${bucket}/${remote_path}" "$dry_flag"
  fi
}

b2-ls() {
  local remote="${WW_B2_READER_REMOTE:-wherewild-localdev-reader}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local path="${1:-}"

  path="${path#./}"; path="${path#/}"
  path="${path#workspace/}"; path="${path#data/}"

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

  path="${path#./}"; path="${path#/}"
  path="${path#workspace/}"; path="${path#data/}"

  local remote_path
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
  local transfers="${WW_RCLONE_TRANSFERS:-16}"
  local stats_interval="${WW_RCLONE_STATS_INTERVAL:-1m}"
  local log_dir="/workspace/logs/rclone"
  local pid_dir="/workspace/logs/pids"
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

  path="${path#./}"; path="${path#/}"
  path="${path#workspace/}"; path="${path#data/}"

  if [[ "$force" -ne 1 && "$dry_run" -ne 1 ]]; then
    echo "b2-push-dir: refuses to overwrite remote without --force (use --dry-run to preview)"
    return 1
  fi

  local source="${local_root}/${path}"
  if [[ ! -d "$source" ]]; then
    echo "b2-push-dir: local directory not found: $source"
    return 1
  fi

  local remote_path
  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}/${path}"
  else
    remote_path="${bucket}/${path}"
  fi

  local log_name="push-dir-${path//\//-}"
  local log_file="${log_dir}/${log_name}.log"
  local pid_file="${pid_dir}/rclone-${log_name}.pid"

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "b2-push-dir: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  echo "b2-push-dir: ${source} → ${remote}:${remote_path}"
  : > "$log_file"
  rclone copy "$source" "${remote}:${remote_path}" \
    --fast-list \
    --transfers "$transfers" \
    --stats "$stats_interval" \
    --stats-log-level INFO \
    --log-file "$log_file" \
    --log-level INFO \
    ${dry_run:+--dry-run} > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-push-dir started (pid ${pid}); log: ${log_file}"
}

b2-wipe() {
  local remote="${WW_B2_WRITER_REMOTE:-wherewild-localdev-writer}"
  local bucket="${WW_B2_BUCKET:-wherewild-data}"
  local prefix="${WW_B2_PREFIX:-data}"
  local dry_run=0

  for arg in "$@"; do
    case "$arg" in
      --dry-run) dry_run=1 ;;
    esac
  done

  local remote_path="${bucket}"
  if [[ -n "$prefix" ]]; then
    remote_path="${bucket}/${prefix}"
  fi

  if [[ "$dry_run" -eq 1 ]]; then
    echo "b2-wipe (dry-run): would purge ${remote}:${remote_path}"
    return 0
  fi

  echo "b2-wipe WARNING:"
  echo "This will DELETE ALL files under ${remote}:${remote_path}"
  echo "Type 'destroy' to proceed:"
  local confirm
  read -r confirm
  if [[ "$confirm" != "destroy" ]]; then
    echo "b2-wipe: aborted"
    return 1
  fi

  local log_dir="/workspace/logs/rclone"
  local pid_dir="/workspace/logs/pids"
  local log_file="${log_dir}/wipe.log"
  local pid_file="${pid_dir}/rclone-wipe.pid"
  mkdir -p "$log_dir" "$pid_dir"

  echo "b2-wipe: purging ${remote}:${remote_path} …"
  : > "$log_file"
  rclone purge "${remote}:${remote_path}" \
    --log-file "$log_file" \
    --log-level INFO > /dev/null 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_file"
  echo "b2-wipe started (pid ${pid}); log: ${log_file}"
}

b2-help() {
  cat <<'EOF'
B2 helpers:

- b2-mount
  Mount remote data read-only at /workspace/.b2-mount.

- b2-umount
  Unmount the B2 mount.

- b2-pull-all
  Copy the entire remote data tree into /workspace/data (background, logs to /workspace/logs/rclone/clone.log).

- b2-pull-sync [--force|--dry-run]
  Sync remote data to /workspace/data; makes local EXACTLY match remote and deletes local extras.

- b2-pull <path> [dest] [--dry-run] [--force]
  Download a single file from B2. Default dest: /workspace/data/<path>.

- b2-push <path> [dest] [--dry-run] [--force]
  Upload a single local file from /workspace/data to B2.

- b2-push-all [--force|--dry-run]
  Copy local data to B2 without deletions (excludes cache/).

- b2-overwrite-remote [--force|--dry-run]
  Sync local data to B2; makes remote EXACTLY match local and deletes remote extras (excludes cache/).

- b2-stop
  Stop any running b2 jobs (mount/copy/sync).

- b2-ls <path>
  List files on the remote at the given path (relative to /data).

- b2-pull-dir <path> [--dry-run]
  Download a directory from B2 into /workspace/data/<path>.

- b2-push-dir <path> [--dry-run] [--force]
  Upload a local directory from /workspace/data/<path> to B2.

- b2-wipe [--dry-run]
  Purge ALL files under the remote data prefix (irreversible — requires typing 'destroy').

- b2-env
  Print WHEREWILD_DATA_ROOT for the mount path.
EOF
}

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

api() {
  local log_dir="/workspace/logs"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/api.pid"

  ww_load_b2_env
  local data_root
  data_root="$(ww_data_root "$@")"
  export WHEREWILD_DATA_ROOT="$data_root"

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "api: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  if [[ -f "$log_dir/api.log" ]]; then
    mv -f "$log_dir/api.log" "$log_dir/api.previous.log"
  fi

  setsid uv run --env-file /workspace/.env uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info \
    > "$log_dir/api.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "api started: http://localhost:8000/docs (data: $data_root)"
}

api-fg() {
  ww_load_b2_env
  local data_root
  data_root="$(ww_data_root "$@")"
  export WHEREWILD_DATA_ROOT="$data_root"
  _uv uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
}

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

pt() {
  local extra=()
  local args=()
  for arg in "$@"; do
    case "$arg" in
      --temporal) extra+=("--live") ;;
      *) args+=("$arg") ;;
    esac
  done
  _uv pytest --cov --cov-report=term-missing "${extra[@]}" "${args[@]}"
}

pl() {
  _uv ruff check . "$@"
}

pp() {
  local lint_only=() shared=()
  for arg in "$@"; do
    case "$arg" in
      --fix|--unsafe-fixes) lint_only+=("$arg") ;;
      *) shared+=("$arg") ;;
    esac
  done
  pl "${lint_only[@]}" "${shared[@]}" && pt "${shared[@]}"
}

_resolve_script() {
  local name="$1"
  name="${name#./}"
  name="${name#/}"
  name="${name%.py}"

  # already rooted at scripts/ or deeper
  if [[ "$name" == scripts/* ]]; then
    echo "$name"
    return
  fi

  # direct match under scripts/
  if [[ -f "/workspace/scripts/${name}.py" ]]; then
    echo "scripts/${name}"
    return
  fi

  # search subdirectories of scripts/
  local found
  found=$(find /workspace/scripts -name "${name}.py" -not -path "*/__pycache__/*" 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    found="${found#/workspace/}"
    echo "${found%.py}"
    return
  fi

  echo "scripts/${name}"
}

pd() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pd: provide a script name"
    return 1
  fi

  module="$(_resolve_script "$module")"
  module="${module//\//.}"

  _uv python -m "$module" "$@"
}

pdb() {
  local module="$1"
  shift

  if [[ -z "$module" ]]; then
    echo "pdb: provide a script name"
    return 1
  fi

  module="$(_resolve_script "$module")"

  local log_dir="/workspace/logs/scripts"
  local pid_dir="/workspace/logs/pids"
  local log_name="${module##*/}"
  local pid_file="$pid_dir/$log_name.pid"
  local log_file="$log_dir/$log_name.log"
  local module_dotted="${module//\//.}"

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "pdb: $log_name already running (use pdbs $log_name to stop)"
    return 1
  fi

  PYTHONUNBUFFERED=1 setsid uv run --env-file /workspace/.env python -u -m "$module_dotted" "$@" > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "pdb started: $log_file"
}

pdbs() {
  local module="$1"

  if [[ -z "$module" ]]; then
    echo "pdbs: provide a script name"
    return 1
  fi

  local module_path
  module_path="$(_resolve_script "$module")"

  local log_name="${module_path##*/}"
  local pid_file="/workspace/logs/pids/$log_name.pid"
  local module_dotted="${module_path//\//.}"

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
      module="$(_resolve_script "$module")"

      log_name="${module##*/}"
      pid_file="$pid_dir/$log_name.pid"
      log_file="$log_dir/$log_name.log"

      module="${module//\//.}"

      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "pdbc: $log_name already running (use pdbs $log_name to stop)"
        break
      fi

      PYTHONUNBUFFERED=1 setsid uv run --env-file /workspace/.env python -u -m "$module" > "$log_file" 2>&1 &
      pid="$!"
      echo "$pid" > "$pid_file"
      wait "$pid"
      rm -f "$pid_file"
    done
  ) &

  echo "pdbc started: $log_dir"
}

sync-gis-layers() {
  local dest="${WW_GIS_LAYERS_DEST:-}"
  if [[ -z "$dest" ]]; then
    echo "sync-gis-layers: WW_GIS_LAYERS_DEST is not set"
    return 1
  fi
  local src="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}/gis/layers"
  local transfers="${WW_RCLONE_TRANSFERS:-8}"
  local log_dir="/workspace/logs/rclone"
  local log_file="${log_dir}/gis_layers_sync.log"
  mkdir -p "$log_dir"
  echo "syncing $src → $dest (log: $log_file)"
  rclone copy "$src" "$dest" \
    --config /workspace/docker/rclone.conf \
    --transfers "$transfers" \
    --stats-one-line \
    --stats 60m \
    --log-file "$log_file" &
  echo "running in background (pid $!)"
  echo "  tail -f $log_file"
}

ww-help() {
  cat <<'EOF'
api [--remote|--local]   start api in background (default: auto-detect mount)
api-fg [--remote|--local] start api in foreground (with reload)
api-stop                 stop api

b2-help   show B2 storage commands

sync-gis-layers          copy gis/layers/ to prod server in background

pt                   run tests with coverage
pt --temporal        run live S3 end-to-end tests
pl                   lint (ruff)
pp                   lint + test (pipeline approximation)

pd  <script>         run script in foreground  (scripts/ prefix assumed)
pdb <script>         run script in background with logging
pdbs <script>        stop a background script
pdbc <s1> <s2> ...   run scripts sequentially in background (chain)
EOF
}
