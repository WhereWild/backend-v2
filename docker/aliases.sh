api() {
  local log_dir="/workspace/logs"
  local pid_dir="/workspace/logs/pids"
  local pid_file="${pid_dir}/api.pid"

  mkdir -p "$log_dir" "$pid_dir"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "api: already running (pid $(cat "$pid_file"))"
    return 0
  fi

  if [[ -f "$log_dir/api.log" ]]; then
    mv -f "$log_dir/api.log" "$log_dir/api.previous.log"
  fi

  setsid uv run uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info \
    > "$log_dir/api.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "api started: http://localhost:8000/docs"
}

api-fg() {
  uv run uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info --reload
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
