if [ -f /workspace/.env ]; then
    set -a
    # shellcheck disable=SC1091
    . /workspace/.env
    set +a
fi

_uv() {
    uv run --env-file /workspace/.env "$@"
}

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

  setsid _uv uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info \
    > "$log_dir/api.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "api started: http://localhost:8000/docs"
}

api-fg() {
  _uv uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info --reload
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
  _uv pytest --cov --cov-report=term-missing "$@"
}

pl() {
  _uv ruff check . "$@"
}

pp() {
  pl "$@" && pt
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

  _uv python -m "$module" "$@"
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

  PYTHONUNBUFFERED=1 setsid _uv python -u -m "$module" "$@" > "$log_file" 2>&1 &
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

      PYTHONUNBUFFERED=1 setsid _uv python -u -m "$module" > "$log_file" 2>&1 &
      pid="$!"
      echo "$pid" > "$pid_file"
      wait "$pid"
      rm -f "$pid_file"
    done
  ) &

  echo "pdbc started: $log_dir"
}

ww-help() {
  cat <<'EOF'
api        start api in background
api-fg     start api in foreground (with reload)
api-stop   stop api

pt         run tests with coverage
pl         lint (ruff)
pp         lint + test (pipeline approximation)

pd  <script>         run script in foreground  (scripts/ prefix assumed)
pdb <script>         run script in background with logging
pdbs <script>        stop a background script
pdbc <s1> <s2> ...   run scripts sequentially in background (chain)
EOF
}
