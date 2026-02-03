alias api='mkdir -p /workspace/logs && cd /workspace && setsid -f uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info --reload --reload-dir /workspace/main.py --reload-dir /workspace/util > /workspace/logs/api.log 2>&1 && echo "api started: http://localhost:8000/docs"'
alias api-fg='cd /workspace && uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info --reload --reload-dir /workspace/main.py --reload-dir /workspace/util'
alias docs='mkdir -p /workspace/logs && cd /workspace && setsid -f mkdocs serve --dev-addr 0.0.0.0:9101 > /workspace/logs/docs.log 2>&1 && echo "docs started: http://localhost:9101/"'
alias api-stop='pkill -f "uvicorn main:app" && echo "api stopped"'
alias docs-stop='pkill -f "mkdocs serve" && echo "docs stopped"'

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

  setsid python -m "$module" "$@" > "$log_file" 2>&1 &
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

      setsid python -m "$module" > "$log_file" 2>&1 &
      pid="$!"
      echo "$pid" > "$pid_file"
      wait "$pid"
      rm -f "$pid_file"
    done
  ) &

  echo "pdbc started: $log_dir"
}
