#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /etc/wherewild_aliases.sh ]]; then
  echo "wherewild-auto-pull: missing /etc/wherewild_aliases.sh"
  exit 1
fi

. /etc/wherewild_aliases.sh

# Default auto-pull hours are interpreted in the container's local timezone.
# Set the TZ environment variable if you need the schedule to run in a specific timezone.
hours_csv="${WHEREWILD_AUTO_PULL_HOURS:-3,9,15,21}"
post_run_sleep_seconds="${WHEREWILD_AUTO_PULL_POST_RUN_SLEEP_SECONDS:-60}"
log_dir="/workspace/logs/auto-pull"
rclone_log_file="/workspace/logs/rclone/clone.log"
mkdir -p "$log_dir"
log_file="${log_dir}/service.log"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$message" | tee -a "$log_file"
}

parse_hours() {
  local raw_hours="$1"
  local hour
  local cleaned=()
  IFS=',' read -r -a parsed_hours <<< "$raw_hours"
  for hour in "${parsed_hours[@]}"; do
    hour="${hour//[[:space:]]/}"
    if [[ -z "$hour" ]]; then
      continue
    fi
    if [[ ! "$hour" =~ ^[0-9]{1,2}$ ]] || (( hour < 0 || hour > 23 )); then
      echo "Invalid WHEREWILD_AUTO_PULL_HOURS entry: $hour" >&2
      return 1
    fi
    cleaned+=("$hour")
  done

  if [[ "${#cleaned[@]}" -eq 0 ]]; then
    echo "WHEREWILD_AUTO_PULL_HOURS must define at least one hour" >&2
    return 1
  fi

  printf '%s\n' "${cleaned[@]}" | sort -n | uniq
}

mapfile -t schedule_hours < <(parse_hours "$hours_csv")

next_run_epoch() {
  local now candidate hour
  now="$(date +%s)"

  for hour in "${schedule_hours[@]}"; do
    candidate="$(date -d "today ${hour}:00:00" +%s)"
    if (( candidate > now )); then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  date -d "tomorrow ${schedule_hours[0]}:00:00" +%s
}

log "Auto-pull scheduler started; hours=${hours_csv} local_data_root=${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"

while true; do
  next_epoch="$(next_run_epoch)"
  now_epoch="$(date +%s)"
  sleep_seconds="$((next_epoch - now_epoch))"
  next_label="$(date -d "@${next_epoch}" '+%Y-%m-%d %H:%M:%S %Z')"

  if (( sleep_seconds > 0 )); then
    log "Sleeping ${sleep_seconds}s until ${next_label}"
    sleep "$sleep_seconds"
  fi

  log "Triggering scheduled b2-pull-all"
  if b2-pull-all >> "$log_file" 2>&1; then
    log "b2-pull-all launch requested; transfer details will be written to ${rclone_log_file}"
  else
    log "b2-pull-all could not be launched; helper output was written to ${log_file}"
  fi

  sleep "$post_run_sleep_seconds"
done
