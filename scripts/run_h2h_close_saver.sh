#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${BARLOWE_H2H_CLOSE_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${BARLOWE_H2H_CLOSE_UV_BIN:-/usr/local/bin/uv}"
RUN_PATTERN="scripts/save_current_h2h_closing_lines.py"
WINDOW_TZ="${BARLOWE_H2H_CLOSE_TZ:-America/New_York}"
WINDOW_START_HOUR="${BARLOWE_H2H_CLOSE_START_HOUR:-11}"
WINDOW_END_HOUR="${BARLOWE_H2H_CLOSE_END_HOUR:-23}"
TARGET_DATE="${BARLOWE_H2H_CLOSE_DATE:-}"
REGIONS="${BARLOWE_H2H_CLOSE_REGIONS:-us}"
MAX_MATCH_HOURS="${BARLOWE_H2H_CLOSE_MAX_MATCH_HOURS:-12}"
ALL_GAMES="${BARLOWE_H2H_CLOSE_ALL_GAMES:-0}"
DB_LOG="${BARLOWE_H2H_CLOSE_DB_LOG:-1}"

DRY_RUN=0
FORCE=0
VALIDATE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    --validate) VALIDATE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

load_from_zsh() {
  local name="$1"
  if [[ -n "${!name:-}" ]] || ! command -v zsh >/dev/null; then
    return
  fi
  local value
  value="$(zsh -ic "printenv $name" 2>/dev/null || true)"
  if [[ -n "$value" ]]; then
    export "$name=$value"
  fi
}

for name in ODDS_API_KEY MLB_DB_NAME MLB_DB_USER MLB_DB_PASSWORD MLB_DB_HOST MLB_DB_PORT MLB_DB_SCHEMA; do
  load_from_zsh "$name"
done

close_args=(run python scripts/save_current_h2h_closing_lines.py --regions "$REGIONS" --max-match-hours "$MAX_MATCH_HOURS")
if [[ -n "$TARGET_DATE" ]]; then
  close_args+=(--date "$TARGET_DATE")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  close_args+=(--dry-run)
fi
if [[ "$ALL_GAMES" == "1" ]]; then
  close_args+=(--all-games)
fi
if [[ "$DB_LOG" != "1" ]]; then
  close_args+=(--no-db-log)
fi

current_hour="$(TZ="$WINDOW_TZ" date +%H)"
current_hour=$((10#$current_hour))
in_window=0
if (( WINDOW_START_HOUR <= WINDOW_END_HOUR )); then
  if (( current_hour >= WINDOW_START_HOUR && current_hour <= WINDOW_END_HOUR )); then
    in_window=1
  fi
else
  if (( current_hour >= WINDOW_START_HOUR || current_hour <= WINDOW_END_HOUR )); then
    in_window=1
  fi
fi

if [[ "$VALIDATE" == "1" ]]; then
  echo "repo: $REPO_DIR"
  echo "uv: $UV_BIN"
  echo "window tz: $WINDOW_TZ"
  echo "window hours inclusive: $WINDOW_START_HOUR-$WINDOW_END_HOUR"
  echo "current window hour: $current_hour"
  echo "in window: $in_window"
  echo "target date: ${TARGET_DATE:-today}"
  echo "regions: $REGIONS"
  echo "max match hours: $MAX_MATCH_HOURS"
  echo "all games: $ALL_GAMES"
  echo "db log: $DB_LOG"
  echo "odds api key present: ${ODDS_API_KEY:+yes}"
  echo "close saver command: $UV_BIN ${close_args[*]}"
  exit 0
fi

cd "$REPO_DIR"

if [[ "$FORCE" != "1" && "$in_window" != "1" ]]; then
  echo "[$(date '+%F %T')] outside h2h close saver window $WINDOW_START_HOUR-$WINDOW_END_HOUR $WINDOW_TZ; skipping"
  exit 0
fi

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] h2h close saver already running; skipping new launch"
  exit 0
fi

if [[ -z "${ODDS_API_KEY:-}" ]]; then
  echo "[$(date '+%F %T')] ODDS_API_KEY is not set; cannot fetch h2h close lines" >&2
  exit 1
fi

echo "[$(date '+%F %T')] saving current h2h board as close lines"
echo "[$(date '+%F %T')] command: $UV_BIN ${close_args[*]}"
exec "$UV_BIN" "${close_args[@]}"
