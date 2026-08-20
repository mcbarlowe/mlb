#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${BARLOWE_PAPER_TRADE_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${BARLOWE_PAPER_TRADE_UV_BIN:-/usr/local/bin/uv}"
RUN_PATTERN="scripts/paper_trade_moneyline.py"
WINDOW_TZ="${BARLOWE_PAPER_TRADE_TZ:-America/New_York}"
WINDOW_START_HOUR="${BARLOWE_PAPER_TRADE_START_HOUR:-5}"
WINDOW_END_HOUR="${BARLOWE_PAPER_TRADE_END_HOUR:-13}"
RUN_DAILY_ETL="${BARLOWE_PAPER_TRADE_RUN_DAILY_ETL:-1}"
ETL_DATE="${BARLOWE_PAPER_TRADE_ETL_DATE:-}"
FILL_CLOSE_LINES="${BARLOWE_PAPER_TRADE_FILL_CLOSE_LINES:-1}"
SETTLE_FIRST="${BARLOWE_PAPER_TRADE_SETTLE_FIRST:-1}"
CSV_LOG="${BARLOWE_PAPER_TRADE_CSV_LOG:-1}"
ALL_GAMES="${BARLOWE_PAPER_TRADE_ALL_GAMES:-0}"
SKIP_ACTIVE_ROSTERS="${BARLOWE_PAPER_TRADE_SKIP_ACTIVE_ROSTERS:-0}"
TARGET_DATE="${BARLOWE_PAPER_TRADE_DATE:-}"
# Empty means use the script default (0.03). The strategy_version label is derived from this
# value, so changing it starts a separate record rather than pooling with the old threshold.
EDGE_THRESHOLD="${BARLOWE_PAPER_TRADE_EDGE:-}"

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

for name in ODDS_API_KEY MLFLOW_TRACKING_URI MLB_DB_NAME MLB_DB_USER MLB_DB_PASSWORD MLB_DB_HOST MLB_DB_PORT MLB_DB_SCHEMA; do
  load_from_zsh "$name"
done

daily_etl_args=(run python scripts/run_daily_postgres_etl.py)
if [[ -n "$ETL_DATE" ]]; then
  daily_etl_args+=(--date "$ETL_DATE")
fi

paper_args=(run python scripts/paper_trade_moneyline.py)
if [[ -n "$TARGET_DATE" ]]; then
  paper_args+=(--date "$TARGET_DATE")
fi
if [[ -n "$EDGE_THRESHOLD" ]]; then
  paper_args+=(--edge-threshold "$EDGE_THRESHOLD")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  paper_args+=(--dry-run)
fi
if [[ "$CSV_LOG" != "1" ]]; then
  paper_args+=(--no-csv-log)
fi
if [[ "$ALL_GAMES" == "1" ]]; then
  paper_args+=(--all-games)
fi
if [[ "$SKIP_ACTIVE_ROSTERS" == "1" ]]; then
  paper_args+=(--skip-active-rosters)
fi

close_line_args=(run python scripts/fill_paper_trade_closing_lines.py)
if [[ "$DRY_RUN" == "1" ]]; then
  close_line_args+=(--dry-run)
fi

settle_args=(run python scripts/settle_paper_trades.py --db --refresh-final-scores)
if [[ "$DRY_RUN" == "1" ]]; then
  settle_args+=(--dry-run)
fi

current_hour="$(TZ="$WINDOW_TZ" date +%H)"
current_hour=$((10#$current_hour))
in_window=0
if (( current_hour >= WINDOW_START_HOUR && current_hour <= WINDOW_END_HOUR )); then
  in_window=1
fi

if [[ "$VALIDATE" == "1" ]]; then
  echo "repo: $REPO_DIR"
  echo "uv: $UV_BIN"
  echo "window tz: $WINDOW_TZ"
  echo "window hours inclusive: $WINDOW_START_HOUR-$WINDOW_END_HOUR"
  echo "current window hour: $current_hour"
  echo "in window: $in_window"
  echo "daily ETL before settlement: $RUN_DAILY_ETL"
  echo "daily ETL date: ${ETL_DATE:-yesterday}"
  echo "fill missing close lines: $FILL_CLOSE_LINES"
  echo "settle first: $SETTLE_FIRST"
  echo "csv log: $CSV_LOG"
  echo "all games: $ALL_GAMES"
  echo "skip active rosters: $SKIP_ACTIVE_ROSTERS"
  echo "target date: ${TARGET_DATE:-today}"
  echo "edge gate: ${EDGE_THRESHOLD:-script default 0.03}"
  echo "odds api key present: ${ODDS_API_KEY:+yes}"
  echo "mlflow uri present: ${MLFLOW_TRACKING_URI:+yes}"
  echo "pipeline order: daily ETL -> fill close lines -> settle paper trades -> opening-line scan"
  echo "daily ETL command: $UV_BIN ${daily_etl_args[*]}"
  echo "close-line fill command: $UV_BIN ${close_line_args[*]}"
  echo "settle command: $UV_BIN ${settle_args[*]}"
  echo "paper command: $UV_BIN ${paper_args[*]}"
  exit 0
fi

cd "$REPO_DIR"

if [[ "$FORCE" != "1" && "$in_window" != "1" ]]; then
  echo "[$(date '+%F %T')] outside paper-trade window $WINDOW_START_HOUR-$WINDOW_END_HOUR $WINDOW_TZ; skipping"
  exit 0
fi

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] paper-trade logger already running; skipping new launch"
  exit 0
fi

if [[ "$RUN_DAILY_ETL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(date '+%F %T')] dry-run: skipping daily ETL refresh"
  else
    echo "[$(date '+%F %T')] refreshing final game data before settlement"
    echo "[$(date '+%F %T')] command: $UV_BIN ${daily_etl_args[*]}"
    if ! "$UV_BIN" "${daily_etl_args[@]}"; then
      echo "[$(date '+%F %T')] daily ETL step failed; continuing to settlement and opening-line scan" >&2
    fi
  fi
fi

if [[ "$FILL_CLOSE_LINES" == "1" ]]; then
  echo "[$(date '+%F %T')] filling missing paper-trade close lines"
  echo "[$(date '+%F %T')] command: $UV_BIN ${close_line_args[*]}"
  if ! "$UV_BIN" "${close_line_args[@]}"; then
    echo "[$(date '+%F %T')] close-line fill step failed; continuing to settlement and opening-line scan" >&2
  fi
fi

if [[ "$SETTLE_FIRST" == "1" ]]; then
  echo "[$(date '+%F %T')] settling paper trades before opening-line scan"
  if ! "$UV_BIN" "${settle_args[@]}"; then
    echo "[$(date '+%F %T')] settlement step failed; continuing to opening-line scan" >&2
  fi
fi

if [[ -z "${ODDS_API_KEY:-}" ]]; then
  echo "[$(date '+%F %T')] ODDS_API_KEY is not set; cannot fetch opening lines" >&2
  exit 1
fi
echo "[$(date '+%F %T')] running paper-trade opening-line scan"
echo "[$(date '+%F %T')] command: $UV_BIN ${paper_args[*]}"
exec "$UV_BIN" "${paper_args[@]}"
