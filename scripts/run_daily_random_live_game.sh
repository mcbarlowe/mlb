#!/bin/bash
set -euo pipefail

REPO_DIR="/Users/matthewbarlowe/code/python/mlb"
UV_BIN="/usr/local/bin/uv"
CAFFEINATE_BIN="/usr/bin/caffeinate"
RUN_PATTERN="scripts/run_live_pipeline.py --random-game"

POST_ENABLED="${BARLOWE_RANDOM_GAME_POST:-1}"
TARGET_DATE="${BARLOWE_RANDOM_GAME_DATE:-}"
LEAD_MINUTES="${BARLOWE_RANDOM_GAME_LEAD_MINUTES:-15}"
POLL_INTERVAL="${BARLOWE_RANDOM_GAME_POLL_INTERVAL:-3}"
OUTCOME_RUN_DIR="${BARLOWE_RANDOM_GAME_OUTCOME_RUN_DIR:-auto}"
SEED="${BARLOWE_RANDOM_GAME_SEED:-}"

cd "$REPO_DIR"

if [[ "${1:-}" == "--validate" ]]; then
  echo "repo: $REPO_DIR"
  echo "uv: $UV_BIN"
  echo "caffeinate: $CAFFEINATE_BIN"
  echo "post enabled: $POST_ENABLED"
  echo "target date: ${TARGET_DATE:-today}"
  echo "lead minutes: $LEAD_MINUTES"
  echo "poll interval: $POLL_INTERVAL"
  echo "outcome run dir: $OUTCOME_RUN_DIR"
  echo "seed: ${SEED:-<random>}"
  echo "bluesky handle present: ${BLUESKY_HANDLE:+yes}"
  echo "bluesky app password present: ${BLUESKY_APP_PASSWORD:+yes}"
  echo "mlflow uri present: ${MLFLOW_TRACKING_URI:+yes}"
  echo "command: $UV_BIN run python scripts/run_live_pipeline.py --random-game --poll-interval $POLL_INTERVAL --lead-minutes $LEAD_MINUTES --outcome-run-dir $OUTCOME_RUN_DIR ${POST_ENABLED:+--post}"
  exit 0
fi

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] live pipeline already running; skipping new daily launch"
  exit 0
fi

args=(run python scripts/run_live_pipeline.py --random-game --poll-interval "$POLL_INTERVAL" --lead-minutes "$LEAD_MINUTES" --outcome-run-dir "$OUTCOME_RUN_DIR")
if [[ -n "$TARGET_DATE" ]]; then
  args+=(--date "$TARGET_DATE")
fi
if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi
if [[ "$POST_ENABLED" == "1" ]]; then
  args+=(--post)
fi

echo "[$(date '+%F %T')] launching daily random game pipeline"
echo "[$(date '+%F %T')] command: $UV_BIN ${args[*]}"
exec "$CAFFEINATE_BIN" -is "$UV_BIN" "${args[@]}"
