#!/bin/bash
set -euo pipefail

REPO_DIR="/Users/matthewbarlowe/code/python/mlb"
UV_BIN="/usr/local/bin/uv"
CAFFEINATE_BIN="/usr/bin/caffeinate"
RUN_PATTERN="scripts/run_daily_sim_slate.py"

POST_ENABLED="${BARLOWE_DAILY_SIM_POST:-1}"
WATCH_STARTERS="${BARLOWE_DAILY_SIM_WATCH_STARTERS:-1}"
POST_PROVIDER="${BARLOWE_DAILY_SIM_POST_PROVIDER:-both}"
ALL_GAMES="${BARLOWE_DAILY_SIM_ALL_GAMES:-0}"
TARGET_DATE="${BARLOWE_DAILY_SIM_DATE:-}"
SIMS="${BARLOWE_DAILY_SIM_SIMS:-2000}"
POLL_INTERVAL_MINUTES="${BARLOWE_DAILY_SIM_POLL_INTERVAL_MINUTES:-15}"
OUTCOME_RUN_DIR="${BARLOWE_DAILY_SIM_OUTCOME_RUN_DIR:-auto}"
OUTPUT_DIR="${BARLOWE_DAILY_SIM_OUTPUT_DIR:-output/sim_cards/daily}"
STATE_DIR="${BARLOWE_DAILY_SIM_STATE_DIR:-output/sim_state}"

cd "$REPO_DIR"
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

if [[ "$POST_ENABLED" == "1" ]]; then
  for name in \
    BLUESKY_HANDLE \
    BLUESKY_APP_PASSWORD \
    BLUESKY_PDS_URL \
    X_API_KEY \
    X_API_KEY_SECRET \
    X_ACCESS_TOKEN \
    X_ACCESS_TOKEN_SECRET \
    X_API_CLIENT_ID \
    X_API_CLIENT_SECRET \
    X_API_OAUTH2_ACCESS_TOKEN \
    X_API_OAUTH2_REFRESH_TOKEN \
    X_API_ACCESS_TOKEN \
    X_API_REFRESH_TOKEN \
    MLFLOW_TRACKING_URI; do
    load_from_zsh "$name"
  done
fi


if [[ "${1:-}" == "--validate" ]]; then
  echo "repo: $REPO_DIR"
  echo "uv: $UV_BIN"
  echo "caffeinate: $CAFFEINATE_BIN"
  echo "post enabled: $POST_ENABLED"
  echo "post provider: $POST_PROVIDER"
  echo "all games: $ALL_GAMES"
  echo "watch starters: $WATCH_STARTERS"
  echo "target date: ${TARGET_DATE:-today}"
  echo "sims: $SIMS"
  echo "poll interval minutes: $POLL_INTERVAL_MINUTES"
  echo "outcome run dir: $OUTCOME_RUN_DIR"
  echo "output dir: $OUTPUT_DIR"
  echo "state dir: $STATE_DIR"
  echo "bluesky handle present: ${BLUESKY_HANDLE:+yes}"
  echo "bluesky app password present: ${BLUESKY_APP_PASSWORD:+yes}"
  echo "x api client id present: ${X_API_CLIENT_ID:+yes}"
  echo "x api client secret present: ${X_API_CLIENT_SECRET:+yes}"
  echo "x api key present: ${X_API_KEY:+yes}"
  echo "x api key secret present: ${X_API_KEY_SECRET:+yes}"
  echo "x access token present: ${X_ACCESS_TOKEN:+yes}"
  echo "x api access token present: ${X_API_ACCESS_TOKEN:+yes}"
  echo "x api oauth2 access token present: ${X_API_OAUTH2_ACCESS_TOKEN:+yes}"
  echo "x api refresh token present: ${X_API_REFRESH_TOKEN:+yes}"
  echo "x api oauth2 refresh token present: ${X_API_OAUTH2_REFRESH_TOKEN:+yes}"
  echo "x access token secret present: ${X_ACCESS_TOKEN_SECRET:+yes}"
  echo "mlflow uri present: ${MLFLOW_TRACKING_URI:+yes}"
  validate_args=(run python scripts/run_daily_sim_slate.py --sims "$SIMS" --poll-interval-minutes "$POLL_INTERVAL_MINUTES" --outcome-run-dir "$OUTCOME_RUN_DIR" --output-dir "$OUTPUT_DIR" --state-dir "$STATE_DIR")
  if [[ -n "$TARGET_DATE" ]]; then
    validate_args+=(--date "$TARGET_DATE")
  fi
  if [[ "$ALL_GAMES" == "1" ]]; then
    validate_args+=(--all-games)
  fi
  if [[ "$POST_ENABLED" == "1" ]]; then
    validate_args+=(--post --post-provider "$POST_PROVIDER")
  fi
  if [[ "$WATCH_STARTERS" == "1" ]]; then
    validate_args+=(--watch-starters)
  fi
  echo "command: $UV_BIN ${validate_args[*]}"
  exit 0
fi

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] daily sim slate pipeline already running; skipping new launch"
  exit 0
fi

args=(run python scripts/run_daily_sim_slate.py --sims "$SIMS" --poll-interval-minutes "$POLL_INTERVAL_MINUTES" --outcome-run-dir "$OUTCOME_RUN_DIR" --output-dir "$OUTPUT_DIR" --state-dir "$STATE_DIR")
if [[ -n "$TARGET_DATE" ]]; then
  args+=(--date "$TARGET_DATE")
fi
if [[ "$ALL_GAMES" == "1" ]]; then
  args+=(--all-games)
fi
if [[ "$POST_ENABLED" == "1" ]]; then
  args+=(--post --post-provider "$POST_PROVIDER")
fi
if [[ "$WATCH_STARTERS" == "1" ]]; then
  args+=(--watch-starters)
fi

echo "[$(date '+%F %T')] launching daily sim slate pipeline"
echo "[$(date '+%F %T')] command: $UV_BIN ${args[*]}"
exec "$CAFFEINATE_BIN" -is "$UV_BIN" "${args[@]}"
