#!/bin/bash
# Materialized from the mlb wheel by mlb-install-agents.
# Edit the packaged template, reinstall, and redeploy; local edits are lost.
set -euo pipefail

STATE_ROOT="${MLB_STATE_ROOT:-__MLB_STATE_ROOT__}"
BIN_DIR="${MLB_BIN_DIR:-__MLB_BIN_DIR__}"
CAFFEINATE_BIN="/usr/bin/caffeinate"
LOCK_DIR="${TMPDIR:-/tmp}/com.barloweanalytics.daily-season-projection.lock"
LOCK_OWNED=0

POST_ENABLED="${BARLOWE_SEASON_PROJECTION_POST:-1}"
POST_PROVIDER="${BARLOWE_SEASON_PROJECTION_POST_PROVIDER:-bluesky}"
SEASON="${BARLOWE_SEASON_PROJECTION_SEASON:-}"
AS_OF="${BARLOWE_SEASON_PROJECTION_AS_OF:-}"
TRIALS="${BARLOWE_SEASON_PROJECTION_TRIALS:-5000}"
TUNE_TRIALS="${BARLOWE_SEASON_PROJECTION_TUNE_TRIALS:-1000}"
OUTPUT_DIR="${BARLOWE_SEASON_PROJECTION_OUTPUT_DIR:-}"
RAW_DATA_PATH="${BARLOWE_SEASON_PROJECTION_RAW_DATA_PATH:-data/raw/livefeeds}"
REFRESH_LOOKBACK_DAYS="${BARLOWE_SEASON_PROJECTION_REFRESH_LOOKBACK_DAYS:-3}"
MAX_REFRESH_GAMES="${BARLOWE_SEASON_PROJECTION_MAX_REFRESH_GAMES:-500}"
CONCURRENCY_LIMIT="${BARLOWE_SEASON_PROJECTION_CONCURRENCY_LIMIT:-15}"
EXTRA_ARGS="${BARLOWE_SEASON_PROJECTION_EXTRA_ARGS:-}"

# The raw feed path and the default output directory are working-directory
# relative, so the state root is the working directory.
cd "$STATE_ROOT"

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

for name in \
  MLB_DB_NAME \
  MLB_DB_USER \
  MLB_DB_PASSWORD \
  MLB_DB_HOST \
  MLB_DB_PORT \
  MLB_DB_SCHEMA \
  MLFLOW_TRACKING_URI; do
  load_from_zsh "$name"
done

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
    X_API_REFRESH_TOKEN; do
    load_from_zsh "$name"
  done
fi

args=(--trials "$TRIALS" --tune-trials "$TUNE_TRIALS" --raw-data-path "$RAW_DATA_PATH" --refresh-lookback-days "$REFRESH_LOOKBACK_DAYS" --max-refresh-games "$MAX_REFRESH_GAMES" --concurrency-limit "$CONCURRENCY_LIMIT")
if [[ -n "$SEASON" ]]; then
  args+=(--season "$SEASON")
fi
if [[ -n "$AS_OF" ]]; then
  args+=(--as-of "$AS_OF")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$POST_ENABLED" == "1" ]]; then
  args+=(--post --post-provider "$POST_PROVIDER")
else
  args+=(--no-post)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  args+=("${extra_args_array[@]}")
fi

if [[ "${1:-}" == "--validate" ]]; then
  echo "state root: $STATE_ROOT"
  echo "console scripts: $BIN_DIR"
  echo "caffeinate: $CAFFEINATE_BIN"
  echo "post enabled: $POST_ENABLED"
  echo "post provider: $POST_PROVIDER"
  echo "season: ${SEASON:-current year}"
  echo "as of: ${AS_OF:-today}"
  echo "trials: $TRIALS"
  echo "tune trials: $TUNE_TRIALS"
  echo "output dir: ${OUTPUT_DIR:-output/season_projection_<season>}"
  echo "raw data path: $RAW_DATA_PATH"
  echo "refresh lookback days: $REFRESH_LOOKBACK_DAYS"
  echo "max refresh games: $MAX_REFRESH_GAMES"
  echo "concurrency limit: $CONCURRENCY_LIMIT"
  echo "bluesky handle present: ${BLUESKY_HANDLE:+yes}"
  echo "bluesky app password present: ${BLUESKY_APP_PASSWORD:+yes}"
  echo "x api client id present: ${X_API_CLIENT_ID:+yes}"
  echo "x api client secret present: ${X_API_CLIENT_SECRET:+yes}"
  echo "x api key present: ${X_API_KEY:+yes}"
  echo "x api key secret present: ${X_API_KEY_SECRET:+yes}"
  echo "x access token present: ${X_ACCESS_TOKEN:+yes}"
  echo "x access token secret present: ${X_ACCESS_TOKEN_SECRET:+yes}"
  echo "x api access token present: ${X_API_ACCESS_TOKEN:+yes}"
  echo "x api oauth2 access token present: ${X_API_OAUTH2_ACCESS_TOKEN:+yes}"
  echo "x api refresh token present: ${X_API_REFRESH_TOKEN:+yes}"
  echo "x api oauth2 refresh token present: ${X_API_OAUTH2_REFRESH_TOKEN:+yes}"
  echo "mlflow uri present: ${MLFLOW_TRACKING_URI:+yes}"
  echo "command: $BIN_DIR/mlb-daily-season-projection ${args[*]}"
  exit 0
fi

cleanup() {
    status=$?
    set +e
    if [[ "$LOCK_OWNED" == "1" ]]; then
        /bin/rm -f "$LOCK_DIR/pid"
        /bin/rmdir "$LOCK_DIR" 2>/dev/null
    fi
    exit "$status"
}
trap cleanup EXIT

# A console script cannot be matched by a pgrep pattern, so the run is
# serialized on an atomically created lock directory keyed by the agent label.
if ! /bin/mkdir "$LOCK_DIR" 2>/dev/null; then
    owner="$(/bin/cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ "$owner" =~ ^[0-9]+$ ]] && /bin/kill -0 "$owner" 2>/dev/null; then
        echo "[$(date '+%F %T')] daily season projection already running; skipping new launch"
        exit 0
    fi
    if [[ -z "$owner" ]]; then
        /bin/sleep 1
        owner="$(/bin/cat "$LOCK_DIR/pid" 2>/dev/null || true)"
        if [[ "$owner" =~ ^[0-9]+$ ]] && /bin/kill -0 "$owner" 2>/dev/null; then
            echo "[$(date '+%F %T')] daily season projection already running; skipping new launch"
            exit 0
        fi
    fi
    stale="${LOCK_DIR}.stale.$$"
    /bin/mv "$LOCK_DIR" "$stale"
    /bin/rm -f "$stale/pid"
    /bin/rmdir "$stale"
    /bin/mkdir "$LOCK_DIR"
fi
printf '%s\n' "$$" >"$LOCK_DIR/pid"
LOCK_OWNED=1

echo "[$(date '+%F %T')] launching daily season projection"
echo "[$(date '+%F %T')] command: $BIN_DIR/mlb-daily-season-projection ${args[*]}"
# Not exec'd: the EXIT trap must still run to release the lock directory.
"$CAFFEINATE_BIN" -is "$BIN_DIR/mlb-daily-season-projection" "${args[@]}"
