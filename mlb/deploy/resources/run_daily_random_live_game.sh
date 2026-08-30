#!/bin/bash
# Materialized from the mlb wheel by mlb-install-agents.
# Edit the packaged template, reinstall, and redeploy; local edits are lost.
set -euo pipefail

STATE_ROOT="${MLB_STATE_ROOT:-__MLB_STATE_ROOT__}"
BIN_DIR="${MLB_BIN_DIR:-__MLB_BIN_DIR__}"
CAFFEINATE_BIN="/usr/bin/caffeinate"
SOCIAL_ENV="${BARLOWE_SOCIAL_ENV:-__MLB_SOCIAL_ENV__}"
LOCK_DIR="${TMPDIR:-/tmp}/com.barloweanalytics.daily-random-live-game.lock"
LOCK_OWNED=0

POST_ENABLED="${BARLOWE_RANDOM_GAME_POST:-1}"
POST_PROVIDER="${BARLOWE_RANDOM_GAME_POST_PROVIDER:-bluesky}"
TARGET_DATE="${BARLOWE_RANDOM_GAME_DATE:-}"
LEAD_MINUTES="${BARLOWE_RANDOM_GAME_LEAD_MINUTES:-15}"
POLL_INTERVAL="${BARLOWE_RANDOM_GAME_POLL_INTERVAL:-3}"
OUTCOME_RUN_DIR="${BARLOWE_RANDOM_GAME_OUTCOME_RUN_DIR:-auto}"
SEED="${BARLOWE_RANDOM_GAME_SEED:-}"

if [[ -f "$SOCIAL_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SOCIAL_ENV"
  set +a
fi

# The models, run directories, and raw feeds this pipeline reads are all
# working-directory relative, so the state root is the working directory.
cd "$STATE_ROOT"

if [[ "${1:-}" == "--validate" ]]; then
  echo "state root: $STATE_ROOT"
  echo "console scripts: $BIN_DIR"
  echo "caffeinate: $CAFFEINATE_BIN"
  echo "post enabled: $POST_ENABLED"
  echo "post provider: $POST_PROVIDER"
  echo "target date: ${TARGET_DATE:-today}"
  echo "lead minutes: $LEAD_MINUTES"
  echo "poll interval: $POLL_INTERVAL"
  echo "outcome run dir: $OUTCOME_RUN_DIR"
  echo "seed: ${SEED:-<random>}"
  echo "bluesky handle present: ${BLUESKY_HANDLE:+yes}"
  echo "bluesky app password present: ${BLUESKY_APP_PASSWORD:+yes}"
  echo "x api client id present: ${X_API_CLIENT_ID:+yes}"
  echo "x api client secret present: ${X_API_CLIENT_SECRET:+yes}"
  echo "x api key present: ${X_API_KEY:+yes}"
  echo "x api key secret present: ${X_API_KEY_SECRET:+yes}"
  echo "x api access token present: ${X_API_ACCESS_TOKEN:+yes}"
  echo "x api oauth2 access token present: ${X_API_OAUTH2_ACCESS_TOKEN:+yes}"
  echo "x api refresh token present: ${X_API_REFRESH_TOKEN:+yes}"
  echo "x api oauth2 refresh token present: ${X_API_OAUTH2_REFRESH_TOKEN:+yes}"
  echo "x access token secret present: ${X_ACCESS_TOKEN_SECRET:+yes}"
  echo "mlflow uri present: ${MLFLOW_TRACKING_URI:+yes}"
  validate_args=(--random-game --poll-interval "$POLL_INTERVAL" --lead-minutes "$LEAD_MINUTES" --outcome-run-dir "$OUTCOME_RUN_DIR")
  if [[ -n "$TARGET_DATE" ]]; then
    validate_args+=(--date "$TARGET_DATE")
  fi
  if [[ -n "$SEED" ]]; then
    validate_args+=(--seed "$SEED")
  fi
  if [[ "$POST_ENABLED" == "1" ]]; then
    validate_args+=(--post --post-provider "$POST_PROVIDER")
  fi
  echo "command: $BIN_DIR/mlb-live-pipeline ${validate_args[*]}"
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
        echo "[$(date '+%F %T')] live pipeline already running; skipping new daily launch"
        exit 0
    fi
    if [[ -z "$owner" ]]; then
        /bin/sleep 1
        owner="$(/bin/cat "$LOCK_DIR/pid" 2>/dev/null || true)"
        if [[ "$owner" =~ ^[0-9]+$ ]] && /bin/kill -0 "$owner" 2>/dev/null; then
            echo "[$(date '+%F %T')] live pipeline already running; skipping new daily launch"
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

# Refresh pitcher movement profiles so profile-consuming models see
# yesterday's outings; a failed refresh falls back to the existing store.
echo "[$(date '+%F %T')] refreshing pitcher movement profiles"
if ! "$BIN_DIR/mlb-build-pitcher-movement-profiles"; then
  echo "[$(date '+%F %T')] profile refresh failed; continuing with existing store"
fi

args=(--random-game --poll-interval "$POLL_INTERVAL" --lead-minutes "$LEAD_MINUTES" --outcome-run-dir "$OUTCOME_RUN_DIR")
if [[ -n "$TARGET_DATE" ]]; then
  args+=(--date "$TARGET_DATE")
fi
if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi
if [[ "$POST_ENABLED" == "1" ]]; then
  args+=(--post --post-provider "$POST_PROVIDER")
fi

echo "[$(date '+%F %T')] launching daily random game pipeline"
echo "[$(date '+%F %T')] command: $BIN_DIR/mlb-live-pipeline ${args[*]}"
# Not exec'd: the EXIT trap must still run to release the lock directory.
"$CAFFEINATE_BIN" -is "$BIN_DIR/mlb-live-pipeline" "${args[@]}"
