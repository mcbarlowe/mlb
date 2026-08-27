#!/bin/bash
# LaunchAgent runner: settle the previous night's paper bets and email the daily
# betting report. Self-contained morning job (backfill -> settle -> email), so it
# supersedes run_daily_paper_pipeline.sh and adds the email + prop coverage.
#
# Secrets are pulled from the shared betting email env and then the interactive
# zsh env, never the plist:
#   MLB_REPORT_GMAIL_APP_PASSWORD or EMAIL_PASSWORD       (required to send)
#   MLB_REPORT_GMAIL_USER or BETTING_ARB_EMAIL_USERNAME   (sender)
#   MLB_REPORT_EMAIL_TO                                   (default mcbarlowe@gmail.com)
set -euo pipefail
REPO_DIR="/Users/matthewbarlowe/code/python/mlb"
UV_BIN="/usr/local/bin/uv"
RUN_PATTERN="scripts/email_daily_betting_report.py"
SHARED_EMAIL_ENV="${BARLOWE_SHARED_EMAIL_ENV:-/Users/matthewbarlowe/.config/betting/arbitrage.env}"
cd "$REPO_DIR"

if [[ -f "$SHARED_EMAIL_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SHARED_EMAIL_ENV"
  set +a
fi


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
  MLB_REPORT_GMAIL_APP_PASSWORD \
  MLB_REPORT_GMAIL_USER \
  MLB_REPORT_EMAIL_TO \
  EMAIL_PASSWORD \
  BETTING_ARB_EMAIL_FROM \
  BETTING_ARB_EMAIL_USERNAME \
  BARLOWE_PAPER_REPORT_EMAIL_TO \
  MLB_DB_NAME \
  MLB_DB_USER \
  MLB_DB_PASSWORD \
  MLB_DB_HOST \
  MLB_DB_PORT \
  MLB_DB_SCHEMA; do
  load_from_zsh "$name"
done

: "${MLB_REPORT_GMAIL_APP_PASSWORD:=${EMAIL_PASSWORD:-}}"
: "${MLB_REPORT_GMAIL_USER:=${BETTING_ARB_EMAIL_USERNAME:-${BETTING_ARB_EMAIL_FROM:-}}}"
if [[ -n "${BARLOWE_PAPER_REPORT_EMAIL_TO:-}" && -z "${MLB_REPORT_EMAIL_TO:-}" ]]; then
  MLB_REPORT_EMAIL_TO="$BARLOWE_PAPER_REPORT_EMAIL_TO"
fi
export MLB_REPORT_GMAIL_APP_PASSWORD MLB_REPORT_GMAIL_USER
if [[ -n "${MLB_REPORT_EMAIL_TO:-}" ]]; then
  export MLB_REPORT_EMAIL_TO
fi

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] report already running; skipping"
  exit 0
fi

echo "[$(date '+%F %T')] backfilling last night + settling bets"
"$UV_BIN" run python scripts/run_daily_postgres_etl.py || \
  echo "[$(date '+%F %T')] backfill/settle step failed (continuing to report)"

echo "[$(date '+%F %T')] emailing daily betting report"
"$UV_BIN" run python scripts/email_daily_betting_report.py
