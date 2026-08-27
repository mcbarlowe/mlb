#!/bin/bash
# LaunchAgent runner: backfill yesterday, settle paper bets, then email the
# daily betting report. The report is gated on a successful ETL/settlement run;
# if that step fails, email a failure notice instead of sending stale numbers.
#
# Secrets are pulled from the shared betting email env and then the interactive
# zsh env, never the plist:
#   MLB_REPORT_GMAIL_APP_PASSWORD or EMAIL_PASSWORD       (required to send)
#   MLB_REPORT_GMAIL_USER or BETTING_ARB_EMAIL_USERNAME   (sender)
#   MLB_REPORT_EMAIL_TO                                   (default mcbarlowe@gmail.com)
set -euo pipefail
REPO_DIR="${MLB_REPORT_REPO_DIR:-/Users/matthewbarlowe/code/python/mlb}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
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

send_etl_failure_email() {
  local log_path="$1"
  local exit_code="$2"
  if [[ "${MLB_REPORT_FAILURE_DRY_RUN:-}" == "1" ]]; then
    echo "[$(date '+%F %T')] dry-run: would email ETL failure report"
    tail -n "${MLB_REPORT_FAILURE_LOG_LINES:-120}" "$log_path"
    return 0
  fi
  /usr/bin/python3 - "$log_path" "$exit_code" <<'PY'
from __future__ import annotations

import os
import smtplib
import socket
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage

log_path = sys.argv[1]
exit_code = sys.argv[2]
recipient = os.environ.get("MLB_REPORT_EMAIL_TO") or "mcbarlowe@gmail.com"
password = os.environ.get("MLB_REPORT_GMAIL_APP_PASSWORD")
sender = os.environ.get("MLB_REPORT_GMAIL_USER") or recipient
if not password:
    raise SystemExit("MLB_REPORT_GMAIL_APP_PASSWORD is required for ETL failure emails")

with open(log_path, encoding="utf-8", errors="replace") as handle:
    lines = handle.readlines()
excerpt = "".join(lines[-120:])
subject = f"MLB daily betting report blocked: ETL failed {datetime.now():%Y-%m-%d}"
body = (
    "Daily betting report was not sent because run_daily_postgres_etl.py failed.\n\n"
    f"Host: {socket.gethostname()}\n"
    f"Repo: {os.getcwd()}\n"
    f"Exit code: {exit_code}\n\n"
    "Last ETL output:\n"
    "----------------\n"
    f"{excerpt}"
)
message = EmailMessage()
message["From"] = sender
message["To"] = recipient
message["Subject"] = subject
message.set_content(body)
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30) as smtp:
    smtp.login(sender, password)
    smtp.send_message(message)
print(f"sent ETL failure notice to {recipient}")
PY
}

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] report already running; skipping"
  exit 0
fi

etl_log="$(mktemp "${TMPDIR:-/tmp}/mlb-daily-etl.XXXXXX.log")"
trap 'rm -f "$etl_log"' EXIT

echo "[$(date '+%F %T')] backfilling last night + settling bets"
if ! "$UV_BIN" run python scripts/run_daily_postgres_etl.py 2>&1 | tee "$etl_log"; then
  etl_status="${PIPESTATUS[0]}"
  echo "[$(date '+%F %T')] backfill/settle step failed; sending failure notice"
  send_etl_failure_email "$etl_log" "$etl_status"
  exit "$etl_status"
fi

echo "[$(date '+%F %T')] emailing daily betting report"
"$UV_BIN" run python scripts/email_daily_betting_report.py
