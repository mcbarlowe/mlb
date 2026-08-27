#!/bin/bash
set -euo pipefail

# Daily Paper Trading Pipeline
# Runs each morning to:
#   1. Backfill yesterday's game results
#   2. Settle completed paper trades
#   3. Generate trading report
#
# Usage: ./scripts/run_daily_paper_pipeline.sh [--date YYYY-MM-DD]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
EMAIL_REPORT="${BARLOWE_PAPER_REPORT_EMAIL:-1}"
EMAIL_TO="${BARLOWE_PAPER_REPORT_EMAIL_TO:-mcbarlowe@gmail.com}"
SHARED_EMAIL_ENV="${BARLOWE_SHARED_EMAIL_ENV:-/Users/matthewbarlowe/.config/betting/arbitrage.env}"

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
  EMAIL_PASSWORD \
  BETTING_ARB_EMAIL_FROM \
  BETTING_ARB_EMAIL_USERNAME \
  BETTING_ARB_SMTP_HOST \
  BETTING_ARB_SMTP_PORT \
  BARLOWE_PAPER_REPORT_EMAIL_TO; do
  load_from_zsh "$name"
done

if [[ -n "${BARLOWE_PAPER_REPORT_EMAIL_TO:-}" ]]; then
  EMAIL_TO="$BARLOWE_PAPER_REPORT_EMAIL_TO"
fi

# Parse arguments
TARGET_DATE=""
VERBOSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      TARGET_DATE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Set target date (default: today, which settles yesterday's games)
if [[ -z "$TARGET_DATE" ]]; then
  TARGET_DATE=$(date -u +%Y-%m-%d)
fi

cd "$REPO_DIR"

echo "=========================================="
echo "Daily Paper Trading Pipeline"
echo "Date: $TARGET_DATE"
echo "=========================================="
echo ""

# Step 1: Backfill yesterday's games
echo "[1/2] Backfilling game results..."
$UV_BIN run python scripts/run_daily_postgres_etl.py --date "$TARGET_DATE"
echo ""

# Step 2: Settle paper trades
echo "[2/2] Settling paper trades..."
settle_args=(run python scripts/settle_daily_paper_trades.py --date "$TARGET_DATE")
if [[ $VERBOSE -eq 1 ]]; then
  settle_args+=(--verbose)
fi
if [[ "$EMAIL_REPORT" == "1" ]]; then
  settle_args+=(--email-report --email-to "$EMAIL_TO")
fi
$UV_BIN "${settle_args[@]}"

echo ""
echo "=========================================="
echo "Pipeline Complete"
echo "=========================================="
