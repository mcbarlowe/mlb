#!/bin/bash
# LaunchAgent runner: shop batter props across books and text new +EV plays.
# Follows the run_daily_sim_slate.sh conventions (zshrc secret loading,
# single-instance guard, absolute uv path).
set -euo pipefail
REPO_DIR="/Users/matthewbarlowe/code/python/mlb"
UV_BIN="/usr/local/bin/uv"
RUN_PATTERN="scripts/shop_batter_props.py"
RECIPIENT="${BARLOWE_PROP_SHOP_RECIPIENT:-matt@barloweanalytics.com}"
MIN_EV="${BARLOWE_PROP_SHOP_MIN_EV:-0.11}"
MIN_GP="${BARLOWE_PROP_SHOP_MIN_GP:-150}"
MARKETS="${BARLOWE_PROP_SHOP_MARKETS:-}"
MARKET_MIN_EV="${BARLOWE_PROP_SHOP_MARKET_MIN_EV:-batter_home_runs=0.15}"
SHRINK_K="${BARLOWE_PROP_SHOP_SHRINK_K:-50}"
HALF_LIFE="${BARLOWE_PROP_SHOP_HALF_LIFE:-400}"
START_HOUR="${BARLOWE_PROP_SHOP_START_HOUR:-8}"
END_HOUR="${BARLOWE_PROP_SHOP_END_HOUR:-23}"
NOTIFY_METHOD="${BARLOWE_PROP_SHOP_NOTIFY_METHOD:-both}"
NTFY_TOPIC="${BARLOWE_PROP_SHOP_NTFY_TOPIC:-barlowe-props-c47d9e2a51b3}"
cd "$REPO_DIR"

HOUR_ET="$(TZ=America/New_York date '+%-H')"
if (( HOUR_ET < START_HOUR || HOUR_ET > END_HOUR )); then
  exit 0
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
load_from_zsh ODDS_API_KEY

if pgrep -f "$RUN_PATTERN" >/dev/null; then
  echo "[$(date '+%F %T')] prop shop already running; skipping"
  exit 0
fi

args=(run python scripts/shop_batter_props.py
      --notify --recipient "$RECIPIENT"
      --notify-method "$NOTIFY_METHOD" --ntfy-topic "$NTFY_TOPIC"
      --min-ev "$MIN_EV" --min-gp "$MIN_GP"
      --market-min-ev "$MARKET_MIN_EV" --shrink-k "$SHRINK_K"
      --recency-half-life "$HALF_LIFE")
if [[ -n "$MARKETS" ]]; then
  args+=(--markets "$MARKETS")
fi
echo "[$(date '+%F %T')] shopping batter props"
"$UV_BIN" "${args[@]}"
echo "[$(date '+%F %T')] settling prop paper bets"
"$UV_BIN" run python scripts/settle_prop_alerts.py --push --ntfy-topic "$NTFY_TOPIC" || \
  echo "[$(date '+%F %T')] settlement failed (non-fatal)"
