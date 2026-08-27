#!/bin/bash
set -euo pipefail

# Install the daily paper trades pipeline as a launchd job
# Usage: ./scripts/install_paper_trades_scheduler.sh [--time HH:MM]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_SOURCE="$SCRIPT_DIR/com.barloweanalytics.paper-trades-daily.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.barloweanalytics.paper-trades-daily.plist"
LOG_DIR="$HOME/.mlb"

# Parse arguments
HOUR=9
MINUTE=0
for arg in "$@"; do
  case "$arg" in
    --time)
      TIME_ARG="$2"
      HOUR=$(echo "$TIME_ARG" | cut -d: -f1)
      MINUTE=$(echo "$TIME_ARG" | cut -d: -f2)
      shift 2
      ;;
    --uninstall)
      echo "Uninstalling daily paper trades job..."
      launchctl bootout gui/$(id -u) "$PLIST_DEST" 2>/dev/null || true
      rm -f "$PLIST_DEST"
      echo "Done. Job uninstalled."
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

echo "=========================================="
echo "Installing Daily Paper Trades Pipeline"
echo "=========================================="
echo ""

# Create log directory
echo "[1/4] Creating log directory..."
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"
echo "      ✓ Logs will be written to: $LOG_DIR/paper_trades.log"
echo ""

# Create LaunchAgents directory if needed
echo "[2/4] Setting up LaunchAgents directory..."
mkdir -p "$HOME/Library/LaunchAgents"
chmod 755 "$HOME/Library/LaunchAgents"
echo "      ✓ Created: $HOME/Library/LaunchAgents"
echo ""

# Copy and modify plist
echo "[3/4] Installing launchd configuration..."
cp "$PLIST_SOURCE" "$PLIST_DEST"
chmod 644 "$PLIST_DEST"
echo "      ✓ Installed: $PLIST_DEST"
echo ""

# Update time if specified
if [[ -n "${TIME_ARG:-}" ]]; then
  echo "      Scheduling for: ${HOUR}:$(printf "%02d" $MINUTE) daily"
  # Use sed to update the time in the plist
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/<integer>9<\/integer><!-- Hour -->/<integer>$HOUR<\/integer>/g" "$PLIST_DEST"
    sed -i '' "s/<integer>0<\/integer><!-- Minute -->/<integer>$MINUTE<\/integer>/g" "$PLIST_DEST"
  fi
fi
echo ""

# Load the job
echo "[4/4] Loading launchd job..."
if launchctl list com.barloweanalytics.paper-trades-daily &>/dev/null; then
  echo "      Job already loaded. Reloading..."
  launchctl bootout gui/$(id -u) "$PLIST_DEST" 2>/dev/null || true
  sleep 1
fi

launchctl bootstrap gui/$(id -u) "$PLIST_DEST"
echo "      ✓ Job loaded successfully"
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Daily paper trades pipeline will run at ${HOUR}:$(printf "%02d" $MINUTE) each day."
echo ""
echo "Commands:"
echo "  - View status:     launchctl list com.barloweanalytics.paper-trades-daily"
echo "  - View logs:       tail -f $LOG_DIR/paper_trades.log"
echo "  - Uninstall:       ./scripts/install_paper_trades_scheduler.sh --uninstall"
echo ""
