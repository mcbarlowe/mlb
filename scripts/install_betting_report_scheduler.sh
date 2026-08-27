#!/bin/bash
set -euo pipefail

# Install the daily betting-report emailer as a launchd job.
# Usage:
#   ./scripts/install_betting_report_scheduler.sh [--time HH:MM]
#   ./scripts/install_betting_report_scheduler.sh --uninstall

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="com.barloweanalytics.daily-betting-report"
PLIST_SOURCE="$SCRIPT_DIR/$LABEL.plist"
PLIST_DEST="$HOME/.LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/BarloweAnalytics"

HOUR=9
MINUTE=30
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      HOUR=$(echo "$2" | cut -d: -f1)
      MINUTE=$(echo "$2" | cut -d: -f2)
      shift 2
      ;;
    --uninstall)
      echo "Uninstalling $LABEL..."
      launchctl bootout "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
      rm -f "$PLIST_DEST"
      echo "Done."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Installing $LABEL (daily at ${HOUR}:$(printf '%02d' "$MINUTE"))"
mkdir -p "$LOG_DIR" "$HOME/.LaunchAgents"
cp "$PLIST_SOURCE" "$PLIST_DEST"
chmod 644 "$PLIST_DEST"

# Set the schedule reliably via PlistBuddy.
PB=/usr/libexec/PlistBuddy
"$PB" -c "Set :StartCalendarInterval:0:Hour $HOUR" "$PLIST_DEST"
"$PB" -c "Set :StartCalendarInterval:0:Minute $MINUTE" "$PLIST_DEST"

# Warn (do not fail) if the Gmail app password is not reachable from the shell.
if ! zsh -ic 'printenv MLB_REPORT_GMAIL_APP_PASSWORD' >/dev/null 2>&1; then
  echo "WARNING: MLB_REPORT_GMAIL_APP_PASSWORD is not set in your shell."
  echo "         Add it to ~/.zshrc before the job can send email:"
  echo "           export MLB_REPORT_GMAIL_APP_PASSWORD='<16-char app password>'"
  echo "           export MLB_REPORT_GMAIL_USER='mcbarlowe@gmail.com'   # sender"
fi

if launchctl list "$LABEL" &>/dev/null; then
  echo "Reloading existing job..."
  launchctl bootout "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
  sleep 1
fi
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"

echo "Installed. Commands:"
echo "  Status:    launchctl list $LABEL"
echo "  Run now:   launchctl kickstart -k gui/$(id -u)/$LABEL"
echo "  Logs:      tail -f $LOG_DIR/betting-report.out.log"
echo "  Uninstall: ./scripts/install_betting_report_scheduler.sh --uninstall"
