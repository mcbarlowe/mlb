# Daily Paper Trading Pipeline

Automated daily settlement of paper trades based on overnight game results.

## Overview

The daily pipeline:
1. **Backfills** completed games from the previous night
2. **Settles** paper trades based on final game scores
3. **Reports** trading summary (win rate, profit, ROI)

## Quick Start

### Manual Execution

Run the pipeline immediately:

```bash
# Settle yesterday's games (automatic date detection)
uv run python scripts/settle_daily_paper_trades.py

# Settle a specific date
uv run python scripts/settle_daily_paper_trades.py --date 2026-08-17

# Show detailed trade results
uv run python scripts/settle_daily_paper_trades.py --verbose
```

### Automated Daily Scheduling

Install as a launchd job to run automatically each morning:

```bash
# Install with default 9:00 AM start time
./scripts/install_paper_trades_scheduler.sh

# Install with custom time (e.g., 8:30 AM)
./scripts/install_paper_trades_scheduler.sh --time 08:30

# Uninstall the scheduled job
./scripts/install_paper_trades_scheduler.sh --uninstall
```

## How It Works

### 1. Backfill Step

The pipeline calls `run_daily_postgres_etl.py` to:
- Download game results from MLB API for yesterday's date
- Ingest final scores and game state into PostgreSQL
- Mark games as "Final" in the database

### 2. Settlement Step

The pipeline calls `settle_daily_paper_trades.py` to:
- Query all unsettled paper trades in the database
- Look up final game results from the `games` and `linescore` tables
- Calculate profit/loss for each trade based on:
  - Opening odds when the bet was placed
  - Closing odds (for CLV calculation)
  - Final game result (home won / away won)
- Update each trade with:
  - `status = "settled"`
  - `profit_units` (in bankroll units)
  - `result` (win/loss)
  - `clv` (closing line value)

### 3. Reporting Step

After settlement, the pipeline displays:
- **Trade count**: Total, settled, pending
- **Performance metrics**: Win rate, ROI, avg CLV, beat-close rate
- **Detailed results** (if `--verbose` flag used)

## Monitoring

### Check Job Status

```bash
# See if job is loaded
launchctl list com.barloweanalytics.paper-trades-daily

# View job configuration
launchctl dumpstate user | grep paper-trades
```

### View Logs

```bash
# Last 50 lines
tail -50 ~/.mlb/paper_trades.log

# Follow live
tail -f ~/.mlb/paper_trades.log

# Errors
tail -f ~/.mlb/paper_trades.err
```

### Manual Test Run

```bash
# Test the full pipeline with verbose output
bash scripts/run_daily_paper_pipeline.sh --verbose --date 2026-08-17
```

## Troubleshooting

### Job Not Running at Scheduled Time

1. **Verify it's loaded:**
   ```bash
   launchctl list com.barloweanalytics.paper-trades-daily
   ```
   
2. **Check system time is correct:**
   ```bash
   date
   ```

3. **Reload the job:**
   ```bash
   launchctl bootout gui/$(id -u) ~/.LaunchAgents/com.barloweanalytics.paper-trades-daily.plist
   sleep 2
   launchctl bootstrap gui/$(id -u) ~/.LaunchAgents/com.barloweanalytics.paper-trades-daily.plist
   ```

### Games Not Settling

1. **Check backfill completed:**
   ```bash
   uv run python scripts/run_daily_postgres_etl.py --date 2026-08-17
   ```

2. **Verify game data in database:**
   ```bash
   psql -d postgres -c "SELECT COUNT(*), abstract_game_state FROM games WHERE game_datetime::date = '2026-08-17' GROUP BY abstract_game_state;"
   ```

3. **Check for missing linescore data:**
   ```bash
   psql -d postgres -c "SELECT COUNT(*) FROM linescore WHERE game_pk IN (SELECT game_pk FROM games WHERE game_datetime::date = '2026-08-17');"
   ```

## Configuration

### Log Location

Logs are written to:
- **Stdout:** `~/.mlb/paper_trades.log`
- **Stderr:** `~/.mlb/paper_trades.err`

### Time Zone

The pipeline uses UTC internally but defaults to processing "yesterday" based on system time. To process a specific date:

```bash
uv run python scripts/settle_daily_paper_trades.py --date 2026-08-16
```

### Bankroll Configuration

The pipeline uses the following defaults (configured in the settlement report):
- **Starting Bankroll:** $2,000
- **Unit Size:** 2% of bankroll = $40
- **Bet Sizing:** 1/4 Kelly with 5% cap

These can be adjusted by modifying the `STARTING_BANKROLL` and `UNIT_SIZE_PERCENT` variables in the settlement scripts if needed.

## Pipeline Scripts

### `run_daily_paper_pipeline.sh`
Master wrapper that runs:
1. `run_daily_postgres_etl.py` — backfill games
2. `settle_daily_paper_trades.py` — settle trades

### `settle_daily_paper_trades.py`
Standalone settlement script. Can be run independently to re-settle trades without backfilling.

### `install_paper_trades_scheduler.sh`
LaunchD job installer. Handles:
- Creating log directories
- Installing launchd plist
- Scheduling the pipeline
- Uninstallation

## Example Output

```
[2026-08-17] Settlement scan: updated=3 missing_final=1 missing_close=0

==========================================================================================
PAPER TRADING SUMMARY
==========================================================================================

Total Trades:        11 (10 settled, 1 pending)
Win Rate:            60.0%
ROI:                 +30.97%

Total Staked:        31.95u
Total Profit:        +9.89u
Avg CLV:             +0.0130
Beat Close Rate:     90.0%

==========================================================================================
```

## Next Steps

- [ ] Install the scheduled job: `./scripts/install_paper_trades_scheduler.sh`
- [ ] Verify it runs at the scheduled time tomorrow morning
- [ ] Set up Slack/email notifications for results (not yet implemented)
- [ ] Add running average tracking (season-to-date stats)
