> **SUPERSEDED — DO NOT ACT ON THESE NUMBERS.** The futures results below were produced before
> the audit in `docs/FUTURES_AUDIT.md`, which found four defects: the de-vig normalised
> multi-winner markets to a single winner (6x inflation on division, 12x on playoffs),
> actual outcomes were hardcoded and several are wrong, `league_championship_prob` sums to
> 4.0 where 2.0 is required, and `teams.division_name` is corrupted. The only market free of
> defects, championship futures, returned -1.08%.

# Historical Futures Betting Backtest

## Overview

Infrastructure for backtesting the championship/division/playoff futures betting strategy on historical seasons using preseason odds.

## Historical Data Challenge

**The-odds-api limitation:** The `/v4/historical` endpoint does NOT support outrights/futures markets. It only supports event-level markets (game spreads, totals, moneylines).

This means we cannot programmatically fetch historical preseason futures odds the way we can fetch historical game odds.

## Solution: Manual CSV Input

Created `scripts/load_historical_futures_from_csv.py` to load manually-gathered historical odds.

### CSV Format

```csv
season,market_type,team_name,bookmaker,american_odds,snapshot_time,source
2024,championship,Los Angeles Dodgers,consensus,+550,2024-03-15T00:00:00Z,manual-preseason
2024,championship,Atlanta Braves,consensus,+650,2024-03-15T00:00:00Z,manual-preseason
...
```

Required columns:
- `season`: Year (e.g. 2024)
- `market_type`: `championship`, `al_pennant`, `nl_pennant`, or `division_<div>` 
- `team_name`: Full team name (case-insensitive, e.g. "Los Angeles Dodgers")
- `bookmaker`: Bookmaker name or "consensus"
- `american_odds`: American odds format (e.g. +550, -200)
- `snapshot_time`: ISO timestamp
- `source`: Data source label

### Loading Historical Odds

```bash
# Dry-run to validate
uv run python scripts/load_historical_futures_from_csv.py data/historical_2024.csv --dry-run

# Load to database
uv run python scripts/load_historical_futures_from_csv.py data/historical_2024.csv --db
```

## Gathering Historical Odds

### Sources for Historical Futures Odds

1. **Wayback Machine** (archive.org)
   - Search for archived pages from major sportsbooks in early March each season
   - Example: `https://web.archive.org/web/20240310*/https://www.draftkings.com/sportsbook/mlb-futures`
   
2. **Sports Reference Sites**
   - Some sports media outlets publish preseason championship odds in articles
   - Check ESPN, The Athletic, CBS Sports archives
   
3. **Vegas Insider Historical Pages**
   - Sometimes maintains historical futures markets
   - `https://www.vegasinsider.com/mlb/odds/futures/`

4. **Odds Archive Services**
   - Paid services like Sports Odds History
   - Academic datasets

5. **Published Predictions**
   - FanGraphs, Baseball Prospectus sometimes publish pre-season betting market consensus

### What to Gather

For each season (2022-2025), ideally get **preseason odds** (mid-March before Opening Day):

**Essential:**
- **Championship** odds for all 30 teams

**Nice to have:**
- **Division winner** odds for each division
- **Pennant winner** (AL/NL) odds
- **Make playoffs** Yes/No odds

Use "consensus" as bookmaker if you're averaging multiple books, or record individual books if available.

## Running Backtest

Once historical odds are loaded:

```bash
# Single season
uv run python scripts/backtest_futures.py --season 2024 --market championship

# Multiple seasons
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market championship

# All markets (championship, pennants, playoffs)
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --all-markets
```

### Backtest Parameters

- `--edge-threshold 0.05`: Minimum 5% edge to bet (default)
- `--kelly-multiplier 0.25`: Quarter-Kelly sizing (default)
- `--max-stake 0.05`: Maximum 5% stake per bet (default)

## Expected Results

Based on the model's +14.6% ROI on 3+ win edge game totals, we expect:

- **Championship odds:** Higher variance (30-team market), likely +8-15% ROI on 5%+ edges
- **Division odds:** Medium variance (5-team markets), likely +10-18% ROI
- **Pennant odds:** Lower variance (15-team markets), likely +12-20% ROI

## Current Status

✅ **Infrastructure complete:**
- CSV loader for historical odds
- Backtest script with Kelly sizing
- Actual outcome comparison
- ROI calculation

⏳ **Historical data needed:**
- 2022 preseason championship odds (30 teams)
- 2023 preseason championship odds (30 teams)
- 2024 preseason championship odds (30 teams) - **SAMPLE LOADED**
- 2025 preseason championship odds (30 teams)

## Sample Data

`data/historical_futures_odds_sample.csv` contains representative 2024 preseason championship odds based on market consensus.

To get real historical odds for backtesting, use the sources above and create similar CSVs for 2022-2025.

## Next Steps

1. **Gather 2022-2023 preseason championship odds** from Wayback Machine
2. **Verify 2024 sample odds** against actual archived markets
3. **Add 2025 preseason odds** (current season)
4. **Run multi-year backtest** to validate model edge
5. **Expand to division/pennant markets** if championship backtest is profitable
