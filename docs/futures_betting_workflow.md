# Futures Betting Infrastructure

Complete workflow for fetching futures odds, generating model-based edges, and tracking paper trades for MLB season futures markets (championship, division, playoff).

## Overview

The futures betting infrastructure consists of:

1. **Odds fetching** - Pull live futures odds from the-odds-api.com
2. **Database storage** - Store odds in `mlb.futures_odds`
3. **Edge calculation** - Generate season projections and compare to market
4. **Paper trading** - Track hypothetical bets in `mlb.futures_paper_trades`
5. **Settlement** - Update results at season end

## Database Schema

### mlb.futures_odds

Stores futures odds snapshots by season/market/team/bookmaker:

```sql
CREATE TABLE futures_odds (
    season integer NOT NULL,
    market_type text NOT NULL,  -- championship, division, playoff
    team_id integer NOT NULL,
    team_name text,
    bookmaker text NOT NULL,
    american_odds integer NOT NULL,
    implied_probability double precision,
    snapshot_time timestamptz NOT NULL,
    source text DEFAULT 'the-odds-api',
    ingested_at timestamptz DEFAULT now(),
    PRIMARY KEY (season, market_type, team_id, bookmaker, snapshot_time)
);
```

### mlb.futures_paper_trades

Tracks paper trade recommendations and results:

```sql
CREATE TABLE futures_paper_trades (
    strategy_version text NOT NULL,
    season integer NOT NULL,
    market_type text NOT NULL,
    team_id integer NOT NULL,
    team_name text,
    paper_date date NOT NULL,
    snapshot_time timestamptz NOT NULL,
    model_probability double precision NOT NULL,
    consensus_market_prob double precision,
    edge double precision,
    best_bookmaker text,
    best_american_odds integer,
    best_decimal_odds double precision,
    staking text NOT NULL,
    stake_units double precision NOT NULL,
    status text DEFAULT 'open',  -- open, won, lost, push, cancelled
    result text,
    profit_units double precision,
    PRIMARY KEY (strategy_version, season, market_type, team_id)
);
```

## Workflow

### 1. Fetch Futures Odds

Pull live odds from the-odds-api outrights endpoint:

```bash
# Dry run to test
uv run python scripts/fetch_futures_odds.py --season 2027 --dry-run

# Store in database
uv run python scripts/fetch_futures_odds.py --season 2027 --db
```

**Environment variables:**
- `ODDS_API_KEY` - Required for API access

**Options:**
- `--season YEAR` - Season year (required)
- `--regions us,uk` - Comma-separated regions (default: us)
- `--max-markets N` - Limit markets for testing
- `--dry-run` - Test without writing to database
- `--db` - Write to mlb.futures_odds table

### 2. Generate Paper Trades

Calculate edges using baseline team-strength model and generate bet recommendations:

```bash
# Preview edges
uv run python scripts/paper_trade_futures_from_db.py --season 2027 --dry-run

# Store paper trades
uv run python scripts/paper_trade_futures_from_db.py --season 2027 --db
```

**Options:**
- `--season YEAR` - Season year (required)
- `--market-type championship|division|playoff` - Market to analyze (default: championship)
- `--edge-threshold 0.05` - Minimum edge (default: 0.05 = 5%)
- `--kelly-multiplier 0.25` - Fractional Kelly (default: 0.25)
- `--kelly-cap 0.05` - Max stake in units (default: 0.05 = 5% of bankroll)
- `--trials 5000` - Monte Carlo trials (default: 5000)
- `--dry-run` - Preview without database write
- `--db` - Write to mlb.futures_paper_trades

The script:
1. Loads latest futures odds from `mlb.futures_odds`
2. Generates season projection using baseline model (no priors)
3. Calculates edges: `edge = model_prob - market_prob`
4. Sizes bets using fractional Kelly: `stake = kelly_multiplier * kelly_fraction`
5. Caps stake at `kelly_cap` (default 5% max)
6. Stores recommendations in `mlb.futures_paper_trades` with `status='open'`

### 3. Settle at Season End

After season completes, settle paper trades based on actual outcomes:

```bash
# Preview settlement
uv run python scripts/settle_futures_paper_trades.py --season 2026 --dry-run

# Write results
uv run python scripts/settle_futures_paper_trades.py --season 2026 --db
```

**Options:**
- `--season YEAR` - Season to settle (required)
- `--strategy-version NAME` - Strategy to settle (default: futures_baseline_model_v1)
- `--dry-run` - Show results without updating database
- `--db` - Write settlement to mlb.futures_paper_trades

The script:
1. Queries `mlb.futures_paper_trades` for open bets
2. Determines outcomes from actual season results:
   - **Championship**: Queries `postseason_results` for World Series winner
   - **Division**: Finds division leaders from final standings
   - **Playoff**: Queries `games` for playoff participants
3. Calculates profit:
   - Won: `profit = stake * payout_multiplier`
   - Lost: `profit = -stake`
4. Updates `status`, `result`, `profit_units` in database

## Historical Performance

Based on 2022-2025 backtest of baseline model vs win totals:

| Metric | Value |
|---|---:|
| Overall ROI | +5.5% |
| 3+ win edge ROI | **+14.6%** |
| Win rate (all) | 57.5% |
| Win rate (3+ edge) | 62.0% |

**Strongest signal**: 3+ win edge threshold on win totals showed 62% win rate and 14.6% ROI.

## Strategy

**Current strategy version**: `futures_baseline_model_v1`

Uses:
- **Model**: Baseline team-strength (Elo + run differential + starters + lineup + bullpen), no priors
- **Sizing**: 0.25-Kelly (fractional Kelly multiplier = 0.25)
- **Cap**: 5% of bankroll max per bet
- **Threshold**: 5% minimum edge (model prob - market prob >= 0.05)

**Recommended filters** based on historical evidence:
- **Championship futures**: Use 5% minimum edge
- **Win totals**: Use 3-win minimum edge (historically 14.6% ROI)
- **Direction bias**: OVER bets outperformed (+9.8% ROI vs +1.6% for UNDERs)

## Maintenance

### Daily (during futures market updates)

```bash
# Fetch latest odds
uv run python scripts/fetch_futures_odds.py --season 2027 --db

# Regenerate paper trades (updates existing if still open)
uv run python scripts/paper_trade_futures_from_db.py --season 2027 --db
```

### Season end

```bash
# Settle completed bets
uv run python scripts/settle_futures_paper_trades.py --season 2026 --db
```

### Analysis queries

```sql
-- Current open paper trades
SELECT market_type, team_name, edge, best_american_odds, stake_units
FROM mlb.futures_paper_trades
WHERE strategy_version = 'futures_baseline_model_v1'
  AND status = 'open'
ORDER BY edge DESC;

-- Historical ROI by season
SELECT season,
       COUNT(*) as bets,
       SUM(stake_units) as staked,
       SUM(profit_units) as profit,
       SUM(profit_units) / NULLIF(SUM(stake_units), 0) * 100 as roi_pct
FROM mlb.futures_paper_trades
WHERE strategy_version = 'futures_baseline_model_v1'
  AND status != 'open'
GROUP BY season
ORDER BY season DESC;

-- Latest odds snapshot
SELECT market_type, team_name, bookmaker, american_odds
FROM mlb.futures_odds
WHERE season = 2027
  AND snapshot_time = (SELECT MAX(snapshot_time) FROM mlb.futures_odds WHERE season = 2027)
ORDER BY market_type, american_odds;
```

## Notes

- **API limits**: the-odds-api has monthly request quotas; check `x-requests-remaining` header
- **Odds updates**: Futures odds change infrequently; daily updates sufficient
- **Model updates**: If team-strength champion model changes, regenerate projections
- **Line shopping**: Best price selection across bookmakers built into paper trade generator
- **No-vig calculation**: Uses simple normalization; assumes efficient market aggregation

## See Also

- `scripts/paper_trade_futures.py` - CSV-based edge calculator (no database)
- `src/betting/futures.py` - Core edge calculation logic
- `test_futures_infrastructure.py` - Test suite
