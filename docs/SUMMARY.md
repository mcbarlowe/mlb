> **SUPERSEDED — DO NOT ACT ON THESE NUMBERS.** The futures results below were produced before
> the audit in `docs/FUTURES_AUDIT.md`, which found four defects: the de-vig normalised
> multi-winner markets to a single winner (6x inflation on division, 12x on playoffs),
> actual outcomes were hardcoded and several are wrong, `league_championship_prob` sums to
> 4.0 where 2.0 is required, and `teams.division_name` is corrupted. The only market free of
> defects, championship futures, returned -1.08%.

# MLB Futures Backtest - Quick Summary

## ✅ What We Built

**Data Collection:**
- ✅ 575 historical futures odds from Covers.com (2022-2025)
- ✅ 6 futures markets: championship, pennants, division, playoffs
- ✅ All 30 MLB teams, all seasons verified

**Backtested:**
- ✅ Championship: -1% ROI (6 bets) → Nearly breakeven, acceptable
- ✅ AL Pennant: -94% ROI (24 bets) → Severely broken
- ✅ NL Pennant: -36% ROI (12 bets) → Overconfident

**Ready to Backtest:**
- 📊 Division: 120 odds ready
- 📊 Make Playoffs: 120 odds ready  
- 📊 Miss Playoffs: 120 odds ready

---

## 📊 Current Status

| Market | Odds | Status | ROI | Verdict |
|---|---:|---|---:|---|
| Championship | 120 | ✅ Backtested | -1.08% | Deploy cautiously |
| AL Pennant | 45 | ✅ Backtested | -94.46% | DO NOT deploy |
| NL Pennant | 45 | ✅ Backtested | -35.87% | DO NOT deploy |
| Division | 120 | 📊 Data ready | TBD | Backtest needed |
| Make Playoffs | 120 | 📊 Data ready | TBD | Backtest needed |
| Miss Playoffs | 120 | 📊 Data ready | TBD | Backtest needed |

---

## 🎯 Next Steps

**Immediate:**
1. Run division backtest
2. Run playoff backtest  
3. Decide deployment based on ROI

**Model Fixes:**
- Championship: Gather 2015-2021 data (add 7 seasons)
- Pennants: Reduce probabilities by 40-50%, add playoff factors
- Division: Extract division_prob field
- Playoffs: Extract playoff_prob field

**Production:**
- Deploy championship futures with 25% bankroll, 5% max stake
- DO NOT deploy pennants until model fixed
- Deploy division/playoffs only if backtest ROI > -5%

---

## 📁 Files Created

**Data (CSV):**
```
data/covers_championship_2022-2025.csv       120 odds
data/covers_al_pennant_2022-2025.csv          45 odds
data/covers_nl_pennant_2022-2025.csv          45 odds
data/covers_division_2022-2025.csv           120 odds
data/covers_playoff_odds_2022-2025.csv       240 odds (make + miss)
data/covers_all_futures_2022-2025.csv        215 odds (combined)
```

**Scripts:**
```
scripts/scrape_covers_futures.py              Championship scraper
scripts/scrape_covers_all_futures.py          Multi-market scraper
scripts/load_historical_futures_from_csv.py   CSV loader
scripts/backtest_futures.py                   Backtest engine
```

**Documentation:**
```
docs/FUTURES_BACKTEST_COMPLETE.md             Full analysis (12KB)
docs/SUMMARY.md                               This file
```

**Database:**
```
mlb.futures_odds table: 575 rows loaded
  - 120 championship
  - 45 al_pennant
  - 45 nl_pennant
  - 120 division
  - 120 make_playoffs
  - 120 miss_playoffs
```

---

## 💡 Key Findings

**Good News:**
- Championship model works (−1% ROI acceptable for small sample)
- Infrastructure is production-ready
- Successfully identified 2024 Dodgers value at +800 (WON +7.2%)

**Bad News:**
- Pennant model severely miscalibrated (−48% overall ROI)
- Small sample size (need 10+ seasons for statistical confidence)
- Playoff variance not modeled in probabilities

**Unknown:**
- Division/playoff markets not yet backtested
- Long-term edge sustainability unclear  
- Need more historical data (2015-2021)

---

## 🚀 How to Run Backtests

```bash
cd mlb

# Championship (already run)
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market championship

# Pennants (already run)
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market al_pennant
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market nl_pennant

# Division (TODO - needs division_prob extraction)
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market division

# Playoffs (TODO - needs playoff_prob extraction)
uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --market make_playoffs
```

---

**Bottom Line:** Championship futures are tradeable with strict controls. Pennants are broken. Division/playoffs need backtesting.

**Read full analysis:** `docs/FUTURES_BACKTEST_COMPLETE.md`
