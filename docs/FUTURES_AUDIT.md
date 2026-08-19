# Futures Backtest Audit

**Corrected result: futures are not profitable.** With the de-vig and the outcomes both fixed,
division futures return **-3.37%** over 100 bets and 12 seasons, make-playoffs **-12.18%** over
95 bets and 9 seasons. The previously reported +71% and +9% were artifacts of four defects.

| Market | Bets | Seasons | Win rate | Corrected ROI | Previously reported |
|---|---:|---:|---:|---:|---:|
| division | 100 | 12 | 22.0% | **-3.37%** | +71% |
| make_playoffs | 95 | 9 | 35.8% | **-12.18%** | +9% |
| championship | 6 | 4 | 16.7% | -1.08% | -1.08% (was already clean) |

All three agree with the moneyline finding of +0.18% (95% CI -4.29% to +4.94%): once the
measurement is correct, every market returns roughly zero to negative.

A caution about sample selection, from this audit itself. Running only 2022-2025 gives division
**+37.66%**; running all twelve available seasons gives **-3.37%**. Those four seasons were the
good stretch, chosen only because the earlier work used them. Season ROIs range from -100% to
+154%, so per-season dispersion is roughly 85 points and the standard error on a twelve-season
mean is about 25 points. The -3.37% is indistinguishable from zero, and so was the +71%.

Do not deploy on the previous futures numbers. `docs/FUTURES_BACKTEST_COMPLETE.md`,
`docs/FINAL_BACKTEST_RESULTS.md`, `docs/EXECUTIVE_SUMMARY.md`, and `docs/SUMMARY.md` state
results produced before this audit and are superseded.

## Season detail, corrected

| Season | Division bets | Division ROI | Playoff bets | Playoff ROI |
|---|---:|---:|---:|---:|
| 2013 | 10 | +154.3% | | |
| 2014 | 9 | -77.0% | | |
| 2015 | 7 | -52.6% | | |
| 2016 | 8 | +57.6% | 9 | +20.8% |
| 2017 | 11 | -93.4% | 9 | -73.5% |
| 2018 | 9 | -61.9% | 11 | -47.1% |
| 2019 | 8 | -100.0% | 10 | -2.8% |
| 2021 | 8 | -51.5% | 12 | -58.7% |
| 2022 | 8 | +71.6% | 14 | +25.2% |
| 2023 | 7 | -43.5% | 9 | -33.1% |
| 2024 | 7 | +118.1% | 9 | +23.3% |
| 2025 | 8 | +19.1% | 12 | +30.3% |
| **Total** | **100** | **-3.37%** | **95** | **-12.18%** |

---

## How the markets account for

| Market | De-vig | Model probabilities | Reported ROI | Cause of the result |
|---|---|---|---:|---|
| championship | correct, 1 slot | correct, sums 1.000 | **-1.08%** | **no defect** |
| AL / NL pennant | correct, 1 slot | **wrong, sums 4.0 where 2.0 is required** | -94% / -36% | 2x inflated model probability |
| division | **wrong, 6x** | correct, sums 6.000 | **+71%** | inflated edge plus fabricated outcomes |
| make_playoffs | **wrong, 12x** | correct, sums 12.000 | **+9%** | inflated edge plus hardcoded outcomes |

The single market that was correct on both sides is roughly breakeven, which agrees with the
moneyline finding of +0.18% (95% CI -4.29% to +4.94%) over 2,458 bets. Every market that looked
profitable contained a defect that manufactured the edge.

---

## Defect 1: de-vig normalised every market to one winner

`scripts/backtest_futures.py`, previously:

```python
vig_factor = sum(all_implied)                      # over all 30 teams
implied_prob = (1.0 / decimal_odds) / vig_factor    # forces the market to total 1.0
```

A market's implied probabilities sum to the number of winning slots it settles, not to 1.
Division settles six winners, make-playoffs twelve, miss-playoffs eighteen. Measured with best
available price per team:

| Market | Teams | Sum of implied | True slots | True overround | Inflation applied |
|---|---:|---:|---:|---:|---:|
| division | 30 | 6.852 | 6 | 1.142 | **6.0x** |
| make_playoffs | 30 | 11.937 | 12 | 0.995 | **12.0x** |
| miss_playoffs | 30 | 19.041 | 18 | 1.058 | **18.0x** |

Worked example, 2023 Seattle at +600. Implied 14.29%. Correct fair value 14.29 / 1.142 =
12.51%. The code returned 14.29 / 6.852 = **2.09%**, against a model probability of 79.2%, for
a reported edge of 77 points. Championship and pennant markets settle one winner, so
normalising to 1.0 is correct there, which is why they were unaffected.

Fixed by declaring the slot count per market and raising rather than guessing when a market is
unknown:

```python
MARKET_SLOTS = {"championship": 1, "al_pennant": 1, "nl_pennant": 1,
                "division": 6, "make_playoffs": 12, "miss_playoffs": 18, ...}
overround = sum(all_implied) / MARKET_SLOTS[market_type]
implied_prob = (1.0 / decimal_odds) / overround
```

## Defect 2: actual outcomes were hardcoded, and several are wrong

`_load_actual_outcomes` carries literal sets with a `# In production, query
mlb.postseason_results table` note. Line 97 ships with a question mark in the comment:

```python
outcomes["division"] = {141, 136, 117, 144, 158, 119}  # TB, SEA, HOU, ATL, MIL, LAD (?)
```

Verified errors:

| Season | Hardcoded | Actual | Effect |
|---|---|---|---|
| 2023 AL East | Tampa Bay | Baltimore | fabricated win |
| 2023 AL West | Seattle listed as winner | Houston | **fabricated win worth +0.300u, the largest single winner that season** |
| 2022 NL Central | Milwaukee | St. Louis | fabricated win |

A backtest cannot be graded against guessed outcomes. These must come from the database.

## Defect 3: pennant probabilities are twice what they can be

Simulated 2023, 2,000 trials, summed across all 30 teams:

| Field | Sum | Required | Status |
|---|---:|---:|---|
| `division_win_prob` | 6.000 | 6 | correct |
| `playoff_prob` | 12.000 | 12 | correct |
| `championship_prob` | 1.000 | 1 | correct |
| `league_championship_prob` | **4.000** | **2** | **2x inflated** |

Two pennants are awarded, so the field must sum to 2. It sums to 4, so every pennant edge was
computed against a doubled model probability. That is why the pennant backtests lost 94% and
36% while using a correct de-vig: the model overstated its own probabilities and Kelly sized
accordingly.

The other three fields are coherent, so the simulation is not broadly broken.

## Defect 4: `teams.division_name` is corrupted

Houston is recorded in `National League Central`. Any grouping or validation that relies on
`teams.division_name` is unreliable, including attempts to derive division winners from
standings. Divisions must be resolved from a trustworthy source before defect 2 can be fixed
properly.

---

## What the model actually claims

Even after correcting the de-vig, division edges remain very large, because the model is
confident: 2023 Seattle 79.2%, 2025 Seattle 82.5%, 2023 Dodgers 93.6%, 2024 Seattle 71.8%. The
probabilities are internally coherent, summing to 6.000, but a preseason simulation asserting a
single team is 80% to win its division is a claim worth validating on its own before it is
staked. Seattle is the model's favourite in four consecutive seasons at 34%, 79%, 72%, 82%.

---

## What must happen before any futures number is trusted

1. Replace `_load_actual_outcomes` with a query against real results. Requires a trustworthy
   division mapping, so defect 4 is a prerequisite.
2. Fix `league_championship_prob` to sum to 2.0.
3. Re-run all markets with the corrected de-vig, already applied.
4. Validate the model's division probabilities against realised division-winner frequencies by
   probability bucket, the same reliability test applied to the moneyline model in
   `scripts/assess_model_calibration.py`.
5. Only then compare against the market, using the residual instrument in
   `scripts/find_market_residual_signal.py` rather than a raw edge threshold, since raw
   disagreement selection produced a 6.3 point winner's curse on the moneyline model.

Sample size is the binding limit regardless: division futures offer roughly 6 bets per season
per the corrected threshold, so 16 seasons is on the order of 100 bets. The moneyline analysis
needed about 13,000 bets to resolve a 2% ROI. Futures variance per bet is far higher. No futures
conclusion from this data will be statistically strong, whatever the point estimate.

---

## Status

Fixed and re-measured:

- **Defect 1**, de-vig: fixed via `MARKET_SLOTS` in `scripts/backtest_futures.py`.
- **Defect 2**, hardcoded outcomes: replaced by `scripts/futures_outcomes.py`, which derives
  division winners and playoff fields from aggregated `mlb.linescore` results, breaks ties on
  head-to-head record, and reports any tie head-to-head cannot resolve. Verified against the
  three real tiebreaks in range: 2022 NL East to Atlanta over the Mets, 2023 AL West to Houston
  over Texas, 2025 AL East to Toronto over the Yankees.
- **Defect 4**, `teams.division_name`: worked around by `TEAM_DIVISION_OVERRIDE` in
  `scripts/futures_outcomes.py`, mapping Houston to the AL West. The reference table row is
  still wrong in the database and should be corrected there; anything else reading
  `division_id` or `division_name` for Houston is affected.

Still open:

- **Defect 3**, `league_championship_prob` sums to 4.0 where 2.0 is required. Pennant markets
  are therefore still not measurable, and their reported -94% and -36% remain uninterpretable.
  Division, make-playoffs, and championship are unaffected, their model fields summing to
  6.000, 12.000, and 1.000 respectively.

Conclusion: division and make-playoffs futures are measured and unprofitable. Combined with
championship at -1.08% and full-game moneyline at +0.18%, no market in this repository has yet
shown an edge once measured correctly.
