# Moneyline Backtest, 2020-2025

**Model:** champion team-strength logistic, refit walk-forward (train 2015..N-1, test N)
**Odds:** `mlb.odds`, per-book, strictly pre-game snapshots
**Consensus:** median across books in decimal space (`backtest_moneyline.consensus_american`)
**Staking:** flat 1u for the headline table

Reproduce:

```bash
uv run python scripts/backtest_moneyline.py --season 2022 \
  --walkforward-train 2015,2016,2017,2018,2019,2020,2021
```

Omitting `--walkforward-train` scores the registered champion **in-sample** and is not a
backtest. Every number below passes the flag.

---

## Results, bet at open, flat 1u

| Season | Games | Bets @5% | ROI @5% | Win% | CLV | Beat close | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 679 | 234 | -3.88% | 44.4% | +0.0029 | 50% | -9.1u |
| 2021 | 1,946 | 429 | +7.28% | 48.3% | +0.0058 | 54% | +31.2u |
| 2022 | 2,184 | 441 | -11.26% | 42.0% | +0.0032 | 53% | -49.7u |
| 2023 | 2,255 | 471 | -2.28% | 49.0% | +0.0203 | 77% | -10.8u |
| 2024 | 2,235 | 283 | -4.09% | 45.2% | +0.0140 | 71% | -11.6u |
| 2025 | 2,307 | 363 | +6.71% | 49.3% | +0.0092 | 64% | +24.3u |

Pooled, by edge threshold:

| Threshold | Bets | Net | ROI |
|---|---:|---:|---:|
| 0% (bet every game) | 11,606 | -444.3u | **-3.83%** |
| 3% | 4,922 | -166.6u | **-3.38%** |
| 5% | 2,221 | -25.7u | **-1.16%** |

---

## Reading

**Positive CLV is real, and it still does not imply profit.** This document has been wrong
about CLV twice, in opposite directions: first calling it evidence of skill, then dismissing it
as pure regression to the mean. Genuine closing lines settle it, and the truth is neither.

With real closes loaded (`line_type='true_close'`, median 4 minutes before first pitch,
9,583 games) two separable questions give opposite answers:

| Test | Model coefficient | Verdict |
|---|---:|---|
| `outcome ~ logit(true_close) + logit(model)` | -0.006 ± 0.132 (z = -0.04) | no information over the close |
| `logit(true_close) ~ logit(open) + logit(model)` | **+0.1156 ± 0.0062 (z = +18.70)** | **genuinely anticipates line movement** |

So the model does hold information the *opening* price lacks, and the market absorbs it by the
close. The regression-to-the-mean dismissal was wrong. But the reason CLV still fails to
convert is subtler and specific to this market:

| Observation point | Lead | Brier | Slope | Intercept |
|---|---|---:|---:|---:|
| open | 19-29h | 0.2392 | +1.032 ± 0.055 | +0.021 ± 0.022 |
| close (fixed-cadence proxy) | ~2.5h | 0.2389 | +1.023 ± 0.053 | +0.013 ± 0.022 |
| true_close | ~4min | 0.2389 | **+1.005 ± 0.053** | **+0.012 ± 0.022** |
| model | n/a | 0.2407 | | |

**The closing line is only 0.0003 Brier better than a price 20+ hours earlier.** The standard
argument that positive CLV implies long-run profit assumes the close is markedly sharper than
the price taken; in MLB moneyline over 2020-2025 that assumption is false. Beating a target
that is barely better than your entry buys almost nothing.

Measured CLV against the true close is +0.0073 with 57% of bets beating it, over 1,777 bets.
Against a shopped hold near 1.4% that is insufficient by roughly half.

Note also that the true close prices at slope 1.005 ± 0.053 with intercept 0.012 ± 0.022 —
cleaner than the proxy, and about as close to a perfectly calibrated forecast as this sample
can resolve.

Median per-book hold is **4.1%** and pooled ROI at the 5% threshold is **-1.16%** betting at
the open against the proxy close.

**Threshold monotonicity does not hold.** ROI rises across -3.83% → -3.38% → -1.16% for the
0/3/5% thresholds, which an earlier revision read as the edge signal being correctly ordered.
Extending the sweep refutes it: with best-price execution the 7% threshold is flat (-0.02% on
917 bets) and the 10% threshold is sharply negative (-14.84% on 177 bets). The apparent
ordering over three points was noise, and it reverses exactly where the selection bias is
strongest.

**Season variance dominates, and the seasonal pattern is not real.** A permutation test over
the settled bets puts the observed spread at p = 0.211 (range) and p = 0.182 (weighted sd),
so one common true ROI explains it. See "The good and bad seasons are the same season".

**Do not read 2024-2025 as the trustworthy subset.** Earlier in-sample champion runs put
2024 at +0.43%; walk-forward puts it at -4.09%. The flip is the leak, not the market.

---

## Data defect found and fixed

The first pass reported -14.2% pooled ROI and 1,021/1,327 bets in 2021/2022. Both were
artifacts of the ingest, not the market.

**Cause.** `/v4/historical/sports/baseball_mlb/odds` returns every event present in a
snapshot, including games already under way. The original split tagged the *latest*
snapshot per game as `close`, so for **74-78% of games** the "closing line" was an in-play
price reflecting the score. An in-play moneyline moves from -120 to -400 on a three-run
first inning.

**Evidence.** Cross-book dispersion of the implied home probability within a single
`(game_pk, line_type)` bucket, before and after restricting to strictly pre-game snapshots:

| Season | median before | p95 before | >10pp before | >10pp after |
|---|---:|---:|---:|---:|
| 2020 | 2.80% | 55.1% | 19.1% | **2.4%** |
| 2021 | 3.96% | 57.3% | 36.8% | **0.9%** |
| 2022 | 15.00% | 75.3% | 56.8% | **0.6%** |
| 2023 | 2.63% | 41.5% | 9.1% | **0.8%** |
| 2024 | 2.33% | 3.8% | 0.0% | 0.0% |
| 2025 | 1.72% | 3.1% | 0.0% | 0.0% |

Contamination tracked bet volume: 2022 was 56.8% contaminated and bet 59% of the slate;
2024-2025 were 0% contaminated and bet 13-16%. Post-fix, all seasons sit at 20-34%.

**Hypotheses tested and rejected along the way.** Per-book overround is normal in every
season (3.4-4.5%), so "historical books charged 11-17% vig" was false — that figure came
from summing home and away prices taken from *different* books, which is not a placeable
bet. Model reliability is equivalent across seasons (MAE 1.45-1.86%, Brier 0.239-0.242),
so "the older models are miscalibrated" was also false. Doubleheaders account for only
4-14% of contaminated games, so team-pair/date collapse in `load_odds_to_db.py` is a minor
residual, not the cause.

**Fix, two places.**
1. `scripts/fetch_all_historical_odds_parallel.py` now drops any event whose
   `commence_time <= snapshot_time` at parse time and reports `skipped_inplay`.
2. `scripts/split_historical_odds_pregame.py` keeps only strictly pre-game snapshots, then
   takes the earliest as `open` and latest as `close`, and drops games with fewer than two
   distinct pre-game snapshots rather than fabricating a zero-CLV pair.

Post-fix leads: `open` at 19-29h pre-game, `close` at 2.2-2.7h pre-game. ~21% of games
dropped for having only one pre-game snapshot, a consequence of the 3-snapshot/day cadence
(16:30, 20:30, 00:30 UTC).

---

## Final evaluation at edge >= 5%

`scripts/evaluate_moneyline_final.py` produces every figure in this section in one pass, so
per-season and pooled numbers cannot drift across code revisions.

Best available configuration: bet at the last strictly-pre-game snapshot, best price among
5 accounts, edge vs the median of per-book de-vigged fair probabilities, flat 1u,
walk-forward model.

| Season | Games | Bets | % slate | Win% | ROI | 95% CI | Net |
|---|---:|---:|---:|---:|---:|---|---:|
| 2020 | 680 | 248 | 36% | 40.7% | -8.61% | [-22.8%, +5.6%] | -21.4u |
| 2021 | 1,946 | 453 | 23% | 47.7% | +10.06% | [-0.8%, +21.2%] | +45.6u |
| 2022 | 2,184 | 505 | 23% | 42.2% | -6.35% | [-16.4%, +4.0%] | -32.1u |
| 2023 | 2,255 | 435 | 19% | 45.7% | -1.84% | [-12.1%, +8.5%] | -8.0u |
| 2024 | 2,421 | 371 | 15% | 45.3% | -1.20% | [-12.5%, +10.9%] | -4.5u |
| 2025 | 2,426 | 446 | 18% | 47.8% | +5.55% | [-4.8%, +16.1%] | +24.8u |
| **Pooled** | | **2,458** | | **45.2%** | **+0.18%** | **[-4.29%, +4.94%]** | **+4.4u** |

Profitable in 2 of 6 seasons. Zero sits inside the pooled interval and inside all six
per-season intervals. Consensus execution on the identical bet list returns -1.92%, so
shopping is worth +2.10pp and is the entire difference between losing and breaking even.

### The model is a worse forecaster than the market on its own picks

ROI understates the problem. On the 2,458 backed sides:

| | Model | Market fair | Actual |
|---|---:|---:|---:|
| Mean probability | 51.5% | 44.5% | **45.2%** |
| Error vs actual | **+6.3pp** | **-0.7pp** | |
| Brier score | 0.2454 | **0.2408** | |

The model claims a **+7.0pp** edge over fair value and realises **+0.7pp** — 10% of the
claim. The market's fair probability is within 0.7pp of the outcome; the model is 6.3pp
high. **Market Brier beats model Brier**, in 5 of 6 seasons individually.

This is adverse selection, not miscalibration. Aggregate reliability is fine (MAE
1.18-1.86%, see `scripts/analyze_model_calibration.py`); the error concentrates in exactly
the subset where the model most disagrees with the price. For this model, large disagreement
with the market is evidence of model error rather than of market error — so the edge filter
is selecting against itself, and raising the threshold selects harder.

The +0.18% decomposes cleanly: a realised +0.7pp edge on a ~44.5% fair probability is about
+1.6% ROI at fair odds, less roughly 1.5% effective hold after shopping, leaving ~+0.1%.
Observed +0.18%. The model contributes almost nothing; shopping contributes the rest.

### The good and bad seasons are the same season

2021 (+10.06%) and 2025 (+5.55%) against 2020 (-8.61%) and 2022 (-6.35%) invites a causal
story. There is none to tell: `scripts/test_season_heterogeneity.py` permutes the 2,458
settled bets across seasons 20,000 times, preserving each season's bet count, and the
observed dispersion is unremarkable.

| Statistic | Observed | p |
|---|---:|---:|
| Range (max - min season ROI) | 18.67pp | **0.211** |
| Bet-count-weighted sd | 6.39pp | **0.182** |

Per-season SE is 5.1-7.3pp, so every deviation from the pooled ROI is within 1.83 SE:

| Season | ROI | Deviation / SE | Market err on backed side | Model err on backed side |
|---|---:|---:|---:|---:|
| 2020 | -8.61% | -1.21 | +3.8% | +12.3% |
| 2021 | +10.06% | **+1.83** | -5.1% | +1.9% |
| 2022 | -6.35% | -1.28 | +2.0% | +9.1% |
| 2023 | -1.84% | -0.37 | +0.4% | +7.1% |
| 2024 | -1.20% | -0.23 | -0.3% | +6.2% |
| 2025 | +5.55% | +0.99 | -3.2% | +3.6% |

One draw at 1.8 SE across six seasons is expected, not anomalous.

**The model was overconfident in all six seasons**, +1.9pp to +12.3pp, including both
profitable ones. So 2021 and 2025 were not seasons where the model was right. What varied is
the *market's* residual on the backed subset: it fell our way in 2021 and 2025 (-5.1%, -3.2%)
and against us in 2020 and 2022 (+3.8%, +2.0%). Market error straddles zero with mean
-0.40%, exactly a calibrated price plus seasonal noise.

`corr(season ROI, market error) = -0.996`, but that is an identity rather than a finding:
the backed side is always the one the model prefers, so realised ROI is a monotone function
of `actual - market_p` on that side. It restates the market residual, it does not explain it.

Regime changes remain plausible a priori — 2020's 60-game schedule and empty parks, 2021's
mid-season foreign-substance enforcement, 2022's universal DH and compressed spring, 2023's
pitch clock and shift restriction. None is detectable at this sample size, so attributing
2021 or 2025 to any of them would be fitting a story to noise.

### Calibration: yes unconditionally, no conditionally

`scripts/assess_model_calibration.py`. Slope and intercept from `outcome ~ logit(p)`, plus
the Murphy decomposition `Brier = reliability - resolution + uncertainty`, on 11,912
walk-forward out-of-sample predictions. Base rate 53.27%, uncertainty 0.2489, so always
predicting the base rate scores Brier 0.2489.

| All games | Slope | Intercept | Reliability | Resolution | Brier | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Model | +0.975 ± 0.051 | -0.005 ± 0.020 | 0.0002 | 0.0079 | 0.2411 | calibrated |
| Market | +1.015 ± 0.048 | +0.003 ± 0.020 | 0.0002 | 0.0095 | 0.2394 | calibrated |

Unconditionally the model is genuinely well calibrated: slope indistinguishable from 1
(z = -0.48), intercept from 0 (z = -0.25), reliability 0.0002. The decile table shows no
bucket off by more than 1.9pp.

**Calibration is not the binding constraint.** A forecast returning the base rate every time
is perfectly calibrated and worthless, so the informative term is resolution: model 0.0079
against market 0.0095. The model carries 83% of the discriminating signal already in the
price, and none of it is additional (blend coefficient -0.022 ± 0.120).

| Backed sides only, n=2,458 | Slope | Intercept | Reliability | Verdict |
|---|---:|---:|---:|---|
| Model | +0.911 ± 0.116 | **-0.255 ± 0.042 (z = -6.09)** | 0.0052 | **MISCALIBRATED** |
| Market | +0.947 ± 0.115 | +0.017 ± 0.048 (z = +0.36) | 0.0005 | calibrated |

Conditional on selection the model fails, and the failure is **entirely in the intercept, not
the slope**: a uniform -0.255 logit shift, about 6.3pp at even money. Ranking within the
subset is fine; the level is biased high. Its reliability is 10x the market's, while the
market stays calibrated on the very games the model most disputes.

This is the winner's curse. Selecting on `model - market` being large selects cases where the
model's own error is positive.

**The bias and the apparent edge are the same quantity**, which forecloses "just recalibrate
it". Applying the -0.255 correction to the backed sides:

| | Mean probability | Mean claimed edge |
|---|---:|---:|
| Model, raw | 51.48% | +7.02% |
| Model, corrected | 45.31% | **+0.85%** |
| Market fair | 44.46% | |
| Actual | 45.16% | |

Correcting the calibration defect collapses the model onto the market and removes the edge it
was selecting on. Brier improves 0.2454 to 0.2414, still short of the market's 0.2408. There
is no recalibration that preserves the edge, because the edge was the defect.

### What would have to change, quantified

The binding constraint is resolution, not calibration, thresholds, or execution. Working
backwards from the observed decomposition (fair probability 44.46% on backed sides, realised
edge +0.70pp, shopped hold 1.40%):

| Target ROI | Required realised edge | Multiple of today |
|---|---:|---:|
| +1% | +1.07% | 1.5x |
| +2% | +1.51% | 2.2x |
| +3% | +1.96% | 2.8x |
| +5% | +2.85% | 4.1x |

A standardised feature with incremental logit coefficient `b` shifts probability by about
`b * p * (1-p)`, so +2-3% ROI needs **0.06-0.08 logits per standard deviation** of genuinely
incremental signal. Measured detection SEs are 0.02-0.04 logits, so the required effect is
2-4x the noise floor: findable at n≈12,000 if it exists, with multiplicity control mandatory.

Resolution is the right frame. Market resolution is 0.0095 and the model's is 0.0079, all of
it redundant. Beating the price requires resolution **above** 0.0095 sourced from information
the price lacks, not a larger share of what it already has.

### Architecture change: anchor on the price

The current design builds a standalone forecast and compares it to the price, which is what
produces the winner's curse worth -0.255 logits of phantom edge. The structural fix is to make
the price the base and model only the residual:

    logit(p) = logit(market_fair) + f(X)

Under this form `f = 0` is the default and every deviation must be earned out of sample, so
selection on disagreement cannot manufacture edge. Given the measured increment of
-0.022 +/- 0.120, `f = 0` is also the current best estimate.

### Feature hypothesis tested and refuted: starter stuff trend

`scripts/build_starter_stuff_features.py`. The pitches table holds 14.2M rows for 2020-2025
with full movement data, none of which the 8-feature team-strength model uses. Hypothesis: a
pitcher's physical stuff degrades before his results-based statistics show it, and the market
prices the results-based statistics.

Constructed strictly pre-game — the pitcher's previous 2 starts against the 6 before those,
with the current start excluded by a shift, differenced home minus away. 34,433 starter-pair
games, 9,811 with odds.

| Feature | n | Raw z | Residual z vs price |
|---|---:|---:|---:|
| Fastball velocity trend | 9,811 | -0.22 | -0.34 |
| Breaking-ball induced vertical break trend | 9,457 | -1.14 | -0.84 |
| Breaking-ball horizontal break trend | 9,457 | +0.09 | +0.27 |

**Refuted, and not for want of power.** Detection SE is 0.0206 logits per sd against a
0.06-0.08 target, so the test carries 3-4x the sensitivity required. Further,
`corr(logit(market_p), fb_velo_edge) = +0.006`: the market does not price this signal and it
does not predict outcomes either (raw z = -0.22). It is orthogonal to both, i.e. noise rather
than suppressed signal.

Thirteen candidate features across two screens have now failed against a price whose
calibration slope is 1.015 +/- 0.048. The prior on the fourteenth should be low.

### Ruled out, with the measurement that ruled it out

| Action | Why not |
|---|---|
| Tune the edge threshold | 7% is flat (-0.02%), 10% is -14.84%; monotonicity was noise |
| Recalibrate the model | The -0.255 correction collapses it onto the market, edge to +0.85% |
| Add more books | Saturates at 5 accounts; the 6th adds +0.06pp |
| Collect more seasons of the same market | Needs ~13,152 bets, about 32 seasons |
| Optimise CLV | Positive CLV here is a selection artifact, not skill |
| Bet earlier at the open | +0.37% vs +0.18%, inside noise; hold is identical at both points |

### Both untested directions, now tested

The two cheapest open questions were whether a thinner market prices worse, and whether real
closing lines change the CLV picture. Both were run; details in the two sections below. Summary:
F5 totals is measurably less efficient but charges more than the inefficiency is worth, and
genuine closing lines confirm the model anticipates line movement yet still cannot beat the
close.


### F5 totals: less efficient, but more expensive than the inefficiency is worth

`scripts/test_f5_totals_efficiency.py` and `scripts/test_f5_line_bias.py`. First-five totals
from `mlb.f5_odds`, 2025 only, outcomes from `mlb.linescore` innings 1-5, five complete innings
required, pushes dropped.

A methodological note first: **calibration of P(over) is nearly uninformative for a totals
market.** Books set the point so the two sides are close to even, leaving resolution at
0.0008-0.0009 against 0.0095 for full-game moneyline, and the mean forecast pinned at 50.1%.
The slope standard error is 0.68-0.71, so the slope test cannot resolve anything. The
informative tests use the line and the hit rate.

| | Per-book hold | Shopped hold (5 books, same point) | Under edge | Breakeven | Tradeable |
|---|---:|---:|---:|---:|---|
| F5 totals, open | 6.52% | 4.05% | +2.49% | 2.03% | marginal |
| F5 totals, close | 6.52% | 3.60% | +1.67% | 1.80% | **no** |
| *Full-game ML* | *4.10%* | *~1.40%* | | | |

Two genuine signs of lower efficiency: a persistent lean to the under, and **no improvement
from open to close** (Brier 0.2498 to 0.2504, and the point moved in only 22% of games) where
full-game moneyline does improve. But the market charges 6.52% per book and still 3.60% after
shopping five books at the same point, against 1.40% for full-game. The lean is roughly 2.0 SE
at open and 1.4 SE at close on one season, and it does not clear the shopping-adjusted
breakeven at the close. Cheaper to trade the efficient market than the expensive inefficient
one.

Do not read the mean-residual statistic as a signal: actual first-five runs exceed the line by
+0.42 runs with z = +5.5, yet the under still hits more often, because runs are right-skewed.
The line sits below the mean and above the median simultaneously. Only the hit rate is
decision-relevant.

F5 moneyline cannot be tested: `home_ml` is null for every open and close row in `mlb.f5_odds`.
`mlb.nrfi_yrfi_odds` holds 5 rows and needs a backfill.

### True closing lines: infrastructure

`scripts/fetch_closing_lines.py`. The historical endpoint stores snapshots every five minutes
and returns the latest at or before the requested timestamp, so requesting
`first_pitch - 2min` lands within about five minutes of the close. Games sharing a start time
all appear in that one snapshot, so the cost is one call per distinct start time rather than
per game: 672-1,801 calls and 6,720-18,010 credits per season.

| Season | Distinct starts | Credits | Median lead | Coverage loaded |
|---|---:|---:|---:|---:|
| 2020 | 672 | 6,720 | 5 min | 841/900 (93.4%) |
| 2021 | 1,710 | 17,100 | 6 min | 2,412/2,430 (99.3%) |
| 2022 | 1,765 | 17,650 | 6 min | 2,429/2,430 (100.0%) |
| 2023 | 1,742 | 17,420 | 4 min | 2,425/2,430 (99.8%) |
| 2024 | 1,780 | 17,800 | 4 min | 2,426/2,430 (99.8%) |
| 2025 | 1,801 | 18,010 | 4 min | 2,427/2,430 (99.9%) |

Total 94,700 credits. Events at or before the snapshot timestamp are discarded using the API's
own commence time, so in-play prices cannot leak in as they did in the first pull.

One bug worth recording because it is easy to repeat: `game_datetime` is timezone-aware in the
session zone, so `strftime("%Y-%m-%dT%H:%M:%SZ")` formats the *local* components and appends a
literal Z, requesting a snapshot hours from first pitch. Symptom was a uniform ~305 minute lead.
Convert with `astimezone(UTC)` before formatting.

---

## Line shopping

`scripts/backtest_moneyline_lineshop.py` (per season) and
`scripts/backtest_moneyline_lineshop_pooled.py` (pooled, sweeps account count).

Design, because the naive version flatters itself:

- **Edge** is measured against a fair probability built by de-vigging **each book's own
  two-sided pair**, then taking the median of those per-book fair probabilities. Pairing is
  never broken across books. Taking the best price on *both* sides gives +1.0% to +2.3%
  overround with 0.5-9.8% of games outright negative, i.e. fabricated arbitrage; de-vigging
  that would inflate every edge.
- **Execution** is the best decimal price on the single chosen side, within a fixed panel.
  Only one side of a game is ever backed. Because the de-vig normalises the two outcomes to
  sum to 1, the away edge is the exact negative of the home edge, so side selection reduces
  to `model > fair_home`.
- **Isolation**: the identical bet list is settled twice, at best price and at consensus
  price. The difference is execution value with selection held constant.

Bets are placed at the **last strictly-pre-game snapshot** (~2.5h out), not `open`. That is
the only market state with stable book coverage across 2020-2025: five books clear 80%
coverage in every season (`betonlineag, draftkings, fanduel, bovada, betrivers`), averaging
10.7-19.5 books/game. The `open` bucket sits 19-29h out and in 2023 carries almost nothing
but DraftKings, so best-of-N there is confounded by panel depth rather than by shopping.

### Execution value scales with accounts, then saturates

Pooled 2020-2025, flat 1u, edge >= 5%:

| Accounts | Bets | ROI best price | ROI consensus | Shop gain | Avg price impr |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,521 | +0.21% | +0.21% | +0.00pp | +0.000% |
| 2 | 2,502 | +0.57% | -0.58% | +1.14pp | +0.515% |
| 3 | 2,478 | +0.06% | -1.58% | +1.64pp | +0.745% |
| 4 | 2,477 | -0.12% | -2.01% | +1.89pp | +0.897% |
| 5 | 2,458 | +0.18% | -1.92% | +2.10pp | +0.962% |
| 6 | 2,478 | +0.24% | -1.92% | +2.16pp | +1.010% |

Shopping is worth a consistent **+1.9 to +2.3pp** in every individual season, and the gain
saturates: 5 to 6 accounts adds only +0.06pp.

**But ROI at best price is flat in the account count** (+0.21% at K=1 vs +0.24% at K=6).
Adding books does two things at once: it improves the fill *and* sharpens the fair-value
benchmark, which changes which bets qualify. At K=1 the benchmark is the same book being
bet, which is a weak benchmark that flatters selection. As the benchmark sharpens, consensus
ROI falls from +0.21% to -1.92% and best-price execution claws back almost exactly the
2.1pp. **The two effects cancel.** Shopping converts a ~4.1% hold into a ~2% hold; the
model's genuine edge is worth roughly the same; the result is breakeven.

### The result is statistically indistinguishable from zero

Bootstrap, 4,000 resamples, panel of 5, best-price execution:

| Threshold | Bets | ROI | 95% CI | Per-bet sd | SE |
|---|---:|---:|---|---:|---:|
| 2% | 7,264 | -1.96% | [-4.44%, +0.71%] | 1.10 | 1.29% |
| 3% | 5,294 | -1.64% | [-4.61%, +1.37%] | 1.11 | 1.53% |
| 5% | 2,458 | +0.18% | [-4.40%, +4.73%] | 1.15 | 2.31% |
| 7% | 917 | -0.02% | [-7.46%, +7.51%] | 1.17 | 3.88% |
| 10% | 177 | -14.84% | [-32.02%, +2.43%] | 1.17 | 8.82% |

Every interval straddles zero. The monotone-improvement-with-threshold pattern visible in
the non-shopped results **does not survive** past 5%: 7% is flat and 10% is sharply negative
on 177 bets. That pattern was noise.

Per-bet standard deviation is ~1.15 units, so resolving a 2% ROI at two standard errors
needs **13,225 bets** — about 32 seasons at the 5% threshold (410 bets/season) or 11 seasons
at the 2% threshold (1,211 bets/season). Six seasons cannot answer this question, and no
amount of re-analysis of these six will change that.

---

## Known limitations

- **Best-price execution assumes the price is actually takeable.** Best-of-5 is a
  price-feed maximum, not a filled bet: no limit checks, no line movement between
  observation and placement, no account restriction. Sharp accounts get limited, and the
  best price often carries the smallest limit. Real fills sit somewhere between the
  consensus and best columns, so the +2.1pp is an upper bound.
- **No stake sizing across correlated games.** Same-day bets share weather, umpire, and
  bullpen-fatigue exposure; flat and Kelly both treat them as independent.
- **Close is the last pre-game snapshot, not the true close.** With snapshots at
  16:30/20:30/00:30 UTC, `close` sits ~2.5h before first pitch, so CLV understates a real
  closing-line comparison.
- **~21% of 2020-2023 games are absent** for lack of two pre-game snapshots. A denser
  snapshot cadence would recover them.
- **2020 is a 60-game COVID season** with 679 usable games; treat separately.

---

## Data inventory

| Season | Games w/ open | Games w/ close | Source |
|---|---:|---:|---|
| 2020 | 680 | 680 | parallel fetch, pre-game split |
| 2021 | 1,947 | 1,947 | parallel fetch, pre-game split |
| 2022 | 2,184 | 2,184 | parallel fetch, pre-game split |
| 2023 | 2,255 | 2,255 | parallel fetch, pre-game split |
| 2024 | 2,236 | 2,422 | pre-existing pipeline |
| 2025 | 2,308 | 2,426 | pre-existing pipeline |

Credits: 18,720 used of 5,000,000. Raw payloads retained at
`data/odds_history/moneyline_{season}.parquet`, so the split can be re-derived without
re-fetching.

---

## Status

**Do not deploy.** Pooled ROI at the 5% threshold with best-of-5 execution is **+0.18%,
95% CI [-4.29%, +4.94%]** over 2,458 bets — statistically indistinguishable from zero, and
profitable in only 2 of 6 seasons.

The stronger objection is not the ROI, it is the forecast comparison. On the backed sides the
market's fair probability lands within 0.7pp of the outcome while the model is 6.3pp high,
and **market Brier beats model Brier** (0.2408 vs 0.2454) in 5 of 6 seasons. The model
realises 10% of its claimed edge. Large model-market disagreement is, for this model,
evidence of model error — so the edge filter selects against itself and a higher threshold
selects harder. That is why the monotone-improvement pattern reverses past 5%.

Line shopping is nonetheless real and worth **+2.1pp**, converting a ~4.1% hold to ~2%. It is
the entire difference between -1.92% and +0.18%. Keep it for any future strategy; it is
orthogonal to model quality.

Waiting for data is not the fix: resolving a 2% ROI needs ~13,152 bets, about 32 seasons at
this bet rate. The binding constraint is the numerator. In rough order of expected value:
improve the forecast where the market is weakest rather than where it disagrees most; test
markets thinner than full-game moneyline; obtain lower-hold pricing. Each changes the edge
itself instead of narrowing an interval around zero.

Futures results are reported separately in `docs/FUTURES_BACKTEST_COMPLETE.md` and were
produced from Covers.com preseason odds through a different path. They are untouched by
this defect but were **not** re-verified in this pass; the same "is the price actionable at
the timestamp claimed" question applies there and is open.
