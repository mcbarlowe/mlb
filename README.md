# MLB Betting Model

An MLB data pipeline, pitch-level Monte Carlo simulator, and team-strength win model, built to
find a tradeable edge in baseball betting markets — and the record of what happened when every
component was measured against real prices.

**Pipeline, schema, setup, and operational runbooks:** [`docs/PIPELINE.md`](docs/PIPELINE.md).
**Full evidence for the claims below:** [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Verdict

**There is no deployable prediction edge in this repository.** One edge is verified, and it is an
execution edge rather than a forecasting one. Six model/market pairs have now been tested against
real closing prices. All six failed.

| Component | Status | Measurement |
|---|---|---|
| Line shopping across books | **VERIFIED** | +2.1pp, all six seasons, saturates at 5 accounts |
| Never bet June | **VERIFIED** | 12/12 negative cells; 91% of the gap is the market sharpening |
| Flat staking | **CORRECT BY DEFAULT** | Kelly amplifies loss when the edge estimate is negative |
| Team-strength moneyline model | **FAILED** | market more accurate on the bet subset, interval excludes zero |
| Full-game totals via pitch sim | **FAILED** | sim Brier worse by +0.0032 on 574 non-push games; ROI −1.5% on 372 bets |
| First-five totals | **FAILED** | no open→close improvement, 3.60% shopped hold |
| Pitcher strikeout props | **FAILED** | no bias across 8,785 graded starts, −0.25% ± 0.47% |
| Division / playoff-field / pennant futures | **FAILED** | flat 1u: −8.16%, −7.98%, +7.24% (fade side, CI [−18.6, +34.8]) after fixing six defects |

The infrastructure is the asset. The measurement discipline is the deliverable. The forecasts are
not tradeable.

---

## Rule 1: shop every price across at least five books

The only reproducible positive result in this project.

| Metric | Single book | Best of 5 |
|---|---:|---:|
| Moneyline hold | 4.10% | ~1.40% |
| Effect on realised ROI | baseline | **+2.1pp** |

Present in every season 2021–2026, ranging +1.9pp to +2.3pp. Gains saturate at five accounts; a
sixth adds nothing measurable. This is structural rather than statistical — taking the best of
several two-sided quotes mechanically reduces the vig paid. It is also the entire difference
between the headline moneyline results of −1.92% and +0.18%.

Panel is `PANEL_PRIORITY[:5]` in `scripts/backtest_moneyline_lineshop.py`. Execution must clear
the best available price on the side being bet, never the consensus.

**Operational caveat.** Best-of-five is a *price-feed* maximum, not a fill. Real accounts carry
lower limits, stale openers get pulled or voided, and shopping is precisely what triggers limits.
Realised execution lands below this ceiling.

---

## Rule 2: never bet June

The best-evidenced structural finding here, and the only subgroup effect that survived every guard
applied to it.

| Guard | Result |
|---|---|
| Bonferroni for 6 months | z = −3.12 against a 2.64 threshold |
| Permutation test on month spread | observed range 18.51%, p = 0.003 |
| Threshold monotonicity | negative at 1/2/3/4/5/7% edge |
| Second training window | negative 6/6 seasons for 2015+ as well as 2018+, so 12/12 |
| Independent metric | corr(monthly Brier gap, monthly ROI) = −0.964, computed on all games |

Pooled June ROI is −10.49% on 983 bets. No other month is unanimous: Mar/Apr 8/12, May 2/12,
Jul 4/12, Aug 5/12, Sep/Oct 6/12. **Only June.**

### The mechanism is the market, not the model

This determines whether the rule generalises, and it inverts the obvious reading.

| Forecaster | June | non-June | vs own baseline |
|---|---:|---:|---:|
| Model | 0.242168 | 0.241986 | +0.000182 |
| Market | 0.238889 | 0.240764 | **−0.001875** |

**91% of the June gap is the market sharpening; 9% is model degradation.** The model is flat in
June — the price gets materially better. June is the cleanest information environment of the
season: roughly 60–70 games of current-season data, rosters and bullpen roles settled, spring
uncertainty gone, and it precedes both July deadline churn and September call-ups and tanking. A
sharp market exploits that; season-to-date aggregates do not gain proportionally.

"Avoid where the counterparty is sharpest" has a causal story. "Avoid where I look weakest" does
not, and would have been the wrong diagnosis here.

### What the rule buys

| Configuration | Bets | Pooled ROI | Bet-subset Brier gap |
|---|---:|---:|---|
| All months | 5,699 | −1.99% | +0.003217 [+0.001702, +0.004696] |
| June removed | 4,716 | **−0.22%** | +0.002517 [+0.000915, +0.004085] |

It converts a clear loser into a breakeven. It does not create an edge: the ROI interval still
includes zero, and the accuracy test remains significant against us.

---

## Rule 3: flat stakes, and only with a negative expectation accepted

Kelly and every fractional-Kelly variant size *up* on larger estimated edge. When the edge
estimate is biased positive — the measured condition here — that concentrates stake on the worst
bets. Flat 1u is correct for a strategy with no established edge, because it minimises the
variance of a known-negative process instead of maximising growth of an imaginary positive one.

Do not run Kelly against these edge estimates. The edge term is not merely uncertain; it is
directionally wrong on the bet subset.

---

## The arithmetic that decides all of it

Breakeven requires the model to be more accurate than the price by enough to cover the shopped
hold. Measured on exactly the games the strategy bets (|edge| ≥ 3%, non-June, six seasons):

```
model Brier   0.244081
market Brier  0.241564
gap          +0.002517    95% CI [+0.000915, +0.004085]    market wins, significant
```

The model is **less** accurate than the price on the games it selects — and not by an
unmeasurable amount. The interval excludes zero across 4,716 bets.

### Selection makes it worse, not better

| Sample | Model minus market Brier |
|---|---:|
| All games, non-June | +0.001222 |
| Bet subset, non-June | **+0.002517** |

Filtering to disagreements **doubles** the deficit. This is the winner's curse: betting large
positive disagreement selects a mixture of *market is wrong* and *model is wrong high*, and the
second term dominates. Confirmed independently by a λ-shrinkage sweep, where shrinking the model
toward the price improved accuracy monotonically while bet volume and profit collapsed to zero
together — accuracy purchased by reproducing the market's own information cannot pay.

The consequence is general: no threshold, month, feature set, or training window fixes this,
because the selection step itself is anti-predictive. Raising the edge threshold past 5% was
measured and fails (7% flat, 10% at −14.84%).

---

## Market selection

From the 34-market hold survey (`scripts/survey_mlb_market_holds.py`), ranked by the edge needed
to break even. Lower is more attackable.

| Market | Per-book | Shopped | Breakeven | Status |
|---|---:|---:|---:|---|
| h2h (moneyline) | 4.10% | 1.40% | **0.70%** | tested, no edge |
| Full-game totals | 4.71% | 2.48% | 1.24% | tested, no edge |
| Run line / spreads | 4.29% | 2.41% | 1.20% | untested, poor prior |
| Pitcher strikeouts | 6.80% | 3.45% | 1.66% | tested, no bias |
| Totals 1st inning (NRFI) | 6.49% | 3.69% | 1.85% | untested, needs backfill |
| First-five totals | 6.52% | 3.60% | 1.80% | tested, no edge |
| Batter props | 5.4–7.1% | 2.5–3.5% | 2.5–3.5% | too expensive |
| Season win totals | no odds stored | n/a | ~4.8% at −110 | tested, no accuracy edge |

**Hold rises monotonically with how exotic the market is.** The cheapest markets are the most
liquid and most sharply priced; the expensive ones frequently have a single book quoting them, so
no shopping is possible. There is no market that is both cheap to trade and inattentively priced.

Do not bet futures. Every tested futures market loses once six defects are corrected:
multi-winner de-vig normalised to a single winner (inflating edges 6×/12×/18×), hardcoded outcomes
containing a fabricated 2023 Seattle division title, a pennant probability summing to 4.0 where
2.0 is required, a corrupted `teams.division_name` placing Houston in the NL Central, an
era-blind playoff field, and a stale-snapshot scrape. See
[`docs/FUTURES_AUDIT.md`](docs/FUTURES_AUDIT.md).

The **era-blind playoff field**: `simulate_season` defaulted to three wild cards per league for
every season, simulating a 12-team field in 2016–2021 when only 10 berths existed. That overstated
every team's playoff probability by exactly 2/30 before 2022 and, because the de-vig target shared
the assumption, fabricated edge on the fade side. The tell was a mean model playoff probability of
exactly 0.4000 in all nine seasons. Both are now era-aware and agree with
`scripts/futures_outcomes.py`; the de-vig check is self-proving, since a correctly normalised
market must show zero bias, and it moved from +0.0370 to −0.0000 [−0.0495, +0.0490].

The **stale-snapshot scrape**: `_find_preseason_column` in `scripts/scrape_covers_all_futures.py`
only ever saw the group-header row, found no month name in it, and fell through to "first column
after Team". The number of preseason snapshots and of blank leading columns both vary by season, so
six of twelve division seasons held in-season prices — 2013, 2021 and 2022 from June 1, and 2014,
2017 and 2023 from May 1, each matching 30/30. Those seasons compared a March-15 projection against
a market holding one to two months of results the model did not have. Fixed by reading the
`Preseason` group colspan and taking the last preseason column; the fixed parser reproduces all six
previously-clean seasons exactly. `scripts/reload_futures_preseason_odds.py` rewrote all thirteen
seasons at their true snapshot, non-destructively, since `load_latest_futures_odds` selects
`MAX(snapshot_time)` and every true date falls after the `03-24` label the stale rows carry.
Division moved from −10.07% to −8.16%, and 2021 flipped sign from −56.2% to +23.1%.

**Pre-registered test.** 2026 preseason division odds are now stored (`2026-03-25` snapshot,
30 teams, source `covers.com-preseason`). 2026 playoff outcomes do not exist until October, so the
test cannot run yet. It is worth running because both instruments agreed in direction on the
2024–2025 window — accuracy moved from a significant market advantage of +0.02439 [+0.00517,
+0.04428] in 2016–2023 to an indistinguishable −0.00360 in 2024–2025, and ROI from −8.83% to
+24.71% — while neither difference reached significance and the window was chosen after seeing the
table. One genuinely out-of-sample season is the cheapest way to break that tie.

**Provenance caveat.** `make_playoffs` and `miss_playoffs` odds carry the `covers.com` source tag
but cannot be reproduced from it: the archive exposes only world-series, pennant and division
pages, and `a=po` silently serves the division page. They do pass a strong internal check, with
per-team implied probabilities summing to 1.037–1.064 across all nine seasons, a realistic 4–6%
two-sided hold, so the prices are genuine market quotes rather than synthesised. Their snapshot
timing is nonetheless unverified, which is exactly the defect that corrupted half the division
history, so treat the +7.24% fade result as unverified rather than merely weak.

Playoff-field calibration is worth recording separately, because it is a live model defect rather
than a betting result. Across 270 team-seasons the mean bias is now exactly zero, but the tails
are badly overdispersed: teams the model rates 0.75–0.90 to reach the playoffs actually do so
61.5% of the time, and teams it rates below 0.10 reach them 9.8% of the time. The season simulator
carries no injuries, trades, or in-season regression, so its probabilities are too extreme. The
exploitable direction is to fade the model's own strong favourites — which its edge rule can never
generate, since it only bets where the model disagrees with the price in the model's favour.

---

## What is explicitly NOT justified

These configurations look profitable and are hindsight selection. They are recorded so nobody
re-derives them and mistakes them for findings.

| Configuration | Reported | Why it fails |
|---|---:|---|
| 2025+2026 only, June removed | +6.59% [+0.94%, +12.59%] | best 2 of 6 seasons chosen after seeing results; paired Brier on those same 1,378 bets is −0.0007 ± 0.003, i.e. nothing |
| May + Aug only | +5.36% [+0.62%, +10.30%] | 15 month-pairs available; the companion Mar/Apr exclusion is 8/12 mixed, refuting the method |
| Excluding Jun + Mar/Apr | +2.00% | Mar/Apr is not unanimous in either training window |
| Recency window 2018+ over 2015+ | +1.40% claimed | paired test gives −0.68pp, CI [−0.048, +0.035], worse in 4 of 6 seasons |
| Futures, 2022–2025 window | +37.66% | the full 12-season range gives −3.37% |
| `miss_playoffs` fade, quarter-Kelly | +16.25% | Kelly sized winning heavy favourites to 0.000u, so 2017 returned exactly −100% from a 55% win rate; flat 1u gives +7.24%, and the era-blind playoff field was fabricating the rest |
| `miss_playoffs` fade, flat 1u | +7.24% | CI [−18.6%, +34.8%]; 5/9 seasons; monotonicity broke once the era bug was fixed; one 19-bet bucket carries it; ~190 seasons needed to resolve |
| Season win totals, 3+ win edge | +16.88% | model MAE 7.840 vs line MAE 7.810, difference +0.030 [−0.814, +0.868], so no accuracy basis; every threshold CI includes zero; 2025 alone carries it at +43.18% |
| Season win totals, all bets | +8.62% | 2/4 seasons positive; assumes −110 with no stored odds and no shopping possible, and real win-total juice is frequently −115/−125 |
| Positive CLV at the 5% threshold | evidence of skill | non-monotone across thresholds; the closing line is only 0.0003 Brier better than the open |

The pattern is identical every time: a subset with a nominally significant interval, an interval
that ignores the selection, and collapse when the sample is extended.

**A configuration selected on ROI must be validated by an accuracy test on the same games.** ROI
discards the probability and keeps only win/loss; paired Brier keeps the magnitude and is the
stronger instrument. When they disagree, the accuracy test wins.

---

## If you bet anyway

Expect to lose approximately the shopped hold. Size from that, not from an edge estimate.

- Flat 1u. No Kelly, no progression.
- Best price across five books, always.
- No June.
- Moneyline only. Every other market is more expensive and none has been shown attackable.
- Treat it as paid entertainment, or as data collection for the movement work below — not as an
  investment.

Measured pooled expectation with all three rules applied: **−0.22% per unit staked**, interval
[−3.25%, +2.92%]. Indistinguishable from breakeven, and the best honest estimate available.

---

## What would change the verdict

One category remains untested with a plausible route to information the price lacks.

**Market-dynamics data.** The oracle test established a real ceiling: perfect foresight of the
closing line, betting at the open, returns **+6.14% [+1.14%, +10.94%]** on 1,566 bets restricted
to games whose line moves at least three points. That is statistically significant, and it is the
only positive result in this project that is not an artifact of selection.

It is unreachable with current inputs. The target is continuous — `logit(close) − logit(open)` —
carrying far more statistical power per game than binary win/loss. Four features survived a
Bonferroni residual screen against it (`model_disagree`, `book_dispersion`, `lead_hours`,
`n_books`), yet the best predicted-movement standard deviation is **0.0206 logits against the
0.1200 required**, a 5.8σ shortfall, and predicted movement never once reaches the threshold
across 6,988 games. ROI degrades monotonically as predicted-movement confidence rises.

Closing that gap needs data this repository does not have and cannot derive: lineup announcement
timestamps, injury-feed latency, per-book move logs, and betting-percentage splits. **That is a
purchasing decision, not a modelling one.** Everything reachable by compute has been tested.

---

## Reproduction

```sh
# Month scoping, ROI by season, and the June mechanism decomposition
uv run python scripts/test_june_scoped.py
uv run python scripts/test_june_scoped.py --seasons 2025,2026

# Month-by-month with all six multiplicity guards
uv run python scripts/analyze_monthly_performance.py

# The training-window recency claim, paired and structurally leak-free
uv run python scripts/test_training_window_recency.py

# Market hold survey across 34 MLB markets
uv run python scripts/survey_mlb_market_holds.py --dates 2025-06-10,2025-07-15,2025-08-05

# Oracle ceiling and the line-movement screen
uv run python scripts/test_beat_the_opener_ceiling.py
uv run python scripts/screen_movement_signal.py

# Moneyline backtest with and without line shopping
uv run python scripts/backtest_moneyline_lineshop.py

# Futures at flat 1u with season-consistency, threshold and odds-bucket guards
uv run python scripts/review_futures_flat.py

# Playoff-field calibration on 270 team-seasons; catches the era-blind wild-card defect
uv run python scripts/test_playoff_calibration.py

# Verify every futures snapshot is a true preseason column, then reload if not
uv run python scripts/reload_futures_preseason_odds.py            # dry run
uv run python scripts/reload_futures_preseason_odds.py --write
```

---

## Model registry

Registered model `mlb-team-strength-win` on the shared MLflow service, experiment
`mlb-model-training-shared`.

| Version | Fit window | Features | Holdout | Gate | Alias |
|---|---|---:|---|---|---|
| v1 | 2021–2024 | 5 | 2025 | passed | `champion` |
| v3 | 2018–2024 | 8 | 2025 | failed | — |
| v4 | 2021–2024 | 8 | 2025 | failed | — |

Versions 3 and 4 are recorded challengers. Neither should be promoted. All four cells of the
feature × window grid fall within 0.000290 Brier of one another, and no pairwise difference is
significant — the grid is smaller than its own noise, and the gap to the market (+0.001572 pooled)
is over five times the entire grid's width.

The gate requires a positive 95% paired date-block lower bound against the incumbent across
walk-forward folds, with no material single-season regression. **That gate is what prevented every
claim in the "not justified" table from reaching deployment**, and it correctly failed v3 despite
a better point estimate and wins in all four folds.

Do not hand-register model versions. A prior artifact was created with
`create_model_version(source="runs:/<id>/")` and no `run_id`, producing an entry that failed 9 of
the loader's 15 validation gates, logged only `train_accuracy`, and could never be loaded. Use
`scripts/evaluate_team_strength.py --log-mlflow`, which logs the ordered feature contract,
training and holdout datasets, coefficients, rolling-fold evidence, bootstrap intervals, and the
promotion result.
