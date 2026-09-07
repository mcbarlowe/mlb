# Bayesian Pitcher Strikeout Model and Backtest Plan

## Status and objective

Execution is deferred until the repository refactor is complete. Do not implement this plan on
both sides of the refactor or preserve compatibility shims for the pre-refactor layout. At the
start of execution, remap the proposed modules below to the refactored boundaries and make one
clean implementation.

The objective is to produce a leak-free pregame probability distribution for a starting pitcher's
strikeout count and test it against actual, strictly pregame sportsbook prices. The result must
answer two separate questions:

1. Does the model predict strikeout props more accurately than the de-vigged market?
2. Do the model's selected bets make money at prices that were actually available?

A negative result is complete work. Bayesian shrinkage can reduce false edges; it cannot create an
edge that is not present.

## Scope

### In scope

- Starting-pitcher strikeout props at the market's primary line.
- Over, under, and push probabilities from a full posterior predictive distribution.
- Historical regular-season outcomes from 2015-2025.
- Historical pregame prop prices from 2023-2025.
- A fixed cross-season sportsbook panel for the headline backtest.
- Walk-forward features and posterior updates using only information available before each odds
  snapshot.
- Flat 1-unit staking for every headline result.
- Model, market, calibration, ROI, and uncertainty comparisons.

### Out of scope

- Live or real-money deployment.
- Kelly sizing or bankroll optimization.
- Parlays and alternate-line ladders.
- Reusing the hitter model's EV gates without strikeout-specific validation.
- Using the realized starting lineup unless a timestamped pregame lineup source is added.
- Adding a heavy probabilistic-programming dependency before the simple Bayesian baseline earns
  that complexity.
- Tuning any model, feature, book panel, or betting gate on 2025 results.

## Verified data inventory

The current repository and PostgreSQL schema provide:

- `mlb.pitching`: strikeouts, batters faced, pitch count, outs, starter status, and pitcher ID.
- `mlb.games`: game identity, teams, game time, season, type, and final state.
- `mlb.players`: pitcher handedness and identity fields.
- `mlb.batting`: opponent strikeout and plate-appearance history.
- 50,386 completed regular-season starts from 2015-2025.
- Only seven of those starts have missing or zero batters-faced values.
- 8,785 currently graded, strictly pregame pitcher-strikeout offers:

| Test season | Graded pitcher-games | Median shopped hold | Median books at primary point |
|---|---:|---:|---:|
| 2023 | 2,353 | 5.31% | 3 |
| 2024 | 3,128 | 4.14% | 6 |
| 2025 | 3,304 | 3.33% | 7 |

Current source artifacts:

- `data/odds_history/props_2023-03-25_2023-09-29.parquet`
- `data/odds_history/props_2024-03-25_2024-09-29.parquet`
- `data/odds_history/props_2025-03-25_2025-09-29.parquet`
- `scripts/fetch_prop_odds.py`: historical per-event odds ingestion with a strictly pregame
  snapshot rule.
- `scripts/analyze_prop_hold.py`: same-book, same-point hold analysis.
- `scripts/test_prop_market_bias.py`: primary-line construction and the existing model-free
  over/under baseline.
- `src/betting/odds.py`: American/decimal conversion and two-way de-vigging.

The full-season files contain pitcher-strikeout prices, so this project can run a genuine
real-price backtest. The current hitter estimator does not yet have equivalent full-season price
coverage.

## Non-negotiable data contract

The existing model-free script grades by `(player_name, date)` and permits a one-day fallback. That
is adequate for an exploratory market-bias screen, but not for a model backtest. The normalized
backtest frame must use exact game and player identities.

Each accepted offer must contain at least:

- `event_id`
- `snapshot_time`
- `commence_time`
- `game_pk`
- `player_id`
- `pitcher_name`
- `home_team_id`, `away_team_id`, and `opponent_team_id`
- `point`
- `bookmaker`
- paired `over_price` and `under_price`
- `actual_strikeouts`

Normalization rules:

1. Map Odds API team names to MLB team IDs.
2. Match an event to one unique `game_pk` using home team, away team, and commence time. The
   implementation must define and test the time tolerance; ambiguous matches are rejected, not
   guessed.
3. Match the offered pitcher to the unique recorded starter in that game and persist `player_id`.
4. Distinguish doubleheaders by game time and teams. Never fall back to date-only grading.
5. Pair over and under within the same book and exact line point. Never manufacture a pair across
   books or points.
6. Define the primary line as the point offered by the most books with a complete two-sided pair.
   Predeclare a deterministic tie-break before outcomes are joined.
7. Preserve unmatched and rejected records with reason codes. Report match coverage before any
   model result.
8. Verify every snapshot is strictly before the corresponding commence time.

Build the fixed sportsbook panel from quote coverage without inspecting outcomes or ROI. The
headline report must use the same panel definition in every season. An all-available-books result
may be reported as a secondary execution scenario, but never substituted for the fixed-panel
headline. This matters because available depth increased from roughly three books per primary
point in 2023 to seven in 2025.

## Modeling strategy

### Required baselines

Implement and retain these before the Bayesian model:

1. **Market baseline:** median of each book's own de-vigged over probability at the primary point.
2. **League count baseline:** season-as-of-date league strikeout distribution.
3. **Rolling empirical-Bayes baseline:** a pitcher's trailing strikeouts per batter faced, shrunk
   toward the as-of-date league rate, combined with a trailing workload estimate.
4. **Model-free execution baselines:** always over and always under at the best fixed-panel price.

The Bayesian model must beat these, not merely a 50% classifier.

### Version 1: two-part hierarchical empirical Bayes

Strikeout count is the product of opportunity and strikeout ability. Model them separately:

\[
B_i \sim \operatorname{NegBinomial}(\mu_{B,i}, \phi_B)
\]

\[
K_i \mid B_i,p_i \sim \operatorname{Binomial}(B_i,p_i)
\]

where `B` is batters faced and `K` is strikeouts. The strikeout-rate component uses partial
pooling:

\[
\operatorname{logit}(p_i)
= \alpha_{season}
+ u_{pitcher,i}
+ v_{opponent,i}
+ \beta^T X_i.
\]

Required v1 information:

- Pitcher strikeouts and batters faced before the snapshot.
- Exponentially decayed recent pitcher history, with league prior weight for small samples.
- Pitcher workload history: batters faced, outs, and pitch count per start.
- Rest since the prior appearance.
- Opponent team strikeouts per plate appearance, computed chronologically before the game.
- Pitcher hand and opponent handedness composition only if the composition is obtainable without
  using the realized postgame lineup.
- Home/away and season environment.

Draw both workload and strikeout rate, then draw strikeout count. Continue posterior simulation
until Monte Carlo error for the over/under probability is below a declared tolerance; use a fixed
seed for reproducibility.

For a line `L`, retain:

\[
P(K>L),\qquad P(K=L),\qquad P(K<L).
\]

For decimal odds `d`, settle expected profit with pushes explicitly:

\[
EV_{over}=P(K>L)(d-1)-P(K<L).
\]

The under calculation is symmetric. A push has zero profit and returns the stake.

This version should be implementable with vectorized array operations and conjugate or
empirical-Bayes updates. Do not add PyMC merely to label the model Bayesian.

### Version 2: full joint hierarchical model, conditional

Consider a PyMC or equivalent joint model only if v1 improves out-of-sample proper scores but
shows a specific deficiency that a joint posterior can address, such as underestimated uncertainty
for low-history pitchers or workload/strikeout-rate dependence. Version 2 is not part of the
initial acceptance criteria. Adding it without a demonstrated v1 limitation is unnecessary model
weight.

### Features explicitly deferred from v1

- Actual batting order.
- Home-plate umpire effect; the current data inventory does not supply a reliable pregame umpire
  contract.
- Pitch-level neural-model outputs.
- Weather and roof status.
- Pitch-mix or velocity changes not expressible as leak-free pregame rolling features.
- Manager-specific hooks unless a simple workload model leaves a repeatable residual.

Add a deferred feature only when a predeclared walk-forward experiment improves real-price ROI or
proper scoring outside the season used to propose it.

## Walk-forward experiment protocol

### Season roles

- **2015-2022:** historical outcomes available for prior estimation and chronological feature
  state.
- **2023 development:** select the v1 prior strengths, decay, workload specification, and one
  primary betting rule.
- **2024 validation:** run the locked 2023 specification once. Bug fixes are allowed; analytical
  changes are not. Any rerun after a bug fix must record the defect and the before/after result.
- **2025 final test:** open only after the 2024 validation decision is recorded. No model, feature,
  panel, or threshold changes after opening it.

The project has already inspected 2025 for a model-free over/under bias. Therefore 2025 is not
pristine to the project as a whole, but it remains unused for selecting this model's features and
betting rule. State that limitation in the final report.

### Chronology invariant

For a prediction at snapshot time `t`, every outcome-derived feature and posterior update must use
rows with game time strictly before `t`. Build the prediction row before applying the current
game's result. Add an automated assertion that deliberately injecting the current result changes
no emitted feature or prediction.

### Bet construction

- Use only the selected primary point for each pitcher-event.
- Compute fair probability from complete same-book pairs, then aggregate across the fixed panel.
- Evaluate both sides at their actual best available fixed-panel price.
- Permit at most one position per pitcher-event. If both sides appear positive because of a true
  cross-book arbitrage, flag it separately rather than letting the model choose both.
- Select the side with the stronger locked decision score.
- Use flat 1-unit stakes.
- Drop no losing observations after selection. DNP, cancellation, and grading rules must be
  declared before validation.
- Report exploratory threshold sweeps separately from the single locked strategy.

The primary betting rule must be written to a versioned configuration artifact after 2023 and
before 2024. It should combine a minimum posterior mean EV with a minimum posterior probability
that EV is positive. Choose both on 2023 only; do not import the hitter strategy's 11%/15% gates.

## Evaluation

### Distribution and probability quality

Report on every eligible primary-line offer and on the selected-bet subset:

- Brier score for the over outcome, excluding pushes from the binary score.
- Log loss for the over outcome, excluding pushes from the binary score.
- Market Brier and log loss on exactly the same observations.
- Brier and log-loss deltas versus market.
- Calibration by predicted-probability bin with estimated versus realized rate.
- Posterior predictive coverage for strikeout-count intervals.
- Mean absolute error and count-distribution log score as secondary diagnostics.

A model can be calibrated but useless if it has no resolution. Report both calibration and its
ability to separate high- and low-strikeout starts.

### Betting quality

Report by season and pooled:

- Eligible offers and selected bets.
- Wins, losses, pushes, and voids.
- Total staked, net units, and flat-stake ROI.
- Mean predicted EV and the realized-ROI gap.
- Mean best price, consensus price, and line-shopping gain.
- ROI by side, line point, odds bucket, month, book, and pitcher-history bucket.
- Maximum drawdown.
- Date-clustered bootstrap confidence interval for ROI.
- Sensitivity to the fixed panel versus all available books.

Do not claim an edge from pooled ROI alone. A single season, price bucket, or longshot must not
carry the conclusion invisibly.

## Advancement and stopping gates

### Data gate

- Every accepted row maps uniquely to `game_pk` and `player_id`.
- No accepted snapshot occurs at or after first pitch.
- No accepted price pair crosses books or line points.
- Rejected and unmatched rows are counted by reason.
- Primary-line and book-panel selection is deterministic.

Failure means fix the frame before fitting any model.

### Open 2024 validation only when

- Baselines reproduce the existing model-free results within explained differences caused by the
  fixed panel and stricter identity matching.
- Chronology and settlement tests pass.
- The 2023 model and primary betting rule are serialized and frozen.

### Open 2025 final test only when

- The locked model beats the market on both Brier score and log loss in 2024, overall or under a
  more specific gate declared before 2024 is scored.
- Selected-bet calibration is not materially overconfident.
- No profit claim depends on a grading, in-play, panel-depth, or alternate-line defect.

If these fail, stop and record that the Bayesian model did not improve on the market. Do not tune
against 2024 and relabel 2025 as untouched without creating a new, explicitly exploratory lineage.

### Tradeable-edge claim

Do not call the strategy tradeable unless the completely locked validation/test evidence shows:

- Better Brier score and log loss than the market on the relevant subset.
- Positive flat-stake ROI in both locked seasons or a clearly justified predeclared pooled test.
- A date-clustered confidence interval that supports a positive rather than merely unresolved
  result.
- Realized ROI reasonably consistent with posterior expected EV.
- No single day or longshot accounts for the entire profit.

Otherwise classify the result as failed or inconclusive.

## Proposed implementation boundaries

Resolve these names against the refactored repository before creating files. Preserve the
responsibility boundaries even if final paths differ:

- **Normalized frame builder:** raw odds plus MLB identities and outcomes; no model logic.
- **Chronological feature builder:** as-of-game pitcher, opponent, and workload state.
- **Bayesian strikeout model:** fit/update/predict posterior distribution; no sportsbook logic.
- **Backtest engine:** price pairing, line selection, bet decision, settlement, and summaries.
- **CLI scripts:** orchestration only.

Likely post-refactor artifacts:

- `src/betting/strikeout_data.py`
- `src/betting/strikeout_model.py`
- `src/betting/strikeout_backtest.py`
- `scripts/build_strikeout_prop_frame.py`
- `scripts/backtest_strikeout_bayes.py`
- `test_strikeout_data.py`
- `test_strikeout_model.py`
- `test_strikeout_backtest.py`
- generated `data/analysis/strikeout_prop_frame.parquet`
- generated `output/strikeout_bayes/`

Do not add the generated frame or result directories to source control. Register an MLflow model
or update `docs/FINDINGS.md` only after the locked experiment is complete.

## Required tests

### Identity and odds tests

- A normal event maps to the correct game and starting pitcher.
- Doubleheaders map by commence time, not date alone.
- Ambiguous games and names are rejected.
- Over/under pairs never cross books or points.
- Primary-line ties resolve deterministically.
- A post-commence snapshot is rejected.
- The fixed book panel does not change by test season.

### Chronology tests

- The current game's outcome is absent from every emitted feature.
- Same-day earlier games are included only when completed before the snapshot.
- A pitcher's first observed start falls back to the declared league prior.
- Opponent rates use only prior plate appearances.

### Model tests

- Posterior over, push, and under probabilities sum to one.
- Raising the strikeout line cannot increase `P(over)` for the same posterior draws.
- More pitcher history narrows posterior uncertainty in a controlled synthetic example.
- A higher opponent strikeout tendency increases predicted strikeout rate, holding workload fixed.
- Fixed seeds reproduce predictions.
- Posterior simulation meets the declared Monte Carlo error tolerance.

### Settlement tests

- Half-point and integer lines grade correctly.
- Pushes return stake and contribute zero profit.
- Better decimal price increases EV without changing event probability.
- Best-price execution uses only the fixed panel.
- Only one position is selected per pitcher-event.
- Flat-stake ROI equals net profit divided by settled non-void stake.

## Execution phases

### Phase 0: post-refactor entry

- [ ] Confirm the refactor is merged and no parallel task owns the betting/model boundaries.
- [ ] Re-read `AGENTS.md`, `.omp/AGENTS.md`, `README.md`, and the refactored betting modules.
- [ ] Map the proposed boundaries to the final repository layout.
- [ ] Re-run the existing strikeout market-bias analysis and record its output as the baseline.
- [ ] Freeze the implementation plan; do not add compatibility layers for removed paths.

### Phase 1: normalized historical frame

- [ ] Implement exact event, game, and pitcher identity resolution.
- [ ] Pair prices and select the primary line deterministically.
- [ ] Produce book-coverage and rejection reports.
- [ ] Select and record the fixed book panel without outcomes.
- [ ] Write the normalized Parquet frame and data-contract tests.

### Phase 2: chronological baselines

- [ ] Implement leak-free state updates and feature emission.
- [ ] Reproduce market, league, rolling empirical-Bayes, always-over, and always-under results.
- [ ] Verify proper scores and settlement against hand-checked fixtures.

### Phase 3: Bayesian v1 development on 2023

- [ ] Implement workload and strikeout-rate posterior components.
- [ ] Validate posterior predictive calibration and simulation stability.
- [ ] Compare v1 with every required baseline.
- [ ] Select one primary betting rule using 2023 only.
- [ ] Serialize the model specification, priors, feature list, panel, and betting rule.

### Phase 4: locked 2024 validation

- [ ] Run the frozen model and strategy once.
- [ ] Produce overall, selected-subset, side, line, odds-bucket, and monthly reports.
- [ ] Record any implementation defect before fixing and rerunning.
- [ ] Apply the 2025-opening gate without changing the model.

### Phase 5: untouched 2025 test

- [ ] Run only if the 2024 gate passes.
- [ ] Produce the same report with no analytical changes.
- [ ] Run date-clustered bootstrap uncertainty and profit-concentration checks.
- [ ] Classify the result as failed, inconclusive, or tradeable under the predeclared gates.

### Phase 6: closeout

- [ ] Record commands, configuration, checksums, and outputs needed to reproduce the result.
- [ ] Add focused MLflow lineage only if it improves reproducibility.
- [ ] Update `README.md` and `docs/FINDINGS.md` with the final result, including a negative result.
- [ ] Keep real-money deployment disabled unless the tradeable-edge gate passes.

## Deliverables

The completed experiment should leave:

1. A normalized, auditable historical strikeout-offer frame.
2. A deterministic chronological feature and posterior-prediction pipeline.
3. Baseline and Bayesian results on identical offers.
4. A locked 2024 validation report and, only if opened, a locked 2025 test report.
5. Exact flat-stake bet records sufficient to recompute every ROI figure.
6. A concise final verdict that distinguishes predictive accuracy, calibration, execution value,
   and statistical uncertainty.
