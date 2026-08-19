# Findings: is there a tradeable edge?

Consolidated state after auditing every market and model in this repository. Supersedes
`docs/EXECUTIVE_SUMMARY.md`, which carried a deployment plan built on numbers that did not
survive review.

**Short answer: no deployable edge. One verified structural edge, zero prediction edge, and
proof that exploitable inefficiency exists but is not reachable with the data we hold.**

---

## Measured results, every market

| Market | Model | Bets | ROI | 95% CI | Note |
|---|---|---:|---:|---|---|
| Full-game moneyline | team-strength, walk-forward | 2,458 | **+0.18%** | -4.29 to +4.94 | best-of-5 execution |
| Full-game moneyline | same, consensus execution | 2,458 | -1.92% | | shopping is the difference |
| Championship futures | season simulation | 6 | -1.08% | | only market never defective |
| Division futures | season simulation | 100 | **-3.37%** | | was reported +71% |
| Make-playoffs futures | season simulation | 95 | **-12.18%** | | was reported +9% |
| F5 totals | market bias only | 1,797 | not tradeable | | 1.67% lean vs 3.60% hold |
| Pennant futures | — | — | unmeasurable | | model field sums to 4.0, needs 2.0 |

Everything lands between roughly -12% and +0.2%. The one market that was never defective,
championship futures, returned -1.08%, which is what the others converged to once corrected.

---

## 1. Verified edge: execution

Line shopping is worth **+2.1 percentage points**, measured at +1.9 to +2.3 in every individual
season, and it is the entire difference between -1.92% and +0.18%. It converts a 4.1% per-book
hold into roughly 1.4%. Gains saturate at five accounts; a sixth adds +0.06pp.

This is structural, not predictive. It requires accounts, not insight, and it is the only
positive finding here that survived every check.

**It is also an upper bound.** Best-of-five is a price-feed maximum, not a filled bet: no limit
modelling, no movement between observation and placement, no account restriction, and the best
price usually carries the smallest limit. Real execution sits between the consensus and best
columns, so true ROI is below +0.18%.

## 2. Zero prediction edge

Five independent measurements agree:

| Test | Result |
|---|---|
| Blend coefficient, model over true close | -0.006 +/- 0.132 |
| Incremental Brier, model on top of market | +0.00000 |
| Incremental Brier, market on top of model | +0.00170 |
| 13 outcome features, residual screen | none survive, at 3-4x required power |
| Threshold sweep, lambda shrinkage, movement selection | zero or negative throughout |

The containment is one-directional: our model's information set is a strict subset of the
price's. Consequently **no transformation of the model can create edge** — shrinkage,
thresholding, reweighting, and sizing only reshuffle which zero-edge bets get placed. The
lambda experiment demonstrated this concretely: out-of-sample gain was +0.24% against a
hindsight +6.65%, and the mechanism turned out to be threshold-raising in disguise.

The model is not badly built. It is well calibrated unconditionally, slope 0.975 +/- 0.051 with
reliability 0.0002, and captures 83% of the market's resolution. Every bit of that 83% is
already in the price.

## 3. Inefficiency exists but is out of reach

An oracle with perfect foresight of the closing line, betting best opening prices, returns
**+6.14% (95% CI +1.14 to +10.94)** restricted to games whose line moves at least three points,
and +11.45% at five points. Both intervals exclude zero. The market is therefore **not**
perfectly efficient: its opening price is beatable, and the money is real.

It is unreachable with what we observe:

| | Logits |
|---|---:|
| Actual move, sd | 0.0946 |
| Our predicted move, sd | 0.0206 |
| Required to capture the value | 0.1200 |

The threshold is 5.8 standard deviations of our prediction, and the largest predicted move
across 6,988 games is 0.1041 — it never once reaches it. Trading the prediction loses at every
threshold and loses **more** as confidence rises, from -1.49% at no threshold to -14.20% at 0.06,
which is the signature of an absent signal rather than a weak one.

Movement is genuinely predictable: four features clear a Bonferroni-corrected screen, with
`model_disagree` at t = +18.86. But that feature supplies 0.0357 of the total 0.041 R-squared, so
predicting movement is 89% betting model disagreement with extra steps. The three genuine
market-structure features contribute 0.0055 combined. Reaching the required magnitude needs
R-squared near 0.6, a 15x improvement, from data we do not have: lineup announcement timestamps,
injury feeds, betting-percentage splits, per-book move logs.

**t = 18.86 and still unprofitable** is the compact statement of this project. Statistical
significance and economic tradeability are different bars.

---

## Why this outcome was likely

The price is not a rival model; it is the capital-weighted aggregation of every participant's
information. Our inputs are public Statcast, Elo, FIP, wOBA, bullpen usage, lineups, weather,
and park factors, all of which every professional syndicate also holds. Any edge findable in
public data on the most liquid baseball market is, by construction, an edge that survived
everyone else looking for it.

This also explains a result that otherwise looks paradoxical. There is real headroom in our
model, resolution 0.0079 against the market's 0.0095, but it is **headroom to converge, not to
beat**. Closing it earns exactly zero because the price already contains it. Shrinking the model
toward the market improves its Brier monotonically while bet volume falls to zero; at the point
of maximum accuracy the model is the market and places no bets at all.

---

## Defects found and corrected

Every previously reported positive result traced to a defect. Recording them because the failure
modes recur.

| Defect | Effect | Status |
|---|---|---|
| Historical odds included in-play snapshots | "closing lines" were post-first-pitch for 74-78% of games; inflated bets 3x and ROI to -14.2% | fixed, strictly pre-game filter |
| Backtests run without `--walkforward-train` | scored the champion model in-sample; 2024 moved +0.43% to -4.09% | fixed |
| Futures de-vig normalised every market to one winner | division edges inflated 6x, playoffs 12x, miss-playoffs 18x | fixed via `MARKET_SLOTS` |
| Futures outcomes hardcoded | 2022 NL Central and 2023 AL East/Central wrong; a fabricated Seattle win was a season's largest winner | fixed, derived from game results |
| `league_championship_prob` sums to 4.0 | pennant edges doubled | **open** |
| `teams.division_name` places Houston in NL Central | division groupings wrong | worked around in code, **still wrong in database** |
| Cross-book pooling of totals at differing points | fabricated a 0.72% hold against a real 4.05% | fixed |
| Claims made from cherry-picked season ranges | division 2022-2025 gives +37.66%, all twelve seasons give -3.37% | both reported |

Three claims in earlier revisions of these documents were wrong and are retracted: that
historical books charged 11-17% vig, that older-season models were miscalibrated, and that
positive CLV demonstrated skill. The last was then over-corrected to "pure artifact," which was
also wrong: the model does genuinely anticipate line movement at z = +18.70. It simply does not
pay, because the close is only 0.0003 Brier better than the open on average.

---

## Market survey: 34 MLB markets ranked by cost


`scripts/survey_mlb_market_holds.py`. Hold needs only prices, no outcomes, so it is the cheapest
possible screen and it sets the bar any model must clear. Breakeven on a two-way market is half
the shopped hold.

| Market | Shopped hold | Books | Breakeven | Status |
|---|---:|---:|---:|---|
| h2h | 2.13% | 11 | 1.07% | tested, no edge |
| totals (full game) | 2.48% | 10 | **1.24%** | market tested, no bias; model run in progress |
| spreads (run line) | 2.41% | 10 | 1.20% | untested |
| h2h_1st_5_innings | 2.80% | 9 | 1.40% | untested |
| pitcher_strikeouts | 3.49% | 7 | 1.74% | tested, no bias on 8,785 starts |
| spreads_1st_5_innings | 3.63% | 5 | 1.81% | untested |
| alternate_totals | 3.66% | 5 | 1.83% | untested, ~92 rungs per game |
| totals_1st_1_innings (NRFI) | 3.69% | 4 | 1.85% | untested |
| team_totals | 4.23% | 5 | 2.11% | untested |
| 7 batter prop markets | 5.0-7.0% | 2-5 | 2.5-3.5% | 2 tested, no bias |
| 6 markets, one book only | no shopping | 1 | untradeable | — |

**The structural finding: hold rises monotonically with how exotic the market is.** The cheapest
markets to trade are the most liquid and most sharply priced ones; markets with less professional
attention charge more for access, up to 7% shopped, and six have a single book quoting them so no
shopping is possible at all. There is no market here that is both cheap to trade and inattentively
priced. That is the market-making equilibrium working against us from both directions.

Books are also internally consistent: the game total minus the sum of the two team totals has a
median of exactly 0.00 for all five books tested, so there is no easy model-free relative-value
edge between related markets. A tail of 7.7% of book-games showed at least a full run of slack on
n=65, which is the one cross-market check worth running at scale since it needs no baseball model.

One correction worth recording. The first survey reported `alternate_spreads` at a **negative**
24% hold, which is impossible. Cause was mine: sides were keyed by position in a dict, so a book
listing the away team first had its away price paired against another book's home price. FanDuel
also quotes the same team at both -1.5 and +1.5, which collided. Fixed by keying sides canonically
(over/under, or home/away against the event's home team) and normalising handicaps to the home
team's signed point. `alternate_spreads` then reads 5.24%.

## Full-game totals: market closed, model closed

The odds were **already in the database** and I nearly spent 73,000 credits re-pulling them.
`mlb.odds_totals` holds 79,247 rows over 2024 and 2025, per book, with open and close.

Market side, `scripts/test_totals_market_bias.py`, 4,631 graded non-push games at the close:

| Season | n | Implied | Actual | Bias | z |
|---|---:|---:|---:|---:|---:|
| 2024 | 2,320 | 50.02% | 50.04% | +0.03% | +0.03 |
| 2025 | 2,311 | 50.04% | 48.20% | -1.84% | -1.77 |
| pooled | 4,631 | 50.03% | 49.13% | -0.91% | -1.23 |

No tradeable bias, and the same signature as strikeout props: one season looks interesting, the
other is flat, pooled is noise. Model-free strategies at best shopped price return -4.16% always
over and -0.61% always under, the latter straddling zero. That closes the market side, making
totals the fifth market where a model-free attack is ruled out.

The **model** side is different, and it produced the first positive signal in this project.
`scripts/sim_totals_eval.py` already existed with a full comparison harness, having only ever been
smoke-tested on one game with three simulations. Run properly on 60 games at 200 sims:

| | Brier | Log loss |
|---|---:|---:|
| sim | **0.2390** | **0.6705** |
| market | 0.2488 | 0.6908 |

Brier gap -0.0098, 95% CI -0.0256 to +0.0048. **Not significant**, z about -1.26 on 60 games, and
the accompanying +15.6% flat-bet ROI on 38 bets is meaningless. But it is the first time any model
here has come out ahead of a price at all: moneyline was 0.2411 against 0.2394 with the model
worse, and F5 totals was 0.2511 against 0.2511.

There is a structural reason totals could genuinely differ. The market pins P(over) near 50% by
choosing where to set the line, so its implied probability carries almost no resolution, spanning
only 46-53% in the sample. The simulation produces a real distribution, 0.41 to 0.67. On the
moneyline the price carried substantial resolution and the model added nothing; here there is room
that does not exist elsewhere.

Do not misread the diagnostic line "mean market 8.49, mean actual 9.20" as a 0.71-run market
error. Run distributions are right-skewed, so the line sits near the median while the mean runs
higher. This is the same artifact already recorded for first-five totals, where the mean residual
carried z = +5.5 while the under still hit more often.

### The signal reversed at scale

The completed 600-game run, `run_id=totals_eval_600g_200s`, 574 non-push games:

| | Brier | Log loss |
|---|---:|---:|
| sim | 0.2530 | 0.6993 |
| market | **0.2498** | **0.6928** |

Brier gap **+0.0032**, 95% CI -0.0023 to +0.0087, z +1.12. Positive means the simulation is
**worse** than the market. Flat-bet at edge above 0.03: 372 bets, 51.6% wins, **ROI -1.5%**.

The reversal occurred inside a single run:

| Sample | n | Brier gap |
|---|---:|---:|
| pilot, seed 11 | 60 | -0.0098 |
| this run, interim | 130 | -0.0079 |
| this run, complete | 574 | **+0.0032** |

The interim and final figures come from the same run and the same seed, so the first 130 games
favoured the simulation and the remaining 444 more than reversed it. The per-side symmetry that
had looked most encouraging also vanished: at n=130 both over and under beat their blind baselines
by about 11.5 points, and at n=574 they carry opposite signs at every threshold above 0.02, which
is the signature of noise rather than skill.

**The leak check is moot.** The run omitted `--pa-calibration`, so the PA calibration may have been
in-sample, which would flatter the simulation. It came out worse than the market regardless, so a
leak-free run can only be worse. The same applies to a 2024 holdout: there is no positive result
left to validate. No further compute is warranted.

Totals was the best-positioned candidate in this project: cheapest untested market at 1.24%
breakeven, a genuinely different question from win probability, purpose-built pitch-level
machinery, odds already loaded, harness already written, and a structural reason to expect room
since the market's P(over) is pinned at 50% plus or minus 2%. It still failed. That is six
model-market pairs tested and six failures.

The process is the finding. A run stopped at 60 games would have reported a -0.0098 Brier edge
with a +15.6% flat-bet ROI and staked money on it.

## Win totals at real prices (added 2026-08-19)

The earlier win-totals test assumed -110 both ways because no prices were stored. Covers
publishes the real preseason per-side prices (one BetMGM snapshot, late March), now scraped for
2013-2026 into `resources/season_win_totals_odds.csv` (`scripts/scrape_covers_win_totals.py`;
outcomes still derived from our own game data, Covers' published wins used only as an integrity
cross-check: 300 checked, 2 mismatches). Real two-way hold is 4.80%, worse than the assumed 4.54%.

Model-free at real prices, 389 team-seasons over 13 full seasons (2020 excluded):

| Strategy | Bets | Win% | ROI | 95% CI |
|---|---:|---:|---:|---|
| always over | 389 | 45.5% | **-13.93%** | [-23.49%, -4.46%] excludes zero |
| always under | 389 | 54.5% | +3.87% | [-5.69%, +13.45%] |

Win-total overs are systematically shaded: blind-over is a genuinely losing strategy at real
prices. Blind-under is positive but its interval straddles zero — thirteen seasons of ~30 bets
is still not enough to promote the under-lean to an edge.

Model strategy at real prices (120 bets, 2022-2025): +8.32% [-8.79%, +25.00%], per-season
-4.4/+0.7/+1.2/+35.7 — every threshold's CI includes zero, 2025 carries the result, and the
accuracy precondition still fails (model MAE 7.814 vs line 7.775, paired diff +0.04 wins,
CI straddles zero). The model also picks under 58% of the time, so part of its positive point
estimate is the market-wide under-lean rather than team-specific skill. Verdict unchanged:
no deployable edge, now established at real prices. 2026 lines are captured preseason, so the
2026 row settles at real prices in October.

## What remains untested

| Item | Cost | Prior |
|---|---|---|
| F5 simulation against F5 totals, full season | ~13 hours compute | low; 40-game pilot showed Brier parity at 0.2511/0.2511 |
| Run lines and spreads | `mlb.f5_odds` has `home_spread`, `home_spread_ml` | untested |
| NRFI/YRFI | table holds 5 rows, needs a backfill | untested |
| Market-dynamics data | a purchase, not a model | the only route with a plausible path to real edge |
| Pennant futures | fix `league_championship_prob` first | low, by analogy |

Sample size is a hard limit on the futures side regardless. Corrected, division futures generate
roughly eight bets per season, so twelve seasons is 100 bets with per-season ROI ranging from
-100% to +154%. The standard error on a twelve-season mean is about 25 points. No futures
conclusion from this data is statistically strong in either direction, including this one.

---

## What the project actually produced

The infrastructure is sound and the measurement discipline is now the asset: strictly pre-game
filtering, enforced walk-forward, slot-aware de-vig, outcomes derived rather than asserted,
residual screens with multiplicity control, permutation tests for seasonal patterns, oracle
ceilings to bound a strategy before building it, and bootstrap intervals on every ROI.

None of the earlier positive results could have survived those checks. That is the deliverable:
a $10,000 deployment was planned against a claimed +71% on a market that returns -3.37%.
