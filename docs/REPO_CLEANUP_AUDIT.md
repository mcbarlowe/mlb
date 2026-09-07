# Repository Cleanup Audit

Last applied: 2026-09-07. Supersedes the 2026-08-20 classification report,
which was written against the `src/` layout and the betting code that
`22a32b0`, `0191fe6`, `ed9c2d2`, and `4e6080b` removed.

Purpose: keep durable MLB infrastructure separate from generated artifacts,
one-off exploration, and code that no longer has a caller.

## Current shape

- Tracked files: 290.
- Python: 122 modules under `mlb/`, 54 scripts under `scripts/`, 50 root `test_*.py`.
- Ignored local artifact trees: `data/` ~208 MB, `models/` ~143 MB. `output/`,
  `catboost_info/`, `mlruns/`, and `mlflow.db` are ignored and currently small
  or absent; the historical `output/` tree was archived to iCloud on 2026-09-05.
- `.gitignore` excludes `/data/`, `/models/`, `/output/`, `/catboost_info/`,
  `/mlflow.db`, `/mlruns/`, `/mlartifacts/`, `/artifacts/`, and `*.log`.

## Applied 2026-09-07

| Change | Reason |
|---|---|
| Deleted `scratch.py` | Throwaway schedule-dump script, no references. |
| Deleted `f5_diagnostics.py` | Queried `{mlb_schema}.f5_odds`; that table exists only in the `betting` schema, so the script could not run and its subject is out of this repository's ownership. |
| Deleted `notebooks/pitch_eda_executed.ipynb` | 2.4 MB of executed output for the tracked `notebooks/pitch_eda.ipynb`. |
| Deleted `at_bat_analysis.png`, `sample_pitch_card.png` | Outputs of `scripts/analyze_at_bat.py` and `scripts/create_sample_pitch_card.py`. |
| Deleted root `com.barloweanalytics.daily-season-projection.plist` | Hand-edited copy with hardcoded user paths, superseded by the templated `mlb/deploy/resources/launchd/` plist that `mlb-install-agents` renders. |
| Untracked `logs/*.log` (4.1 MB) | Local run output; `*.log` was already ignored. |
| Moved `cleaning_pitch_data.ipynb` to `notebooks/` | Root clutter; `notebooks/` is already excluded from pytest and ruff. |
| Moved `GUMBOPDF3-29.pdf` to `docs/references/gumbo_documentation.pdf` | It is the MLBAM GUMBO feed specification, which is the format `mlb/data/game_feed_data.py` parses. Reference material, not junk. |
| Removed `psycopg2-binary` | No `psycopg2` import anywhere; the code is `psycopg` v3 throughout. |
| Moved `xgboost` to the `training` extra | `PitchXGBoostModel._require_xgboost` already imports it lazily and raises a clear message when absent, and no deployed agent trains XGBoost. |
| Repointed documented commands at console entry points | `scripts/run_daily_postgres_etl.py`, `run_daily_season_projection.py`, `run_daily_sim_slate.py`, `run_live_pipeline.py`, `backtest_season_projections.py`, `build_pitcher_movement_profiles.py`, and the two `publish_*` scripts became `mlb-*` entry points in `e415bba`; README, `docs/PIPELINE.md`, and `.omp/AGENTS.md` still told readers to run the deleted files. |

Already handled by the betting split, so no longer listed as open: the
evidence scripts under `scripts/test_*.py`, the failed-strategy futures and
Kalshi scripts, and the superseded betting docs. `jupyter` and `seaborn` moved
to optional extras in `e415bba`.

Root `AGENTS.md` and `CLAUDE.md` stay. They are the discovery convention for
coding agents and both are mirrored into Notion; the earlier suggestion to
delete them as "thin stubs" was wrong.

## Still open

### Modules with no inbound reference

Re-verified 2026-09-07 against every tracked `.py`, `.md`, `.toml`, and `.sh`:

- `mlb/endpoints/game_status.py`
- `mlb/endpoints/live_feed.py`
- `mlb/endpoints/logical_events.py`
- `mlb/endpoints/schedule_types.py`
- `mlb/endpoints/sky.py`
- `mlb/endpoints/timestamps.py`
- `mlb/endpoints/venues.py`
- `mlb/endpoints/wind_direction.py`
- `mlb/ml/lstm_predictor.py`
- `mlb/model_evaluation/market_inputs.py`
- `mlb/model_evaluation/moneyline_inputs.py`

`mlb/endpoints/live_feed.py` duplicates `GameFeed` plus `GameFeedData` and is
the clearest deletion. The other seven endpoint wrappers are only worth keeping
if the intent is a complete Stats API client library; decide that once rather
than case by case. `mlb/ml/lstm_predictor.py` duplicates
`mlb/ml/pitch_predictor.py`. All of `mlb/model_evaluation/` is unreachable: its
two modules reproduce leak-free moneyline inputs for historical evaluation and
nothing imports either.

Before deleting any of them, re-run the reference check over tracked files and
run the narrow affected tests; keep batches small.

### Scripts that need `data/processed/livefeeds`

`analyze_at_bat.py`, `create_sample_pitch_card.py`, `evaluate_combined_model.py`,
`example_pitch_prediction.py`, `generate_*_pitch_cards.py`, and
`save_feature_engine_for_lstm.py` default to that tree, which was deleted on
2026-09-05 after every pitch key was proved present in `mlb.pitches`. They are
not broken — `mlb/etl/get_live_feeds.py` regenerates the tree from raw feeds —
but they cannot run against a fresh clone without that step.

### Test layout

50 `test_*.py` files sit at the repository root, which makes real tests, helper
modules, and fixtures visually indistinguishable. Moving them into
`tests/{etl,ml,outcome,sim,live,analysis}/` and switching `testpaths` from `.`
to `tests` is worth doing, but it produces a large diff and should land on its
own after code changes settle.

### Plan documents

`docs/pitcher_strikeout_model_plan.md` describes a `strikeout_model.py` design
that shipped instead as the five-market `mlb/pitcher_props/` package, and
`docs/training_plan.md` references a `scripts/evaluate_model.py` that does not
exist. Both feed the Notion Tasks database, so reconcile them there rather than
deleting.

## Verification used for cleanup batches

- Reference check over tracked files before deleting any module.
- `uv run pytest -q` for the affected subset, then the full suite.
- `uv sync --group dev` plus an import check after any dependency change.
- `uv run ruff check .` and `uv run basedpyright`.
