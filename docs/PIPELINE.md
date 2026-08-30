# MLB Data Pipeline

A comprehensive ETL pipeline for extracting, transforming, and loading MLB baseball game data from the MLB Stats API into a local PostgreSQL analytical schema.

> Pipeline, schema, model-production, and operational reference. All sportsbook
> and exchange ingestion, normalized market datasets, pricing, market-vs-model
> research, execution, ledgers, settlement, ROI, alerts, and reports are owned
> by the sibling `betting` repository.

## Features

- **Complete Game Data Extraction**: Fetch schedules and detailed live feed data from MLB Stats API
- **Pitch-Level Analytics**: Extract granular pitch-by-pitch data including PITCHf/x metrics, spin rates, break angles, and velocities
- **Star Schema Database**: Normalized relational database with proper foreign key constraints
- **Boxscore Statistics**: Full batting, pitching, and fielding statistics for every game
- **Dimension Tables**: Teams, venues, players, and reference data properly normalized
- **High Performance**: Processes 21,000+ games with 6M+ pitches in a local PostgreSQL schema
- **Data Quality**: Foreign key constraints ensure referential integrity across all tables

## Database Schema

The pipeline creates a star schema optimized for analytical queries:

### Dimension Tables
- **teams** (45 rows): Team information including league, division, and venue
- **venues** (82 rows): Stadium details with location, capacity, and field dimensions
- **players** (10,001 rows): Player biographical and physical information
- **Reference tables**: Positions, pitch types, event types, game types

### Fact Tables
- **games** (21,643 rows): Game-level facts with FKs to teams and venues
- **pitches** (5,999,516 rows): Pitch-by-pitch data with PITCHf/x metrics, linked to games
- **linescore** (381,170 rows): Inning-by-inning scoring data
- **batting** (184,534 rows): Player batting statistics per game
- **pitching** (38,773 rows): Player pitching statistics per game
- **fielding** (46,103 rows): Player fielding statistics per game

### Key Relationships

Foreign key relationships ensure data integrity:

- **games.away_team_id** → teams.team_id
- **games.home_team_id** → teams.team_id
- **games.venue_id** → venues.venue_id
- **pitches.game_pk** → games.game_pk

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and targets a local PostgreSQL database by default.

```bash
# Clone the repository
git clone <your-repo-url>
cd mlb

# Install runtime and development tooling
uv sync --group dev
```

Default database target:

- database: `postgres`
- schema: `mlb`
- host: local socket / libpq defaults
Override the target with `MLB_DB_NAME`, `MLB_DB_USER`, `MLB_DB_PASSWORD`, `MLB_DB_HOST`, `MLB_DB_PORT`, and `MLB_DB_SCHEMA`.

## Usage

### 1. Extract Game Schedules

First, fetch game schedules for desired seasons:

```python
from datetime import UTC, datetime
from pathlib import Path
import json

from mlb.endpoints.schedule import Schedule

schedule_api = Schedule()
for season in range(2009, datetime.now(tz=UTC).year + 1):
    data = schedule_api.get(season=season, sportId=1)

    output_path = Path(f"data/raw/schedules/schedule_{season}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f)
```

### 2. Extract Live Feed Data

Extract detailed game data for all scheduled games:

```bash
python mlb/etl/get_live_feeds.py
```

This will:
- Read all schedule files from `data/raw/schedules/`
- Extract live feed JSON for each game
- Save to `data/raw/livefeeds/{season}/{game_id}.json`

### 3. Load Data into PostgreSQL

Run the complete ETL pipeline to populate the configured PostgreSQL schema:

```bash
uv run python -m src.etl.load_to_database
```

This process:
1. Creates all database tables in the configured schema
2. Loads reference data (positions, pitch types, etc.)
3. Processes all live feed JSON files
4. Loads dimension data (teams, venues, players)
5. Loads fact data (games, pitches, boxscore stats)
6. Creates indexes for query performance
7. Vacuums and analyzes managed tables

By default the ETL writes to the `mlb` schema in the local `postgres` database.

### 3a. Resumable Backfill

Use the resumable backfill script when you want a long-running load that:
- downloads missing season schedules asynchronously
- downloads missing live feed JSON files asynchronously
- shows `tqdm` progress while files are fetched and while games are backfilled
- records completion state in `backfill_game_progress`
- skips already completed games on the next run
- rewrites a game's fact rows inside a transaction before marking it complete

```bash
uv run python scripts/backfill_postgres.py
```

For the initial historical load, use the bulk historical mode:

```bash
uv run python scripts/backfill_postgres.py --bulk-historical
```

Bulk mode keeps season-level resume state in `bulk_backfill_progress`, but it skips work by checking which `game_pk` values already exist in `mlb.games`: fully loaded seasons are skipped on rerun, and partial seasons continue with only the missing games.

Pass a different live-feed root explicitly if needed:

```bash
uv run python scripts/backfill_postgres.py /path/to/livefeeds
```

### 4. Query the Database

```python
from mlb.database import PostgresConfig, PostgresHandler

db_config = PostgresConfig.from_env()

with PostgresHandler(db_config) as db:
    query = """
    SELECT
        pitch_type,
        COUNT(*) as pitches,
        ROUND(AVG(pitch_start_speed), 1) as avg_speed,
        ROUND(AVG(spin_rate), 0) as avg_spin
    FROM pitches
    WHERE pitch_start_speed IS NOT NULL
    GROUP BY pitch_type
    ORDER BY pitches DESC
    LIMIT 10
    """

    result = db.query(query)
    print(result)
```

### 5. Verify Database Contents

Use the provided verification script:

```bash
uv run python verify_database.py
```

### 6. Backtest Season Projections

Use the season projection backtest to evaluate preseason division and playoff forecasts against final standings. The script writes model, flat-schedule baseline, improvement, calibration, summary CSVs, and optional playoff-probability graphics. The optional `--market-win-totals` input is a caller-supplied model prior; market collection, normalization, and storage remain outside this repository.

```bash
uv run python scripts/backtest_season_projections.py \
  --seasons 2022 2023 2024 2025 \
  --trials 5000 \
  --tune-trials 1000 \
  --out output/season_projection_default_backtest.csv \
  --calibration-out output/season_projection_default_backtest_calibration.csv \
  --summary-out output/season_projection_default_backtest_summary.csv \
  --graphics-out-dir output/season_projection_default_graphics
```

Daily live-season graphics:

```bash
# Refresh pre-cutoff games, render current-season JPEGs, and dry-run the social post
uv run python scripts/run_daily_season_projection.py

# Publish the generated playoff-probability and playoff-stage graphics to X
uv run python scripts/run_daily_season_projection.py --post --post-provider x
```

The daily runner bounds data mutation to current-season regular-season games that are either non-final before `--as-of` or within `--refresh-lookback-days` of it. It overwrites those raw live-feed JSON files, force-refreshes only those `game_pk`s into PostgreSQL, refuses to project while pre-`--as-of` games remain non-final, writes `output/season_projection_<season>/season_<season>_model_*.{csv,jpg}`, and records the posted ID in the output directory so a same-day relaunch does not duplicate the X post.


Optional tuning knobs:
- `--team-prior-scale-grid`: candidate prior-season team-strength offsets.
- `--market-win-totals`: optional external model-prior CSV with `season`, `win_total`, and `team_id`, `abbreviation`, or `team_name`; this repository never fetches or stores the source market data.
- `--market-prior-scale-grid`: candidate scales for that optional preseason prior.
- `--market-prior-min-tune-seasons`: prior seasons with external priors required before tuning nonzero scales.
- `--schedule-strength-scale-grid`: candidate remaining-schedule-strength offsets.
- `--calibrate-playoff-probs`: fit anchored playoff-probability calibration on prior seasons.
- `--graphics-out-dir`: optional directory for per-season playoff-probability and playoff-stage JPEG graphics.


## Project Structure

```
mlb/
├── data/
│   ├── raw/
│   │   ├── schedules/              # Season schedules (JSON)
│   │   └── livefeeds/              # Raw game data (JSON)
│   │       └── {season}/
│   │           └── {game_id}.json
│   ├── processed/
│   │   └── livefeeds/              # Transformed data (Parquet)
│   └── exports/                    # Optional Parquet exports from Postgres tables
├── mlb/
│   ├── endpoints/                  # MLB API endpoint classes
│   │   ├── base_api.py            # Base API interface
│   │   ├── schedule.py            # Schedule endpoint
│   │   ├── live_feed.py           # Live feed endpoint
│   │   └── ...                    # Reference data endpoints
│   ├── data/                       # Data transformation classes
│   │   ├── game_feed_data.py      # Pitch data transformer
│   │   ├── game_data.py           # Game fact transformer
│   │   ├── team_data.py           # Team dimension transformer
│   │   ├── venue_data.py          # Venue dimension transformer
│   │   ├── player_data.py         # Player dimension transformer
│   │   ├── boxscore_data.py       # Boxscore statistics transformer
│   │   └── linescore_data.py      # Linescore transformer
│   ├── database/
│   │   └── postgres_handler.py    # PostgreSQL schema operations
│   └── etl/
│       ├── get_live_feeds.py       # Live feed extraction script
│       ├── load_to_database.py     # One-shot PostgreSQL load script
│       └── postgres_backfill.py    # Resumable PostgreSQL backfill logic
├── scripts/
│   └── backfill_postgres.py        # CLI entry point for resumable backfills
├── examples/                       # Usage examples
├── test_game_dimensions.py         # Star schema relationship tests
├── test_game_pitches_relation.py   # FK constraint tests
├── test_postgres_backfill.py       # Resume/backfill regression test
├── verify_database.py              # Database verification
└── README.md
```

## Data Dictionary

### Pitches Table (Core Analytical Table)

Key columns include:

**Game Context:**
- `game_pk`: Game identifier (FK to games)
- `season`: Season year
- `game_date`: Date of game
- `inning`, `half_inning`: Inning information

**Players:**
- `pitcher_id`, `pitcher_name`, `throw_side`: Pitcher details
- `batter_id`, `batter_name`, `bat_side`: Batter details

**Pitch Metrics:**
- `pitch_type`: Pitch type (FF, SL, CH, etc.)
- `pitch_start_speed`: Release velocity (mph)
- `spin_rate`: Spin rate (rpm)
- `spin_direction`: Spin axis angle
- `break_vertical`, `break_horizontal`: Break measurements (inches)
- `px`, `pz`: Pitch location coordinates
- `pitch_zone`: Strike zone location (1-14)

**PITCHf/x Physics:**
- `x0`, `y0`, `z0`: Initial position (ft)
- `vx0`, `vy0`, `vz0`: Initial velocity (ft/s)
- `ax`, `ay`, `az`: Acceleration (ft/s squared)
- `pfxX`, `pfxZ`: Horizontal/vertical movement

**Outcome:**
- `event`: Play result (single, strikeout, etc.)
- `description`: Pitch call
- `is_strike`, `is_ball`, `is_in_play`: Outcome flags

## Development

### Adding New API Endpoints

All API endpoints inherit from `BaseAPI`:

```python
from mlb.endpoints.base_api import BaseAPI

class MyEndpoint(BaseAPI):
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/my_endpoint"

    def get(self, **params):
        return self._request("GET", params=params)
```

### Adding New Transformers

Data transformers follow a consistent pattern:

```python
import pandas as pd

class MyDataTransformer:
    def __init__(self):
        self.data_types = {
            "field1": int,
            "field2": str,
            # ... define schema
        }

    def transform(self, data: dict) -> pd.DataFrame:
        # Extract and flatten data
        records = self._extract_records(data)
        df = pd.DataFrame(records)
        return df.astype(self.data_types)

    def save_to_db(self, df: pd.DataFrame, db_handler, if_exists: str = "append"):
        db_handler.insert_dataframe(df, "my_table", if_exists=if_exists)
```

### Code Quality

```bash
# Format code
uv run black .

# Lint the PostgreSQL ETL surface
uv run ruff check mlb/database mlb/etl/get_live_feeds.py mlb/etl/load_to_database.py mlb/etl/postgres_backfill.py scripts/backfill_postgres.py examples/database_examples.py test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py verify_database.py

# Type-check the PostgreSQL ETL surface with the local .venv
uv run basedpyright
```

### Running Tests

```bash
# Sample-data regression tests against PostgreSQL
uv run pytest -q test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py

# Verify the configured PostgreSQL schema
uv run python verify_database.py
```

### MLflow Training

Train the documented pitch type and conditioned location models with MLflow tracking from PostgreSQL:

```bash
uv run python scripts/train_models_with_mlflow.py
```

Defaults:

- training data source: `postgres`
- tracking URI: `http://10.0.0.171:5001`
- experiment: `mlb-model-training-shared`
- artifact transport: remote clients upload through the iMac MLflow HTTP server
- artifact storage on iMac: `/Users/matthewbarlowe/mlflow-artifacts/`
- train seasons: `2018, 2019, 2021, 2022, 2023`
- validation season: `2024`
- test season: `2025`

Shared multi-machine setup:

- MLflow metadata lives in the local PostgreSQL `mlflow` schema on the iMac; use the direct Postgres URI only as the iMac server's `--backend-store-uri`.
- The server's SQLAlchemy backend URI uses `postgresql+psycopg2`; MLflow 3.14 binds model-version strings incompatibly with the `psycopg` v3 driver and breaks registered-model detail pages.
- The iMac runs `mlflow server` on `http://10.0.0.171:5001` with artifact serving enabled and `/Users/matthewbarlowe/mlflow-artifacts/` as `--artifacts-destination`.
- The LaunchAgent allows `http://10.0.0.171:5001` as a CORS origin; without it, browser run searches fail even though direct MLflow client queries succeed.
- `mlb-model-training-shared` is the production/import experiment for the live stack (pitch type, location, and outcome models).
- `mlb-model-training` is the legacy direct-database experiment. Keep its historical runs, but do not target it for new multi-machine training.
- The MacBook Pro and MacBook Air should use `MLFLOW_TRACKING_URI=http://10.0.0.171:5001`, not the direct Postgres URI, so artifact uploads go through the iMac server.
- Training entrypoints default to the shared HTTP URI and experiment. If you intentionally want a local SQLite run, pass it explicitly with `--mlflow-tracking-uri sqlite:///...`.
- `scripts/run_outcome_training.sh` remains available as a convenience helper; it resolves `MLFLOW_TRACKING_URI` from the current shell or `~/.zshrc` and then launches `scripts/train_outcome_models.py`.
- To register an existing paired outcome run, use `uv run python scripts/import_outcome_models_to_mlflow.py --run-dir <run-dir> --profiles-dir <profiles-dir> --set-champion`.

Outcome-model versioning and promotion:

```bash
# Train and retain immutable candidate versions.
uv run python scripts/train_outcome_models.py --register-models

# Train, register, and promote the pair when both stages pass.
uv run python scripts/train_outcome_models.py --set-champion
```

- Stage A is registered as `mlb-pitch-result-stage-a`; Stage B is registered as `mlb-in-play-event-stage-b`. Each native CatBoost package includes its model, feature/class contract, metrics, profile stores, input example, signature, and explicit dependencies.
- Both versions share one `outcome_release_id`, source run, contract version, and pinned `sim_inputs_run_id`. Production resolves the two `champion` aliases and rejects partial or incompatible alias updates.
- Promotion requires both stages to beat their conditional validation and test log-loss baselines and to carry finite validation/test Brier, log-loss, and accuracy metrics. Failed candidates remain versioned but do not move either alias.
- The production simulator downloads the immutable registered-model versions into `models/outcome/mlflow_cache/` and loads the exact simulator-input run pinned by that release.

Win-model versioning and comparison:

```bash
uv run python scripts/evaluate_team_strength.py --log-mlflow --set-champion
```

- Every evaluation logs the fitted estimator, exact ordered feature contract, training/holdout datasets, coefficients, rolling-fold evidence, paired date-block bootstrap intervals, promotion result, and dependencies.
- Contract v2 adds recency- and age-adjusted projected-lineup wOBA (scaled per 10 wOBA points) plus individual bullpen FIP and recent-workload availability to the existing Elo, run-form, and starter features. Active rosters filter injuries, transactions, and call-ups; unseen players use league priors.
- Win-probability estimators use the logged-model name `win_probability_model` and `model_collection=win_probability_models`; `model_family=team_strength_win` identifies this implementation. The stable registered-model identifier remains `mlb-team-strength-win`.
- Every candidate creates an immutable registered-model version. Production resolves only the `champion` alias. The loader supports both the five-feature v1 contract and eight-feature v2 contract, so a v2 candidate can be logged without invalidating the current v1 champion.
- `--set-champion` advances the alias only when the candidate's 95% paired date-block lower bounds improve both Brier score and log loss over walk-forward v1 and league-home-rate baselines across at least three seasons, with no material single-season regression. Failed candidates remain versioned and the command exits nonzero.
- Compare runs through `rolling_evaluation.json`, the `rolling_*` MLflow metrics, and immutable registered-model version tags. `latest_logged_version` identifies the newest candidate; `champion_version` identifies production.
- `models/`, `output/`, `catboost_info/`, and local MLflow stores are generated working data and are intentionally ignored by Git. Production outcome artifacts bootstrap from shared MLflow when absent; other local model paths must be produced by training or downloaded explicitly.

The PostgreSQL training source must already contain the required seasons before this command will run.

Use `--low-memory` for historical retrains on memory-constrained machines; it streams season-sized chunks instead of materializing the entire training window at once.
The regression tests use `example_json_files/example_live_feed.json`; they do not require a populated `data/` checkout.

### Live Next-Pitch Prediction

Predict the next pitch of in-progress games and publish pitch cards to Bluesky, X, or both:

```bash
# Dry run for today's schedule: waits for first pitch, polls live games,
# saves cards to output/live_cards/<game_pk>/ without posting
uv run python scripts/run_live_pipeline.py

# Post to X
uv run python scripts/run_live_pipeline.py --post --post-provider x

# Cross-post to Bluesky and X
uv run python scripts/run_live_pipeline.py --post --post-provider both

# Follow one game only
uv run python scripts/run_live_pipeline.py --game-pk 823514
```

How it works:

1. Reads the MLB schedule for the target date and sleeps until `--lead-minutes` before the earliest first pitch.
2. Polls each game's live feed every `--poll-interval` seconds (via `DailyPipeline.monitor_all_games`), starting each game automatically when it goes live.
3. On every new pitch state, builds the current at-bat sequence plus a pending-pitch row, predicts the next pitch type and location, and renders a pitch card.
4. Posts the card once per at-bat by default (`--post-cadence pitch` posts on every pitch; `--max-posts-per-game` caps volume).

Model defaults point to local working copies in `models/attention_full/run_20260119_124719` (pitch type) and `models/pitch_type_location_20260121_003206` (location). These generated files are not version-controlled; train or download the artifacts before a fresh checkout runs this pipeline, and override them with `--pitch-type-model` / `--location-model` when needed.

Bluesky posting uses `BLUESKY_HANDLE` (e.g. `pitchbot.bsky.social`) and `BLUESKY_APP_PASSWORD` (create one at bsky.app -> Settings -> App Passwords; never use the main account password). X posting accepts either OAuth 1.0a user-context credentials (`X_API_KEY`, `X_API_KEY_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`) or OAuth 2.0 user-context credentials (`X_API_CLIENT_ID`, `X_API_CLIENT_SECRET`, plus `X_API_ACCESS_TOKEN` or `X_API_OAUTH2_ACCESS_TOKEN`; add `X_API_REFRESH_TOKEN` or `X_API_OAUTH2_REFRESH_TOKEN` if you want unattended posting to survive token expiry). Without `--post`, the pipeline runs in dry-run mode and only saves card images.

### Daily Game Simulation Board

Generate one morning board image covering every preview game on the slate:

```bash
# Dry run: build the board image and write probable-starter state locally
uv run python scripts/run_daily_sim_slate.py

# Cross-post the board to Bluesky and X, then keep polling preview games for probable-starter changes
uv run python scripts/run_daily_sim_slate.py --post --post-provider both --watch-starters
```

How it works:

1. Resolves the outcome-model artifacts with `--outcome-run-dir auto`: shared MLflow production runs first when `MLFLOW_TRACKING_URI` is set, then `models/outcome/latest_run.txt`, then the newest local `models/outcome/run_*` directory.
2. Loads the gate-passed `champion_version` of the registered MLflow model selected by `--win-model-name` (default `mlb-team-strength-win`), then rebuilds its chronological Elo, run-form, starter, lineup-projection, individual-bullpen, and recent-workload state from PostgreSQL through games before the slate date. Missing or inconsistent champion metadata stops the job rather than silently fitting another model.
3. Fetches that day's preview games from the MLB schedule with hydrated probable starters.
4. Uses the team-strength model for published win odds and the pitch/outcome Monte Carlo chain for score distributions, then renders the combined board to `output/sim_cards/daily/daily_sim_<date>.jpg`.
5. Stores the probable-starter snapshot and published board ID in `output/sim_state/daily_sim_<date>.json`. A same-day restart reuses that post instead of publishing a duplicate; with `--watch-starters`, it resumes polling and posts a fresh one-game card only when a probable starter changes.

The promotion gate uses at least three walk-forward season folds through the held-out season. A candidate must have positive 95% paired date-block bootstrap lower bounds for Brier-score and log-loss improvement over both the v1 champion contract and the fixed league-home-rate baseline, with no material single-season regression:

```bash
uv run python scripts/evaluate_team_strength.py
```

`scripts/run_daily_sim_slate.sh` is the launchd-friendly wrapper; it reads `BARLOWE_DAILY_SIM_*` environment variables for posting, posting provider, polling cadence, state/output directories, optional date overrides, and `BARLOWE_DAILY_SIM_WIN_MODEL`. The wrapper defaults to `BARLOWE_DAILY_SIM_POST_PROVIDER=both` and `mlb-team-strength-win`. The installed `com.barloweanalytics.daily-sim-slate` LaunchAgent runs at 09:00 local time and uses the shared MLflow HTTP service. `scripts/run_daily_random_live_game.sh` uses the same provider mechanism for `BARLOWE_RANDOM_GAME_*`. `scripts/run_daily_season_projection.sh` uses `BARLOWE_SEASON_PROJECTION_*`; the installed `com.barloweanalytics.daily-season-projection` LaunchAgent runs at 08:30 local time and posts the daily season projection to X by default.


## Performance Notes

- **PostgreSQL Performance**: Local relational queries stay fast once indexes are built on key columns
- **ETL Speed**: Processes ~21,000 games in approximately 30-45 minutes
- **Memory Efficient**: Processes games one at a time to minimize memory usage
- **Indexes**: Automatically creates indexes on frequently queried columns (game_pk, player_ids)

## Data Sources

All data is sourced from the official MLB Stats API:
- **Base URL**: `https://statsapi.mlb.com/api/v1/`
- **Documentation**: [MLB Stats API Docs](https://appac.github.io/mlb-data-api-docs/)
- **Rate Limiting**: Be respectful of API rate limits when extracting data

## License

This project is for educational and analytical purposes. MLB data is subject to [MLB's copyright and data usage policies](https://www.mlb.com/official-information/terms-of-use).

## Acknowledgments

- MLB Stats API for providing comprehensive baseball data
- PostgreSQL for providing the local relational storage engine
- The baseball analytics community for inspiring this work

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
