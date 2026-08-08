# MLB Data Pipeline

A comprehensive ETL pipeline for extracting, transforming, and loading MLB baseball game data from the MLB Stats API into a local PostgreSQL analytical schema.

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

from src.endpoints.schedule import Schedule

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
python src/etl/get_live_feeds.py
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
from src.database import PostgresConfig, PostgresHandler

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
├── src/
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
from src.endpoints.base_api import BaseAPI

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
uv run ruff check src/database src/etl/get_live_feeds.py src/etl/load_to_database.py src/etl/postgres_backfill.py scripts/backfill_postgres.py examples/database_examples.py test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py verify_database.py

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
- tracking URI: `sqlite:///$(pwd)/mlflow.db`
- experiment: `mlb-model-training`
- train seasons: `2018, 2019, 2021, 2022, 2023`
- validation season: `2024`
- test season: `2025`

The PostgreSQL training source must already contain the required seasons before this command will run.

Use `--low-memory` for historical retrains on memory-constrained machines; it streams season-sized chunks instead of materializing the entire training window at once.

You can override the tracking backend with `MLFLOW_TRACKING_URI` or `--mlflow-tracking-uri`.

The regression tests use `example_json_files/example_live_feed.json`; they do not require a populated `data/` checkout.

### Live Next-Pitch Prediction

Predict the next pitch of in-progress games and publish pitch cards to Twitter:

```bash
# Dry run for today's schedule: waits for first pitch, polls live games,
# saves cards to output/live_cards/<game_pk>/ without tweeting
uv run python scripts/run_live_pipeline.py

# Actually post to Twitter (requires credentials, see below)
uv run python scripts/run_live_pipeline.py --post

# Follow one game only
uv run python scripts/run_live_pipeline.py --game-pk 823514
```

How it works:

1. Reads the MLB schedule for the target date and sleeps until `--lead-minutes` before the earliest first pitch.
2. Polls each game's live feed every `--poll-interval` seconds (via `DailyPipeline.monitor_all_games`), starting each game automatically when it goes live.
3. On every new pitch state, builds the current at-bat sequence plus a pending-pitch row, predicts the next pitch type and location, and renders a pitch card.
4. Posts the card once per at-bat by default (`--post-cadence pitch` posts on every pitch; `--max-posts-per-game` caps volume).

Model defaults are the trained pair in `models/attention_full/run_20260119_124719` (pitch type) and `models/pitch_type_location_20260121_003206` (location); override with `--pitch-type-model` / `--location-model` when newer models finish training.

Twitter posting uses these environment variables: `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET`. Without `--post`, the pipeline runs in dry-run mode and only saves card images.

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
