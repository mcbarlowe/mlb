# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLB data pipeline that extracts, transforms, and loads (ETL) baseball game data from the MLB Stats API. The project focuses on collecting detailed pitch-level data from MLB games and transforming it into structured datasets for analysis.

## Development Commands

### Package Management
- **Install dependencies**: `uv sync` (project uses uv for dependency management)
- **Add a dependency**: `uv add <package-name>`
- **Add a dev dependency**: `uv add --group dev <package-name>`

### Code Formatting
- **Format code**: `black .` or `black <file>`

### Running Scripts
- **Run Python files**: `python <script.py>` (ensure virtual environment is activated)
- **Run Jupyter notebooks**: `jupyter notebook` or `jupyter lab`

### Data Processing
- **Extract live feeds**: `python src/etl/get_live_feeds.py`
- **Process live feed data**: Runs transformation on raw live feed data to parquet format

## Architecture

### Data Flow Pipeline

1. **Schedule Extraction** → 2. **Live Feed Extraction** → 3. **Data Transformation** → 4. **Parquet Storage**

### Directory Structure

```
data/
├── raw/
│   ├── schedules/          # Schedule JSONs by season (2018-2025)
│   └── livefeeds/          # Raw game data organized by season
│       └── {season}/
│           └── {game_id}.json
└── processed/
    └── livefeeds/          # Transformed parquet files
        └── {season}/
            └── {game_id}.parquet
```

### API Endpoint Classes

All API endpoint classes inherit from `BaseAPI` (src/endpoints/base_api.py) which defines the contract:
- `get()` - Handle GET requests
- `post()` - Handle POST requests (not implemented)

**Implemented Endpoints:**
- `Schedule` (src/endpoints/schedule.py) - Fetches game schedules
  - Base URL: `https://statsapi.mlb.com/api/v1/schedule/games`
  - Parameters: `sportId`, `season`

- `ScheduleTypes` (src/endpoints/schedule_types.py) - Fetches available schedule types
  - Base URL: `https://statsapi.mlb.com/api/v1/scheduleTypes`

- `LiveFeed` (src/endpoints/live_feed.py) - Fetches and transforms live game data
  - Base URL: `https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live`
  - Uses gzip encoding for efficiency
  - Implements full ETL pipeline with extract/transform methods

### ETL Pattern in LiveFeed

The `LiveFeed` class implements a complete ETL pattern:

1. **Extract** (`extract(game_id)`) - Fetches raw JSON from MLB API
2. **Transform** (`transform(data, game_id, season)`) - Processes JSON into structured DataFrame
   - `_process_plays_data()` - Iterates through all plays
   - `_process_play_data()` - Extracts at-bat level information
   - `_process_pitch_data()` - Extracts detailed pitch metrics
3. **Load** (`load()`) - Placeholder for data persistence

### Data Transformation Logic

The transformation pipeline flattens nested MLB API JSON into pitch-level rows containing:

**Play/At-Bat Context:**
- Event outcome, scores, inning, outs
- Batter/pitcher IDs, names, handedness
- Base runner states

**Pitch Details:**
- Pitch type, count, outcome
- Start/end speed, spin rate, spin direction
- Strike zone coordinates (px, pz, zone)
- Break metrics (angle, length, vertical/horizontal)
- PITCHf/x coordinates (x, y, z positions and velocities)

**Key Transformation Notes:**
- Handles missing runner data by adding default False values
- Enforces strict data types via predefined `data_types` dict (src/endpoints/live_feed.py:171-234)
- Merges play-level and pitch-level data for each pitch row

## Key Implementation Details

### Error Handling
- All API endpoints wrap requests with try/except for `HTTPError` and `RequestException`
- ETL scripts use try/except blocks with tqdm.write() for error logging during batch processing

### Data Processing Pipeline (src/etl/get_live_feeds.py)

**Extraction Flow:**
1. Reads schedule JSON files from `data/raw/schedules/`
2. Extracts season from filename pattern `schedule_{season}.json`
3. Iterates through each game with progress bar (tqdm)
4. Saves raw live feed JSON to `data/raw/livefeeds/{season}/{game_id}.json`

**Processing Flow:**
1. Reads all raw live feed JSONs recursively
2. Transforms each using `LiveFeed.transform()`
3. Saves as parquet to `data/processed/livefeeds/{season}/{game_id}.parquet`
4. Logs errors without stopping batch process

### Dependencies
- **Data manipulation**: pandas, polars, pyarrow
- **API calls**: requests (with gzip compression support)
- **Progress tracking**: tqdm with logging integration
- **Visualization**: seaborn (for notebooks)
- **Notebooks**: jupyter

## Development Notes

### When Adding New API Endpoints:
1. Create class inheriting from `BaseAPI`
2. Override `__init__()` to set `self.base_url`
3. Override `get()` and/or `post()` methods
4. Implement proper error handling for HTTP requests
5. Return JSON response from successful requests

### When Modifying Data Schema:
- Update the `data_types` dict in `LiveFeed.transform()` (src/endpoints/live_feed.py:171-234)
- Ensure all new fields are extracted in `_process_pitch_data()` or `_process_play_data()`
- Handle missing data gracefully (see runner data handling pattern)

### Data Pipeline Execution Order:
1. Run schedule extraction first to populate `data/raw/schedules/`
2. Use `live_feed_etl()` to extract raw live feeds
3. Use `process_live_feed_data()` to transform to parquet
