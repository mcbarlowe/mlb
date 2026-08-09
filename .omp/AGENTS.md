# OMP repo notes

- Package manager: `uv`. Bootstrap with `uv sync --group dev`.
- Default database target: local PostgreSQL `dbname=postgres schema=mlb`. Override with `MLB_DB_NAME`, `MLB_DB_USER`, `MLB_DB_PASSWORD`, `MLB_DB_HOST`, `MLB_DB_PORT`, and `MLB_DB_SCHEMA`.
- Fast verification commands:
  - `uv run pytest -q test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py`
  - `uv run python verify_database.py`
  - `uv run ruff check src/database src/etl/get_live_feeds.py src/etl/load_to_database.py src/etl/postgres_backfill.py scripts/backfill_postgres.py examples/database_examples.py test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py verify_database.py`
  - `uv run basedpyright`
- `pyrightconfig.json` intentionally scopes type-checking to the PostgreSQL ETL surface so OMP diagnostics stay actionable.
- Resumable backfill command: `uv run python scripts/backfill_postgres.py` (downloads missing schedules/live feeds asynchronously, then resumes the Postgres backfill)
- Bulk historical mode: `uv run python scripts/backfill_postgres.py --bulk-historical` (batches inserts by season and skips any games already present in `mlb.games`; a season is skipped entirely once all of its game files are already loaded)
- MLflow training command: `uv run python scripts/train_models_with_mlflow.py` (uses PostgreSQL as the training data source by default and logs to a local SQLite MLflow backend at `mlflow.db`; override the tracking backend with `MLFLOW_TRACKING_URI` or `--mlflow-tracking-uri`)
- Outcome model training: `uv run python scripts/train_outcome_models.py` (Stage A pitch-result + Stage B in-play-event CatBoost models from PostgreSQL; `--quick` trains a 1-season smoke in ~4 minutes; logs to the same MLflow backend)
- Sample-data regression tests use `example_json_files/example_live_feed.json`; they do not require a populated `data/` tree.
- Treat `data/`, `models/`, `output/`, `catboost_info/`, and executed notebooks as local/generated artifacts unless the task explicitly targets them.
- ETL/database code lives in `src/endpoints`, `src/data`, `src/database`, and `src/etl`.
- ML training code lives in `src/ml` and `scripts/`; it depends on `torch` and `catboost` and is separate from the Postgres ETL path.
- Pitch outcome models live in `src/outcome` (Stage A/B CatBoost + leak-free dataset builder); deliberately independent of `src/ml` pitch prediction models — shared contract is pitch-type codes + plate coordinates only. Tests: `uv run pytest -q test_outcome_models.py`.
