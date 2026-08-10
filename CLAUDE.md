# CLAUDE.md

Project-specific agent guidance lives in `.omp/AGENTS.md`.
Use `README.md` for architecture and user-facing workflow details.

Verified repo commands:
- `uv sync --group dev`
- `uv run pytest -q test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py`
- `uv run python verify_database.py`
- `uv run ruff check src/database src/etl/get_live_feeds.py src/etl/load_to_database.py src/etl/postgres_backfill.py scripts/backfill_postgres.py examples/database_examples.py test_game_dimensions.py test_game_pitches_relation.py test_postgres_backfill.py verify_database.py`
- `uv run basedpyright`
