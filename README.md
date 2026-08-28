# MLB Data and Models

MLB owns baseball data ingestion, normalized PostgreSQL facts and market data, feature preparation, model training, model inference, and simulation. Betting execution is intentionally outside this repository.

## Ownership boundary

MLB owns:

- MLB Stats API ingestion and resumable backfills
- the `mlb` PostgreSQL schema
- normalized historical MLB market datasets
- pitch, outcome, team-strength, prop-probability, and simulation models
- versioned model/result contracts consumed by downstream services

The sibling `betting` repository owns sportsbook and Kalshi access, price selection, Kelly staking, paper ledgers, settlement, ROI and bankroll calculations, alerts, reports, and betting LaunchAgents. MLB code must not write the `betting` schema or import the betting package.

## Setup

```bash
uv sync --group dev
```

PostgreSQL uses `MLB_DB_NAME`, `MLB_DB_USER`, `MLB_DB_PASSWORD`, `MLB_DB_HOST`, `MLB_DB_PORT`, and `MLB_DB_SCHEMA`.

## Daily ETL

```bash
uv run python scripts/run_daily_postgres_etl.py
uv run python scripts/run_daily_postgres_etl.py --date 2026-08-27
```

The command exits nonzero for fetch, processing, unresolved-game, or backfill failures. It performs no betting settlement or reporting.

Historical backfill:

```bash
uv run python scripts/backfill_postgres.py
uv run python scripts/backfill_postgres.py --bulk-historical
```

## Published contracts

Install the additive read-only result views in an existing database:

```bash
uv run python scripts/install_betting_data_contracts.py
```

Model-only prediction producers:

```bash
uv run python scripts/publish_moneyline_predictions.py \
  --date 2026-08-28 --output-json /tmp/moneyline.json

uv run python scripts/publish_prop_predictions.py \
  --date 2026-08-28 \
  --request-json /tmp/prop-request.json \
  --output-json /tmp/prop-predictions.json
```

These producers do not fetch sportsbook prices, select bets, size stakes, alert, or write paper ledgers.

## Verification

```bash
uv run pytest -q
uv run ruff check src scripts test_*.py
uv run basedpyright
```

Detailed ETL, schema, model-training, simulation, and deployment guidance: [`docs/PIPELINE.md`](docs/PIPELINE.md).
