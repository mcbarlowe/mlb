#!/usr/bin/env python3
"""Download game feeds and ingest them into PostgreSQL.

This is the main daily data pipeline:
1. Download missing schedules and live-feed JSON files
2. Backfill PostgreSQL from raw feeds (with stale-progress reset)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the previous day's games and ingest them into PostgreSQL.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to process in YYYY-MM-DD format. Defaults to yesterday.",
    )
    return parser.parse_args()


def resolve_target_date(date_arg: str | None) -> date:
    if date_arg is None:
        return datetime.now(tz=UTC).date() - timedelta(days=1)
    return datetime.strptime(date_arg, "%Y-%m-%d").replace(tzinfo=UTC).date()


def completed_game_pks(pipeline_summary: dict) -> list[int]:
    pks: list[int] = []
    for item in pipeline_summary.get("completed", []):
        if isinstance(item, dict):
            if item.get("error"):
                continue
            game_pk = item.get("game_pk")
        else:
            game_pk = item
        if game_pk is not None:
            pks.append(int(game_pk))
    return pks


def pipeline_failure_counts(pipeline_summary: dict) -> dict[str, int]:
    completed_errors = sum(
        isinstance(item, dict) and bool(item.get("error"))
        for item in pipeline_summary.get("completed", [])
    )
    fetch_errors = sum(
        isinstance(item, dict) and bool(item.get("error"))
        for item in pipeline_summary.get("other", [])
    )
    unresolved = len(pipeline_summary.get("live", [])) + len(
        pipeline_summary.get("scheduled", [])
    )
    blocking_states = sum(
        isinstance(item, dict)
        and item.get("state") not in {"postponed", "cancelled"}
        and not item.get("error")
        for item in pipeline_summary.get("other", [])
    )
    return {
        "completed_errors": completed_errors,
        "fetch_errors": fetch_errors,
        "unresolved": unresolved,
        "blocking_states": blocking_states,
    }


def main() -> int:
    from src.database import PostgresConfig
    from src.etl.daily_pipeline import run_daily_pipeline
    from src.etl.postgres_backfill import run_postgres_backfill

    args = parse_args()
    target_date = resolve_target_date(args.date)

    print(f"Running daily ETL for {target_date.isoformat()}")
    pipeline_summary = run_daily_pipeline(
        target_date=target_date,
        skip_existing=False,
        poll_live=False,
    )
    failures = pipeline_failure_counts(pipeline_summary)

    db_config = PostgresConfig.from_env()
    completed_pks = completed_game_pks(pipeline_summary)
    backfill_summary = run_postgres_backfill(
        db_config,
        Path("data/raw/livefeeds"),
        force_game_pks=completed_pks,
    )

    print("\nDaily pipeline summary")
    print(f"- date: {target_date.isoformat()}")
    print(f"- games scheduled: {pipeline_summary['total_games']}")
    print(f"- games skipped existing: {len(pipeline_summary.get('skipped', []))}")
    print(f"- games processed: {len(pipeline_summary.get('completed', []))}")
    print(f"- completed game errors: {failures['completed_errors']}")
    print(f"- fetch errors: {failures['fetch_errors']}")
    print(f"- unresolved live/scheduled games: {failures['unresolved']}")
    print(f"- blocking game states: {failures['blocking_states']}")

    print("\nDatabase backfill summary")
    print(f"- target: {db_config.describe()}")
    print(f"- discovered raw files: {backfill_summary.discovered_files}")
    print(f"- processed games: {backfill_summary.processed_games}")
    print(f"- skipped already complete: {backfill_summary.skipped_completed}")
    print(f"- failed games: {backfill_summary.failed_games}")

    if any(failures.values()) or backfill_summary.failed_games > 0:
        print("\nDaily ETL failed")
        for name, count in failures.items():
            if count:
                print(f"- {name.replace('_', ' ')}: {count}")
        if backfill_summary.failed_games > 0:
            print(f"- backfill failed games: {backfill_summary.failed_games}")
        return 1

    print("\nDaily ETL completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
