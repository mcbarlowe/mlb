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


def main() -> None:
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

    print("\nDatabase backfill summary")
    print(f"- target: {db_config.describe()}")
    print(f"- discovered raw files: {backfill_summary.discovered_files}")
    print(f"- processed games: {backfill_summary.processed_games}")
    print(f"- skipped already complete: {backfill_summary.skipped_completed}")
    print(f"- failed games: {backfill_summary.failed_games}")


if __name__ == "__main__":
    main()
