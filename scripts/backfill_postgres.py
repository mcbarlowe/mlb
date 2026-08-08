from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import PostgresConfig
from src.etl.postgres_backfill import (
    BulkBackfillSummary,
    default_seasons,
    sync_and_backfill_postgres,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download missing schedule/live feed JSON files and backfill the configured "
            "PostgreSQL schema."
        ),
    )
    parser.add_argument(
        "raw_data_path",
        nargs="?",
        default="data/raw/livefeeds",
        help="Root directory where live feed JSON files are stored and resumed.",
    )
    parser.add_argument(
        "--bulk-historical",
        action="store_true",
        help=(
            "Use season-batched historical loading. Existing games in mlb.games are skipped, "
            "so reruns continue with only the missing games."
        ),
    )
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    db_config = PostgresConfig.from_env()
    selected_seasons = None
    if args.start_season is not None or args.end_season is not None:
        selected_seasons = default_seasons(
            start_season=args.start_season or 2009,
            end_season=args.end_season,
        )

    summary = sync_and_backfill_postgres(
        db_config,
        Path(args.raw_data_path),
        seasons=selected_seasons,
        bulk_historical=args.bulk_historical,
    )

    live_feed_success = sum(item["success"] for item in summary.live_feed_stats.values())
    live_feed_errors = sum(item["error"] for item in summary.live_feed_stats.values())
    live_feed_skipped = sum(item["skipped"] for item in summary.live_feed_stats.values())

    print("\nPipeline summary")
    print(f"- target: {db_config.describe()}")
    print(f"- schedules downloaded: {summary.schedules.downloaded}")
    print(f"- schedules skipped existing: {summary.schedules.skipped_existing}")
    print(f"- live feeds downloaded: {live_feed_success}")
    print(f"- live feeds skipped existing: {live_feed_skipped}")
    print(f"- live feed download errors: {live_feed_errors}")

    if isinstance(summary.backfill, BulkBackfillSummary):
        print(f"- discovered seasons: {summary.backfill.discovered_seasons}")
        print(f"- processed seasons: {summary.backfill.processed_seasons}")
        print(f"- skipped already complete: {summary.backfill.skipped_completed}")
        print(f"- failed seasons: {summary.backfill.failed_seasons}")
    else:
        print(f"- discovered raw files: {summary.backfill.discovered_files}")
        print(f"- processed games: {summary.backfill.processed_games}")
        print(f"- skipped already complete: {summary.backfill.skipped_completed}")
        print(f"- failed games: {summary.backfill.failed_games}")


if __name__ == "__main__":
    main()
