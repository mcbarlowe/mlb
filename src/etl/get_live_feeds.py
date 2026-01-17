"""
ETL script for extracting live feed data from MLB Stats API.

This module provides both sync and async methods for downloading game data.
The async version is significantly faster (10x+) due to concurrent downloads.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import aiofiles
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.endpoints.game_feed import GameFeed


async def live_feed_etl_async(
    concurrency_limit: int = 15,
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict[str, int]]:
    """
    Extract live feed data concurrently using async/await.

    This is significantly faster than sequential downloads due to concurrent requests.
    Expected 10x+ speedup for full season downloads.

    Args:
        concurrency_limit: Maximum concurrent requests (default 15)
        skip_existing: Skip games that already have JSON files (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Statistics per season {season: {"success": N, "error": M, "skipped": K}}
    """
    game_file_path = Path("data/raw/schedules/")
    stats = {}

    for file in sorted(game_file_path.glob("*.json")):
        season = file.stem.split("_")[1]

        # Filter by seasons if specified
        if seasons and season not in seasons:
            continue

        live_feeds_path = Path(f"data/raw/livefeeds/{season}")
        live_feeds_path.mkdir(parents=True, exist_ok=True)

        with open(file, "r") as f:
            games = json.load(f)

        # Collect all game PKs from this season
        game_pks = []
        skipped = 0
        for date in games.get("dates", []):
            for game in date.get("games", []):
                game_pk = game["gamePk"]

                # Skip if file exists and skip_existing is True
                if skip_existing:
                    output_file = live_feeds_path / f"{game_pk}.json"
                    if output_file.exists():
                        skipped += 1
                        continue

                game_pks.append(game_pk)

        if not game_pks:
            print(f"Season {season}: All games already downloaded ({skipped} files exist)")
            stats[season] = {"success": 0, "error": 0, "skipped": skipped}
            continue

        print(f"Season {season}: Fetching {len(game_pks)} games ({skipped} already exist)...")

        # Track progress and errors
        success_count = 0
        error_count = 0
        pbar = tqdm(total=len(game_pks), desc=f"Downloading {season}")

        async def save_game(game_pk: int, data: dict) -> None:
            """Save game data to JSON file."""
            nonlocal success_count
            output_file = live_feeds_path / f"{game_pk}.json"
            async with aiofiles.open(output_file, "w") as f:
                await f.write(json.dumps(data, indent=2))
            success_count += 1
            pbar.update(1)

        def on_error(game_pk: int, error: Exception) -> None:
            """Handle fetch errors."""
            nonlocal error_count
            error_count += 1
            pbar.update(1)
            tqdm.write(f"Error fetching game {game_pk}: {error}")

        async with GameFeed(concurrency_limit=concurrency_limit) as game_feed:
            # Create tasks for all games
            async def fetch_and_save(game_pk: int) -> None:
                try:
                    data = await game_feed.get_async(game_pk)
                    await save_game(game_pk, data)
                except Exception as e:
                    on_error(game_pk, e)

            # Run all fetches concurrently
            tasks = [fetch_and_save(game_pk) for game_pk in game_pks]
            await asyncio.gather(*tasks)

        pbar.close()
        print(f"Season {season}: {success_count} succeeded, {error_count} failed")
        stats[season] = {"success": success_count, "error": error_count, "skipped": skipped}

    return stats


def live_feed_etl(
    concurrency_limit: int = 15,
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict[str, int]]:
    """
    Extract live feed data and save to JSON files.

    This function uses async/await internally for concurrent downloads,
    providing 10x+ speedup compared to sequential downloads.

    Args:
        concurrency_limit: Maximum concurrent requests (default 15)
        skip_existing: Skip games that already have JSON files (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Statistics per season {season: {"success": N, "error": M, "skipped": K}}
    """
    return asyncio.run(
        live_feed_etl_async(
            concurrency_limit=concurrency_limit,
            skip_existing=skip_existing,
            seasons=seasons,
        )
    )


async def process_live_feed_data_async(
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict[str, int]]:
    """
    Process raw live feed data and transform to parquet files (async I/O).

    Uses async file I/O for reading JSON files, though transformation is CPU-bound.

    Args:
        skip_existing: Skip games that already have parquet files (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Statistics per season {season: {"success": N, "error": M, "skipped": K}}
    """
    from src.data.game_feed_data import GameFeedData

    game_feed_data = GameFeedData()
    live_feed_raw_path = Path("data/raw/livefeeds/")
    stats = {}

    # Group files by season
    all_files = list(live_feed_raw_path.glob("**/*.json"))
    files_by_season: dict[str, list[Path]] = {}

    for file in all_files:
        season = file.parent.stem
        if seasons and season not in seasons:
            continue
        if season not in files_by_season:
            files_by_season[season] = []
        files_by_season[season].append(file)

    for season, files in sorted(files_by_season.items()):
        processed_path = Path(f"data/processed/livefeeds/{season}")
        processed_path.mkdir(parents=True, exist_ok=True)

        # Filter to files that need processing
        files_to_process = []
        skipped = 0

        for file in files:
            game_pk = int(file.stem)
            output_file = processed_path / f"{game_pk}.parquet"
            if skip_existing and output_file.exists():
                skipped += 1
                continue
            files_to_process.append((file, game_pk, output_file))

        if not files_to_process:
            print(f"Season {season}: All games already processed ({skipped} files exist)")
            stats[season] = {"success": 0, "error": 0, "skipped": skipped}
            continue

        print(f"Season {season}: Processing {len(files_to_process)} games ({skipped} already exist)...")

        success_count = 0
        error_count = 0

        with logging_redirect_tqdm():
            for file, game_pk, output_file in tqdm(
                files_to_process, desc=f"Processing {season}"
            ):
                try:
                    # Read JSON (async would help with I/O but transformation is CPU-bound)
                    async with aiofiles.open(file, "r") as f:
                        content = await f.read()
                    live_feed_json = json.loads(content)

                    # Transform data (CPU-bound)
                    data_df = game_feed_data.transform(live_feed_json, game_pk, season)

                    # Save to parquet
                    game_feed_data.save(data_df, output_file, format="parquet")
                    success_count += 1

                except Exception as e:
                    error_count += 1
                    tqdm.write(f"Error processing {file}: {e}")

        print(f"Season {season}: {success_count} succeeded, {error_count} failed")
        stats[season] = {"success": success_count, "error": error_count, "skipped": skipped}

    return stats


def process_live_feed_data(
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict[str, int]]:
    """
    Process raw live feed data and transform to parquet files.

    Args:
        skip_existing: Skip games that already have parquet files (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Statistics per season {season: {"success": N, "error": M, "skipped": K}}
    """
    return asyncio.run(
        process_live_feed_data_async(
            skip_existing=skip_existing,
            seasons=seasons,
        )
    )


async def full_etl_async(
    concurrency_limit: int = 15,
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Run the complete ETL pipeline: download -> transform.

    Args:
        concurrency_limit: Maximum concurrent API requests (default 15)
        skip_existing: Skip games that already exist (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Combined statistics for download and processing phases
    """
    print("=" * 60)
    print("MLB Live Feed ETL Pipeline")
    print("=" * 60)

    # Phase 1: Download
    print("\n[Phase 1] Downloading game data from MLB API...")
    download_stats = await live_feed_etl_async(
        concurrency_limit=concurrency_limit,
        skip_existing=skip_existing,
        seasons=seasons,
    )

    # Phase 2: Transform
    print("\n[Phase 2] Transforming data to parquet format...")
    process_stats = await process_live_feed_data_async(
        skip_existing=skip_existing,
        seasons=seasons,
    )

    print("\n" + "=" * 60)
    print("ETL Pipeline Complete")
    print("=" * 60)

    return {"download": download_stats, "process": process_stats}


def full_etl(
    concurrency_limit: int = 15,
    skip_existing: bool = True,
    seasons: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Run the complete ETL pipeline: download -> transform.

    Args:
        concurrency_limit: Maximum concurrent API requests (default 15)
        skip_existing: Skip games that already exist (default True)
        seasons: Optional list of seasons to process. If None, processes all.

    Returns:
        dict: Combined statistics for download and processing phases
    """
    return asyncio.run(
        full_etl_async(
            concurrency_limit=concurrency_limit,
            skip_existing=skip_existing,
            seasons=seasons,
        )
    )


if __name__ == "__main__":
    # Run the full ETL pipeline
    full_etl()
