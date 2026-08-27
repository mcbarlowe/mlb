"""
Example usage of the Daily Pipeline for MLB game data.

This script demonstrates various ways to use the DailyPipeline class
for fetching game data by date and handling live game polling.
"""

import asyncio
from datetime import date

from src.etl.daily_pipeline import (
    DailyPipeline,
    GameState,
    monitor_full_day,
    run_daily_pipeline,
    run_date_range,
)


def example_1_basic_usage():
    """
    Basic usage: Process all games for a specific date.

    This is the simplest way to use the pipeline. It will:
    - Fetch all games scheduled for the date
    - Process completed games immediately
    - Poll live games until they complete (if any)
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage - Single Date")
    print("=" * 60)

    # Process a specific date (historical)
    # This will NOT poll since games are already completed
    results = run_daily_pipeline(
        target_date=date(2023, 6, 5),
        skip_existing=False,  # Re-download even if files exist
    )

    print(f"\nProcessed {len(results.get('completed', []))} completed games")
    print(f"Skipped {len(results.get('skipped', []))} existing games")


def example_2_today_with_polling():
    """
    Process today's games with live polling.

    NOTE: This uses run_daily_pipeline which only polls games that are LIVE
    at the moment you start. Games scheduled for later won't be picked up.

    For full-day monitoring that catches all games, use example_2b instead.
    """
    print("\n" + "=" * 60)
    print("Example 2: Today's Games with Live Polling (snapshot)")
    print("=" * 60)

    results = run_daily_pipeline(
        target_date=date.today(),
        poll_interval=30.0,  # Poll every 30 seconds
        poll_live=True,      # Enable polling for live games
    )

    print(f"\nTotal games: {results['total_games']}")
    print(f"Completed: {len(results.get('completed', []))}")
    print(f"Live (now completed): {len(results.get('live', {}))}")
    print(f"Still scheduled: {len(results.get('scheduled', []))}")


def example_2b_full_day_monitoring():
    """
    Monitor ALL games for the entire day.

    This is the RECOMMENDED approach for live day monitoring.
    Unlike example_2, this will:
    - Poll currently live games
    - Monitor scheduled games and automatically start polling when they begin
    - Continue until ALL games for the day have completed

    Example scenario:
    - You start at 10am
    - 1pm game starts -> automatically picks it up and starts polling
    - 1pm game ends -> processes and saves
    - 4pm game starts -> automatically picks it up and starts polling
    - 7pm games start -> picks them all up
    - Continues until the last game of the day finishes
    """
    print("\n" + "=" * 60)
    print("Example 2b: Full Day Monitoring (recommended)")
    print("=" * 60)

    results = monitor_full_day(
        target_date=date.today(),
        poll_interval=30.0,  # Poll every 30 seconds
    )

    print(f"\nTotal games: {results['total_games']}")
    print(f"Processed: {len(results.get('processed', {}))}")
    print(f"Skipped (existing): {len(results.get('skipped', []))}")


def example_3_date_range():
    """
    Process multiple dates for historical data backfill.

    This is useful for catching up on games you missed or
    backfilling historical data.
    """
    print("\n" + "=" * 60)
    print("Example 3: Date Range Processing")
    print("=" * 60)

    # Process a week of games
    start = date(2023, 7, 1)
    end = date(2023, 7, 7)

    results = run_date_range(
        start_date=start,
        end_date=end,
        skip_existing=True,  # Skip games we already have
    )

    total_games = sum(r.get("total_games", 0) for r in results.values())
    print(f"\nProcessed {len(results)} dates")
    print(f"Total games: {total_games}")


async def example_4_custom_callbacks():
    """
    Using callbacks to react to game events in real-time.

    This is useful for building real-time dashboards or
    triggering notifications when games complete.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Callbacks")
    print("=" * 60)

    def on_game_complete(game_pk: int, data: dict):
        """Called when a game completes."""
        home = data.get("gameData", {}).get("teams", {}).get("home", {}).get("name", "Unknown")
        away = data.get("gameData", {}).get("teams", {}).get("away", {}).get("name", "Unknown")
        linescore = data.get("liveData", {}).get("linescore", {})
        home_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)

        print(f"\n🏆 GAME COMPLETE: {away} {away_runs} @ {home} {home_runs}")

    def on_game_update(game_pk: int, data: dict, state: GameState):
        """Called on each poll update."""
        if state == GameState.LIVE:
            linescore = data.get("liveData", {}).get("linescore", {})
            inning = linescore.get("currentInning", "?")
            inning_state = linescore.get("inningState", "")
            print(f"  Game {game_pk}: {inning_state} {inning}")

    async with DailyPipeline(
        poll_interval=30.0,
        on_game_complete=on_game_complete,
        on_game_update=on_game_update,
    ) as pipeline:
        results = await pipeline.run(
            target_date=date.today(),
            poll_live=True,
        )

    return results


async def example_5_manual_polling_control():
    """
    Manual control over the polling process.

    Use a stop_event to cancel polling programmatically,
    for example after a timeout or user interrupt.
    """
    print("\n" + "=" * 60)
    print("Example 5: Manual Polling Control")
    print("=" * 60)

    # Create a stop event
    stop_event = asyncio.Event()

    # Set a timeout to stop polling after 5 minutes
    async def stop_after_timeout():
        await asyncio.sleep(300)  # 5 minutes
        print("\nTimeout reached - stopping poll...")
        stop_event.set()

    async with DailyPipeline(poll_interval=30.0) as pipeline:
        # Start timeout task
        timeout_task = asyncio.create_task(stop_after_timeout())

        try:
            results = await pipeline.run(
                target_date=date.today(),
                poll_live=True,
                stop_event=stop_event,
            )
        finally:
            timeout_task.cancel()

    return results


async def example_6_fetch_without_polling():
    """
    Fetch game states without polling.

    Useful for checking what games are happening without
    committing to waiting for them to complete.
    """
    print("\n" + "=" * 60)
    print("Example 6: Fetch Without Polling")
    print("=" * 60)

    async with DailyPipeline() as pipeline:
        results = await pipeline.run(
            target_date=date.today(),
            poll_live=False,  # Don't wait for live games
        )

    print(f"\nGames today: {results['total_games']}")
    print(f"Currently live: {len(results.get('live', {}))}")
    print(f"Scheduled for later: {len(results.get('scheduled', []))}")

    # The live games dict contains their current state
    # You could process these later or start polling separately


def example_7_skip_existing():
    """
    Demonstrate skip_existing behavior.

    The pipeline can skip games that have already been
    downloaded to avoid redundant API calls.
    """
    print("\n" + "=" * 60)
    print("Example 7: Skip Existing Files")
    print("=" * 60)

    # First run - download all games
    print("\nFirst run - downloading all games...")
    results1 = run_daily_pipeline(
        target_date=date(2023, 6, 5),
        skip_existing=False,
    )
    print(f"Downloaded: {len(results1.get('completed', []))}")

    # Second run - skip existing files
    print("\nSecond run - skipping existing files...")
    results2 = run_daily_pipeline(
        target_date=date(2023, 6, 5),
        skip_existing=True,
    )
    print(f"Downloaded: {len(results2.get('completed', []))}")
    print(f"Skipped: {len(results2.get('skipped', []))}")


if __name__ == "__main__":
    import sys

    # Run a specific example based on command line arg
    examples = {
        "1": example_1_basic_usage,
        "2": example_2_today_with_polling,
        "2b": example_2b_full_day_monitoring,
        "3": example_3_date_range,
        "4": lambda: asyncio.run(example_4_custom_callbacks()),
        "5": lambda: asyncio.run(example_5_manual_polling_control()),
        "6": lambda: asyncio.run(example_6_fetch_without_polling()),
        "7": example_7_skip_existing,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Daily Pipeline Examples")
        print("-" * 40)
        print("Usage: python examples/daily_pipeline_example.py [example_number]")
        print()
        print("Available examples:")
        print("  1  - Basic usage: single date processing")
        print("  2  - Today's games with live polling (snapshot only)")
        print("  2b - FULL DAY monitoring (recommended for live days)")
        print("  3  - Date range processing for backfill")
        print("  4  - Custom callbacks for real-time updates")
        print("  5  - Manual polling control with stop event")
        print("  6  - Fetch game states without polling")
        print("  7  - Skip existing files demo")
        print()
        print("Running Example 1 as default...")
        example_1_basic_usage()
