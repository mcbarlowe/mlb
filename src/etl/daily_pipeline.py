"""
Daily and Live Game Data Pipeline for MLB Stats API.

This module provides date-based ETL functionality for:
- Fetching all games for a specific date (completed or live)
- Polling live games until completion
- Automatic handling of game completion and data transformation

Uses async/await throughout for concurrent operations.
"""

import asyncio
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import aiofiles
from tqdm import tqdm

from src.data.game_feed_data import GameFeedData
from src.endpoints.game_feed import GameFeed
from src.endpoints.schedule import Schedule


class GameState(Enum):
    """Game state classifications based on MLB API status codes."""

    SCHEDULED = "scheduled"      # Game hasn't started
    LIVE = "live"               # Game in progress
    FINAL = "final"             # Game completed
    POSTPONED = "postponed"     # Game postponed
    SUSPENDED = "suspended"     # Game suspended
    CANCELLED = "cancelled"     # Game cancelled
    UNKNOWN = "unknown"         # Unknown status


# Status code mappings from MLB API
FINAL_STATUS_CODES = {"F", "FT", "FR", "FO"}  # Final, Final (Tied), etc.
LIVE_STATUS_CODES = {"I", "MA", "MB", "MC", "MD", "ME", "MF", "MG", "MH", "MI"}  # In Progress, Manager Challenge variations
SCHEDULED_STATUS_CODES = {"S", "PW", "PI"}  # Scheduled, Pre-Game Warmup, Pre-Game Intros
POSTPONED_STATUS_CODES = {"PD", "DR"}  # Postponed, Delayed/Rainout
SUSPENDED_STATUS_CODES = {"SU", "UR"}  # Suspended, Unknown Resume
CANCELLED_STATUS_CODES = {"C", "CR"}  # Cancelled


def classify_game_state(status_code: str) -> GameState:
    """
    Classify a game's state based on its status code.

    Args:
        status_code: The statusCode from gameData.status

    Returns:
        GameState enum value
    """
    if status_code in FINAL_STATUS_CODES:
        return GameState.FINAL
    elif status_code in LIVE_STATUS_CODES:
        return GameState.LIVE
    elif status_code in SCHEDULED_STATUS_CODES:
        return GameState.SCHEDULED
    elif status_code in POSTPONED_STATUS_CODES:
        return GameState.POSTPONED
    elif status_code in SUSPENDED_STATUS_CODES:
        return GameState.SUSPENDED
    elif status_code in CANCELLED_STATUS_CODES:
        return GameState.CANCELLED
    else:
        return GameState.UNKNOWN


def extract_game_status(game_data: dict) -> tuple[str, GameState]:
    """
    Extract status code and game state from game data.

    Args:
        game_data: The full game JSON from the live feed API

    Returns:
        Tuple of (status_code, GameState)
    """
    status_code = game_data.get("gameData", {}).get("status", {}).get("statusCode", "")
    return status_code, classify_game_state(status_code)


class DailyPipeline:
    """
    Pipeline for fetching and processing MLB games by date.

    Supports:
    - Fetching all games for a specific date
    - Processing completed games immediately
    - Polling live games until completion
    - Concurrent async operations throughout

    Usage:
        async with DailyPipeline() as pipeline:
            # Fetch and process all games for today
            results = await pipeline.run(date.today())

            # Or fetch games for a specific date
            results = await pipeline.run(date(2024, 7, 15))
    """

    def __init__(
        self,
        concurrency_limit: int = 15,
        poll_interval: float = 30.0,
        output_dir: Path = Path("data/raw/livefeeds"),
        processed_dir: Path = Path("data/processed/livefeeds"),
        on_game_complete: Optional[Callable[[int, dict], None]] = None,
        on_game_update: Optional[Callable[[int, dict, GameState], None]] = None,
    ):
        """
        Initialize the daily pipeline.

        Args:
            concurrency_limit: Maximum concurrent API requests
            poll_interval: Seconds between polls for live games
            output_dir: Directory for raw JSON output
            processed_dir: Directory for processed parquet output
            on_game_complete: Optional callback when a game completes (game_pk, data)
            on_game_update: Optional callback on each poll update (game_pk, data, state)
        """
        self.concurrency_limit = concurrency_limit
        self.poll_interval = poll_interval
        self.output_dir = output_dir
        self.processed_dir = processed_dir
        self.on_game_complete = on_game_complete
        self.on_game_update = on_game_update

        self._schedule: Optional[Schedule] = None
        self._game_feed: Optional[GameFeed] = None
        self._transformer = GameFeedData()

    async def __aenter__(self):
        """Initialize async resources."""
        self._schedule = Schedule(concurrency_limit=self.concurrency_limit)
        self._game_feed = GameFeed(concurrency_limit=self.concurrency_limit)
        await self._schedule.__aenter__()
        await self._game_feed.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if self._game_feed:
            await self._game_feed.__aexit__(exc_type, exc_val, exc_tb)
        if self._schedule:
            await self._schedule.__aexit__(exc_type, exc_val, exc_tb)

    async def get_games_for_date(self, target_date: date) -> list[dict]:
        """
        Fetch the schedule for a specific date.

        Args:
            target_date: The date to fetch games for (YYYY-MM-DD format)

        Returns:
            List of game objects from the schedule API
        """
        date_str = target_date.strftime("%Y-%m-%d")
        schedule_data = await self._schedule.get_async(
            sportId=1,
            date=date_str,
        )

        games = []
        for date_entry in schedule_data.get("dates", []):
            games.extend(date_entry.get("games", []))

        return games

    async def fetch_game(self, game_pk: int) -> dict:
        """
        Fetch live feed data for a single game.

        Args:
            game_pk: The game's primary key

        Returns:
            The full game JSON data
        """
        return await self._game_feed.get_async(game_pk)

    async def save_game_json(self, game_pk: int, data: dict, season: str) -> Path:
        """
        Save game data to JSON file.

        Args:
            game_pk: The game's primary key
            data: The game JSON data
            season: The season year

        Returns:
            Path to the saved file
        """
        season_dir = self.output_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)

        output_file = season_dir / f"{game_pk}.json"
        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(data, indent=2))

        return output_file

    async def transform_game(self, game_pk: int, data: dict, season: str) -> Path:
        """
        Transform game data to parquet format.

        Args:
            game_pk: The game's primary key
            data: The game JSON data
            season: The season year

        Returns:
            Path to the saved parquet file
        """
        season_dir = self.processed_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)

        output_file = season_dir / f"{game_pk}.parquet"

        # Transform to DataFrame
        df = self._transformer.transform(data, game_pk, season)

        # Save to parquet
        self._transformer.save(df, output_file, format="parquet")

        return output_file

    async def process_completed_game(
        self,
        game_pk: int,
        data: dict,
        season: str,
    ) -> dict:
        """
        Process a completed game: save JSON and transform to parquet.

        Args:
            game_pk: The game's primary key
            data: The game JSON data
            season: The season year

        Returns:
            Dict with processing results
        """
        json_path = await self.save_game_json(game_pk, data, season)
        parquet_path = await self.transform_game(game_pk, data, season)

        if self.on_game_complete:
            self.on_game_complete(game_pk, data)

        return {
            "game_pk": game_pk,
            "json_path": str(json_path),
            "parquet_path": str(parquet_path),
        }

    async def poll_live_games(
        self,
        live_games: dict[int, dict],
        season: str,
        stop_event: Optional[asyncio.Event] = None,
        scheduled_games: Optional[set[int]] = None,
    ) -> dict[int, dict]:
        """
        Poll live games until they complete, optionally monitoring scheduled games.

        Args:
            live_games: Dict of game_pk -> initial game data
            season: The season year
            stop_event: Optional event to stop polling early
            scheduled_games: Optional set of scheduled game PKs to monitor for start

        Returns:
            Dict of game_pk -> final processing results
        """
        results = {}
        active_games = dict(live_games)
        pending_games = set(scheduled_games) if scheduled_games else set()

        if not active_games and not pending_games:
            return {}

        print(f"\nPolling {len(active_games)} live games, monitoring {len(pending_games)} scheduled (interval: {self.poll_interval}s)...")

        while active_games or pending_games:
            # Check for stop signal
            if stop_event and stop_event.is_set():
                print("Polling stopped by external signal")
                break

            # Check if any scheduled games have started
            if pending_games:
                games_now_live = []
                check_tasks = [
                    self._poll_single_game(game_pk, season)
                    for game_pk in pending_games
                ]
                check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

                for game_pk, result in zip(list(pending_games), check_results):
                    if isinstance(result, Exception):
                        continue

                    data, state = result

                    if state == GameState.LIVE:
                        print(f"Game {game_pk} has started - adding to active polling")
                        active_games[game_pk] = data
                        games_now_live.append(game_pk)
                    elif state == GameState.FINAL:
                        # Game completed while we weren't polling (unlikely but possible)
                        print(f"Game {game_pk} already completed - processing...")
                        try:
                            process_result = await self.process_completed_game(
                                game_pk, data, season
                            )
                            results[game_pk] = process_result
                        except Exception as e:
                            print(f"Error processing game {game_pk}: {e}")
                            results[game_pk] = {"game_pk": game_pk, "error": str(e)}
                        games_now_live.append(game_pk)
                    elif state in (GameState.POSTPONED, GameState.CANCELLED):
                        print(f"Game {game_pk} is {state.value} - removing from monitoring")
                        games_now_live.append(game_pk)

                # Remove games that are no longer pending
                for game_pk in games_now_live:
                    pending_games.discard(game_pk)

            # Poll all active games concurrently
            if active_games:
                poll_tasks = [
                    self._poll_single_game(game_pk, season)
                    for game_pk in active_games
                ]
                poll_results = await asyncio.gather(*poll_tasks, return_exceptions=True)

                # Process results
                games_to_remove = []
                for game_pk, result in zip(list(active_games.keys()), poll_results):
                    if isinstance(result, Exception):
                        print(f"Error polling game {game_pk}: {result}")
                        continue

                    data, state = result

                    # Call update callback if provided
                    if self.on_game_update:
                        self.on_game_update(game_pk, data, state)

                    if state == GameState.FINAL:
                        # Game completed - process and remove from active
                        print(f"Game {game_pk} completed - processing...")
                        try:
                            process_result = await self.process_completed_game(
                                game_pk, data, season
                            )
                            results[game_pk] = process_result
                        except Exception as e:
                            print(f"Error processing completed game {game_pk}: {e}")
                            results[game_pk] = {"game_pk": game_pk, "error": str(e)}

                        games_to_remove.append(game_pk)

                    elif state in (GameState.POSTPONED, GameState.CANCELLED, GameState.SUSPENDED):
                        # Game not continuing - remove from polling
                        print(f"Game {game_pk} is {state.value} - removing from poll")
                        games_to_remove.append(game_pk)

                # Remove completed/stopped games
                for game_pk in games_to_remove:
                    del active_games[game_pk]

            if active_games or pending_games:
                # Show status and wait before next poll
                status_parts = []
                if active_games:
                    status_parts.append(f"Live: {len(active_games)}")
                if pending_games:
                    status_parts.append(f"Scheduled: {len(pending_games)}")
                print(f"  [{', '.join(status_parts)}] waiting {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)

        return results

    async def _poll_single_game(self, game_pk: int, season: str) -> tuple[dict, GameState]:
        """
        Poll a single game and return its current state.

        Args:
            game_pk: The game's primary key
            season: The season year

        Returns:
            Tuple of (game_data, GameState)
        """
        data = await self.fetch_game(game_pk)
        _, state = extract_game_status(data)
        return data, state

    async def monitor_all_games(
        self,
        target_date: date,
        skip_existing: bool = True,
        stop_event: Optional[asyncio.Event] = None,
        on_game_start: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        """
        Monitor all games for a date until every game completes.

        This is the recommended method for live day monitoring. It will:
        - Process any already-completed games immediately
        - Poll currently live games
        - Monitor scheduled games and automatically start polling when they begin
        - Continue until ALL games for the day have completed

        Args:
            target_date: The date to monitor
            skip_existing: Skip games that already have JSON files
            stop_event: Optional event to stop monitoring early
            on_game_start: Optional callback when a scheduled game starts (game_pk, data)

        Returns:
            Dict with processing statistics and results

        Example:
            async with DailyPipeline(poll_interval=30.0) as pipeline:
                # Monitor all of today's games until they're all done
                results = await pipeline.monitor_all_games(date.today())
        """
        self._on_game_start = on_game_start
        date_str = target_date.strftime("%Y-%m-%d")
        season = str(target_date.year)

        print(f"\n{'='*60}")
        print(f"Full Day Monitor: {date_str}")
        print(f"{'='*60}")

        # Fetch schedule for the date
        print(f"\nFetching schedule for {date_str}...")
        schedule_games = await self.get_games_for_date(target_date)
        print(f"Found {len(schedule_games)} games scheduled")

        if not schedule_games:
            return {
                "date": date_str,
                "total_games": 0,
                "processed": {},
                "skipped": [],
                "cancelled": [],
            }

        # Setup directories
        season_dir = self.output_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)

        # Categorize all games
        completed_games = []
        live_games = {}
        scheduled_games = set()
        other_games = []
        skipped_games = []

        print("Fetching initial game states...")
        game_pks = [g["gamePk"] for g in schedule_games]

        fetch_tasks = []
        for game_pk in game_pks:
            if skip_existing:
                existing_file = season_dir / f"{game_pk}.json"
                if existing_file.exists():
                    skipped_games.append(game_pk)
                    continue
            fetch_tasks.append((game_pk, self.fetch_game(game_pk)))

        results_list = await asyncio.gather(
            *[t[1] for t in fetch_tasks],
            return_exceptions=True,
        )

        for (game_pk, _), result in zip(fetch_tasks, results_list):
            if isinstance(result, Exception):
                print(f"Error fetching game {game_pk}: {result}")
                other_games.append({"game_pk": game_pk, "error": str(result)})
                continue

            _, state = extract_game_status(result)

            if state == GameState.FINAL:
                completed_games.append((game_pk, result))
            elif state == GameState.LIVE:
                live_games[game_pk] = result
            elif state == GameState.SCHEDULED:
                scheduled_games.add(game_pk)
            else:
                other_games.append({"game_pk": game_pk, "state": state.value})

        print(f"\nInitial game states:")
        print(f"  - Completed: {len(completed_games)}")
        print(f"  - Live: {len(live_games)}")
        print(f"  - Scheduled: {len(scheduled_games)}")
        print(f"  - Other (postponed/cancelled): {len(other_games)}")
        print(f"  - Skipped (existing): {len(skipped_games)}")

        # Process already-completed games first
        processed_results = {}
        if completed_games:
            print(f"\nProcessing {len(completed_games)} already-completed games...")
            with tqdm(total=len(completed_games), desc="Processing completed") as pbar:
                for game_pk, data in completed_games:
                    try:
                        result = await self.process_completed_game(game_pk, data, season)
                        processed_results[game_pk] = result
                    except Exception as e:
                        print(f"Error processing game {game_pk}: {e}")
                        processed_results[game_pk] = {"game_pk": game_pk, "error": str(e)}
                    pbar.update(1)

        # Now poll live games AND monitor scheduled games
        if live_games or scheduled_games:
            poll_results = await self.poll_live_games(
                live_games=live_games,
                season=season,
                stop_event=stop_event,
                scheduled_games=scheduled_games,
            )
            processed_results.update(poll_results)

        print(f"\n{'='*60}")
        print(f"Full Day Monitor Complete: {date_str}")
        print(f"  Total processed: {len(processed_results)}")
        print(f"  Skipped: {len(skipped_games)}")
        print(f"{'='*60}")

        return {
            "date": date_str,
            "season": season,
            "total_games": len(schedule_games),
            "processed": processed_results,
            "skipped": skipped_games,
            "other": other_games,
        }

    async def run(
        self,
        target_date: date,
        skip_existing: bool = True,
        poll_live: bool = True,
        stop_event: Optional[asyncio.Event] = None,
    ) -> dict:
        """
        Run the full pipeline for a specific date.

        Args:
            target_date: The date to process games for
            skip_existing: Skip games that already have JSON files
            poll_live: Whether to poll live games until completion
            stop_event: Optional event to stop live polling early

        Returns:
            Dict with processing statistics and results
        """
        date_str = target_date.strftime("%Y-%m-%d")
        season = str(target_date.year)

        print(f"\n{'='*60}")
        print(f"Daily Pipeline: {date_str}")
        print(f"{'='*60}")

        # Fetch schedule for the date
        print(f"\nFetching schedule for {date_str}...")
        schedule_games = await self.get_games_for_date(target_date)
        print(f"Found {len(schedule_games)} games scheduled")

        if not schedule_games:
            return {
                "date": date_str,
                "total_games": 0,
                "completed": [],
                "live": [],
                "scheduled": [],
                "other": [],
            }

        # Categorize games by state
        season_dir = self.output_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)

        completed_games = []
        live_games = {}
        scheduled_games = []
        other_games = []
        skipped_games = []

        # Fetch all games concurrently to check their status
        print("Fetching game data to determine states...")
        game_pks = [g["gamePk"] for g in schedule_games]

        fetch_tasks = []
        for game_pk in game_pks:
            # Check if we should skip
            if skip_existing:
                existing_file = season_dir / f"{game_pk}.json"
                if existing_file.exists():
                    skipped_games.append(game_pk)
                    continue
            fetch_tasks.append((game_pk, self.fetch_game(game_pk)))

        # Gather all fetches
        results = await asyncio.gather(
            *[t[1] for t in fetch_tasks],
            return_exceptions=True,
        )

        for (game_pk, _), result in zip(fetch_tasks, results):
            if isinstance(result, Exception):
                print(f"Error fetching game {game_pk}: {result}")
                other_games.append({"game_pk": game_pk, "error": str(result)})
                continue

            _, state = extract_game_status(result)

            if state == GameState.FINAL:
                completed_games.append((game_pk, result))
            elif state == GameState.LIVE:
                live_games[game_pk] = result
            elif state == GameState.SCHEDULED:
                scheduled_games.append(game_pk)
            else:
                other_games.append({"game_pk": game_pk, "state": state.value})

        print(f"\nGame states:")
        print(f"  - Completed: {len(completed_games)}")
        print(f"  - Live: {len(live_games)}")
        print(f"  - Scheduled: {len(scheduled_games)}")
        print(f"  - Other: {len(other_games)}")
        print(f"  - Skipped (existing): {len(skipped_games)}")

        # Process completed games
        completed_results = []
        if completed_games:
            print(f"\nProcessing {len(completed_games)} completed games...")
            with tqdm(total=len(completed_games), desc="Processing completed") as pbar:
                for game_pk, data in completed_games:
                    try:
                        result = await self.process_completed_game(game_pk, data, season)
                        completed_results.append(result)
                    except Exception as e:
                        print(f"Error processing game {game_pk}: {e}")
                        completed_results.append({"game_pk": game_pk, "error": str(e)})
                    pbar.update(1)

        # Poll live games if requested
        live_results = {}
        if poll_live and live_games:
            live_results = await self.poll_live_games(live_games, season, stop_event)

        print(f"\n{'='*60}")
        print("Pipeline Complete")
        print(f"{'='*60}")

        return {
            "date": date_str,
            "season": season,
            "total_games": len(schedule_games),
            "completed": completed_results,
            "live": live_results,
            "scheduled": scheduled_games,
            "other": other_games,
            "skipped": skipped_games,
        }


async def run_daily_pipeline_async(
    target_date: Optional[date] = None,
    concurrency_limit: int = 15,
    poll_interval: float = 30.0,
    skip_existing: bool = True,
    poll_live: bool = True,
) -> dict:
    """
    Run the daily pipeline for a specific date.

    Convenience function that manages the async context.

    Args:
        target_date: Date to process (defaults to today)
        concurrency_limit: Maximum concurrent API requests
        poll_interval: Seconds between polls for live games
        skip_existing: Skip games that already have JSON files
        poll_live: Whether to poll live games until completion

    Returns:
        Dict with processing statistics and results
    """
    if target_date is None:
        target_date = date.today()

    async with DailyPipeline(
        concurrency_limit=concurrency_limit,
        poll_interval=poll_interval,
    ) as pipeline:
        return await pipeline.run(
            target_date=target_date,
            skip_existing=skip_existing,
            poll_live=poll_live,
        )


def run_daily_pipeline(
    target_date: Optional[date] = None,
    concurrency_limit: int = 15,
    poll_interval: float = 30.0,
    skip_existing: bool = True,
    poll_live: bool = True,
) -> dict:
    """
    Run the daily pipeline for a specific date.

    Synchronous wrapper around the async pipeline.

    Args:
        target_date: Date to process (defaults to today)
        concurrency_limit: Maximum concurrent API requests
        poll_interval: Seconds between polls for live games
        skip_existing: Skip games that already have JSON files
        poll_live: Whether to poll live games until completion

    Returns:
        Dict with processing statistics and results

    Example:
        # Process today's games
        results = run_daily_pipeline()

        # Process a specific date
        results = run_daily_pipeline(date(2024, 7, 15))

        # Process without polling live games
        results = run_daily_pipeline(date(2024, 7, 15), poll_live=False)
    """
    return asyncio.run(
        run_daily_pipeline_async(
            target_date=target_date,
            concurrency_limit=concurrency_limit,
            poll_interval=poll_interval,
            skip_existing=skip_existing,
            poll_live=poll_live,
        )
    )


async def run_date_range_async(
    start_date: date,
    end_date: date,
    concurrency_limit: int = 15,
    skip_existing: bool = True,
) -> dict[str, dict]:
    """
    Process multiple dates sequentially.

    Args:
        start_date: First date to process (inclusive)
        end_date: Last date to process (inclusive)
        concurrency_limit: Maximum concurrent API requests
        skip_existing: Skip games that already have JSON files

    Returns:
        Dict of date_str -> processing results
    """
    from datetime import timedelta

    results = {}
    current_date = start_date

    async with DailyPipeline(concurrency_limit=concurrency_limit) as pipeline:
        while current_date <= end_date:
            result = await pipeline.run(
                target_date=current_date,
                skip_existing=skip_existing,
                poll_live=False,  # Don't poll for historical dates
            )
            results[current_date.strftime("%Y-%m-%d")] = result
            current_date += timedelta(days=1)

    return results


def run_date_range(
    start_date: date,
    end_date: date,
    concurrency_limit: int = 15,
    skip_existing: bool = True,
) -> dict[str, dict]:
    """
    Process multiple dates sequentially.

    Synchronous wrapper around the async function.

    Args:
        start_date: First date to process (inclusive)
        end_date: Last date to process (inclusive)
        concurrency_limit: Maximum concurrent API requests
        skip_existing: Skip games that already have JSON files

    Returns:
        Dict of date_str -> processing results

    Example:
        from datetime import date
        results = run_date_range(date(2024, 7, 1), date(2024, 7, 7))
    """
    return asyncio.run(
        run_date_range_async(
            start_date=start_date,
            end_date=end_date,
            concurrency_limit=concurrency_limit,
            skip_existing=skip_existing,
        )
    )


async def monitor_full_day_async(
    target_date: Optional[date] = None,
    concurrency_limit: int = 15,
    poll_interval: float = 30.0,
    skip_existing: bool = True,
) -> dict:
    """
    Monitor all games for a date until every game completes.

    This is the recommended method for live day monitoring. Unlike run_daily_pipeline,
    this will:
    - Poll currently live games
    - Monitor scheduled games and automatically start polling when they begin
    - Continue until ALL games for the day have completed

    Use this when you want to start monitoring in the morning and have it run
    all day, picking up games as they start (1pm games, 4pm games, 7pm games, etc.)

    Args:
        target_date: Date to monitor (defaults to today)
        concurrency_limit: Maximum concurrent API requests
        poll_interval: Seconds between polls for live/scheduled games
        skip_existing: Skip games that already have JSON files

    Returns:
        Dict with processing statistics and results
    """
    if target_date is None:
        target_date = date.today()

    async with DailyPipeline(
        concurrency_limit=concurrency_limit,
        poll_interval=poll_interval,
    ) as pipeline:
        return await pipeline.monitor_all_games(
            target_date=target_date,
            skip_existing=skip_existing,
        )


def monitor_full_day(
    target_date: Optional[date] = None,
    concurrency_limit: int = 15,
    poll_interval: float = 30.0,
    skip_existing: bool = True,
) -> dict:
    """
    Monitor all games for a date until every game completes.

    Synchronous wrapper around the async function.

    This is the recommended method for live day monitoring. Unlike run_daily_pipeline,
    this will:
    - Poll currently live games
    - Monitor scheduled games and automatically start polling when they begin
    - Continue until ALL games for the day have completed

    Use this when you want to start monitoring in the morning and have it run
    all day, picking up games as they start (1pm games, 4pm games, 7pm games, etc.)

    Args:
        target_date: Date to monitor (defaults to today)
        concurrency_limit: Maximum concurrent API requests
        poll_interval: Seconds between polls for live/scheduled games
        skip_existing: Skip games that already have JSON files

    Returns:
        Dict with processing statistics and results

    Example:
        from datetime import date

        # Monitor all of today's games
        results = monitor_full_day()

        # Monitor a specific date (will wait for all games to complete)
        results = monitor_full_day(date(2024, 7, 15))
    """
    return asyncio.run(
        monitor_full_day_async(
            target_date=target_date,
            concurrency_limit=concurrency_limit,
            poll_interval=poll_interval,
            skip_existing=skip_existing,
        )
    )


if __name__ == "__main__":
    # Example: Monitor all of today's games
    results = monitor_full_day()
    print(f"\nResults: {json.dumps(results, indent=2, default=str)}")
