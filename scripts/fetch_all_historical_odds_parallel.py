"""Fetch historical MLB moneyline odds in parallel with rate limiting.

Fetches 2020-2023 odds in parallel using asyncio with smart rate limiting
to maximize throughput without overwhelming The Odds API.

Usage:
    uv run python scripts/fetch_all_historical_odds_parallel.py
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"

# Season definitions
SEASONS = {
    2020: ("2020-07-23", "2020-09-27"),  # COVID short season
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
}

# Snapshot times (UTC)
SNAPSHOT_TIMES = ["16:30", "20:30"]  # Opening and closing lines
NEXT_DAY_TIMES = ["00:30"]  # Late West Coast games


def generate_snapshots(start: str, end: str) -> list[str]:
    """Generate snapshot timestamps for a date range."""
    d0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    d1 = datetime.fromisoformat(end).replace(tzinfo=UTC)
    stamps: list[str] = []
    
    day = d0
    while day <= d1:
        # Same-day snapshots
        for hhmm in SNAPSHOT_TIMES:
            h, m = (int(x) for x in hhmm.split(":"))
            stamps.append(day.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        # Next-day snapshots for late games
        for hhmm in NEXT_DAY_TIMES:
            h, m = (int(x) for x in hhmm.split(":"))
            nxt = day + timedelta(days=1)
            stamps.append(nxt.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        day += timedelta(days=1)
    
    return sorted(set(stamps))


class RateLimiter:
    """Smart rate limiter that adapts to API responses."""
    
    def __init__(self, max_concurrent: int = 10, min_delay: float = 0.1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_delay = min_delay
        self.last_request = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit slot."""
        await self.semaphore.acquire()
        
        async with self.lock:
            # Ensure minimum delay between requests
            now = time.time()
            elapsed = now - self.last_request
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
            self.last_request = time.time()
    
    def release(self):
        """Release rate limit slot."""
        self.semaphore.release()


async def fetch_snapshot(
    session: aiohttp.ClientSession,
    api_key: str,
    timestamp: str,
    season: int,
    limiter: RateLimiter,
    retries: int = 3,
) -> tuple[dict | None, int]:
    """Fetch one historical snapshot."""
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": timestamp,
    }
    
    await limiter.acquire()
    
    try:
        for attempt in range(retries):
            try:
                async with session.get(BASE_URL, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        credits = int(resp.headers.get("x-requests-last", 10))
                        return data, credits
                    
                    elif resp.status == 429:  # Rate limit
                        wait = int(resp.headers.get("retry-after", 5))
                        await asyncio.sleep(wait)
                        continue
                    
                    elif resp.status in (401, 422):
                        text = await resp.text()
                        raise SystemExit(f"API error {resp.status}: {text[:200]}")
                    
                    else:
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None, 10
            
            except TimeoutError:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None, 10
    
    finally:
        limiter.release()
    
    return None, 10


async def fetch_season(
    session: aiohttp.ClientSession,
    api_key: str,
    season: int,
    start: str,
    end: str,
    limiter: RateLimiter,
    output_dir: Path,
) -> dict:
    """Fetch all snapshots for one season."""
    snapshots = generate_snapshots(start, end)
    output_file = output_dir / f"moneyline_{season}.parquet"
    
    results = []
    total_credits = 0
    failed = 0
    skipped_inplay = 0
    
    # Progress bar for this season
    pbar = tqdm(
        total=len(snapshots),
        desc=f"{season}",
        position=season - 2020,
        leave=True,
    )
    
    for timestamp in snapshots:
        data, credits = await fetch_snapshot(session, api_key, timestamp, season, limiter)
        total_credits += credits
        
        if data is not None and "data" in data:
            snapshot_ts = data.get("timestamp", timestamp)
            
            # Discard events already under way: the historical endpoint returns
            # every event present in the snapshot, including in-play games whose
            # prices reflect the score rather than a pre-game market.
            snap_dt = datetime.fromisoformat(snapshot_ts)

            for game in data["data"]:
                game_id = game.get("id")
                commence = game.get("commence_time")
                home = game.get("home_team")
                away = game.get("away_team")

                if not commence:
                    continue
                if datetime.fromisoformat(commence) <= snap_dt:
                    skipped_inplay += 1
                    continue
                
                for book in game.get("bookmakers", []):
                    bookmaker = book.get("key")
                    
                    for market in book.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        
                        outcomes = market.get("outcomes", [])
                        home_price = None
                        away_price = None
                        
                        for outcome in outcomes:
                            if outcome.get("name") == home:
                                home_price = outcome.get("price")
                            elif outcome.get("name") == away:
                                away_price = outcome.get("price")
                        
                        if home_price is not None and away_price is not None:
                            results.append({
                                "snapshot_time": snapshot_ts,
                                "game_id": game_id,
                                "commence_time": commence,
                                "home_team": home,
                                "away_team": away,
                                "bookmaker": bookmaker,
                                "home_ml": home_price,
                                "away_ml": away_price,
                            })
        else:
            failed += 1
        
        pbar.update(1)
    
    pbar.close()
    
    # Save to parquet
    if results:
        df = pl.DataFrame(results)
        df.write_parquet(output_file)
    
    return {
        "season": season,
        "snapshots": len(snapshots),
        "games": len(results),
        "credits": total_credits,
        "failed": failed,
        "skipped_inplay": skipped_inplay,
        "output": str(output_file),
    }


async def main():
    """Fetch all seasons in parallel."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY not set in environment")
    
    output_dir = Path("data/odds_history")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Fetching Historical MLB Moneyline Odds (2020-2023)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Seasons: {len(SEASONS)}")
    print("Concurrent requests: 10 (rate limited)")
    print()
    
    # Create shared session and rate limiter
    limiter = RateLimiter(max_concurrent=10, min_delay=0.1)
    
    async with aiohttp.ClientSession() as session:
        # Launch all seasons in parallel
        tasks = [
            fetch_season(session, api_key, season, start, end, limiter, output_dir)
            for season, (start, end) in SEASONS.items()
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)
    
    total_credits = 0
    total_snapshots = 0
    total_games = 0
    
    for result in results:
        print(f"\n{result['season']}:")
        print(f"  Snapshots fetched: {result['snapshots']}")
        print(f"  Game records: {result['games']:,}")
        print(f"  Credits used: {result['credits']:,}")
        print(f"  Failed: {result['failed']}")
        print(f"  Output: {result['output']}")
        
        total_credits += result["credits"]
        total_snapshots += result["snapshots"]
        total_games += result["games"]
    
    print("\nTOTAL:")
    print(f"  Snapshots: {total_snapshots:,}")
    print(f"  Game records: {total_games:,}")
    print(f"  Credits used: {total_credits:,}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Rate: {total_snapshots/elapsed:.1f} snapshots/sec")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Load to database: uv run python scripts/load_odds_to_db.py data/odds_history/moneyline_*.parquet")
    print("  2. Run backtests: uv run python scripts/backtest_moneyline.py --season 2020 2021 2022 2023")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
