"""Fetch genuine closing lines by targeting each distinct first-pitch time.

The existing history pull uses three fixed UTC snapshots per day, which puts its latest
strictly-pre-game price a median 2.5h before first pitch. That is not a closing line, so CLV
measured against it is unreliable, and the positive CLV previously reported turned out to be a
selection artifact rather than skill.

The historical endpoint stores snapshots every five minutes and returns the latest snapshot at
or before the requested timestamp. Requesting ``first_pitch - LEAD_MINUTES`` therefore lands
within about five minutes of the actual close. Games sharing a start time are all present in
that one snapshot, so the cost is one call per distinct start time rather than per game: about
1,800 calls and 18,000 credits per season.

Events whose ``commence_time`` is at or before the snapshot timestamp are discarded, using the
API's own commence time rather than the local schedule, so in-play prices cannot leak in.

    uv run python scripts/fetch_closing_lines.py --season 2025
    uv run python scripts/fetch_closing_lines.py --season 2025 --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
LEAD_MINUTES = 2      # request this far before first pitch; snapshots are 5 minutes apart
MAX_CONCURRENT = 8
MIN_DELAY = 0.08


def distinct_start_times(season: int) -> list[datetime]:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT date_trunc('minute', game_datetime) AS start
            FROM {c.schema}.games
            WHERE season::int = %s AND game_type = 'R' AND game_datetime IS NOT NULL
            ORDER BY start
            """,
            (season,),
        )
        out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


class Limiter:
    def __init__(self, concurrent: int, min_delay: float) -> None:
        self.sem = asyncio.Semaphore(concurrent)
        self.min_delay = min_delay
        self.last = 0.0
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self.sem.acquire()
        async with self.lock:
            gap = time.time() - self.last
            if gap < self.min_delay:
                await asyncio.sleep(self.min_delay - gap)
            self.last = time.time()

    async def __aexit__(self, *_exc: object) -> None:
        self.sem.release()


async def fetch_one(
    session: aiohttp.ClientSession, key: str, target: datetime, limiter: Limiter
) -> tuple[list[dict], int]:
    # game_datetime is timezone-aware in the session zone. Formatting its local components
    # with a literal Z would request a snapshot hours away from first pitch.
    stamp = target.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "apiKey": key, "regions": "us", "markets": "h2h",
        "oddsFormat": "american", "date": stamp,
    }
    async with limiter:
        for attempt in range(4):
            try:
                async with session.get(BASE_URL, params=params, timeout=30) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(int(resp.headers.get("retry-after", 5)))
                        continue
                    if resp.status in (401, 422):
                        raise SystemExit(f"API {resp.status}: {(await resp.text())[:200]}")
                    if resp.status != 200:
                        await asyncio.sleep(2**attempt)
                        continue
                    payload = await resp.json()
                    used = int(resp.headers.get("x-requests-last", 10))
                    break
            except TimeoutError:
                if attempt == 3:
                    return [], 10
                await asyncio.sleep(2**attempt)
        else:
            return [], 10

    snapshot = payload.get("timestamp")
    if not snapshot:
        return [], used
    snap_dt = datetime.fromisoformat(snapshot)

    rows: list[dict] = []
    for event in payload.get("data", []):
        commence = event.get("commence_time")
        if not commence:
            continue
        # Strictly pre-game only, judged by the API's own commence time.
        if datetime.fromisoformat(commence) <= snap_dt:
            continue
        home, away = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                if prices.get(home) is None or prices.get(away) is None:
                    continue
                rows.append({
                    "snapshot_time": snapshot,
                    "game_id": event.get("id"),
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": book.get("key"),
                    "home_ml": prices[home],
                    "away_ml": prices[away],
                })
    return rows, used


async def run(season: int, limit: int | None, dry_run: bool) -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY not set")

    starts = distinct_start_times(season)
    if limit:
        starts = starts[:limit]
    targets = [s - timedelta(minutes=LEAD_MINUTES) for s in starts]
    print(f"season {season}: {len(targets):,} distinct start times, "
          f"~{len(targets) * 10:,} credits, lead {LEAD_MINUTES} min")
    if dry_run:
        for t in targets[:5]:
            print(f"  would request {t:%Y-%m-%dT%H:%M:%SZ}")
        return

    limiter = Limiter(MAX_CONCURRENT, MIN_DELAY)
    began = time.time()
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(fetch_one(session, key, t, limiter) for t in targets)
        )

    rows = [r for chunk, _ in results for r in chunk]
    credits = sum(used for _, used in results)
    frame = pl.DataFrame(rows, strict=False)

    # One row per (game, book): the snapshot nearest first pitch wins.
    if len(frame):
        frame = (
            frame.with_columns(
                pl.col("snapshot_time").str.to_datetime().alias("_snap"),
                pl.col("commence_time").str.to_datetime().alias("_start"),
            )
            .with_columns((pl.col("_start") - pl.col("_snap")).alias("_lead"))
            .sort("_lead")
            .unique(subset=["game_id", "bookmaker"], keep="first")
        )
        lead_min = frame["_lead"].dt.total_minutes()
        print(f"lead to first pitch: median {lead_min.median():.0f} min, "
              f"p90 {lead_min.quantile(0.9):.0f} min, max {lead_min.max():.0f} min")
        frame = frame.drop("_snap", "_start", "_lead")

    out = Path(f"data/odds_history/moneyline_{season}_trueclose.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    print(f"rows {len(frame):,} over {frame['game_id'].n_unique():,} games "
          f"-> {out}")
    print(f"credits used {credits:,}, elapsed {(time.time() - began) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.season, args.limit, args.dry_run))


if __name__ == "__main__":
    main()
