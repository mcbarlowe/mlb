#!/usr/bin/env python3
"""Fetch historical Pinnacle moneyline odds, aligned to the stored US snapshots.

Pinnacle is absent from all stored odds rows because every prior pull used ``regions=us`` and
Pinnacle sits in ``eu``. That matters: measured on 1,651 matched 2025 games, Pinnacle's Brier is
0.2368 against 0.2415 for a five-book US median, so every "market" comparison in this repository
was graded against the softer reference.

Alignment is by construction, not by inferring a cadence. A first attempt assumed the stored
``open`` rows came from a daily 18:00 UTC pull, because the API reports snapshots at 17:55:37. That
value is Central time, so the real snapshot is 22:55 UTC, and it is a *day-ahead* pull: 22:55 UTC on
day D covers day D+1's games. Requesting 18:00 UTC returned same-day prices a median of 19 hours
later than the US rows being compared against, which manufactured a +13% edge out of pure
look-ahead. Requesting the timestamps already stored in the database makes that impossible.

    uv run python scripts/fetch_pinnacle_history.py --season 2025 --dry-run
    uv run python scripts/fetch_pinnacle_history.py --season 2025
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.database import PostgresConfig

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
TARGET_BOOKS = ("pinnacle",)


def existing_snapshots(season: int, line_type: str, panel: list[str]) -> list[datetime]:
    """Exact snapshot timestamps the stored US prices carry for that season."""
    cfg = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=cfg.dbname, user=cfg.user, password=cfg.password,
        host=cfg.host, port=cfg.port, connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT o.snapshot_time
            FROM {cfg.schema}.odds o JOIN {cfg.schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND o.line_type = %s AND o.bookmaker = ANY(%s)
              AND o.snapshot_time IS NOT NULL
            ORDER BY 1
            """,
            (season, line_type, panel),
        )
        stamps = [r[0].astimezone(UTC) for r in cur.fetchall()]
    conn.close()
    return stamps


async def fetch_snapshot(
    session: aiohttp.ClientSession,
    key: str,
    stamp: datetime,
    limiter: asyncio.Semaphore,
    books: tuple[str, ...],
) -> tuple[list[dict], str | None]:
    """One historical snapshot at an exact timestamp."""
    params = {
        "apiKey": key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    async with limiter:
        for attempt in range(4):
            try:
                async with session.get(BASE_URL, params=params, timeout=45) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(int(resp.headers.get("retry-after", 5)))
                        continue
                    if resp.status in (401, 422):
                        raise SystemExit(f"API {resp.status}: {(await resp.text())[:200]}")
                    if resp.status != 200:
                        await asyncio.sleep(2**attempt)
                        continue
                    payload = await resp.json()
                    break
            except TimeoutError:
                if attempt == 3:
                    return [], None
                await asyncio.sleep(2**attempt)
        else:
            return [], None

    served = payload.get("timestamp")
    rows: list[dict] = []
    for game in payload.get("data", []):
        for book in game.get("bookmakers", []):
            if book.get("key") not in books:
                continue
            markets = book.get("markets", [])
            if not markets:
                continue
            outcomes = markets[0].get("outcomes", [])
            if len(outcomes) != 2:
                continue
            prices = {o.get("name"): o.get("price") for o in outcomes}
            home = prices.get(game.get("home_team"))
            away = prices.get(game.get("away_team"))
            if home is None or away is None:
                continue
            rows.append(
                {
                    "snapshot_time": served,
                    "game_id": game.get("id"),
                    "commence_time": game.get("commence_time"),
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "bookmaker": book["key"],
                    "home_ml": int(home),
                    "away_ml": int(away),
                }
            )
    return rows, served


async def run(season: int, line_type: str, limit: int, books: tuple[str, ...],
              out: Path, dry_run: bool) -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY is not set")
    panel = list(PANEL_PRIORITY[:5])
    stamps = existing_snapshots(season, line_type, panel)
    if not stamps:
        raise SystemExit(f"no stored {line_type} snapshots for {season}")
    print(f"season {season} {line_type}: {len(stamps)} stored snapshot timestamps")
    print(f"  first {stamps[0].isoformat()}  last {stamps[-1].isoformat()}")
    print(f"  estimated cost: {len(stamps) * 10:,} credits")
    if dry_run:
        print("dry run: probing the first three snapshots only")
        stamps = stamps[:3]

    limiter = asyncio.Semaphore(limit)
    rows: list[dict] = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_snapshot(session, key, s, limiter, books) for s in stamps]
        for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            got, _served = await coro
            rows.extend(got)
            if done % 25 == 0 or done == len(stamps):
                print(f"  {done}/{len(stamps)} snapshots, {len(rows)} rows")

    if not rows:
        print("no rows returned")
        return
    frame = pl.DataFrame(rows)
    # One row per game, keeping the EARLIEST snapshot. A game appears in every snapshot between
    # its posting and its start, and the loader's primary key is
    # (game_pk, bookmaker, market, line_type) with no snapshot component, so whichever row is
    # processed last silently wins. That kept a price roughly 45 minutes before first pitch while
    # the US rows it is compared against are 17-27 hours out, reintroducing the same look-ahead
    # the exact-timestamp alignment was meant to remove. The earliest snapshot is the day-ahead
    # opener, which is what line_type='open' means for the stored US prices.
    before = len(frame)
    frame = (
        frame.sort("snapshot_time")
        .group_by("game_id", maintain_order=True)
        .first()
    )
    print(f"deduped to earliest snapshot per game: {before} -> {len(frame)} rows")
    print(f"rows {len(frame)}, distinct games {frame['game_id'].n_unique()}, "
          f"books {sorted(frame['bookmaker'].unique().to_list())}")
    if dry_run:
        print("dry run: nothing written")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    print(f"wrote {out}")
    print()
    print("load with:")
    print(f"  uv run python scripts/load_odds_to_db.py --stage {out} "
          f"--season {season} --line-type {line_type}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--line-type", default="open", choices=("open", "close", "true_close"))
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--books", default=",".join(TARGET_BOOKS))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    books = tuple(b.strip() for b in args.books.split(",") if b.strip())
    out = Path(args.out) if args.out else Path(
        f"data/odds_history/pinnacle_{args.season}_{args.line_type}.parquet"
    )
    asyncio.run(run(args.season, args.line_type, args.limit, books, out, args.dry_run))


if __name__ == "__main__":
    main()
