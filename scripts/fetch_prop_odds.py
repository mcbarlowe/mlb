"""Fetch historical MLB player prop odds, strictly pre-game.

Props live on the per-event endpoint rather than the bulk odds endpoint, so the cost is one call
per event rather than one call covering the whole slate: 10 credits per market requested, per
event. Four markets across a full season is roughly 97,000 credits.

Props are the least contested markets available here, and the repository already holds the
machinery that matters for them: 14.2M pitches, PA-outcome calibration, pitch-level simulation.
A pitcher strikeout distribution is a direct output of that.

Two safeguards, both learned from earlier defects in this project:

  - Every request targets ``commence_time - LEAD_HOURS`` and any event whose commence time is at
    or before the returned snapshot timestamp is discarded, judged by the API's own commence
    time. An earlier pull silently captured in-play prices for 74-78% of games.
  - Prices are stored per book and per line point. Prop lines differ across books, so pooling
    across points fabricates a near-zero hold, which is a mistake this project has already made
    once on totals.

    uv run python scripts/fetch_prop_odds.py --start 2025-06-01 --end 2025-06-14
    uv run python scripts/fetch_prop_odds.py --start 2025-06-01 --end 2025-06-02 --dry-run
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

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb"
MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")
LEAD_HOURS = 3.0
MAX_CONCURRENT = 6
MIN_DELAY = 0.12


class Limiter:
    def __init__(self, n: int, delay: float) -> None:
        self.sem = asyncio.Semaphore(n)
        self.delay = delay
        self.last = 0.0
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self.sem.acquire()
        async with self.lock:
            gap = time.time() - self.last
            if gap < self.delay:
                await asyncio.sleep(self.delay - gap)
            self.last = time.time()

    async def __aexit__(self, *_e: object) -> None:
        self.sem.release()


async def get_json(session, url, params, limiter, tries=4):
    async with limiter:
        for attempt in range(tries):
            try:
                async with session.get(url, params=params, timeout=40) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(int(resp.headers.get("retry-after", 5)))
                        continue
                    if resp.status == 422:
                        body = await resp.text()
                        # Prop coverage begins partway through 2023, so a market being
                        # unavailable at a given date is expected rather than fatal.
                        if "HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE" in body:
                            return None, 0
                        raise SystemExit(f"API 422: {body[:200]}")
                    if resp.status == 401:
                        raise SystemExit(f"API 401: {(await resp.text())[:200]}")
                    if resp.status != 200:
                        await asyncio.sleep(2**attempt)
                        continue
                    return await resp.json(), int(resp.headers.get("x-requests-last", 0))
            except TimeoutError:
                if attempt == tries - 1:
                    return None, 0
                await asyncio.sleep(2**attempt)
    return None, 0


async def day_events(session, key, day: datetime, limiter):
    stamp = day.replace(hour=12, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload, used = await get_json(
        session, f"{BASE}/events", {"apiKey": key, "date": stamp}, limiter
    )
    if not payload:
        return [], used
    out = []
    for e in payload.get("data", []):
        commence = e.get("commence_time")
        if not commence:
            continue
        start = datetime.fromisoformat(commence)
        if start.date() != day.date():
            continue
        out.append((e["id"], start, e.get("home_team"), e.get("away_team")))
    return out, used


async def event_props(session, key, eid, start, markets, limiter):
    target = start - timedelta(hours=LEAD_HOURS)
    payload, used = await get_json(
        session,
        f"{BASE}/events/{eid}/odds",
        {
            "apiKey": key, "regions": "us", "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": target.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        limiter,
    )
    if not payload:
        return [], used, 0
    snapshot = payload.get("timestamp")
    data = payload.get("data") or {}
    commence = data.get("commence_time")
    if not snapshot or not commence:
        return [], used, 0
    # Strictly pre-game, judged by the API's own commence time.
    if datetime.fromisoformat(commence) <= datetime.fromisoformat(snapshot):
        return [], used, 1

    rows = []
    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            key_m = market.get("key")
            if key_m not in markets:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "event_id": eid,
                    "snapshot_time": snapshot,
                    "commence_time": commence,
                    "home_team": data.get("home_team"),
                    "away_team": data.get("away_team"),
                    "bookmaker": book.get("key"),
                    "market": key_m,
                    "player": outcome.get("description"),
                    "side": outcome.get("name"),
                    "point": outcome.get("point"),
                    "price": outcome.get("price"),
                })
    return rows, used, 0


async def run(start: str, end: str, markets, dry: bool) -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY not set")
    d0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    d1 = datetime.fromisoformat(end).replace(tzinfo=UTC)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    print(f"{len(days)} days, markets {markets}")
    if dry:
        print(f"estimated cost: {len(days)} events calls + "
              f"~{len(days) * 15} event-odds calls x {10 * len(markets)} credits "
              f"= ~{len(days) + len(days) * 15 * 10 * len(markets):,} credits")
        return

    limiter = Limiter(MAX_CONCURRENT, MIN_DELAY)
    began = time.time()
    total_credits = 0
    all_rows: list[dict] = []
    skipped_inplay = 0
    async with aiohttp.ClientSession() as session:
        listings = await asyncio.gather(
            *(day_events(session, key, d, limiter) for d in days)
        )
        events = []
        for evs, used in listings:
            total_credits += used
            events += evs
        print(f"events found: {len(events)}")

        results = await asyncio.gather(
            *(event_props(session, key, eid, start_dt, markets, limiter)
              for eid, start_dt, _h, _a in events)
        )
    for rows, used, skipped in results:
        all_rows += rows
        total_credits += used
        skipped_inplay += skipped

    frame = pl.DataFrame(all_rows, strict=False)
    out = Path(f"data/odds_history/props_{start}_{end}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    print(f"rows {len(frame):,}  events with data "
          f"{frame['event_id'].n_unique() if len(frame) else 0}  "
          f"skipped in-play {skipped_inplay}")
    print(f"credits {total_credits:,}  elapsed {(time.time() - began) / 60:.1f} min")
    print(f"-> {out}")
    if len(frame):
        print()
        print(frame.group_by("market").agg(
            pl.len().alias("rows"),
            pl.col("bookmaker").n_unique().alias("books"),
            pl.col("player").n_unique().alias("players"),
        ).sort("market"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.start, args.end, tuple(args.markets.split(",")), args.dry_run))


if __name__ == "__main__":
    main()
