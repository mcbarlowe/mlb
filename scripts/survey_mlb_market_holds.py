"""Survey every MLB market the API offers and rank them by hold.

The hold sets the bar any model must clear, and it is measurable from prices alone with no
outcomes required, so it is the cheapest possible screen. A market is worth modelling only if its
shopped hold is low enough that a plausible edge clears half of it.

What this project has measured so far, for calibration of expectations:

    full-game moneyline      1.40% shopped   breakeven 0.70%
    pitcher strikeouts       3.33% shopped   breakeven 1.66%   no bias on 8,785 starts
    first-five totals        3.60% shopped   breakeven 1.80%
    batter hits/total bases  5.36-5.38%      breakeven 2.68%

Every market is pulled at a strictly pre-game snapshot, and any event already under way is
discarded using the API's own commence time. Hold is computed inside a single
(event, market, book, player, line) group; pooling across books or across line points fabricates
a near-zero hold, a mistake made twice in this project already.

Cost is 10 credits per market per event, so a few days across a wide market list is a few thousand
credits against millions available.

    uv run python scripts/survey_mlb_market_holds.py --dates 2025-06-10,2025-06-11,2025-06-12
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, american_to_prob

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb"
LEAD_HOURS = 3.0

# Every MLB market the API documents, grouped so requests stay a sane size.
MARKET_BATCHES = [
    ("h2h", "spreads", "totals"),
    ("team_totals", "alternate_spreads", "alternate_totals"),
    ("h2h_1st_1_innings", "spreads_1st_1_innings", "totals_1st_1_innings"),
    ("h2h_1st_3_innings", "spreads_1st_3_innings", "totals_1st_3_innings"),
    ("h2h_1st_5_innings", "spreads_1st_5_innings", "totals_1st_5_innings"),
    ("h2h_1st_7_innings", "spreads_1st_7_innings", "totals_1st_7_innings"),
    ("pitcher_strikeouts", "pitcher_outs", "pitcher_hits_allowed"),
    ("pitcher_walks", "pitcher_earned_runs", "pitcher_record_a_win"),
    ("batter_hits", "batter_total_bases", "batter_home_runs"),
    ("batter_rbis", "batter_runs_scored", "batter_singles"),
    ("batter_doubles", "batter_triples", "batter_walks"),
    ("batter_strikeouts", "batter_stolen_bases", "batter_hits_runs_rbis"),
]


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


async def get_json(session, url, params, limiter, tries=3):
    async with limiter:
        for attempt in range(tries):
            try:
                async with session.get(url, params=params, timeout=45) as resp:
                    body_status = resp.status
                    if body_status == 429:
                        await asyncio.sleep(int(resp.headers.get("retry-after", 5)))
                        continue
                    if body_status == 422:
                        return None, 0, "unavailable"
                    if body_status == 401:
                        raise SystemExit("API 401: bad key")
                    if body_status != 200:
                        await asyncio.sleep(2**attempt)
                        continue
                    return (
                        await resp.json(),
                        int(resp.headers.get("x-requests-last", 0)),
                        "ok",
                    )
            except TimeoutError:
                if attempt == tries - 1:
                    return None, 0, "timeout"
                await asyncio.sleep(2**attempt)
    return None, 0, "failed"


async def events_for(session, key, date: str, limiter):
    stamp = f"{date}T12:00:00Z"
    payload, used, _ = await get_json(
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
        if start.strftime("%Y-%m-%d") != date:
            continue
        out.append((e["id"], start))
    return out, used


async def fetch(session, key, eid, start, batch, limiter):
    target = (start - timedelta(hours=LEAD_HOURS)).astimezone(UTC)
    payload, used, status = await get_json(
        session,
        f"{BASE}/events/{eid}/odds",
        {
            "apiKey": key, "regions": "us", "markets": ",".join(batch),
            "oddsFormat": "american",
            "date": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        limiter,
    )
    if not payload:
        return [], used, status
    data = payload.get("data") or {}
    snapshot, commence = payload.get("timestamp"), data.get("commence_time")
    if not snapshot or not commence:
        return [], used, "empty"
    if datetime.fromisoformat(commence) <= datetime.fromisoformat(snapshot):
        return [], used, "inplay"
    rows = []
    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                rows.append((
                    eid, market.get("key"), book.get("key"),
                    o.get("description"), o.get("point"), o.get("name"), o.get("price"),
                    data.get("home_team"),
                ))
    return rows, used, "ok"


def holds(rows):
    """Per-market per-book overround and best-price overround, grouped within one line.

    Sides are keyed canonically, over/under for line markets and home/away for team markets,
    because keying by position lets a book that lists the away team first pair an away price
    with another book's home price. Team handicaps are normalised to the home team's signed
    point so that home -1.5 and away +1.5 group together, and a book quoting the same team at
    both -1.5 and +1.5 does not collide.
    """
    groups: dict[tuple, dict[str, int]] = defaultdict(dict)
    for eid, market, book, desc, point, name, price, home_team in rows:
        if price is None:
            continue
        side = str(name).lower()
        if side in ("over", "under"):
            groups[(eid, market, desc, point, book)][side] = int(price)
            continue
        is_home = str(name) == str(home_team)
        signed = None
        if point is not None:
            signed = float(point) if is_home else -float(point)
        groups[(eid, market, desc, signed, book)]["home" if is_home else "away"] = int(price)

    per_book: dict[str, list[float]] = defaultdict(list)
    by_line: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for (eid, market, desc, point, _book), sides in groups.items():
        if len(sides) != 2:
            continue
        first = sides.get("over", sides.get("home"))
        second = sides.get("under", sides.get("away"))
        if first is None or second is None:
            continue
        per_book[market].append(
            american_to_prob(first) + american_to_prob(second) - 1.0
        )
        by_line[(eid, market, desc, point)].append((first, second))

    shopped: dict[str, list[float]] = defaultdict(list)
    depth: dict[str, list[int]] = defaultdict(list)
    for (_eid, market, _d, _p), quotes in by_line.items():
        if len(quotes) < 2:
            continue
        shopped[market].append(
            1.0 / max(american_to_decimal(a) for a, _ in quotes)
            + 1.0 / max(american_to_decimal(b) for _, b in quotes)
            - 1.0
        )
        depth[market].append(len(quotes))
    return per_book, shopped, depth


async def run(dates: list[str]) -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY not set")
    limiter = Limiter(6, 0.12)
    credits = 0
    rows: list[tuple] = []
    statuses: dict[str, int] = defaultdict(int)
    began = time.time()

    async with aiohttp.ClientSession() as session:
        listings = await asyncio.gather(
            *(events_for(session, key, d, limiter) for d in dates)
        )
        events = []
        for evs, used in listings:
            credits += used
            events += evs
        print(f"events: {len(events)} across {len(dates)} dates")

        tasks = [
            fetch(session, key, eid, start, batch, limiter)
            for eid, start in events
            for batch in MARKET_BATCHES
        ]
        print(f"requests: {len(tasks)}  (10 credits per market, "
              f"{sum(len(b) for b in MARKET_BATCHES)} markets per event)")
        results = await asyncio.gather(*tasks)

    for r, used, status in results:
        rows += r
        credits += used
        statuses[status] += 1

    print(f"credits {credits:,}  elapsed {(time.time() - began) / 60:.1f} min  "
          f"statuses {dict(statuses)}")
    print()

    per_book, shopped, depth = holds(rows)
    print(f"{'market':>26} | {'pairs':>6} | {'per-book':>9} | {'shopped':>8} | "
          f"{'gain':>8} | {'books':>5} | {'breakeven':>9}")
    print("-" * 92)
    ranked = sorted(
        per_book,
        key=lambda m: statistics.median(shopped[m]) if shopped[m] else 9.9,
    )
    for market in ranked:
        pb = statistics.median(per_book[market])
        if not shopped[market]:
            print(f"{market:>26} | {len(per_book[market]):6d} | {pb:8.2%} | "
                  f"{'single book':>8} | {'':>8} | {1:5d} | {'':>9}")
            continue
        sh = statistics.median(shopped[market])
        dp = statistics.median(depth[market])
        print(f"{market:>26} | {len(per_book[market]):6d} | {pb:8.2%} | {sh:7.2%} | "
              f"{(pb - sh) * 100:+7.2f}pp | {dp:5.0f} | {sh / 2:8.2%}")

    print()
    print("Reference, already measured in this project:")
    print(f"{'full-game moneyline':>26} | {'':>6} | {0.0410:8.2%} | {0.0140:7.2%} | "
          f"{'':>8} | {5:5d} | {0.0070:8.2%}")
    print(f"{'pitcher strikeouts':>26} | {'':>6} | {0.0652:8.2%} | {0.0333:7.2%} | "
          f"{'':>8} | {6:5d} | {0.0166:8.2%}")
    print()
    print("Lower breakeven is better. Anything materially above the 0.70% of full-game")
    print("moneyline is a more expensive market to attack, not a weaker one.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", default="2025-06-10,2025-06-11,2025-06-12")
    args = ap.parse_args()
    asyncio.run(run([d.strip() for d in args.dates.split(",")]))


if __name__ == "__main__":
    main()
