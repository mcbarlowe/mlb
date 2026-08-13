"""Fetch historical MLB moneyline (h2h) odds from The Odds API and stage them.

Read-only w.r.t. our database: this only calls the Odds API and writes a staging
parquet of per-book near-closing lines. A separate loader (run after the schema
migration) resolves team_ids + game_pk and upserts into ``mlb.odds``.

Strategy: one historical snapshot returns the whole slate, so we sample a few
snapshots per day and, per game, keep the latest snapshot at-or-before first
pitch (near-closing). Cost is ~10 credits/snapshot, not per game.

Env: ODDS_API_KEY. Usage:
    uv run python scripts/fetch_odds_history.py --start 2024-03-20 --end 2024-10-01 \
        --out /tmp/odds_2024_stage.parquet
    (add --dry-run-days 1 to validate one day cheaply first)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime, timedelta

import polars as pl
import requests

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
# UTC snapshot times per slate date; the trailing early-morning slot catches
# late West-coast games that start after midnight UTC.
SNAPSHOT_HHMM = ["16:30", "20:30"]
NEXT_DAY_HHMM = ["00:30"]


def snapshot_times(start: str, end: str, times: list[str], next_times: list[str]) -> list[str]:
    d0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    d1 = datetime.fromisoformat(end).replace(tzinfo=UTC)
    stamps: list[str] = []
    day = d0
    while day <= d1:
        for hhmm in times:
            h, m = (int(x) for x in hhmm.split(":"))
            stamps.append(day.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for hhmm in next_times:
            h, m = (int(x) for x in hhmm.split(":"))
            nxt = day + timedelta(days=1)
            stamps.append(nxt.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        day += timedelta(days=1)
    return sorted(set(stamps))


def fetch_snapshot(api_key: str, when: str, markets: str = "h2h", attempts: int = 4) -> tuple[dict, int]:
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "date": when,
    }
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(BASE, params=params, timeout=30)
            if resp.status_code == 200:
                cost = int(resp.headers.get("x-requests-last", 0))
                return resp.json(), cost
            if resp.status_code in (401, 422):
                raise SystemExit(f"Odds API {resp.status_code}: {resp.text[:200]}")
            last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(3 * attempt)
    raise RuntimeError(f"snapshot {when} failed: {last}")


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-03-20")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--out", default="/tmp/odds_2024_stage.parquet")
    parser.add_argument("--dry-run-days", type=int, default=0,
                        help="only fetch this many days (validation)")
    parser.add_argument("--times", default="16:30,20:30",
                        help="same-day UTC snapshot times (comma HH:MM)")
    parser.add_argument("--next-times", default="00:30",
                        help="next-day UTC snapshot times for late games (comma HH:MM)")
    parser.add_argument("--markets", choices=("h2h", "totals"), default="h2h")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY not set in environment")

    times = [t for t in args.times.split(",") if t]
    next_times = [t for t in args.next_times.split(",") if t]
    stamps = snapshot_times(args.start, args.end, times, next_times)
    if args.dry_run_days:
        stamps = stamps[: args.dry_run_days * max(1, len(times) + len(next_times))]

    # per game_id -> best (latest snapshot <= commence) record
    best: dict[str, dict] = {}
    total_cost = 0
    for i, when in enumerate(stamps, 1):
        payload, cost = fetch_snapshot(api_key, when, args.markets)
        total_cost += cost
        snap_ts = payload.get("timestamp")
        snap_dt = _iso(snap_ts) if snap_ts else _iso(when)
        for game in payload.get("data", []):
            commence = game.get("commence_time")
            if not commence:
                continue
            if snap_dt > _iso(commence):
                continue  # snapshot after first pitch: not a pre-game line
            gid = game["id"]
            prev = best.get(gid)
            if prev is not None and snap_dt <= prev["_snap_dt"]:
                continue
            best[gid] = {
                "_snap_dt": snap_dt,
                "game_id": gid,
                "commence_time": commence,
                "away_team": game.get("away_team"),
                "home_team": game.get("home_team"),
                "snapshot_time": snap_ts,
                "bookmakers": game.get("bookmakers", []),
            }
        if i % 20 == 0 or i == len(stamps):
            print(
                f"[{i}/{len(stamps)}] {when}  games={len(best)}  credits={total_cost}",
                flush=True,
            )
        time.sleep(0.2)

    # explode per-book into rows (schema depends on market)
    rows: list[dict] = []
    for rec in best.values():
        for book in rec["bookmakers"]:
            mkt = next(
                (m for m in book.get("markets", []) if m.get("key") == args.markets),
                None,
            )
            if not mkt:
                continue
            commence_dt = _iso(rec["commence_time"])
            base = {
                "game_id": rec["game_id"],
                "commence_time": rec["commence_time"],
                "game_date_utc": commence_dt.date().isoformat(),
                "away_team": rec["away_team"],
                "home_team": rec["home_team"],
                "bookmaker": book["key"],
                "snapshot_time": rec["snapshot_time"],
                "book_last_update": mkt.get("last_update"),
            }
            if args.markets == "h2h":
                prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                home_ml = prices.get(rec["home_team"])
                away_ml = prices.get(rec["away_team"])
                if home_ml is None or away_ml is None:
                    continue
                rows.append({**base, "home_ml": int(home_ml), "away_ml": int(away_ml)})
            else:  # totals
                outs = {o.get("name"): o for o in mkt.get("outcomes", [])}
                over, under = outs.get("Over"), outs.get("Under")
                if not over or not under or over.get("point") is None:
                    continue
                rows.append({
                    **base,
                    "total_point": float(over["point"]),
                    "over_ml": int(over["price"]),
                    "under_ml": int(under["price"]),
                })

    frame = pl.DataFrame(rows)
    frame.write_parquet(args.out)
    n_games = frame["game_id"].n_unique() if frame.height else 0
    print(
        f"\nDONE: {frame.height} book-rows across {n_games} games -> {args.out}"
        f"  (credits used: {total_cost}, remaining budget check via headers)"
    )


if __name__ == "__main__":
    main()
