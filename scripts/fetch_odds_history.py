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


def snapshot_times(start: str, end: str) -> list[str]:
    d0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    d1 = datetime.fromisoformat(end).replace(tzinfo=UTC)
    stamps: list[str] = []
    day = d0
    while day <= d1:
        for hhmm in SNAPSHOT_HHMM:
            h, m = (int(x) for x in hhmm.split(":"))
            stamps.append(day.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for hhmm in NEXT_DAY_HHMM:
            h, m = (int(x) for x in hhmm.split(":"))
            nxt = day + timedelta(days=1)
            stamps.append(nxt.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%SZ"))
        day += timedelta(days=1)
    return sorted(set(stamps))


def fetch_snapshot(api_key: str, when: str, attempts: int = 4) -> tuple[dict, int]:
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
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
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY not set in environment")

    stamps = snapshot_times(args.start, args.end)
    if args.dry_run_days:
        stamps = stamps[: args.dry_run_days * (len(SNAPSHOT_HHMM) + len(NEXT_DAY_HHMM))]

    # per game_id -> best (latest snapshot <= commence) record
    best: dict[str, dict] = {}
    total_cost = 0
    for i, when in enumerate(stamps, 1):
        payload, cost = fetch_snapshot(api_key, when)
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

    # explode per-book into rows
    rows: list[dict] = []
    for rec in best.values():
        for book in rec["bookmakers"]:
            h2h = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            prices = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            home_ml = prices.get(rec["home_team"])
            away_ml = prices.get(rec["away_team"])
            if home_ml is None or away_ml is None:
                continue
            commence_dt = _iso(rec["commence_time"])
            rows.append({
                "game_id": rec["game_id"],
                "commence_time": rec["commence_time"],
                "game_date_utc": commence_dt.date().isoformat(),
                "away_team": rec["away_team"],
                "home_team": rec["home_team"],
                "bookmaker": book["key"],
                "home_ml": int(home_ml),
                "away_ml": int(away_ml),
                "snapshot_time": rec["snapshot_time"],
                "book_last_update": h2h.get("last_update"),
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
