#!/usr/bin/env python3
"""Reload futures odds using the true final-preseason snapshot for each season.

Six of twelve division-futures seasons in ``mlb.futures_odds`` hold in-season prices rather than
preseason ones, each matching a specific later column 30/30:

    2013 -> Jun 1    2014 -> May 1    2017 -> May 1
    2021 -> Jun 1    2022 -> Jun 1    2023 -> May 1

The cause was ``_find_preseason_column`` in ``scripts/scrape_covers_all_futures.py``, which only
ever saw the group-header row, found no month name in it, and fell through to "first column after
Team". The number of preseason snapshots and the number of blank leading columns both vary by
season, so that rule landed on May or June prices. It is fixed to read the ``Preseason`` group
colspan and take the *last* preseason column, and the fixed parser reproduces all six clean
seasons exactly, 30/30 each.

Comparing a March-15 model projection against a June-1 price is not a small error. By June the
market has two months of results the model does not, so disagreement is dominated by information
the model is missing rather than by mispricing.

This writes corrected rows rather than deleting anything. ``load_latest_futures_odds`` selects
``MAX(snapshot_time)``, and every true preseason date falls after the ``03-24`` label the existing
rows carry, so the corrected rows supersede them while the originals stay available for audit.

    uv run python scripts/reload_futures_preseason_odds.py                 # dry run
    uv run python scripts/reload_futures_preseason_odds.py --write
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_historical_futures_from_csv import _load_team_id_map
from scripts.scrape_covers_all_futures import scrape_market
from src.database import PostgresConfig, PostgresHandler
from src.market_data.futures_odds_store import (
    ensure_futures_odds_table,
    insert_futures_odds,
    load_latest_futures_odds,
)
from src.market_data.pricing import american_to_prob

SEASONS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
MARKETS = ["division", "championship", "al_pennant", "nl_pennant"]
SOURCE = "covers.com-preseason"
MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_snapshot(label: str, season: int) -> datetime | None:
    """Turn a column label such as 'Mar 25' into a timestamp in the season's year.

    A preseason label naming October through December belongs to the prior calendar year, so it is
    rejected: the wanted snapshot is the last one before opening day, which is always in spring.
    """
    parts = label.replace(",", " ").split()
    if len(parts) < 2:
        return None
    month = MONTHS.get(parts[0][:3])
    if month is None or month >= 10:
        return None
    try:
        day = int(parts[1])
    except ValueError:
        return None
    return datetime(season, month, day, tzinfo=UTC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Commit rows to the database.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    ap.add_argument("--markets", default=",".join(MARKETS))
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s]
    markets = [m for m in args.markets.split(",") if m]

    team_ids = _load_team_id_map()
    staged: list[dict[str, object]] = []
    unmatched: set[str] = set()

    print(f"{'season':>7} | {'market':>13} | {'teams':>5} | {'snapshot':>12} | status")
    print("-" * 66)
    for season in seasons:
        for market in markets:
            rows = scrape_market(season, market)
            if not rows:
                print(f"{season:>7} | {market:>13} | {0:5d} |            - | no rows")
                continue
            label = str(rows[0].get("snapshot_date", ""))
            stamp = parse_snapshot(label, season)
            if stamp is None:
                print(f"{season:>7} | {market:>13} | {len(rows):5d} | {label:>12} | "
                      f"REJECTED, not a spring date")
                continue
            kept = 0
            for row in rows:
                name = str(row["team_name"]).upper().strip()
                team_id = team_ids.get(name)
                if team_id is None:
                    unmatched.add(str(row["team_name"]))
                    continue
                odds = int(row["american_odds"])
                staged.append(
                    {
                        "season": season,
                        "market_type": market,
                        "team_id": team_id,
                        "team_name": row["team_name"],
                        "bookmaker": "covers-consensus",
                        "american_odds": odds,
                        "implied_probability": american_to_prob(odds),
                        "snapshot_time": stamp,
                        "source": SOURCE,
                    }
                )
                kept += 1
            print(f"{season:>7} | {market:>13} | {kept:5d} | "
                  f"{stamp.date().isoformat():>12} | staged")

    print()
    print(f"staged {len(staged)} rows across {len(seasons)} seasons and {len(markets)} markets")
    if unmatched:
        print(f"unmatched team names ({len(unmatched)}): {sorted(unmatched)[:12]}")

    if not args.write:
        print()
        print("dry run, nothing written. Re-run with --write to commit.")
        return

    cfg = PostgresConfig.from_env()
    with PostgresHandler(cfg) as pg:
        ensure_futures_odds_table(pg)
        inserted = insert_futures_odds(pg, staged)
        print(f"inserted {inserted} rows with source={SOURCE!r}")
        print()
        print("Verifying that load_latest_futures_odds now resolves the corrected snapshot:")
        print(f"  {'season':>7} | {'snapshot':>12} | {'source':>24} | teams")
        print("  " + "-" * 58)
        for season in seasons:
            got = load_latest_futures_odds(pg, season=season, market_type="division")
            if not got:
                print(f"  {season:>7} |            - |                        - | 0")
                continue
            stamp = got[0]["snapshot_time"]
            src = str(got[0]["source"])
            flag = "" if src == SOURCE else "   <-- STILL STALE"
            print(f"  {season:>7} | {str(stamp)[:10]:>12} | {src:>24} | {len(got)}{flag}")


if __name__ == "__main__":
    main()
