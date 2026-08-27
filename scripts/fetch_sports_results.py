"""Fetch normalized NBA, NCAAF, and NHL final scores from ESPN.

The Odds API historical odds endpoint does not provide historical final scores.
This script fills that side of the backtest join with ESPN event results and a
single normalized CSV contract.

Example:
    uv run python scripts/fetch_sports_results.py \
      --sports nba ncaaf nhl \
      --start 2024-01-01 --end 2024-01-31 \
      --output data/sports_results/results_2024_01.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.results import (
    SPORT_CONFIGS,
    CompletedGameResult,
    date_range,
    event_ids_from_core_payload,
    parse_espn_summary,
)

CORE_BASE = "https://sports.core.api.espn.com/v2/sports/{path}/events"
SUMMARY_BASE = "https://site.api.espn.com/apis/site/v2/sports/{path}/summary"

FIELDS = [
    "sport",
    "source",
    "source_event_id",
    "season",
    "week",
    "game_date",
    "commence_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_won",
    "neutral_site",
    "status",
]


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    cache_path: Path,
    refresh: bool,
    attempts: int = 4,
) -> dict[str, Any]:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                payload = response.json()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, sort_keys=True))
                return payload
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
        time.sleep(attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_event_ids(
    session: requests.Session,
    *,
    sport: str,
    target_date: date,
    cache_dir: Path,
    refresh: bool,
) -> list[str]:
    config = SPORT_CONFIGS[sport]
    payload = fetch_json(
        session,
        CORE_BASE.format(path=config.espn_core_path),
        params={"dates": target_date.strftime("%Y%m%d"), "limit": 1000},
        cache_path=cache_dir / sport / f"{target_date:%Y%m%d}_events.json",
        refresh=refresh,
    )
    return event_ids_from_core_payload(payload)


def fetch_summary(
    session: requests.Session,
    *,
    sport: str,
    event_id: str,
    target_date: date,
    cache_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    config = SPORT_CONFIGS[sport]
    return fetch_json(
        session,
        SUMMARY_BASE.format(path=config.espn_site_path),
        params={"event": event_id},
        cache_path=cache_dir / sport / f"{target_date:%Y%m%d}_{event_id}_summary.json",
        refresh=refresh,
    )


def fetch_results(
    *,
    sports: list[str],
    start: date,
    end: date,
    cache_dir: Path,
    refresh: bool,
    include_incomplete: bool,
    sleep_seconds: float,
    limit_days: int | None,
) -> list[CompletedGameResult]:
    days = date_range(start, end)
    if limit_days is not None:
        days = days[:limit_days]

    results: list[CompletedGameResult] = []
    with requests.Session() as session:
        for sport in sports:
            for target_date in days:
                event_ids = fetch_event_ids(
                    session,
                    sport=sport,
                    target_date=target_date,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
                print(f"{SPORT_CONFIGS[sport].code} {target_date:%Y-%m-%d}: {len(event_ids)} events")
                for event_id in event_ids:
                    payload = fetch_summary(
                        session,
                        sport=sport,
                        event_id=event_id,
                        target_date=target_date,
                        cache_dir=cache_dir,
                        refresh=refresh,
                    )
                    parsed = parse_espn_summary(
                        payload,
                        sport=SPORT_CONFIGS[sport].code,
                        include_incomplete=include_incomplete,
                    )
                    if parsed is not None:
                        results.append(parsed)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
    return sorted(results, key=lambda row: (row.sport, row.game_date, row.commence_time, row.home_team, row.away_team))


def write_results(path: Path, results: list[CompletedGameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["game_date"] = result.game_date.isoformat()
            writer.writerow(row)
    print(f"\nwrote results -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sports", nargs="+", choices=sorted(SPORT_CONFIGS), default=sorted(SPORT_CONFIGS))
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=Path("data/sports_results/espn_results.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/sports_results/espn_scoreboards"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit-days", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sleep < 0.0:
        raise SystemExit("--sleep must be non-negative")
    if args.limit_days is not None and args.limit_days < 1:
        raise SystemExit("--limit-days must be positive")

    results = fetch_results(
        sports=list(args.sports),
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        include_incomplete=args.include_incomplete,
        sleep_seconds=args.sleep,
        limit_days=args.limit_days,
    )
    write_results(args.output, results)
    print(f"completed games: {len(results)}")


if __name__ == "__main__":
    main()
