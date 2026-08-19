#!/usr/bin/env python3
"""Fetch historical MLB futures odds for backtesting.

Uses the-odds-api historical endpoint to retrieve preseason futures odds
for past seasons. Useful for backtesting model performance on championship,
division, and playoff futures markets.

Usage:
    uv run python scripts/fetch_futures_odds_history.py --season 2022 --date 2022-03-20 --dry-run
    uv run python scripts/fetch_futures_odds_history.py --season 2022 --date 2022-03-20 --db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.fetch_futures_odds import (
    MARKET_TYPE_MAP,
    TEAM_NAME_ALIASES,
    _normalize_market_type,
    _normalize_team_name,
)
from src.betting.futures_odds_store import (
    ensure_futures_odds_table,
    insert_futures_odds,
)
from src.betting.odds import american_to_prob
from src.database import PostgresConfig, PostgresHandler

HISTORICAL_OUTRIGHTS_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/outrights"


def _fetch_historical_futures(
    api_key: str,
    *,
    date: str,
    regions: str = "us",
) -> tuple[object, dict[str, str]]:
    """Fetch historical futures odds for a specific date.
    
    Args:
        api_key: the-odds-api key
        date: ISO date (YYYY-MM-DD) for historical snapshot
        regions: Comma-separated regions
    
    Returns:
        (response_payload, headers)
    """
    response = requests.get(
        HISTORICAL_OUTRIGHTS_URL,
        params={
            "apiKey": api_key,
            "regions": regions,
            "date": f"{date}T12:00:00Z",  # Noon UTC on the target date
            "dateFormat": "iso",
        },
        timeout=60,
    )
    response.raise_for_status()
    
    payload = response.json()
    return payload, dict(response.headers)


def _parse_historical_futures(
    payload: object,
    *,
    season: int,
    snapshot_date: str,
    team_id_map: dict[str, int],
) -> list[dict[str, object]]:
    """Parse historical outrights response into futures_odds rows.
    
    Similar to _parse_futures_odds but for historical data.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload, got {type(payload).__name__}")
    
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Expected 'data' to be a list")
    
    rows: list[dict[str, object]] = []
    
    # Use the timestamp from the API response if available
    timestamp = payload.get("timestamp", snapshot_date)
    
    for event in data:
        if not isinstance(event, dict):
            continue
        
        sport_key = event.get("sport_key")
        if sport_key != "baseball_mlb":
            continue
        
        bookmakers = event.get("bookmakers", [])
        
        for bookmaker_data in bookmakers:
            if not isinstance(bookmaker_data, dict):
                continue
            
            bookmaker = bookmaker_data.get("key", "unknown")
            markets = bookmaker_data.get("markets", [])
            
            for market in markets:
                if not isinstance(market, dict):
                    continue
                
                market_key = market.get("key", "")
                market_type = _normalize_market_type(market_key)
                
                if market_type is None:
                    continue
                
                outcomes = market.get("outcomes", [])
                
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    
                    team_name_raw = outcome.get("name", "")
                    team_name = _normalize_team_name(team_name_raw)
                    team_id = team_id_map.get(team_name)
                    
                    if team_id is None:
                        print(
                            f"Warning: unknown team {team_name_raw!r} in {market_key}",
                            file=sys.stderr,
                        )
                        continue
                    
                    american_odds = outcome.get("price")
                    if american_odds is None:
                        continue
                    
                    try:
                        implied_prob = american_to_prob(int(american_odds))
                    except (ValueError, TypeError):
                        print(
                            f"Warning: invalid odds {american_odds} for {team_name}",
                            file=sys.stderr,
                        )
                        continue
                    
                    rows.append({
                        "season": season,
                        "market_type": market_type,
                        "team_id": team_id,
                        "team_name": team_name,
                        "bookmaker": bookmaker,
                        "american_odds": int(american_odds),
                        "implied_probability": implied_prob,
                        "snapshot_time": timestamp,
                        "source": "the-odds-api-historical",
                    })
    
    return rows


def _load_team_id_map() -> dict[str, int]:
    """Load team name to team_id mapping."""
    from src.betting.ingest import team_abbrev_to_id
    
    mapping = team_abbrev_to_id()
    
    name_to_id: dict[str, int] = {}
    for key, team_id in mapping.items():
        name_to_id[key] = team_id
    
    return name_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year for these futures",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Historical date to fetch (YYYY-MM-DD), e.g. 2022-03-20 for opening day",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Comma-separated regions (default: us)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but don't write to database",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write to Postgres mlb.futures_odds table",
    )
    parser.add_argument(
        "--response-json",
        type=Path,
        help="Save API response to this JSON file",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Read from saved JSON instead of API (for testing)",
    )
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
    
    # Get API key
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key and args.from_json is None:
        raise SystemExit("ODDS_API_KEY environment variable required unless --from-json provided")
    
    # Fetch odds
    if args.from_json:
        print(f"Reading from {args.from_json}")
        payload = json.loads(args.from_json.read_text())
        headers = {}
    else:
        print(f"Fetching historical futures for {args.date}")
        payload, headers = _fetch_historical_futures(
            api_key,
            date=args.date,
            regions=args.regions,
        )
        
        if args.response_json:
            args.response_json.write_text(json.dumps(payload, indent=2))
            print(f"Saved response to {args.response_json}")
        
        print(f"API response received")
        if "x-requests-remaining" in headers:
            print(f"  Requests remaining: {headers['x-requests-remaining']}")
        if "x-requests-used" in headers:
            print(f"  Requests used: {headers['x-requests-used']}")
    
    # Parse
    team_id_map = _load_team_id_map()
    
    rows = _parse_historical_futures(
        payload,
        season=args.season,
        snapshot_date=f"{args.date}T12:00:00Z",
        team_id_map=team_id_map,
    )
    
    print(f"Parsed {len(rows)} historical futures odds rows")
    
    if not rows:
        print("Warning: No futures odds found for this date")
        print("Historical futures may not be available for all dates")
        return
    
    # Show breakdown by market type
    from collections import Counter
    market_counts = Counter(row["market_type"] for row in rows)
    print(f"\nBreakdown by market:")
    for market_type, count in sorted(market_counts.items()):
        print(f"  {market_type}: {count} odds")
    
    if args.dry_run:
        print("\nSample rows:")
        for row in rows[:10]:
            print(
                f"  {row['market_type']:20s} {row['team_name']:25s} "
                f"{row['bookmaker']:15s} {row['american_odds']:>6d}"
            )
        print("\n--dry-run: not writing to database")
        return
    
    if not args.db:
        print("Use --db to write to database or --dry-run to test")
        return
    
    # Write to database
    db_config = PostgresConfig.from_env()
    
    with PostgresHandler(db_config) as pg:
        ensure_futures_odds_table(pg)
        inserted = insert_futures_odds(pg, rows)
        print(f"\nInserted/updated {inserted} historical futures odds in mlb.futures_odds")


if __name__ == "__main__":
    main()
