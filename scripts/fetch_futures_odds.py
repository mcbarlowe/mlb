#!/usr/bin/env python3
"""Fetch MLB futures odds from the-odds-api and store in Postgres.

Fetches championship, division, and playoff futures (outrights) for the
upcoming MLB season. Requires ODDS_API_KEY environment variable.

Usage:
    uv run python scripts/fetch_futures_odds.py --season 2027 --dry-run
    uv run python scripts/fetch_futures_odds.py --season 2027 --db
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.futures_odds_store import (
    ensure_futures_odds_table,
    insert_futures_odds,
)
from src.betting.odds import american_to_prob
from src.database import PostgresConfig, PostgresHandler

OUTRIGHTS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/outrights"

# Map the-odds-api market names to our normalized market_type
MARKET_TYPE_MAP = {
    "h2h": None,  # Ignore head-to-head (not a future)
    "outrights": "championship",  # Default outright = World Series
    "world series winner": "championship",
    "world series": "championship",
    "championship": "championship",
    "to win world series": "championship",
    "american league winner": "league_championship",
    "national league winner": "league_championship",
    "league championship": "league_championship",
    "to make playoffs": "playoff",
    "playoff": "playoff",
    "postseason": "playoff",
    "division winner": "division",
    "to win division": "division",
}

# Team name normalization (handle API inconsistencies)
TEAM_NAME_ALIASES = {
    "Cleveland Guardians": "Cleveland Guardians",
    "Cleveland Indians": "Cleveland Guardians",
    "Miami Marlins": "Miami Marlins",
    "Florida Marlins": "Miami Marlins",
    "Los Angeles Angels": "Los Angeles Angels",
    "Anaheim Angels": "Los Angeles Angels",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
}


def _fetch_futures_odds(
    api_key: str,
    *,
    regions: str = "us",
    max_markets: int | None = None,
) -> tuple[object, dict[str, str]]:
    """Fetch futures/outrights from the-odds-api.
    
    Returns:
        (response_payload, headers)
    """
    response = requests.get(
        OUTRIGHTS_URL,
        params={
            "apiKey": api_key,
            "regions": regions,
            "dateFormat": "iso",
        },
        timeout=30,
    )
    response.raise_for_status()
    
    payload = response.json()
    
    # Optionally limit markets for testing
    if max_markets is not None and isinstance(payload, list):
        payload = payload[:max_markets]
    
    return payload, dict(response.headers)


def _normalize_market_type(raw_market: str) -> str | None:
    """Map API market name to our market_type enum.
    
    Returns None if this isn't a futures market we track.
    """
    normalized = raw_market.strip().lower()
    return MARKET_TYPE_MAP.get(normalized)


def _normalize_team_name(name: str) -> str:
    """Apply team name aliases."""
    return TEAM_NAME_ALIASES.get(name.strip(), name.strip())


def _parse_futures_odds(
    payload: object,
    *,
    season: int,
    snapshot_time: str,
    team_id_map: dict[str, int],
) -> list[dict[str, object]]:
    """Parse outrights response into futures_odds rows.
    
    Args:
        payload: JSON response from the-odds-api outrights endpoint
        season: Season year these futures are for
        snapshot_time: ISO timestamp of this snapshot
        team_id_map: {team_name: team_id} mapping
    
    Returns:
        List of row dicts ready for insertion
    """
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload, got {type(payload).__name__}")
    
    rows: list[dict[str, object]] = []
    
    for event in payload:
        if not isinstance(event, dict):
            continue
        
        # Each event is a futures market (e.g. "World Series Winner")
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
                    # Not a futures market we track
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
                        "snapshot_time": snapshot_time,
                        "source": "the-odds-api",
                    })
    
    return rows


def _load_team_id_map() -> dict[str, int]:
    """Load team name to team_id mapping from resources."""
    # Use the same mapping as other odds scripts
    from src.betting.ingest import team_abbrev_to_id
    
    mapping = team_abbrev_to_id()
    
    # Invert to {team_name: team_id}
    name_to_id: dict[str, int] = {}
    for key, team_id in mapping.items():
        # The mapping includes abbreviations and full names as keys
        name_to_id[key] = team_id
    
    return name_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year for these futures (e.g. 2027)",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Comma-separated regions (default: us)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        help="Limit number of markets for testing",
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
        "--odds-json",
        type=Path,
        help="Read from file instead of API (for testing)",
    )
    parser.add_argument(
        "--db-log",
        type=Path,
        help="Write database URI to this file before connecting",
    )
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key and args.odds_json is None:
        raise SystemExit("ODDS_API_KEY environment variable required unless --odds-json provided")
    
    # Fetch odds
    if args.odds_json:
        print(f"Reading odds from {args.odds_json}")
        payload = json.loads(args.odds_json.read_text())
        headers = {}
    else:
        print(f"Fetching futures odds from {OUTRIGHTS_URL}")
        payload, headers = _fetch_futures_odds(
            api_key,
            regions=args.regions,
            max_markets=args.max_markets,
        )
        print(f"API response: {len(payload)} markets")
        if "x-requests-remaining" in headers:
            print(f"  Requests remaining: {headers['x-requests-remaining']}")
    
    # Parse
    snapshot_time = datetime.now(UTC).isoformat()
    team_id_map = _load_team_id_map()
    
    rows = _parse_futures_odds(
        payload,
        season=args.season,
        snapshot_time=snapshot_time,
        team_id_map=team_id_map,
    )
    
    print(f"Parsed {len(rows)} futures odds rows")
    
    if args.dry_run:
        print("\nSample rows:")
        for row in rows[:5]:
            print(f"  {row['market_type']:20s} {row['team_name']:25s} {row['bookmaker']:15s} {row['american_odds']:>6d}")
        print("\n--dry-run: not writing to database")
        return
    
    if not args.db:
        print("Use --db to write to database or --dry-run to test")
        return
    
    # Write to database
    db_config = PostgresConfig.from_env()
    if args.db_log:
        args.db_log.write_text(db_config.uri())
    
    with PostgresHandler(db_config) as pg:
        ensure_futures_odds_table(pg)
        inserted = insert_futures_odds(pg, rows)
        print(f"Inserted/updated {inserted} futures odds rows in mlb.futures_odds")


if __name__ == "__main__":
    main()
