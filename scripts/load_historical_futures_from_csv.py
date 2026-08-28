#!/usr/bin/env python3
"""Load historical futures odds from CSV files into mlb.futures_odds.

Reads CSV files with columns: season, market_type, team_name, bookmaker,
american_odds, snapshot_time, source.

Usage:
    uv run python scripts/load_historical_futures_from_csv.py data/historical_futures_odds_sample.csv --dry-run
    uv run python scripts/load_historical_futures_from_csv.py data/historical_futures_odds_sample.csv --db
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig, PostgresHandler
from src.market_data.futures_odds_store import (
    ensure_futures_odds_table,
    insert_futures_odds,
)
from src.market_data.pricing import american_to_prob
from src.market_data.team_mapping import team_abbrev_to_id


def _load_team_id_map() -> dict[str, int]:
    """Load team name to team_id mapping."""
    mapping = team_abbrev_to_id()
    
    # Build name lookup
    name_to_id: dict[str, int] = {}
    for key, team_id in mapping.items():
        name_to_id[key] = team_id
    
    # Add common aliases
    name_to_id["ST LOUIS CARDINALS"] = name_to_id.get("ST. LOUIS CARDINALS", 138)
    
    # Historical team names
    if "CLEVELAND GUARDIANS" in name_to_id:
        name_to_id["CLEVELAND INDIANS"] = name_to_id["CLEVELAND GUARDIANS"]  # 2022 rename
    if "MIAMI MARLINS" in name_to_id:
        name_to_id["FLORIDA MARLINS"] = name_to_id["MIAMI MARLINS"]  # 2012 rename
    
    return name_to_id


def _parse_csv(csv_path: Path, team_id_map: dict[str, int]) -> list[dict[str, object]]:
    """Parse historical futures odds CSV."""
    rows = []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        
        for row_dict in reader:
            team_name = row_dict["team_name"]
            team_id = team_id_map.get(team_name.upper())
            
            if team_id is None:
                print(f"Warning: unknown team {team_name!r}, skipping", file=sys.stderr)
                continue
            
            american_odds = int(row_dict["american_odds"])
            implied_prob = american_to_prob(american_odds)
            
            rows.append({
                "season": int(row_dict["season"]),
                "market_type": row_dict["market_type"],
                "team_id": team_id,
                "team_name": team_name,
                "bookmaker": row_dict["bookmaker"],
                "american_odds": american_odds,
                "implied_probability": implied_prob,
                "snapshot_time": row_dict["snapshot_time"],
                "source": row_dict.get("source", "manual-csv"),
            })
    
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV file with historical futures odds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display but don't write to database",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write to Postgres mlb.futures_odds table",
    )
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        raise SystemExit(f"CSV file not found: {args.csv_file}")
    
    print(f"Loading historical futures odds from {args.csv_file}")
    
    team_id_map = _load_team_id_map()
    rows = _parse_csv(args.csv_file, team_id_map)
    
    print(f"Parsed {len(rows)} historical futures odds rows")
    
    if not rows:
        print("No valid rows found")
        return
    
    # Show breakdown
    from collections import Counter
    
    season_counts = Counter(row["season"] for row in rows)
    market_counts = Counter(row["market_type"] for row in rows)
    
    print("\nBreakdown by season:")
    for season, count in sorted(season_counts.items()):
        print(f"  {season}: {count} odds")
    
    print("\nBreakdown by market:")
    for market_type, count in sorted(market_counts.items()):
        print(f"  {market_type}: {count} odds")
    
    if args.dry_run:
        print("\nSample rows:")
        for row in rows[:10]:
            print(
                f"  {row['season']} {row['market_type']:20s} {row['team_name']:25s} "
                f"{row['american_odds']:>7d} -> {row['implied_probability']:.4f}"
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
