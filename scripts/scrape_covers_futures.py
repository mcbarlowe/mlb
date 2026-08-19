#!/usr/bin/env python3
"""Scrape historical futures odds from Covers.com Sports Odds History.

Fetches preseason championship odds for backtest seasons 2022-2025.

Usage:
    uv run python scripts/scrape_covers_futures.py --season 2024 --output data/covers_2024_championship.csv
    uv run python scripts/scrape_covers_futures.py --seasons 2022 2023 2024 2025 --output data/covers_all_championship.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

COVERS_BASE_URL = "https://www.covers.com/sportsoddshistory/mlb-main/"


def _parse_american_odds(odds_str: str) -> int | None:
    """Parse American odds from string like '+550' or '-130'."""
    if not odds_str or odds_str.strip() == "":
        return None
    
    # Remove any whitespace
    odds_str = odds_str.strip()
    
    # Try to parse as integer (with + or - prefix)
    try:
        return int(odds_str.replace(",", ""))
    except ValueError:
        return None


def scrape_championship_odds(season: int) -> list[dict[str, object]]:
    """Scrape championship odds for a season from Covers.com.
    
    Returns list of dicts with keys:
        team_name, preseason_odds, snapshot_date
    """
    url = f"{COVERS_BASE_URL}?y={season}&sa=mlb&a=ws"
    
    print(f"Fetching {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the odds table
    table = soup.find("table")
    if not table:
        raise ValueError(f"No table found on page for season {season}")
    
    rows = []
    header_row = None
    
    # Find header row to identify columns
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        
        # Check if this is the header row (contains "Preseason", "Regular season", etc.)
        first_cell_text = cells[0].get_text(strip=True)
        if "Team" in first_cell_text or first_cell_text == "":
            # This is likely the header
            header_texts = [cell.get_text(strip=True) for cell in cells]
            
            # Find preseason columns - look for dates in March/April
            preseason_col_idx = None
            for idx, text in enumerate(header_texts):
                # Look for dates like "Mar 28" which is typically final preseason
                if "Mar" in text or "Apr" in text:
                    # Take the last March/early April date as preseason
                    preseason_col_idx = idx
                    snapshot_date_str = text
                    break
            
            if preseason_col_idx is None:
                # Fall back to first numbered column after "Team"
                for idx in range(1, len(header_texts)):
                    if header_texts[idx]:
                        preseason_col_idx = idx
                        snapshot_date_str = header_texts[idx]
                        break
            
            header_row = (header_texts, preseason_col_idx, snapshot_date_str)
            continue
        
        # Skip if we haven't found header yet
        if header_row is None:
            continue
        
        _, preseason_idx, snapshot_str = header_row
        
        # This is a data row
        team_name_cell = cells[0]
        team_name = team_name_cell.get_text(strip=True)
        
        # Skip empty rows or result rows
        if not team_name or "**" in team_name or "WINNER" in team_name:
            continue
        
        # Get preseason odds
        if preseason_idx and preseason_idx < len(cells):
            odds_cell = cells[preseason_idx]
            odds_str = odds_cell.get_text(strip=True)
            odds = _parse_american_odds(odds_str)
            
            if odds is not None:
                rows.append({
                    "team_name": team_name,
                    "american_odds": odds,
                    "snapshot_date": snapshot_str,
                })
    
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        help="Single season to scrape",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Multiple seasons to scrape",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/covers_championship_odds.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = sorted(args.seasons)
    else:
        # Default: scrape backtest seasons
        seasons = [2022, 2023, 2024, 2025]
    
    all_rows = []
    
    for season in seasons:
        try:
            season_rows = scrape_championship_odds(season)
            print(f"  Found {len(season_rows)} teams with odds for {season}")
            
            # Add season column
            for row in season_rows:
                row["season"] = season
                row["market_type"] = "championship"
                row["bookmaker"] = "covers-consensus"
                row["source"] = "covers.com"
                
                # Convert snapshot_date to ISO format (approximate)
                # Most preseason odds are from late March
                row["snapshot_time"] = f"{season}-03-25T00:00:00Z"
            
            all_rows.extend(season_rows)
        
        except Exception as e:
            print(f"  Error scraping {season}: {e}", file=sys.stderr)
            continue
    
    if not all_rows:
        print("No odds found")
        return
    
    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["season", "market_type", "team_name", "bookmaker", "american_odds", "snapshot_time", "source"]
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    
    print(f"\nWrote {len(all_rows)} odds to {args.output}")
    
    # Show breakdown
    from collections import Counter
    season_counts = Counter(row["season"] for row in all_rows)
    print(f"\nBreakdown by season:")
    for season in sorted(season_counts.keys()):
        print(f"  {season}: {season_counts[season]} teams")


if __name__ == "__main__":
    main()
