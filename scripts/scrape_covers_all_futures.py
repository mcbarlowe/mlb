#!/usr/bin/env python3
"""Scrape all futures markets from Covers.com for backtesting.

Fetches championship, AL/NL pennant, division, and playoff odds.

Usage:
    uv run python scripts/scrape_covers_all_futures.py --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

COVERS_BASE_URL = "https://www.covers.com/sportsoddshistory/mlb-main/"
COVERS_DIV_URL = "https://www.covers.com/sportsoddshistory/mlb-div/"

MARKET_CONFIGS = {
    "championship": {"url_template": "mlb-main/?y={year}&sa=mlb&a=ws", "base_url": COVERS_BASE_URL},
    "al_pennant": {"url_template": "mlb-main/?y={year}&sa=mlb&a=al", "base_url": COVERS_BASE_URL},
    "nl_pennant": {"url_template": "mlb-main/?y={year}&sa=mlb&a=nl", "base_url": COVERS_BASE_URL},
    "division": {"url_template": "mlb-div/?y={year}&sa=mlb&a=div", "base_url": COVERS_DIV_URL},
}


def _parse_american_odds(odds_str: str) -> int | None:
    """Parse American odds from string like '+550' or '-130'."""
    if not odds_str or odds_str.strip() == "":
        return None
    
    odds_str = odds_str.strip()
    
    try:
        return int(odds_str.replace(",", ""))
    except ValueError:
        return None


def _preseason_index(table) -> tuple[int | None, str]:
    """Index of the final preseason odds column, and its date label.

    The table carries two header rows. The first holds group labels with colspans, e.g.
    ``Team(1) | Preseason(4) | Regular season(6) | Result(1)``; the second holds a date per
    column. The wanted column is the *last* preseason one, the snapshot closest to opening day.

    Reading the group row's colspan is the only reliable way to find that boundary. The count of
    preseason snapshots varies by season - 2025 carried five, 2026 carries four - so any rule based
    on matching a March date, or on taking the first column after ``Team``, silently reads the wrong
    snapshot. Taking the first column reads the previous October, roughly five months stale, and
    returns nothing at all in a season whose earliest column is blank.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return None, ""
    group_cells = rows[0].find_all(["th", "td"])
    span = 0
    for cell in group_cells:
        label = cell.get_text(strip=True)
        width = int(cell.get("colspan", 1) or 1)
        if "Preseason" in label:
            span = width
            break
    if span < 1:
        return None, ""
    date_cells = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
    # Date row has no Team cell, so its index i corresponds to data-cell index i + 1.
    label = date_cells[span - 1] if span - 1 < len(date_cells) else ""
    return span, label


def _iter_tables(soup):
    """Yield (table, section_heading) for every odds table on the page.

    Division pages carry one table per division, so reading only the first table drops five
    sixths of the market.
    """
    for table in soup.find_all("table"):
        heading = table.find_previous(["h1", "h2", "h3", "h4", "strong", "b"])
        yield table, heading.get_text(strip=True) if heading else ""


def scrape_market(season: int, market_type: str) -> list[dict[str, object]]:
    """Scrape one futures market for a season, taking the final preseason snapshot."""
    config = MARKET_CONFIGS[market_type]
    url = config["base_url"] + config["url_template"].format(year=season)

    print(f"  Fetching {market_type}: {url}")

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 Chrome/120 Safari/537.36"},
        )
        response.raise_for_status()
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, object]] = []

    for table, section in _iter_tables(soup):
        preseason_idx, snapshot_str = _preseason_index(table)
        if preseason_idx is None:
            continue
        for row in table.find_all("tr")[2:]:
            cells = row.find_all(["th", "td"])
            if len(cells) <= preseason_idx:
                continue
            team_name = cells[0].get_text(strip=True)
            if (
                not team_name
                or "**" in team_name
                or "WINNER" in team_name
                or "Division" in team_name
            ):
                continue
            odds = _parse_american_odds(cells[preseason_idx].get_text(strip=True))
            if odds is None:
                continue
            division = section or None
            if "(" in team_name and ")" in team_name:
                head, _, tail = team_name.partition("(")
                team_name = head.strip()
                division = tail.replace(")", "").strip()
            row_data: dict[str, object] = {
                "team_name": team_name,
                "american_odds": odds,
                "snapshot_date": snapshot_str,
            }
            if division:
                row_data["division"] = division
            rows.append(row_data)

    print(f"    Found {len(rows)} teams (final preseason column: {snapshot_str or 'n/a'})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Seasons to scrape (default: 2022-2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=list(MARKET_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which markets to scrape",
    )
    args = parser.parse_args()
    
    if "all" in args.markets:
        markets = list(MARKET_CONFIGS.keys())
    else:
        markets = args.markets
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rows_by_market = {market: [] for market in markets}
    
    for season in sorted(args.seasons):
        print(f"\nSeason {season}:")
        
        for market_type in markets:
            market_rows = scrape_market(season, market_type)
            
            # Add metadata
            for row in market_rows:
                row["season"] = season
                row["market_type"] = market_type
                row["bookmaker"] = "covers-consensus"
                row["source"] = "covers.com"
                row["snapshot_time"] = f"{season}-03-25T00:00:00Z"
            
            all_rows_by_market[market_type].extend(market_rows)
    
    # Write separate CSV for each market
    for market_type, rows in all_rows_by_market.items():
        if not rows:
            print(f"\nNo data for {market_type}, skipping")
            continue
        
        output_file = args.output_dir / f"covers_{market_type}_{min(args.seasons)}-{max(args.seasons)}.csv"
        
        fieldnames = ["season", "market_type", "team_name", "bookmaker", "american_odds", "snapshot_time", "source"]
        if any("division" in row for row in rows):
            fieldnames.insert(3, "division")
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Wrote {len(rows)} {market_type} odds to {output_file}")
        
        # Show breakdown
        from collections import Counter
        season_counts = Counter(row["season"] for row in rows)
        print(f"  Breakdown: {dict(sorted(season_counts.items()))}")
    
    # Also write combined file
    all_rows = []
    for rows in all_rows_by_market.values():
        all_rows.extend(rows)
    
    if all_rows:
        combined_file = args.output_dir / f"covers_all_futures_{min(args.seasons)}-{max(args.seasons)}.csv"
        
        fieldnames = ["season", "market_type", "team_name", "bookmaker", "american_odds", "snapshot_time", "source"]
        
        with open(combined_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\n✓ Wrote {len(all_rows)} total odds to {combined_file}")


if __name__ == "__main__":
    main()
