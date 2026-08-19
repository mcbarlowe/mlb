#!/usr/bin/env python3
"""Batch fetch historical futures odds for backtest seasons 2022-2025.

Fetches preseason futures odds for each season to enable backtesting
the model's championship, division, and playoff predictions.

Fetches multiple snapshots per season:
- Early preseason (mid-March)
- Just before Opening Day
- Mid-season if available

Usage:
    uv run python scripts/backfill_futures_history.py --dry-run
    uv run python scripts/backfill_futures_history.py --db
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Season -> list of (date, description) to fetch
HISTORICAL_SNAPSHOTS = {
    2022: [
        ("2022-03-20", "preseason"),
        ("2022-04-05", "opening-week"),
    ],
    2023: [
        ("2023-03-15", "preseason"),
        ("2023-03-28", "opening-week"),
    ],
    2024: [
        ("2024-03-10", "preseason"),
        ("2024-03-26", "opening-week"),
    ],
    2025: [
        ("2025-03-12", "preseason"),
        ("2025-03-25", "opening-week"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass through to fetch_futures_odds_history.py",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write to database (pass through to fetcher)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Specific seasons to fetch (default: all)",
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    fetcher = script_dir / "fetch_futures_odds_history.py"
    
    if not fetcher.exists():
        raise SystemExit(f"Fetcher script not found: {fetcher}")
    
    seasons_to_fetch = args.seasons if args.seasons else sorted(HISTORICAL_SNAPSHOTS.keys())
    
    print(f"Fetching historical futures odds for seasons: {seasons_to_fetch}")
    print()
    
    failed_fetches = []
    successful_fetches = []
    
    for season in seasons_to_fetch:
        if season not in HISTORICAL_SNAPSHOTS:
            print(f"Warning: no snapshots defined for season {season}, skipping")
            continue
        
        snapshots = HISTORICAL_SNAPSHOTS[season]
        
        for date_str, label in snapshots:
            print(f"{'='*70}")
            print(f"Season {season} - {label} ({date_str})")
            print(f"{'='*70}")
            
            cmd = [
                "uv", "run", "python",
                str(fetcher),
                "--season", str(season),
                "--date", date_str,
            ]
            
            if args.dry_run:
                cmd.append("--dry-run")
            elif args.db:
                cmd.append("--db")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=script_dir.parent,
                    check=False,
                    capture_output=False,
                )
                
                if result.returncode == 0:
                    successful_fetches.append((season, date_str, label))
                    print(f"✓ Success: {season} {label}\n")
                else:
                    failed_fetches.append((season, date_str, label))
                    print(f"✗ Failed: {season} {label} (exit code {result.returncode})\n")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                sys.exit(1)
            except Exception as e:
                failed_fetches.append((season, date_str, label))
                print(f"✗ Error: {season} {label}: {e}\n")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successful: {len(successful_fetches)}")
    print(f"Failed: {len(failed_fetches)}")
    
    if successful_fetches:
        print(f"\n✓ Fetched:")
        for season, date_str, label in successful_fetches:
            print(f"  - {season} {label} ({date_str})")
    
    if failed_fetches:
        print(f"\n✗ Failed:")
        for season, date_str, label in failed_fetches:
            print(f"  - {season} {label} ({date_str})")
        sys.exit(1)


if __name__ == "__main__":
    main()
