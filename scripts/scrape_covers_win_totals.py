"""Scrape historical MLB season win-total lines WITH real prices from Covers.

Source: https://www.covers.com/sportsoddshistory/mlb-win/?y={year}&sa=mlb&t=win
Each season page carries one preseason snapshot (typically BetMGM, late March):
team, win-total line, over odds, under odds, settlement game number, actual
wins, and Covers' own graded result.

Writes ``resources/season_win_totals_odds.csv`` with columns:
  season, team_name, abbreviation, team_id, win_total, over_odds, under_odds,
  actual_wins, result

This replaces the price-less ``season_win_totals_2022_2025.csv`` assumption of
-110 both ways with the real (asymmetric) juice, and captures the current
season's lines before they disappear so future settlements happen at real
prices.

Usage: uv run python scripts/scrape_covers_win_totals.py [--start 2013] [--end 2026]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import psycopg
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

URL = "https://www.covers.com/sportsoddshistory/mlb-win/?y={year}&sa=mlb&t=win"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
ROW_RE = re.compile(
    r"Team=([^\"&]+)\"[^>]*>[^<]+</a>\s*</td>\s*"
    r"<td[^>]*>\s*([0-9.]+)\s*</td>\s*"
    r"<td[^>]*>\s*([+-]?\d+)\s*</td>\s*"
    r"<td[^>]*>\s*([+-]?\d+)\s*</td>\s*"
    r"<td[^>]*>[^<]*</td>\s*"
    r"<td[^>]*>\s*(\d*)\s*</td>\s*"
    r"<td[^>]*>\s*(Over|Under|Push|)\s*</td>",
    re.IGNORECASE,
)
NAME_FIXES = {
    "St Louis Cardinals": "St. Louis Cardinals",
    "Cleveland Indians": "Cleveland Guardians",
    "Athletics": "Oakland Athletics",
    "Sacramento Athletics": "Oakland Athletics",
}


def team_directory() -> dict[str, tuple[str, int]]:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT team_name, abbreviation, team_id FROM {c.schema}.teams "
            f"WHERE sport_id = 1"
        )
        rows = cur.fetchall()
    conn.close()
    return {str(n): (str(a), int(t)) for n, a, t in rows}


def scrape_season(year: int) -> list[dict]:
    resp = requests.get(URL.format(year=year), headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []
    html = resp.text
    out = []
    for m in ROW_RE.finditer(html):
        raw_name = unquote(m.group(1)).replace("+", " ").strip()
        out.append({
            "season": year,
            "team_name": NAME_FIXES.get(raw_name, raw_name),
            "win_total": float(m.group(2)),
            "over_odds": int(m.group(3)),
            "under_odds": int(m.group(4)),
            "actual_wins": int(m.group(5)) if m.group(5) else None,
            "result": m.group(6).capitalize() or None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--out", default="resources/season_win_totals_odds.csv")
    args = ap.parse_args()

    directory = team_directory()
    rows: list[dict] = []
    for year in range(args.start, args.end + 1):
        season_rows = scrape_season(year)
        matched = 0
        for r in season_rows:
            ab_id = directory.get(r["team_name"])
            if ab_id is None:
                print(f"  {year}: unmatched team name {r['team_name']!r}")
                continue
            r["abbreviation"], r["team_id"] = ab_id
            rows.append(r)
            matched += 1
        print(f"{year}: {matched} teams"
              + ("" if season_rows else "  (no table / page missing)"))
        time.sleep(1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "season", "team_name", "abbreviation", "team_id", "win_total",
            "over_odds", "under_odds", "actual_wins", "result",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
