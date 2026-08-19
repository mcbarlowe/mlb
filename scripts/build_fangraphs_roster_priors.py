from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.request import urlopen

FANGRAPHS_PROJECTED_STANDINGS_URL = "https://www.fangraphs.com/standings/projected-standings"

_TEAM_DETAILS = {
    "Angels": ("LAA", "Los Angeles Angels"),
    "Astros": ("HOU", "Houston Astros"),
    "Athletics": ("OAK", "Oakland Athletics"),
    "Blue Jays": ("TOR", "Toronto Blue Jays"),
    "Braves": ("ATL", "Atlanta Braves"),
    "Brewers": ("MIL", "Milwaukee Brewers"),
    "Cardinals": ("STL", "St. Louis Cardinals"),
    "Cubs": ("CHC", "Chicago Cubs"),
    "Diamondbacks": ("ARI", "Arizona Diamondbacks"),
    "Dodgers": ("LAD", "Los Angeles Dodgers"),
    "Giants": ("SF", "San Francisco Giants"),
    "Guardians": ("CLE", "Cleveland Guardians"),
    "Mariners": ("SEA", "Seattle Mariners"),
    "Marlins": ("MIA", "Miami Marlins"),
    "Mets": ("NYM", "New York Mets"),
    "Nationals": ("WSH", "Washington Nationals"),
    "Orioles": ("BAL", "Baltimore Orioles"),
    "Padres": ("SD", "San Diego Padres"),
    "Phillies": ("PHI", "Philadelphia Phillies"),
    "Pirates": ("PIT", "Pittsburgh Pirates"),
    "Rangers": ("TEX", "Texas Rangers"),
    "Rays": ("TB", "Tampa Bay Rays"),
    "Red Sox": ("BOS", "Boston Red Sox"),
    "Reds": ("CIN", "Cincinnati Reds"),
    "Rockies": ("COL", "Colorado Rockies"),
    "Royals": ("KC", "Kansas City Royals"),
    "Tigers": ("DET", "Detroit Tigers"),
    "Twins": ("MIN", "Minnesota Twins"),
    "White Sox": ("CWS", "Chicago White Sox"),
    "Yankees": ("NYY", "New York Yankees"),
}

_FIELDNAMES = [
    "season",
    "abbreviation",
    "team_name",
    "projected_wins",
    "total_games",
    "source",
    "fangraphs_short_name",
    "fangraphs_team_id",
    "fg_current_wins",
    "fg_games_played",
    "fg_remaining_games",
    "fg_rest_of_season_win_probability",
    "fg_projected_final_wins",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build roster-prior CSV rows from FanGraphs Depth Charts projected standings."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--url", default=FANGRAPHS_PROJECTED_STANDINGS_URL)
    parser.add_argument(
        "--html-file",
        type=Path,
        default=None,
        help="Optional saved FanGraphs HTML file; skips network fetch when supplied.",
    )
    parser.add_argument(
        "--source-label",
        default=None,
        help="Source label written into the roster-prior CSV. Defaults to the URL.",
    )
    parser.add_argument(
        "--total-games",
        type=int,
        default=162,
        help="Full-season game count used to convert ROS win probability to projected_wins.",
    )
    return parser.parse_args()


def build_roster_prior_rows(
    standings: Sequence[Mapping[str, object]],
    *,
    season: int,
    source: str,
    total_games: int = 162,
) -> list[dict[str, str]]:
    if total_games <= 0:
        raise ValueError("total_games must be positive")

    rows: list[dict[str, str]] = []
    for standing in standings:
        short_name = _required_text(standing, "shortName")
        if short_name not in _TEAM_DETAILS:
            raise ValueError(f"Unsupported FanGraphs team name: {short_name}")
        abbreviation, team_name = _TEAM_DETAILS[short_name]
        rest_of_season_win_probability = _required_float(standing, "rxWP")
        projected_final_wins = _required_float(standing, "xW")
        projected_wins = rest_of_season_win_probability * total_games
        rows.append(
            {
                "season": str(season),
                "abbreviation": abbreviation,
                "team_name": team_name,
                "projected_wins": _format_float(projected_wins),
                "total_games": str(total_games),
                "source": source,
                "fangraphs_short_name": short_name,
                "fangraphs_team_id": _text(standing.get("teamId")),
                "fg_current_wins": _text(standing.get("W")),
                "fg_games_played": _text(standing.get("G")),
                "fg_remaining_games": _text(standing.get("GL")),
                "fg_rest_of_season_win_probability": _format_float(
                    rest_of_season_win_probability
                ),
                "fg_projected_final_wins": _format_float(projected_final_wins),
            }
        )
    rows.sort(key=lambda row: row["abbreviation"])
    return rows


def load_projected_standings_from_html(html_text: str) -> list[dict[str, object]]:
    payload = _next_data_payload(html_text)
    standings = _find_projected_standings(payload)
    if len(standings) != 30:
        raise ValueError(f"Expected 30 projected standings rows; got {len(standings)}")
    return standings


def write_roster_prior_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _next_data_payload(html_text: str) -> object:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("FanGraphs HTML does not contain __NEXT_DATA__")
    return json.loads(html.unescape(match.group(1)))


def _find_projected_standings(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        if _is_projected_standings(payload):
            return [dict(item) for item in payload]
        for item in payload:
            rows = _find_projected_standings(item)
            if rows:
                return rows
    if isinstance(payload, dict):
        for value in payload.values():
            rows = _find_projected_standings(value)
            if rows:
                return rows
    return []


def _is_projected_standings(value: Sequence[object]) -> bool:
    if len(value) < 30:
        return False
    required_keys = {"shortName", "rxWP", "xW", "teamId", "W", "G", "GL"}
    return all(isinstance(item, dict) and required_keys <= set(item) for item in value)


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = _text(row.get(key))
    if not value:
        raise ValueError(f"Missing FanGraphs field {key}")
    return value


def _required_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing FanGraphs field {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid FanGraphs field {key}") from exc


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    args = parse_args()
    html_text = args.html_file.read_text() if args.html_file is not None else _fetch_text(args.url)
    standings = load_projected_standings_from_html(html_text)
    rows = build_roster_prior_rows(
        standings,
        season=args.season,
        source=args.source_label or args.url,
        total_games=args.total_games,
    )
    write_roster_prior_csv(args.out, rows)
    print(f"wrote {args.out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
