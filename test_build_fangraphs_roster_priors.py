from __future__ import annotations

import csv
import json

from scripts.build_fangraphs_roster_priors import (
    build_roster_prior_rows,
    load_projected_standings_from_html,
    write_roster_prior_csv,
)


def test_load_projected_standings_from_fangraphs_next_data_payload() -> None:
    standings = [
        {
            "shortName": "Dodgers",
            "teamId": 22,
            "W": 74,
            "G": 124,
            "GL": 38,
            "rxWP": 0.60,
            "xW": 96.8,
        }
        for _ in range(30)
    ]
    payload = {"props": {"pageProps": {"dehydratedState": {"queries": [{"state": {"data": standings}}]}}}}
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}"
        "</script>"
    )

    rows = load_projected_standings_from_html(html)

    assert len(rows) == 30
    assert rows[0]["shortName"] == "Dodgers"


def test_build_roster_prior_rows_uses_ros_win_probability() -> None:
    rows = build_roster_prior_rows(
        [
            {
                "shortName": "Dodgers",
                "teamId": 22,
                "W": 74,
                "G": 124,
                "GL": 38,
                "rxWP": 0.60,
                "xW": 96.8,
            }
        ],
        season=2026,
        source="fangraphs",
    )

    assert rows == [
        {
            "season": "2026",
            "abbreviation": "LAD",
            "team_name": "Los Angeles Dodgers",
            "projected_wins": "97.200000",
            "total_games": "162",
            "source": "fangraphs",
            "fangraphs_short_name": "Dodgers",
            "fangraphs_team_id": "22",
            "fg_current_wins": "74",
            "fg_games_played": "124",
            "fg_remaining_games": "38",
            "fg_rest_of_season_win_probability": "0.600000",
            "fg_projected_final_wins": "96.800000",
        }
    ]


def test_write_roster_prior_csv(tmp_path) -> None:
    path = tmp_path / "roster_priors.csv"
    rows = build_roster_prior_rows(
        [
            {
                "shortName": "Dodgers",
                "teamId": 22,
                "W": 74,
                "G": 124,
                "GL": 38,
                "rxWP": 0.60,
                "xW": 96.8,
            }
        ],
        season=2026,
        source="fangraphs",
    )

    write_roster_prior_csv(path, rows)

    with path.open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["abbreviation"] == "LAD"
    assert written[0]["projected_wins"] == "97.200000"
