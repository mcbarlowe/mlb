from __future__ import annotations

import pytest

from mlb.sim.market_priors import (
    MarketWinTotal,
    load_market_win_totals_csv,
    market_prior_offsets_from_win_totals,
)
from mlb.sim.season import TeamInfo


def _team(team_id: int, abbreviation: str, team_name: str) -> TeamInfo:
    return TeamInfo(
        team_id=team_id,
        abbreviation=abbreviation,
        team_name=team_name,
        league_name="AL",
        division_name="East",
    )


def test_market_prior_offsets_use_target_season_win_totals():
    records = [
        MarketWinTotal(2023, 1, 60.0),
        MarketWinTotal(2024, 1, 100.0),
        MarketWinTotal(2024, 2, 62.0),
    ]

    offsets = market_prior_offsets_from_win_totals(
        records,
        prediction_season=2024,
    )

    assert offsets[1] > 0.0
    assert offsets[2] < 0.0
    assert offsets[1] > offsets[2]


def test_market_prior_offsets_reject_impossible_win_totals():
    with pytest.raises(ValueError, match="win_total"):
        market_prior_offsets_from_win_totals(
            [MarketWinTotal(2024, 1, 162.0)],
            prediction_season=2024,
        )


def test_load_market_win_totals_csv_maps_team_labels(tmp_path):
    path = tmp_path / "win_totals.csv"
    path.write_text(
        "season,abbreviation,team_name,win_total,total_games,source\n"
        "2024,AAA,,99.5,162,test\n"
        "2024,,Beta Bears,72.5,162,test\n"
    )
    teams = {
        1: _team(1, "AAA", "Alpha Ants"),
        2: _team(2, "BBB", "Beta Bears"),
    }

    records = load_market_win_totals_csv(path, teams=teams)

    assert records == (
        MarketWinTotal(2024, 1, 99.5, source="test"),
        MarketWinTotal(2024, 2, 72.5, source="test"),
    )
