from __future__ import annotations

import math

import pytest

from mlb.sim.roster_priors import (
    RosterPrior,
    load_roster_prior_offsets,
    load_roster_priors_csv,
    roster_prior_offsets_from_rows,
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


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def test_roster_prior_offsets_prefer_direct_win_probability() -> None:
    offsets = roster_prior_offsets_from_rows(
        [
            RosterPrior(
                season=2024,
                team_id=1,
                win_probability=0.60,
                projected_wins=70.0,
            ),
            RosterPrior(season=2023, team_id=2, win_probability=0.70),
        ],
        prediction_season=2024,
    )

    assert offsets == {1: pytest.approx(_logit(0.60))}


def test_roster_prior_offsets_fall_back_to_projected_wins() -> None:
    offsets = roster_prior_offsets_from_rows(
        [RosterPrior(season=2024, team_id=1, projected_wins=90.0, total_games=162)],
        prediction_season=2024,
    )

    assert offsets[1] == pytest.approx(_logit(90.0 / 162.0))


def test_roster_prior_offsets_score_roster_quality_components() -> None:
    offsets = roster_prior_offsets_from_rows(
        [
            RosterPrior(
                season=2024,
                team_id=1,
                projected_war=50.0,
                returning_pa_share=0.90,
                returning_ip_share=0.85,
                lineup_woba=0.345,
                rotation_fip=3.60,
                bullpen_fip=3.90,
            ),
            RosterPrior(
                season=2024,
                team_id=2,
                projected_war=14.0,
                returning_pa_share=0.60,
                returning_ip_share=0.55,
                lineup_woba=0.285,
                rotation_fip=4.80,
                bullpen_fip=4.50,
            ),
        ],
        prediction_season=2024,
    )

    assert offsets[1] > 0.0
    assert offsets[2] < 0.0
    assert offsets[1] == pytest.approx(-offsets[2])


def test_load_roster_priors_csv_maps_team_labels(tmp_path) -> None:
    path = tmp_path / "roster_priors.csv"
    path.write_text(
        "season,abbreviation,team_name,win_probability,projected_war,source\n"
        "2024,AAA,,0.610,,manual\n"
        "2024,,Beta Bears,,42.0,manual\n"
    )
    teams = {
        1: _team(1, "AAA", "Alpha Ants"),
        2: _team(2, "BBB", "Beta Bears"),
    }

    rows = load_roster_priors_csv(path, teams=teams)

    assert rows == (
        RosterPrior(2024, 1, win_probability=0.610, source="manual"),
        RosterPrior(2024, 2, projected_war=42.0, source="manual"),
    )


def test_load_roster_prior_offsets_accepts_team_id_without_mapping(tmp_path) -> None:
    path = tmp_path / "roster_priors.csv"
    path.write_text("season,team_id,projected_wins,total_games\n2024,7,88,162\n")

    offsets = load_roster_prior_offsets(path, prediction_season=2024)

    assert offsets == {7: pytest.approx(_logit(88.0 / 162.0))}


def test_roster_prior_offsets_accept_mapping_rows_with_team_id() -> None:
    offsets = roster_prior_offsets_from_rows(
        [
            {
                "season": "2024",
                "team_id": "7",
                "lineup_woba": "0.345",
                "rotation_fip": "3.60",
            }
        ],
        prediction_season=2024,
    )

    assert offsets[7] > 0.0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (RosterPrior(2024, 1, win_probability=1.0), "win_probability"),
        (RosterPrior(2024, 1, projected_wins=162.0), "projected_wins"),
        (
            RosterPrior(2024, 1, projected_wins=88.0, total_games=0),
            "total_games",
        ),
        (RosterPrior(2024, 1, returning_pa_share=1.2), "returning_pa_share"),
        (RosterPrior(2024, 1, rotation_fip=9.0), "rotation_fip"),
        (RosterPrior(2024, 1), "supported scoring field"),
    ],
)
def test_roster_prior_offsets_validate_bounds(row: RosterPrior, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        roster_prior_offsets_from_rows([row], prediction_season=2024)


def test_load_roster_priors_csv_rejects_unknown_label_without_mapping(tmp_path) -> None:
    path = tmp_path / "roster_priors.csv"
    path.write_text("season,abbreviation,win_probability\n2024,AAA,0.55\n")

    with pytest.raises(ValueError, match="team_id"):
        load_roster_priors_csv(path)
