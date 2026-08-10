from __future__ import annotations

from pathlib import Path

import polars as pl

from src.sim.bullpen import (
    build_team_bullpen_hands,
    bullpen_arm_id,
    bullpen_for_team,
    relabel_reliever_rows,
    save_team_bullpens,
)
from src.sim.game import BULLPEN_ARM


def _raw() -> pl.DataFrame:
    # Game 1: team 108 pitches top (starter 11, reliever 12),
    # team 117 pitches bottom (starter 21 only).
    return pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 1, 1, 1],
            "at_bat_index": [0, 1, 5, 6, 10, 11],
            "pitch_number": [1, 1, 1, 1, 1, 1],
            "half_inning": ["top", "top", "bottom", "bottom", "top", "top"],
            "home_team_id": [108] * 6,
            "away_team_id": [117] * 6,
            "pitcher_id": [11, 11, 21, 21, 12, 12],
            "throw_side": ["R", "R", "L", "L", "L", "L"],
        }
    )


def test_relabel_reliever_rows_flags_only_relievers():
    relabeled = relabel_reliever_rows(_raw())
    # Only pitcher 12's rows (team 108's reliever) survive, relabeled -108.
    assert relabeled.height == 2
    assert set(relabeled["pitcher_id"].to_list()) == {bullpen_arm_id(108)}


def test_team_bullpen_hands_majority():
    relabeled = relabel_reliever_rows(_raw())
    hands = build_team_bullpen_hands(relabeled)
    assert hands == {108: "L"}


def test_bullpen_for_team_round_trip(tmp_path: Path):
    path = tmp_path / "team_bullpens.json"
    save_team_bullpens({108: "L"}, path)

    import src.sim.bullpen as bullpen_module

    bullpen_module._TEAM_BULLPENS_CACHE = None
    arm = bullpen_for_team(108, path)
    assert arm.player_id == -108
    assert arm.throw_side == "L"
    # Unknown team and missing id fall back to the league arm.
    assert bullpen_for_team(999, path) == BULLPEN_ARM
    assert bullpen_for_team(None, path) == BULLPEN_ARM
    bullpen_module._TEAM_BULLPENS_CACHE = None
