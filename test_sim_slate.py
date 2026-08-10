from __future__ import annotations

import json
from pathlib import Path

from src.sim.game import GameResult
from src.sim.slate import (
    DailySlateState,
    ProbablePitcher,
    SlateGame,
    SlatePrediction,
    StarterChange,
    build_daily_board_caption,
    build_update_caption,
    changed_games,
    load_daily_slate_state,
    save_daily_slate_state,
    snapshot_state,
    starter_changes,
)


def _game(
    *,
    game_pk: int = 1001,
    away_name: str | None = "Away Arm",
    away_id: int | None = 11,
    home_name: str | None = "Home Arm",
    home_id: int | None = 22,
) -> SlateGame:
    return SlateGame(
        game_pk=game_pk,
        slate_date="2026-08-09",
        game_datetime="2026-08-09T18:35:00Z",
        status="Preview",
        away_team_id=110,
        home_team_id=111,
        away_abbrev="BAL",
        home_abbrev="BOS",
        venue="Fenway Park",
        away_probable=ProbablePitcher(player_id=away_id, full_name=away_name),
        home_probable=ProbablePitcher(player_id=home_id, full_name=home_name),
    )


def _prediction() -> SlatePrediction:
    return SlatePrediction(
        game=_game(),
        results=[GameResult(away_runs=3, home_runs=5, innings=9)],
        away_starter="Away Arm",
        home_starter="Home Arm",
        stats={
            "home_win_probability": 0.62,
            "mean_away_runs": 3.4,
            "mean_home_runs": 4.8,
        },
    )


def test_starter_changes_detects_both_sides():
    previous = _game()
    current = _game(away_name="New Away", away_id=33, home_name="New Home", home_id=44)

    changes = starter_changes(previous, current)

    assert changes == [
        StarterChange(side="away", previous="Away Arm", current="New Away"),
        StarterChange(side="home", previous="Home Arm", current="New Home"),
    ]


def test_changed_games_matches_by_game_pk():
    previous_games = {_game(game_pk=1001).game_pk: _game(game_pk=1001)}
    current_games = [_game(game_pk=1001, away_name="Replacement", away_id=77), _game(game_pk=1002)]

    updates = changed_games(previous_games, current_games)

    assert len(updates) == 1
    changed_game, reasons = updates[0]
    assert changed_game.game_pk == 1001
    assert reasons == [
        StarterChange(side="away", previous="Away Arm", current="Replacement")
    ]


def test_build_update_caption_mentions_reason_and_projection():
    prediction = _prediction()
    caption = build_update_caption(
        prediction,
        [StarterChange(side="away", previous="Away Arm", current="Replacement")],
    )

    assert "Updated BAL @ BOS" in caption
    assert "BAL probable starter Away Arm → Replacement" in caption
    assert "New sim: BOS 62% to win" in caption
    assert "Projected score BAL 3.4 - BOS 4.8" in caption
    assert len(caption) <= 300


def test_build_daily_board_caption_mentions_updates_when_enabled():
    caption = build_daily_board_caption(
        "2026-08-09",
        games_summary="12 games",
        include_update_note=True,
    )

    assert "12 games" in caption
    assert "Updates will follow" in caption
    assert len(caption) <= 300


def test_daily_slate_state_round_trip(tmp_path: Path):
    state_path = tmp_path / "daily_sim_2026-08-09.json"
    board_path = tmp_path / "board.jpg"
    state = snapshot_state(
        "2026-08-09",
        board_path=board_path,
        board_post_id="at://post|cid",
        games=[_game()],
    )

    save_daily_slate_state(state_path, state)
    loaded = load_daily_slate_state(state_path)

    assert loaded == state
    assert DailySlateState.from_dict(json.loads(state_path.read_text())) == state
