from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from src.sim.game import Batter, GameResult, GameSimulator
from src.sim.slate import (
    DailySlateState,
    ProbablePitcher,
    SlateGame,
    SlatePrediction,
    StarterChange,
    active_roster_ids,
    build_daily_board_caption,
    build_projected_lineups,
    build_update_caption,
    changed_games,
    load_daily_slate_state,
    save_daily_slate_state,
    simulate_slate_game,
    snapshot_state,
    starter_changes,
)
from src.sim.team_strength import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    TeamStrengthPredictor,
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
    current_games = [
        _game(game_pk=1001, away_name="Replacement", away_id=77),
        _game(game_pk=1002),
    ]

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
    assert "New model: BOS 62% to win" in caption
    assert "Projected score BAL 3.4 - BOS 4.8" in caption
    assert len(caption) <= 300


def test_slate_uses_team_strength_for_published_win_probability(monkeypatch):
    lineups = {
        "away": SimpleNamespace(starter=SimpleNamespace(player_id=11)),
        "home": SimpleNamespace(starter=SimpleNamespace(player_id=22)),
    }
    monkeypatch.setattr(
        "src.sim.slate.build_projected_lineups",
        lambda game, season, announced_lineups=None: (
            lineups,
            {"away": "Away Arm", "home": "Home Arm"},
        ),
    )
    monkeypatch.setattr("src.sim.slate._announced_batters", lambda _game_pk: {})

    class Simulator:
        def simulate_many(self, _away, _home, _n_sims):
            return [GameResult(away_runs=2, home_runs=5, innings=9)]

    class WinPredictor:
        feature_names = LEGACY_FEATURE_NAMES

        def predict_home_probability(self, **kwargs):
            assert kwargs == {
                "season": 2026,
                "away_team_id": 110,
                "home_team_id": 111,
                "away_starter_id": 11,
                "home_starter_id": 22,
            }
            return 0.61

    prediction = simulate_slate_game(
        _game(),
        cast(GameSimulator, Simulator()),
        season=2026,
        n_sims=1,
        win_predictor=cast(TeamStrengthPredictor, WinPredictor()),
    )

    assert prediction.stats["home_win_probability"] == 0.61
    assert prediction.stats["home_win_probability_raw"] == 1.0


def test_confirmed_lineup_overrides_projection_per_side(monkeypatch):
    announced = [Batter(player_id, "R") for player_id in range(1, 10)]
    monkeypatch.setattr(
        "src.sim.slate._announced_batters",
        lambda _game_pk: {"away": announced},
    )
    monkeypatch.setattr(
        "src.sim.slate._projected_batters",
        lambda team_id, slate_date, season: [
            Batter(team_id * 100 + index, "R") for index in range(9)
        ],
    )
    monkeypatch.setattr("src.sim.slate._pitch_hand", lambda _player_id: "R")

    lineups, _ = build_projected_lineups(_game(), season=2026)

    assert [batter.player_id for batter in lineups["away"].batters] == list(
        range(1, 10)
    )
    assert [batter.player_id for batter in lineups["home"].batters] == [
        11100 + index for index in range(9)
    ]


def test_active_roster_ids_separates_batters_pitchers_and_two_way_players(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.sim.slate._fetch_json",
        lambda _url, _params: {
            "roster": [
                {
                    "person": {"id": 1},
                    "position": {"type": "Infielder", "abbreviation": "SS"},
                },
                {
                    "person": {"id": 2},
                    "position": {"type": "Pitcher", "abbreviation": "P"},
                },
                {
                    "person": {"id": 3},
                    "position": {"type": "Two-Way Player", "abbreviation": "TWP"},
                },
            ]
        },
    )

    batter_ids, pitcher_ids = active_roster_ids(110, "2026-08-09")

    assert batter_ids == (1, 3)
    assert pitcher_ids == (2, 3)


def test_slate_passes_live_roster_context_to_v2_predictor(monkeypatch):
    lineups = {
        "away": SimpleNamespace(
            starter=SimpleNamespace(player_id=11),
            batters=[SimpleNamespace(player_id=101), SimpleNamespace(player_id=102)],
        ),
        "home": SimpleNamespace(
            starter=SimpleNamespace(player_id=22),
            batters=[SimpleNamespace(player_id=201), SimpleNamespace(player_id=202)],
        ),
    }
    monkeypatch.setattr(
        "src.sim.slate.build_projected_lineups",
        lambda game, season, announced_lineups=None: (
            lineups,
            {"away": "Away Arm", "home": "Home Arm"},
        ),
    )
    monkeypatch.setattr(
        "src.sim.slate._announced_batters",
        lambda _game_pk: {
            "away": [SimpleNamespace(player_id=102)],
        },
    )
    monkeypatch.setattr(
        "src.sim.slate.active_roster_ids",
        lambda team_id, slate_date: (
            ((101, 103), (11, 31, 32)) if team_id == 110 else ((201, 203), (22, 41, 42))
        ),
    )

    class Simulator:
        def simulate_many(self, _away, _home, _n_sims):
            return [GameResult(away_runs=2, home_runs=5, innings=9)]

    class WinPredictor:
        feature_names = FEATURE_NAMES

        def predict_home_probability(self, **kwargs):
            assert kwargs == {
                "season": 2026,
                "away_team_id": 110,
                "home_team_id": 111,
                "away_starter_id": 11,
                "home_starter_id": 22,
                "prediction_date": date(2026, 8, 9),
                "away_batter_ids": (101, 102),
                "home_batter_ids": (201, 202),
                "away_active_batter_ids": (101, 103, 102),
                "home_active_batter_ids": (201, 203),
                "away_reliever_ids": (11, 31, 32),
                "home_reliever_ids": (22, 41, 42),
            }
            return 0.64

    prediction = simulate_slate_game(
        _game(),
        cast(GameSimulator, Simulator()),
        season=2026,
        n_sims=1,
        win_predictor=cast(TeamStrengthPredictor, WinPredictor()),
    )

    assert prediction.stats["home_win_probability"] == 0.64


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
