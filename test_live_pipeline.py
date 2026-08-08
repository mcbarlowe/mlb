from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from src.live.game_state import build_live_snapshot
from src.live.pipeline import (
    LiveGamePredictionService,
    choose_random_game,
    eligible_games,
    seconds_until_monitoring,
)
from src.live.predictor import mixture_density_grid
from src.live.publisher import DryRunPublisher, PredictionPost


def _pitch_event(pitch_number: int, balls: int, strikes: int, code: str) -> dict:
    return {
        "pitchNumber": pitch_number,
        "playId": f"play-{pitch_number}",
        "details": {
            "description": "called_strike" if strikes else "ball",
            "isInPlay": False,
            "isStrike": strikes > 0,
            "isBall": strikes == 0,
            "type": {"code": code, "description": code},
        },
        "count": {"balls": balls, "strikes": strikes},
        "about": {"outs": 1},
        "pitchData": {
            "startSpeed": 94.2,
            "strikeZoneTop": 3.4,
            "strikeZoneBottom": 1.6,
            "coordinates": {"pX": 0.12, "pZ": 2.4},
            "breaks": {},
        },
    }


def _play(at_bat_index: int, pitch_events: list[dict]) -> dict:
    return {
        "result": {
            "event": "Strikeout",
            "eventType": "strikeout",
            "description": "strikes out swinging.",
            "awayScore": 1,
            "homeScore": 2,
            "isOut": True,
        },
        "about": {"atBatIndex": at_bat_index, "halfInning": "top", "inning": 4},
        "matchup": {
            "batter": {"id": 660271, "fullName": "Shohei Ohtani"},
            "batSide": {"code": "L"},
            "pitcher": {"id": 543037, "fullName": "Gerrit Cole"},
            "pitchHand": {"code": "R"},
        },
        "runners": [],
        "pitchIndex": list(range(len(pitch_events))),
        "playEvents": pitch_events,
    }


def _live_feed(current_pitches: int, balls: int, strikes: int) -> dict:
    completed_play = _play(11, [_pitch_event(1, 0, 1, "FF"), _pitch_event(2, 0, 2, "SL")])
    current_play = _play(
        12,
        [
            _pitch_event(number + 1, 0, number + 1, "FF")
            for number in range(current_pitches)
        ],
    )
    current_play["about"]["halfInning"] = "bottom"
    current_play["count"] = {"balls": balls, "strikes": strikes, "outs": 2}

    return {
        "gameData": {
            "game": {"pk": 999901, "season": "2026", "type": "R"},
            "datetime": {"dateTime": "2026-08-08T23:10:00Z", "dayNight": "night"},
            "teams": {
                "away": {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
                "home": {"id": 111, "name": "Boston Red Sox", "abbreviation": "BOS"},
            },
            "venue": {"id": 3, "name": "Fenway Park"},
            "weather": {"condition": "Clear", "temp": "71", "wind": "8 mph, Out To CF"},
            "status": {"statusCode": "I"},
        },
        "liveData": {
            "plays": {
                "allPlays": [completed_play, current_play],
                "currentPlay": current_play,
            },
            "linescore": {
                "currentInning": 4,
                "isTopInning": False,
                "teams": {"home": {"runs": 2}, "away": {"runs": 1}},
                "offense": {"first": {"id": 1}},
            },
        },
    }


def test_build_live_snapshot_appends_pending_pitch():
    snapshot = build_live_snapshot(_live_feed(current_pitches=2, balls=1, strikes=2))

    assert snapshot is not None
    assert snapshot.game_pk == 999901
    assert snapshot.at_bat_index == 12
    assert snapshot.next_pitch_number == 3
    assert (snapshot.balls, snapshot.strikes) == (1, 2)
    assert snapshot.pitch_key == (12, 2, 1, 2)

    at_bat = snapshot.frame.filter(snapshot.frame["at_bat_index"] == 12)
    assert at_bat.height == 3
    pending = at_bat.sort("pitch_number").row(-1, named=True)
    assert pending["pitch_type_code"] is None
    assert pending["count_after_pitch"] == "1-2"
    assert pending["is_runner_on_first"] is True

    # Prior at-bat rows are retained for game-level cumulative features.
    assert snapshot.frame.filter(snapshot.frame["at_bat_index"] == 11).height == 2

    context = snapshot.context
    assert context.inning_half == "Bot"
    assert context.pitcher_name == "Gerrit Cole"
    assert context.count_str == "1-2"
    assert context.home_team == "BOS"


def test_build_live_snapshot_handles_first_pitch_of_at_bat():
    snapshot = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))

    assert snapshot is not None
    assert snapshot.next_pitch_number == 1
    assert snapshot.pitch_key == (12, 0, 0, 0)


def test_build_live_snapshot_returns_none_for_final_game():
    feed = _live_feed(current_pitches=1, balls=0, strikes=1)
    feed["gameData"]["status"]["statusCode"] = "F"

    assert build_live_snapshot(feed) is None


def test_seconds_until_monitoring_waits_for_first_pitch():
    now = datetime(2026, 8, 8, 16, 0, tzinfo=UTC)
    games = [
        {"gameDate": "2026-08-08T17:05:00Z"},
        {"gameDate": "2026-08-08T18:10:00Z"},
    ]

    delay = seconds_until_monitoring(games, now=now, lead=timedelta(minutes=15))
    assert delay == 50 * 60

    started = seconds_until_monitoring(
        games, now=datetime(2026, 8, 8, 17, 30, tzinfo=UTC), lead=timedelta(minutes=15)
    )
    assert started == 0.0


def test_should_post_respects_at_bat_cadence():
    service = LiveGamePredictionService.__new__(LiveGamePredictionService)
    service.post_cadence = "at_bat"
    service.max_posts_per_game = 40
    service._last_posted_at_bat = {}
    service._posts_per_game = {}

    first = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))
    assert first is not None
    assert service.should_post(first) is True

    service._last_posted_at_bat[first.game_pk] = first.at_bat_index
    later_same_ab = build_live_snapshot(_live_feed(current_pitches=1, balls=0, strikes=1))
    assert later_same_ab is not None
    assert service.should_post(later_same_ab) is False

    service.post_cadence = "pitch"
    assert service.should_post(later_same_ab) is True

    service._posts_per_game[later_same_ab.game_pk] = 40
    assert service.should_post(later_same_ab) is False


def _bare_service(post_cadence: str, seed: int = 7) -> LiveGamePredictionService:
    service = LiveGamePredictionService.__new__(LiveGamePredictionService)
    service.post_cadence = post_cadence
    service.max_posts_per_game = 40
    service.random_pitch_ceiling = 4
    service._rng = random.Random(seed)
    service._last_posted_at_bat = {}
    service._posts_per_game = {}
    service._ab_target_pitch = {}
    return service


def test_should_post_random_pitch_targets_one_pitch_per_at_bat():
    service = _bare_service("random_pitch")

    first = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))
    assert first is not None

    target = service._target_pitch_for(first)
    assert 1 <= target <= 4
    # Redraws are stable for the same at-bat
    assert service._target_pitch_for(first) == target

    assert service.should_post(first) is (first.next_pitch_number == target)

    mid = build_live_snapshot(_live_feed(current_pitches=1, balls=0, strikes=1))
    assert mid is not None
    assert service.should_post(mid) is (mid.next_pitch_number == target)

    # Once the at-bat has posted, later pitches in it never post again
    service._last_posted_at_bat[first.game_pk] = first.at_bat_index
    assert service.should_post(mid) is False


def test_random_pitch_target_catches_up_when_joining_mid_at_bat():
    service = _bare_service("random_pitch")

    mid = build_live_snapshot(_live_feed(current_pitches=2, balls=1, strikes=1))
    assert mid is not None
    assert mid.next_pitch_number == 3

    target = service._target_pitch_for(mid)
    assert 3 <= target <= 4


def test_mixture_density_grid_peaks_near_component_mean():
    pi = np.array([1.0])
    mu = np.array([[0.5, 2.5]])
    sigma = np.array([[0.3, 0.3]])
    rho = np.array([0.0])

    px_grid, pz_grid, density = mixture_density_grid(pi, mu, sigma, rho, grid_size=101)
    peak = np.unravel_index(np.argmax(density), density.shape)

    assert abs(px_grid[peak[1]] - 0.5) < 0.06
    assert abs(pz_grid[peak[0]] - 2.5) < 0.06
    assert density.min() >= 0.0


def test_dry_run_publisher_records_posts(tmp_path: Path):
    publisher = DryRunPublisher()
    card = tmp_path / "card.png"
    card.write_bytes(b"png")

    result = publisher.publish(PredictionPost(text="next pitch", image_path=card))
    assert result == str(card)
    assert publisher.published[0].text == "next pitch"



def _schedule_game(game_pk: int, status_code: str) -> dict:
    return {
        "gamePk": game_pk,
        "gameDate": "2026-08-08T23:10:00Z",
        "status": {"statusCode": status_code},
        "teams": {
            "away": {"team": {"name": "Away"}},
            "home": {"team": {"name": "Home"}},
        },
    }


def test_eligible_games_filters_terminal_states():
    games = [
        _schedule_game(1, "S"),
        _schedule_game(2, "PW"),
        _schedule_game(3, "I"),
        _schedule_game(4, "F"),
        _schedule_game(5, "PD"),
        _schedule_game(6, "C"),
    ]

    kept = [game["gamePk"] for game in eligible_games(games)]
    assert kept == [1, 2, 3]


def test_choose_random_game_is_seed_deterministic():
    games = [_schedule_game(pk, "S") for pk in (10, 20, 30, 40)]

    first = choose_random_game(games, random.Random(42))
    second = choose_random_game(games, random.Random(42))
    assert first is not None and second is not None
    assert first["gamePk"] == second["gamePk"]

    assert choose_random_game([_schedule_game(1, "F")], random.Random(1)) is None
