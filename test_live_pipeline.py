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
from src.live.publisher import DryRunPublisher, PredictionPost, ResultPost


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


def test_pending_row_carries_previous_pitch_speed():
    snapshot = build_live_snapshot(_live_feed(current_pitches=2, balls=1, strikes=2))

    assert snapshot is not None
    pending = (
        snapshot.frame.filter(snapshot.frame["at_bat_index"] == 12)
        .sort("pitch_number")
        .row(-1, named=True)
    )
    # Fixture pitches all have startSpeed 94.2; the pending (unthrown) pitch
    # must inherit it so velocity_delta is 0 instead of a phantom 90-fill drop.
    assert pending["pitch_start_speed"] == 94.2


def test_build_live_snapshot_skips_completed_at_bat():
    feed = _live_feed(current_pitches=1, balls=0, strikes=1)
    feed["liveData"]["plays"]["currentPlay"]["about"]["isComplete"] = True

    assert build_live_snapshot(feed) is None


def test_build_live_snapshot_handles_first_pitch_of_at_bat():
    snapshot = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))

    assert snapshot is not None
    assert snapshot.next_pitch_number == 1
    assert snapshot.pitch_key == (12, 0, 0, 0)


def test_pending_row_feature_engineering_uses_prior_pitches_in_ab():
    from src.live.predictor import LiveNextPitchPredictor

    predictor = LiveNextPitchPredictor(
        "models/attention_full/run_20260119_124719",
        "models/pitch_type_location_20260121_003206",
    )

    snapshot = build_live_snapshot(_live_feed(current_pitches=2, balls=1, strikes=2))
    assert snapshot is not None

    at_bat = predictor._at_bat_features(snapshot).sort("pitch_number")
    pending = at_bat.row(-1, named=True)

    assert pending["pitch_number"] == 3
    assert pending["first_pitch"] == 0
    assert pending["prev_pitch_type_idx"] == 0  # FF
    assert pending["n_fastballs_in_ab"] == 2
    assert pending["n_breaking_in_ab"] == 0
    assert pending["same_pitch_streak"] == 1


def test_pending_row_feature_engineering_handles_first_pitch_without_history():
    from src.live.predictor import LiveNextPitchPredictor

    predictor = LiveNextPitchPredictor(
        "models/attention_full/run_20260119_124719",
        "models/pitch_type_location_20260121_003206",
    )

    snapshot = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))
    assert snapshot is not None

    at_bat = predictor._at_bat_features(snapshot).sort("pitch_number")
    pending = at_bat.row(-1, named=True)

    assert pending["pitch_number"] == 1
    assert pending["first_pitch"] == 1
    assert pending["prev_pitch_type_idx"] == -1
    assert pending["n_fastballs_in_ab"] == 0
    assert pending["n_breaking_in_ab"] == 0
    assert pending["same_pitch_streak"] == 0

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
    service._last_posted_inning = {}
    service._posts_per_game = {}
    service._ab_target_pitch = {}
    service._pending_posted_predictions = {}
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

def test_should_post_half_inning_only_once_per_half():
    service = _bare_service("half_inning")
    service._last_posted_inning = {}

    first = build_live_snapshot(_live_feed(current_pitches=0, balls=0, strikes=0))
    assert first is not None
    assert service.should_post(first) is True

    service._last_posted_inning[first.game_pk] = first.inning_key
    later_same_half = build_live_snapshot(_live_feed(current_pitches=1, balls=0, strikes=1))
    assert later_same_half is not None
    assert service.should_post(later_same_half) is False


def test_dry_run_publisher_records_result_replies(tmp_path: Path):
    publisher = DryRunPublisher()
    card = tmp_path / "result.png"
    card.write_bytes(b"png")

    result = publisher.publish_result(
        ResultPost(text="pitch result", image_path=card, reply_to="at://post|cid")
    )
    assert result == str(card)
    assert publisher.published[0].text == "pitch result"


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



def _synthetic_prediction():
    from src.live.predictor import mixture_density_grid
    from src.ml.pitch_predictor import PitchPrediction

    probs = np.zeros(11)
    probs[0] = 0.62  # FF
    probs[4] = 0.30  # SL
    probs[3] = 0.08  # CH
    pi = np.array([1.0])
    mu = np.array([[0.3, 2.6]])
    sigma = np.array([[0.4, 0.4]])
    rho = np.array([0.0])
    px_grid, pz_grid, density = mixture_density_grid(pi, mu, sigma, rho)
    return PitchPrediction(
        type_probabilities=probs,
        predicted_type_idx=0,
        predicted_type="FF",
        top_3_types=[("FF", 0.62), ("SL", 0.30), ("CH", 0.08)],
        location_point=np.array([0.3, 2.6]),
        location_mode=np.array([0.3, 2.6]),
        px_grid=px_grid,
        pz_grid=pz_grid,
        location_density=density,
        mixture_weights=pi,
        mixture_means=mu,
        mixture_stds=sigma,
    )


def _synthetic_context():
    from src.ml.pitch_predictor import GameContext

    return GameContext(
        pitcher_name="Test Pitcher",
        batter_name="Test Batter",
        pitcher_hand="R",
        batter_hand="L",
        home_team="NYY",
        away_team="ATL",
        inning=6,
        inning_half="Bot",
        balls=2,
        strikes=1,
        outs=1,
        date="2026-08-08",
        runner_on_1b=True,
        score_home=3,
        score_away=2,
        pitch_number=4,
        pitcher_id=None,
        batter_id=None,
    )


def test_build_card_html_contains_key_fields():
    from src.live.card_html import build_card_html

    html = build_card_html(_synthetic_prediction(), _synthetic_context(), 0.74)

    assert "Test Pitcher" in html
    assert "Test Batter" in html
    assert "Four-Seam Fastball" in html
    assert "62%" in html
    assert "IN-ZONE <b>74%</b>" in html
    assert "BOT 6" in html
    assert 'class="resultbar"' not in html


def test_build_card_html_result_variant_marks_actual():
    from src.live.card_html import build_card_html

    html = build_card_html(
        _synthetic_prediction(),
        _synthetic_context(),
        0.74,
        actual_pitch_type="SL",
        actual_location=(0.1, 2.2),
        pitch_result="Called Strike",
    )

    assert 'class="resultbar"' in html
    assert "ACTUAL: SLIDER" in html
    assert "CALLED STRIKE" in html
    assert "MODEL SAID FF" in html


def test_panel_xy_keeps_zone_inside_panel():
    from src.live.card_html import (
        PANEL_H,
        PANEL_W,
        ZONE_BOTTOM_FT,
        ZONE_HALF_W_FT,
        ZONE_TOP_FT,
        _panel_xy,
    )

    for px, pz in [
        (-ZONE_HALF_W_FT, ZONE_TOP_FT),
        (ZONE_HALF_W_FT, ZONE_BOTTOM_FT),
    ]:
        x, y = _panel_xy(px, pz)
        assert 0 <= x <= PANEL_W
        assert 0 <= y <= PANEL_H


def test_shrink_below_blob_limit_keeps_small_files(tmp_path: Path):
    from PIL import Image as PILImage

    from src.live.card_html import _shrink_below_blob_limit

    small = tmp_path / "small.png"
    PILImage.new("RGB", (100, 100), (20, 30, 40)).save(small, format="PNG")

    out = _shrink_below_blob_limit(small)
    assert out == small
    assert out.suffix == ".png"


def test_shrink_below_blob_limit_falls_back_to_jpeg(tmp_path: Path):
    from PIL import Image as PILImage

    from src.live.card_html import _shrink_below_blob_limit

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(1350, 2400, 3), dtype=np.uint8)
    big = tmp_path / "big.png"
    PILImage.fromarray(noise).save(big, format="PNG")
    original_size = big.stat().st_size
    assert original_size > 900_000

    out = _shrink_below_blob_limit(big)
    assert out.suffix == ".jpg"
    assert out.stat().st_size < original_size
    assert not big.exists()


def test_html_renderer_end_to_end(tmp_path: Path):
    import pytest

    from src.live.card_html import HtmlCardRenderer, render_card_png

    renderer = HtmlCardRenderer()
    try:
        try:
            renderer._ensure_page()
        except Exception as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        out = render_card_png(
            _synthetic_prediction(),
            _synthetic_context(),
            0.74,
            tmp_path / "card.png",
            renderer,
        )
        assert out.exists()
        assert out.stat().st_size > 20_000
        # Real cards must stay under the Bluesky blob ceiling.
        assert out.stat().st_size <= 900_000
    finally:
        renderer.close()