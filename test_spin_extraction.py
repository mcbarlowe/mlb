"""Regression tests for spin extraction from live feeds.

The legacy extraction read spin from the wrong level of ``pitchData``
(always null); these tests pin the fixed behavior using the sample feed.
"""

import json
from pathlib import Path

from scripts.fix_pitches_spin import extract_game_spin
from mlb.data.game_feed_data import GameFeedData

EXAMPLE_FEED = Path("example_json_files/example_live_feed.json")


def test_game_feed_data_extracts_spin_from_breaks():
    feed = json.loads(EXAMPLE_FEED.read_text())
    plays = feed["liveData"]["plays"]["allPlays"]
    pitch_event = next(
        ev
        for play in plays
        for ev in play.get("playEvents", [])
        if ev.get("isPitch") and ev.get("pitchData", {}).get("breaks")
    )
    row = GameFeedData()._process_pitch_data(pitch_event)
    breaks = pitch_event["pitchData"]["breaks"]
    assert row["spin_rate"] == breaks["spinRate"]
    assert row["spin_direction"] == breaks["spinDirection"]


def test_repair_extraction_covers_the_whole_feed():
    extracted = extract_game_spin(EXAMPLE_FEED)
    assert extracted is not None
    game_pk, rows = extracted
    assert game_pk > 0
    assert len(rows) > 200  # a full game of pitches
    with_spin = [r for r in rows if r[2] is not None]
    # The overwhelming majority of tracked pitches carry spin.
    assert len(with_spin) / len(rows) > 0.9
    for _, _, spin, direction in with_spin[:50]:
        assert 500.0 < spin < 3600.0
        assert direction is None or direction.isdigit()
