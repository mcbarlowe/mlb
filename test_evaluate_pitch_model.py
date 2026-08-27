from __future__ import annotations

import math

import polars as pl

from scripts.evaluate_pitch_model import historical_rows_as_live_inputs, iter_sequences


def test_iter_sequences_orders_non_pitch_events_by_timestamp() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "at_bat_index": [2, 2, 2],
            "pitch_number": [1, 0, 2],
            "pitch_start_time": [
                "2025-01-01T00:00:01Z",
                "2025-01-01T00:00:02Z",
                "2025-01-01T00:00:03Z",
            ],
            "count_after_pitch": ["0-1", "0-1", "1-1"],
            "is_runner_on_first": [False, False, False],
            "is_runner_on_second": [False, False, False],
            "is_runner_on_third": [False, False, False],
            "pitcher_id": [10, 10, 10],
            "feature": [1.0, 2.0, 3.0],
            "pitch_type_idx": [0.0, float("nan"), 4.0],
        }
    )

    sequence = next(iter_sequences(frame, ["feature"], ["pitch_type_idx"], 20))

    assert sequence["aux"]["balls"] == [0, 0, 0]
    assert sequence["aux"]["strikes"] == [0, 1, 1]
    assert sequence["targets"][0, 0] == 0
    assert math.isnan(sequence["targets"][1, 0])
    assert sequence["targets"][2, 0] == 4


def test_historical_rows_as_live_inputs_rewrites_post_pitch_counts() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "at_bat_index": [2, 2, 2],
            "pitch_number": [1, 2, 3],
            "pitch_start_time": [
                "2025-01-01T00:00:01Z",
                "2025-01-01T00:00:02Z",
                "2025-01-01T00:00:03Z",
            ],
            "count_after_pitch": ["1-0", "1-1", "2-1"],
            "is_runner_on_first": [False, True, True],
            "is_runner_on_second": [False, False, False],
            "is_runner_on_third": [False, False, False],
        }
    )

    live_inputs = historical_rows_as_live_inputs(frame)

    assert live_inputs["count_after_pitch"].to_list() == ["0-0", "1-0", "1-1"]
    assert live_inputs["_balls_before"].to_list() == [0, 1, 1]
    assert live_inputs["_strikes_before"].to_list() == [0, 0, 1]
    assert live_inputs["_stretch"].to_list() == [False, True, True]
