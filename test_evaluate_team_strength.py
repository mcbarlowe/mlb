from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_team_strength import (
    _resolve_season_windows,
    evaluate_rolling_seasons,
    paired_block_improvement_interval,
)
from mlb.sim.team_strength import FEATURE_NAMES


def test_logged_training_window_must_match_terminal_rolling_fold() -> None:
    with pytest.raises(ValueError, match="terminal rolling fold"):
        _resolve_season_windows(
            test_season=2025,
            train_seasons=(2015, 2016),
            rolling_seasons=(2022, 2023, 2024, 2025),
            rolling_window=4,
        )

    train, rolling = _resolve_season_windows(
        test_season=2025,
        train_seasons=None,
        rolling_seasons=None,
        rolling_window=3,
    )

    assert train == (2022, 2023, 2024)
    assert rolling == (2022, 2023, 2024, 2025)


def test_paired_block_interval_rejects_uncertain_point_improvement() -> None:
    outcomes = np.ones(20, dtype=float)
    comparator = np.full(20, 0.6, dtype=float)
    candidate = np.array([0.7] * 12 + [0.5] * 8, dtype=float)
    blocks = np.array([f"2025-04-{day:02d}" for day in range(1, 21)])

    interval = paired_block_improvement_interval(
        candidate,
        comparator,
        outcomes,
        blocks,
        metric="brier",
        resamples=2000,
        seed=7,
    )

    assert interval.estimate > 0.0
    assert interval.lower < 0.0


def test_rolling_gate_requires_confident_improvement_over_v1_and_home_rate() -> None:
    rows: list[dict[str, float | int | str]] = []
    for season in range(2018, 2025):
        for index in range(60):
            home_won = index % 2
            row: dict[str, float | int | str] = {
                "game_pk": season * 1000 + index,
                "season": season,
                "game_date": f"{season}-04-{index % 28 + 1:02d}",
                "home_won": home_won,
            }
            row.update(dict.fromkeys(FEATURE_NAMES, 0.0))
            row["lineup_woba_edge"] = 1.0 if home_won else -1.0
            rows.append(row)
    frame = pd.DataFrame(rows)

    evaluation = evaluate_rolling_seasons(
        frame,
        seasons=(2022, 2023, 2024),
        train_window=4,
        home_rate_baseline=0.5,
        bootstrap_resamples=500,
        bootstrap_seed=42,
        max_season_regression=0.001,
    )

    assert evaluation.gate_passed
    assert len(evaluation.folds) == 3
    assert evaluation.gate_checks["no_material_season_regression"]
    assert all(interval.lower > 0.0 for interval in evaluation.intervals.values())
