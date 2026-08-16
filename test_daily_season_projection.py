from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from scripts.run_daily_season_projection import (
    _caption_date,
    _default_caption,
    _projection_command,
    _projection_outputs,
    _schedule_snapshot_from_rows,
    _x_url_from_post_id,
)


def _row(
    game_pk: int,
    game_date: date,
    status: str,
    away_runs: int | None = None,
    home_runs: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        game_pk=game_pk,
        game_date=game_date,
        status=status,
        away_runs=away_runs,
        home_runs=home_runs,
    )


def test_schedule_snapshot_refreshes_recent_and_stale_games():
    snapshot = _schedule_snapshot_from_rows(
        [
            _row(1, date(2026, 8, 10), "Final", 4, 3),
            _row(2, date(2026, 8, 12), "Preview"),
            _row(3, date(2026, 8, 15), "Final", 2, 1),
            _row(4, date(2026, 8, 16), "Preview"),
            _row(5, date(2026, 8, 20), "Preview"),
        ],
        as_of=date(2026, 8, 16),
        refresh_lookback_days=3,
    )

    assert snapshot.total_games == 5
    assert snapshot.final_games == 2
    assert snapshot.stale_before_as_of == (2,)
    assert snapshot.refresh_game_pks == (2, 3, 4)
    assert snapshot.status_counts == {"Final": 2, "Preview": 3}


def test_default_caption_matches_public_post_style():
    assert _caption_date(date(2026, 8, 16)) == "Aug. 16"
    assert _default_caption(2026, date(2026, 8, 16)) == (
        "2026 MLB season projection as of Aug. 16.\n\n"
        "Playoff odds + playoff stage view."
    )


def test_projection_command_writes_expected_outputs(tmp_path):
    outputs = _projection_outputs(2026, tmp_path)
    command = _projection_command(
        args=SimpleNamespace(
            season=2026,
            trials=100,
            tune_trials=20,
            no_tune_simulation_params=True,
            calibrate_playoff_probs=False,
            market_win_totals=None,
        ),
        as_of=date(2026, 8, 16),
        outputs=outputs,
    )

    assert "scripts/backtest_season_projections.py" in command
    assert command[command.index("--as-of") + 1] == "2026-08-16"
    assert command[command.index("--out") + 1].endswith("season_2026_model_projection.csv")
    assert "--no-tune-simulation-params" in command


def test_x_url_from_post_id_supports_plain_and_multi_ids():
    assert _x_url_from_post_id("2089093870017458335") == (
        "https://x.com/i/web/status/2089093870017458335"
    )
    assert _x_url_from_post_id('multi:{"bluesky":"at://post","x":"2089093870017458335"}') == (
        "https://x.com/i/web/status/2089093870017458335"
    )
    assert _x_url_from_post_id("at://post") is None
