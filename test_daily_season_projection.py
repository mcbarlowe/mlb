from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import mlb.cli.run_daily_season_projection as season_projection
from mlb.cli.run_daily_season_projection import (
    ScheduleSnapshot,
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
    coded_game_state: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        game_pk=game_pk,
        game_date=game_date,
        status=status,
        coded_game_state=coded_game_state,
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
    assert snapshot.remaining_games == 2
    assert snapshot.status_counts == {"Final": 2, "Preview": 3}


def test_schedule_snapshot_treats_cancelled_game_as_resolved():
    # StatsAPI marks a rain cancellation abstract "Final" with no linescore
    # (game 823490, 2026-09-27); it will never gain a score, so it must not
    # block every later run as a stale non-final game.
    snapshot = _schedule_snapshot_from_rows(
        [
            _row(1, date(2026, 9, 27), "Final", 4, 3),
            _row(2, date(2026, 9, 27), "Final", coded_game_state="C"),
            _row(3, date(2026, 9, 27), "Final"),
        ],
        as_of=date(2026, 9, 28),
        refresh_lookback_days=3,
    )

    assert snapshot.final_games == 2
    assert snapshot.stale_before_as_of == (3,)


def test_remaining_games_ignores_cancelled_games_on_or_after_as_of():
    snapshot = _schedule_snapshot_from_rows(
        [
            _row(1, date(2026, 9, 27), "Final", 4, 3),
            _row(2, date(2026, 9, 28), "Final", coded_game_state="C"),
            _row(3, date(2026, 9, 28), "Preview"),
        ],
        as_of=date(2026, 9, 28),
        refresh_lookback_days=3,
    )

    assert snapshot.remaining_games == 1


def test_season_over_skips_projection_and_post(monkeypatch, capsys):
    finished = ScheduleSnapshot(
        total_games=2430,
        final_games=2430,
        remaining_games=0,
        stale_before_as_of=(),
        refresh_game_pks=(),
        status_counts={"Final": 2430},
    )
    monkeypatch.setattr(season_projection, "_ensure_fresh_inputs", lambda **_k: finished)

    def _must_not_run(**_kwargs):
        raise AssertionError("a finished season must not be projected or posted")

    monkeypatch.setattr(season_projection, "_run_projection", _must_not_run)
    monkeypatch.setattr(season_projection, "_publish_outputs", _must_not_run)

    season_projection.main(["--season", "2026", "--as-of", "2026-09-28", "--post"])

    assert "skipping projection and post" in capsys.readouterr().out


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

    assert command[0].endswith("mlb-backtest-season-projections")
    assert command[command.index("--as-of") + 1] == "2026-08-16"
    assert command[command.index("--out") + 1].endswith("season_2026_model_projection.csv")
    assert "--no-tune-simulation-params" in command


def test_projection_command_passes_optional_prior_controls(tmp_path):
    outputs = _projection_outputs(2026, tmp_path)
    market_path = tmp_path / "market_totals.csv"
    roster_path = tmp_path / "roster_priors.csv"

    command = _projection_command(
        args=SimpleNamespace(
            season=2026,
            trials=100,
            tune_trials=20,
            no_tune_simulation_params=True,
            calibrate_playoff_probs=False,
            market_win_totals=market_path,
            market_prior_scale=0.75,
            market_prior_decay_games=30.0,
            roster_priors=roster_path,
            roster_prior_scale=0.50,
            roster_prior_decay_games=14.0,
        ),
        as_of=date(2026, 8, 16),
        outputs=outputs,
    )

    assert command[command.index("--market-win-totals") + 1] == str(market_path)
    assert command[command.index("--market-prior-scale") + 1] == "0.75"
    assert command[command.index("--market-prior-decay-games") + 1] == "30.0"
    assert command[command.index("--roster-priors") + 1] == str(roster_path)
    assert command[command.index("--roster-prior-scale") + 1] == "0.5"
    assert command[command.index("--roster-prior-decay-games") + 1] == "14.0"


def test_x_url_from_post_id_supports_plain_and_multi_ids():
    assert _x_url_from_post_id("2089093870017458335") == (
        "https://x.com/i/web/status/2089093870017458335"
    )
    assert _x_url_from_post_id('multi:{"bluesky":"at://post","x":"2089093870017458335"}') == (
        "https://x.com/i/web/status/2089093870017458335"
    )
    assert _x_url_from_post_id("at://post") is None
