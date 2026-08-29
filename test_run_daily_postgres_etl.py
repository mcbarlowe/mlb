from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace

import pytest


def _install_etl_fakes(
    monkeypatch,
    *,
    completed,
    failed_games,
    other=None,
    live=None,
    scheduled=None,
):
    module = import_module("scripts.run_daily_postgres_etl")
    observed: dict[str, object] = {}

    def fake_run_daily_pipeline(*, target_date, skip_existing, poll_live):
        observed["pipeline_args"] = {
            "target_date": target_date,
            "skip_existing": skip_existing,
            "poll_live": poll_live,
        }
        return {
            "total_games": (
                len(completed) + len(other or []) + len(live or []) + len(scheduled or [])
            ),
            "skipped": [],
            "completed": completed,
            "other": other or [],
            "live": live or [],
            "scheduled": scheduled or [],
        }

    class FakeConfig:
        def describe(self):
            return "dbname=postgres schema=mlb host=local-socket"

    def fake_run_postgres_backfill(config, path, *, force_game_pks=None):
        observed["backfill_args"] = {
            "config": config,
            "path": path,
            "force_game_pks": force_game_pks,
        }
        return SimpleNamespace(
            discovered_files=len(completed),
            processed_games=len(force_game_pks or []),
            skipped_completed=0,
            failed_games=failed_games,
        )

    daily_pipeline_module = import_module("mlb.etl.daily_pipeline")
    database_module = import_module("mlb.database")
    backfill_module = import_module("mlb.etl.postgres_backfill")
    monkeypatch.setattr(
        daily_pipeline_module, "run_daily_pipeline", fake_run_daily_pipeline
    )
    monkeypatch.setattr(
        database_module.PostgresConfig, "from_env", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(
        backfill_module, "run_postgres_backfill", fake_run_postgres_backfill
    )
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(date=None))
    return module, observed


def test_daily_postgres_etl_is_pure_etl_and_defaults_to_yesterday(
    monkeypatch, capsys
):
    module, observed = _install_etl_fakes(
        monkeypatch,
        completed=[{"game_pk": 2}, {"game_pk": 3}],
        failed_games=0,
    )
    target_date = module.resolve_target_date(None)

    assert module.main() == 0

    assert observed["pipeline_args"] == {
        "target_date": target_date,
        "skip_existing": False,
        "poll_live": False,
    }
    assert observed["backfill_args"]["force_game_pks"] == [2, 3]
    assert target_date == datetime.now(tz=UTC).date() - timedelta(days=1)
    output = capsys.readouterr().out
    assert "Daily ETL completed successfully" in output
    assert "Settling" not in output
    assert "paper-trade" not in output


@pytest.mark.parametrize(
    ("completed", "failed_games", "failure_summary"),
    [
        (
            [{"game_pk": 2}, {"game_pk": 3, "error": "download failed"}],
            0,
            "- completed errors: 1",
        ),
        (
            [{"game_pk": 2}],
            2,
            "- backfill failed games: 2",
        ),
    ],
)
def test_daily_postgres_etl_returns_failure_after_summaries(
    monkeypatch, capsys, completed, failed_games, failure_summary
):
    module, observed = _install_etl_fakes(
        monkeypatch,
        completed=completed,
        failed_games=failed_games,
    )

    assert module.main() == 1

    assert observed["backfill_args"]["force_game_pks"] == [2]
    output = capsys.readouterr().out
    assert failure_summary in output
    assert output.index("Daily pipeline summary") < output.index("Daily ETL failed")
    assert output.index("Database backfill summary") < output.index("Daily ETL failed")


@pytest.mark.parametrize(
    ("extra", "failure_summary"),
    [
        ({"other": [{"game_pk": 4, "error": "fetch failed"}]}, "- fetch errors: 1"),
        ({"live": [{"game_pk": 4}]}, "- unresolved: 1"),
        ({"scheduled": [{"game_pk": 4}]}, "- unresolved: 1"),
        ({"other": [{"game_pk": 4, "state": "suspended"}]}, "- blocking states: 1"),
    ],
)
def test_daily_postgres_etl_blocks_report_on_partial_pipeline(
    monkeypatch, capsys, extra, failure_summary
):
    module, _ = _install_etl_fakes(
        monkeypatch,
        completed=[{"game_pk": 2}],
        failed_games=0,
        **extra,
    )

    assert module.main() == 1
    assert failure_summary in capsys.readouterr().out


def test_postponed_and_cancelled_games_are_nonblocking_pipeline_states():
    module = import_module("scripts.run_daily_postgres_etl")

    assert module.pipeline_failure_counts(
        {
            "completed": [],
            "live": [],
            "scheduled": [],
            "other": [
                {"game_pk": 1, "state": "postponed"},
                {"game_pk": 2, "state": "cancelled"},
            ],
        }
    ) == {
        "completed_errors": 0,
        "fetch_errors": 0,
        "unresolved": 0,
        "blocking_states": 0,
    }


def test_completed_game_pks_accepts_legacy_ints_and_processed_dicts():
    module = import_module("scripts.run_daily_postgres_etl")

    assert module.completed_game_pks(
        {
            "completed": [
                1,
                {"game_pk": "2"},
                {"game_pk": 3, "error": "failed"},
                {"other": 4},
            ]
        }
    ) == [1, 2]


def test_resolve_target_date_parses_explicit_date():
    module = import_module("scripts.run_daily_postgres_etl")
    assert module.resolve_target_date("2024-07-15") == date(2024, 7, 15)
