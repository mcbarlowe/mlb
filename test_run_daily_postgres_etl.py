from datetime import UTC, date, datetime, timedelta
from importlib import import_module


def test_daily_postgres_etl_defaults_to_yesterday(monkeypatch):
    module = import_module("scripts.run_daily_postgres_etl")

    observed: dict[str, object] = {}

    def fake_run_daily_pipeline(*, target_date, skip_existing, poll_live):
        observed["pipeline_args"] = {
            "target_date": target_date,
            "skip_existing": skip_existing,
            "poll_live": poll_live,
        }
        return {
            "total_games": 3,
            "skipped": [1],
            "completed": [2, 3],
        }

    class FakeConfig:
        def describe(self):
            return "dbname=postgres schema=mlb host=local-socket"

    class FakeBackfillSummary:
        discovered_files = 3
        processed_games = 2
        skipped_completed = 1
        failed_games = 0

    daily_pipeline_module = import_module("src.etl.daily_pipeline")
    database_module = import_module("src.database")
    backfill_module = import_module("src.etl.postgres_backfill")
    monkeypatch.setattr(daily_pipeline_module, "run_daily_pipeline", fake_run_daily_pipeline)
    monkeypatch.setattr(database_module.PostgresConfig, "from_env", classmethod(lambda cls: FakeConfig()))
    monkeypatch.setattr(backfill_module, "run_postgres_backfill", lambda config, path: FakeBackfillSummary())

    target_date = module.resolve_target_date(None)

    fake_args = type("Args", (), {"date": None})()
    monkeypatch.setattr(module, "parse_args", lambda: fake_args)

    module.main()

    assert observed["pipeline_args"] == {
        "target_date": target_date,
        "skip_existing": True,
        "poll_live": False,
    }
    assert target_date == datetime.now(tz=UTC).date() - timedelta(days=1)



def test_resolve_target_date_parses_explicit_date():
    module = import_module("scripts.run_daily_postgres_etl")
    assert module.resolve_target_date("2024-07-15") == date(2024, 7, 15)
