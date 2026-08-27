import sys
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace


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
            "completed": [{"game_pk": 2}, {"game_pk": 3}, {"game_pk": 4, "error": "boom"}],
        }

    class FakeConfig:
        def describe(self):
            return "dbname=postgres schema=mlb host=local-socket"

    class FakeBackfillSummary:
        discovered_files = 3
        processed_games = 2
        skipped_completed = 1
        failed_games = 0

    class FakePostgresHandler:
        def __init__(self, config):
            observed["handler_config"] = config

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    daily_pipeline_module = import_module("src.etl.daily_pipeline")
    database_module = import_module("src.database")
    backfill_module = import_module("src.etl.postgres_backfill")
    monkeypatch.setattr(
        daily_pipeline_module, "run_daily_pipeline", fake_run_daily_pipeline
    )
    monkeypatch.setattr(
        database_module.PostgresConfig, "from_env", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(database_module, "PostgresHandler", FakePostgresHandler)

    paper_store_module = import_module("src.betting.paper_trade_store")
    prop_settlement_module = import_module("src.betting.prop_settlement")
    monkeypatch.setattr(paper_store_module, "ensure_paper_trades_table", lambda db: None)
    monkeypatch.setattr(paper_store_module, "load_paper_trade_rows", list)
    monkeypatch.setattr(
        paper_store_module,
        "update_paper_trade_settlement_rows",
        lambda rows, *, db_config: None,
    )
    monkeypatch.setattr(
        prop_settlement_module,
        "summarize_prop_bet_rows",
        lambda rows: SimpleNamespace(settled_rows=0, open_rows=0, void_rows=0),
    )
    monkeypatch.setitem(
        sys.modules,
        "settle_paper_trades",
        SimpleNamespace(_settle_rows=lambda rows, *, db_config: (rows, 0, [], [])),
    )
    monkeypatch.setitem(
        sys.modules,
        "settle_prop_alerts",
        SimpleNamespace(
            load_prop_bet_rows=lambda db_config: [],
            settle_open_prop_bets=lambda db_config: [],
        ),
    )
    def fake_run_postgres_backfill(
        config, path, *, force_game_pks=None
    ):
        observed["backfill_args"] = {
            "config": config,
            "path": path,
            "force_game_pks": force_game_pks,
        }
        return FakeBackfillSummary()

    monkeypatch.setattr(
        backfill_module, "run_postgres_backfill", fake_run_postgres_backfill
    )

    target_date = module.resolve_target_date(None)

    fake_args = type("Args", (), {"date": None})()
    monkeypatch.setattr(module, "parse_args", lambda: fake_args)

    module.main()

    assert observed["pipeline_args"] == {
        "target_date": target_date,
        "skip_existing": False,
        "poll_live": False,
    }
    assert observed["backfill_args"]["force_game_pks"] == [2, 3]
    assert target_date == datetime.now(tz=UTC).date() - timedelta(days=1)




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
