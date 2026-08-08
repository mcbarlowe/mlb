import json
from dataclasses import replace
from pathlib import Path

from src.database import PostgresConfig, PostgresHandler
from src.etl import postgres_backfill
from src.etl.postgres_backfill import (
    BACKFILL_PROGRESS_TABLE,
    BULK_BACKFILL_PROGRESS_TABLE,
    BackfillSummary,
    BulkBackfillSummary,
    run_postgres_backfill,
    run_postgres_bulk_backfill,
)

SAMPLE_SOURCE = Path("example_json_files/example_live_feed.json")
TEST_SCHEMA = "mlb_test_backfill"



def _seed_reference_tables(db_config: PostgresConfig) -> None:
    with PostgresHandler(db_config) as db:
        db.reset_schema()
        db.create_all_tables()
        db.connection.execute(
            """
            INSERT INTO positions (code, name, type, abbreviation)
            VALUES ('1', 'Pitcher', 'Pitcher', 'P')
            """
        )



def test_postgres_backfill_resumes_completed_games(tmp_path):
    season_dir = tmp_path / "2022"
    season_dir.mkdir()
    sample_path = season_dir / "631220.json"
    sample_path.write_text(SAMPLE_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    db_config = replace(PostgresConfig.from_env(), schema=TEST_SCHEMA)
    _seed_reference_tables(db_config)

    first_run = run_postgres_backfill(db_config, tmp_path)
    assert first_run.discovered_files == 1
    assert first_run.processed_games == 1
    assert first_run.skipped_completed == 0
    assert first_run.failed_games == 0

    with PostgresHandler(db_config) as db:
        progress_df = db.query(
            f"SELECT source_key, status FROM {BACKFILL_PROGRESS_TABLE} ORDER BY source_key"
        )
        assert progress_df.to_dict("records") == [
            {"source_key": "2022/631220.json", "status": "complete"}
        ]
        assert db.get_row_count("games") == 1
        assert db.get_row_count("pitches") == 259

    second_run = run_postgres_backfill(db_config, tmp_path)
    assert second_run.discovered_files == 1
    assert second_run.processed_games == 0
    assert second_run.skipped_completed == 1
    assert second_run.failed_games == 0



def test_postgres_bulk_backfill_resumes_completed_seasons(tmp_path):
    season_dir = tmp_path / "2022"
    season_dir.mkdir()
    sample_path = season_dir / "631220.json"
    sample_path.write_text(SAMPLE_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    db_config = replace(PostgresConfig.from_env(), schema="mlb_test_bulk_backfill")
    _seed_reference_tables(db_config)

    first_run = run_postgres_bulk_backfill(db_config, tmp_path)
    assert first_run.discovered_seasons == 1
    assert first_run.processed_seasons == 1
    assert first_run.skipped_completed == 0
    assert first_run.failed_seasons == 0

    with PostgresHandler(db_config) as db:
        progress_df = db.query(
            f"SELECT season, status FROM {BULK_BACKFILL_PROGRESS_TABLE} ORDER BY season"
        )
        assert progress_df.to_dict("records") == [
            {"season": 2022, "status": "complete"}
        ]
        assert db.get_row_count("games") == 1
        assert db.get_row_count("pitches") == 259

    second_run = run_postgres_bulk_backfill(db_config, tmp_path)
    assert second_run.discovered_seasons == 1
    assert second_run.processed_seasons == 0
    assert second_run.skipped_completed == 1
    assert second_run.failed_seasons == 0



def test_download_missing_schedules_skips_existing_files(tmp_path, monkeypatch):
    schedule_dir = tmp_path / "schedules"
    schedule_dir.mkdir()
    (schedule_dir / "schedule_2022.json").write_text("{}", encoding="utf-8")

    calls: list[dict[str, int]] = []

    class FakeSchedule:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_async(self, **kwargs):
            calls.append(kwargs)
            return {"season": kwargs["season"], "dates": []}

    monkeypatch.setattr(postgres_backfill, "Schedule", FakeSchedule)

    summary = postgres_backfill.download_missing_schedules(schedule_dir, ["2022", "2023"])
    assert summary.downloaded == 1
    assert summary.skipped_existing == 1
    assert calls == [{"sportId": 1, "season": 2023}]

    downloaded_schedule = json.loads((schedule_dir / "schedule_2023.json").read_text(encoding="utf-8"))
    assert downloaded_schedule == {"season": 2023, "dates": []}



def test_sync_and_backfill_uses_download_and_backfill_stages(tmp_path, monkeypatch):
    raw_data_path = tmp_path / "raw" / "livefeeds"
    db_config = replace(PostgresConfig.from_env(), schema="mlb_test_sync")

    observed: dict[str, object] = {}

    def fake_download_raw_data(raw_path: Path, seasons: list[str]):
        observed["download_args"] = {
            "raw_data_path": raw_path,
            "seasons": seasons,
        }
        return (
            postgres_backfill.ScheduleDownloadSummary(downloaded=1, skipped_existing=0),
            {"2022": {"success": 1, "error": 0, "skipped": 0}},
        )

    def fake_run_postgres_backfill(passed_config: PostgresConfig, passed_raw_data_path: Path):
        observed["backfill_args"] = {
            "schema": passed_config.schema,
            "raw_data_path": passed_raw_data_path,
        }
        return BackfillSummary(
            discovered_files=1,
            processed_games=1,
            skipped_completed=0,
            failed_games=0,
        )

    monkeypatch.setattr(postgres_backfill, "download_raw_data", fake_download_raw_data)
    monkeypatch.setattr(postgres_backfill, "run_postgres_backfill", fake_run_postgres_backfill)

    summary = postgres_backfill.sync_and_backfill_postgres(
        db_config,
        raw_data_path,
        seasons=["2022"],
    )

    assert observed["download_args"] == {
        "raw_data_path": raw_data_path,
        "seasons": ["2022"],
    }
    assert observed["backfill_args"] == {
        "schema": "mlb_test_sync",
        "raw_data_path": raw_data_path,
    }
    assert summary.schedules.downloaded == 1
    assert isinstance(summary.backfill, BackfillSummary)
    assert summary.backfill.processed_games == 1



def test_sync_and_backfill_uses_bulk_runner_when_requested(tmp_path, monkeypatch):
    raw_data_path = tmp_path / "raw" / "livefeeds"
    db_config = replace(PostgresConfig.from_env(), schema="mlb_test_bulk_sync")

    observed: dict[str, object] = {}

    def fake_download_raw_data(raw_path: Path, seasons: list[str]):
        observed["download_args"] = {
            "raw_data_path": raw_path,
            "seasons": seasons,
        }
        return (
            postgres_backfill.ScheduleDownloadSummary(downloaded=1, skipped_existing=0),
            {"2022": {"success": 1, "error": 0, "skipped": 0}},
        )

    def fake_run_postgres_bulk_backfill(passed_config: PostgresConfig, passed_raw_data_path: Path):
        observed["bulk_backfill_args"] = {
            "schema": passed_config.schema,
            "raw_data_path": passed_raw_data_path,
        }
        return BulkBackfillSummary(
            discovered_seasons=1,
            processed_seasons=1,
            skipped_completed=0,
            failed_seasons=0,
        )

    monkeypatch.setattr(postgres_backfill, "download_raw_data", fake_download_raw_data)
    monkeypatch.setattr(postgres_backfill, "run_postgres_bulk_backfill", fake_run_postgres_bulk_backfill)

    summary = postgres_backfill.sync_and_backfill_postgres(
        db_config,
        raw_data_path,
        seasons=["2022"],
        bulk_historical=True,
    )

    assert observed["download_args"] == {
        "raw_data_path": raw_data_path,
        "seasons": ["2022"],
    }
    assert observed["bulk_backfill_args"] == {
        "schema": "mlb_test_bulk_sync",
        "raw_data_path": raw_data_path,
    }
    assert summary.schedules.downloaded == 1
    assert isinstance(summary.backfill, BulkBackfillSummary)
    assert summary.backfill.processed_seasons == 1
