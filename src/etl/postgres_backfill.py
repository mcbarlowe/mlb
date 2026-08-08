from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiofiles
import pandas as pd
import psycopg
from tqdm import tqdm

from src.data import BoxscoreData, GameFeedData, LinescoreData, PlayerData, TeamData
from src.data.game_data import GameData
from src.data.venue_data import VenueData
from src.database import PostgresConfig, PostgresHandler
from src.endpoints.schedule import Schedule
from src.etl.get_live_feeds import live_feed_etl_async
from src.etl.load_to_database import create_indexes, load_reference_data_to_db

BACKFILL_PROGRESS_TABLE = "backfill_game_progress"
BULK_BACKFILL_PROGRESS_TABLE = "bulk_backfill_progress"
DEFAULT_START_SEASON = 2009
BULK_GAME_CHUNK_SIZE = 250


@dataclass(frozen=True)
class BackfillSummary:
    discovered_files: int
    processed_games: int
    skipped_completed: int
    failed_games: int


@dataclass(frozen=True)
class BulkBackfillSummary:
    discovered_seasons: int
    processed_seasons: int
    skipped_completed: int
    failed_seasons: int


@dataclass(frozen=True)
class ScheduleDownloadSummary:
    downloaded: int
    skipped_existing: int


@dataclass(frozen=True)
class PipelineSummary:
    schedules: ScheduleDownloadSummary
    live_feed_stats: dict[str, dict[str, int]]
    backfill: BackfillSummary | BulkBackfillSummary


@dataclass
class DimensionState:
    team_ids: set[int]
    venue_ids: set[int]
    player_ids: set[int]


@dataclass(frozen=True)
class PendingGame:
    file_path: Path
    source_key: str
    season: int
    game_pk: int


@dataclass(frozen=True)
class PendingSeason:
    season: int
    total_games: int
    files: list[Path]


class PostgresBackfill:
    def __init__(self, db_config: PostgresConfig, raw_data_path: Path):
        self.db_config = db_config
        self.raw_data_path = raw_data_path
        self.pitch_transformer = GameFeedData()
        self.linescore_transformer = LinescoreData()
        self.boxscore_transformer = BoxscoreData()
        self.team_transformer = TeamData()
        self.venue_transformer = VenueData()
        self.game_transformer = GameData()
        self.player_transformer = PlayerData()

    def run(self) -> BackfillSummary:
        self._ensure_schema_ready()

        processed_games = 0
        failed_games = 0

        with PostgresHandler(self.db_config) as db:
            self._ensure_progress_table(db)
            dimension_state = self._load_dimension_state(db)
            pending_games, skipped_completed = self._build_pending_games(db)

            tqdm.write(
                f"Discovered {len(pending_games) + skipped_completed} game files; "
                f"{skipped_completed} already complete."
            )

            for pending_game in tqdm(pending_games, desc="Backfilling games", unit="game"):
                try:
                    self._process_game(db, pending_game, dimension_state)
                    processed_games += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, psycopg.Error) as exc:
                    failed_games += 1
                    tqdm.write(f"Failed {pending_game.source_key}: {exc}")

        create_indexes(self.db_config)
        with PostgresHandler(self.db_config) as db:
            db.vacuum()

        return BackfillSummary(
            discovered_files=len(pending_games) + skipped_completed,
            processed_games=processed_games,
            skipped_completed=skipped_completed,
            failed_games=failed_games,
        )

    def _ensure_schema_ready(self) -> None:
        with PostgresHandler(self.db_config) as db:
            db.create_all_tables()
            self._ensure_progress_table(db)
            should_load_reference_data = db.get_row_count("positions") == 0

        if should_load_reference_data:
            load_reference_data_to_db(self.db_config)

    def _ensure_progress_table(self, db: PostgresHandler) -> None:
        db.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BACKFILL_PROGRESS_TABLE} (
                source_key VARCHAR PRIMARY KEY,
                game_pk INTEGER NOT NULL,
                season INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                last_error VARCHAR,
                loaded_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        db.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{BACKFILL_PROGRESS_TABLE}_status
            ON {BACKFILL_PROGRESS_TABLE}(status)
            """
        )

    def _load_dimension_state(self, db: PostgresHandler) -> DimensionState:
        return DimensionState(
            team_ids=self._existing_int_values(db, "teams", "team_id"),
            venue_ids=self._existing_int_values(db, "venues", "venue_id"),
            player_ids=self._existing_int_values(db, "players", "player_id"),
        )

    def _build_pending_games(self, db: PostgresHandler) -> tuple[list[PendingGame], int]:
        completed_keys = self._completed_source_keys(db)
        pending_games: list[PendingGame] = []
        skipped_completed = 0

        for file_path in self._discover_game_files():
            source_key = self._source_key(file_path)
            if source_key in completed_keys:
                skipped_completed += 1
                continue
            pending_games.append(
                PendingGame(
                    file_path=file_path,
                    source_key=source_key,
                    season=int(file_path.parent.name),
                    game_pk=int(file_path.stem),
                )
            )

        return pending_games, skipped_completed

    def _discover_game_files(self) -> list[Path]:
        return sorted(self.raw_data_path.glob("**/*.json"))

    def _source_key(self, file_path: Path) -> str:
        return file_path.relative_to(self.raw_data_path).as_posix()

    def _completed_source_keys(self, db: PostgresHandler) -> set[str]:
        progress_df = db.query(
            f"SELECT source_key FROM {BACKFILL_PROGRESS_TABLE} WHERE status = 'complete'"
        )
        if progress_df.empty:
            return set()
        return set(progress_df["source_key"].tolist())

    def _existing_int_values(self, db: PostgresHandler, table_name: str, column_name: str) -> set[int]:
        values_df = db.query(f"SELECT {column_name} FROM {table_name}")
        if values_df.empty:
            return set()
        return {int(value) for value in values_df[column_name].dropna().tolist()}

    def _process_game(self, db: PostgresHandler, pending_game: PendingGame, dimension_state: DimensionState) -> None:
        with pending_game.file_path.open("r", encoding="utf-8") as handle:
            game_data = json.load(handle)

        teams_df = self.team_transformer.transform(game_data)
        new_teams_df = teams_df[~teams_df["team_id"].isin(dimension_state.team_ids)]
        new_team_ids = (
            {int(team_id) for team_id in new_teams_df["team_id"].tolist()}
            if not new_teams_df.empty
            else set()
        )

        venue_df = self.venue_transformer.transform(game_data)
        venue_id_value = venue_df.iloc[0]["venue_id"] if not venue_df.empty else None
        venue_id = int(venue_id_value) if pd.notna(venue_id_value) else None
        should_insert_venue = venue_id is not None and venue_id not in dimension_state.venue_ids

        players_df = self.player_transformer.transform(game_data)
        new_players_df = players_df[~players_df["player_id"].isin(dimension_state.player_ids)]
        new_player_ids = (
            {int(player_id) for player_id in new_players_df["player_id"].tolist()}
            if not new_players_df.empty
            else set()
        )

        game_df = self.game_transformer.transform(game_data)
        pitches_df = self.pitch_transformer.transform(
            game_data,
            game_id=pending_game.game_pk,
            season=pending_game.season,
        )
        linescore_df = self.linescore_transformer.transform(game_data, game_pk=pending_game.game_pk)
        boxscore_data = self.boxscore_transformer.transform_all(game_data, game_pk=pending_game.game_pk)

        self._mark_progress(
            db,
            pending_game.source_key,
            pending_game.game_pk,
            pending_game.season,
            status="running",
            last_error=None,
        )
        try:
            with db.connection.transaction():
                self._delete_existing_game_rows(db, pending_game.game_pk)

                if not new_teams_df.empty:
                    self.team_transformer.save_to_db(new_teams_df, db)
                if should_insert_venue:
                    self.venue_transformer.save_to_db(venue_df, db)
                if not new_players_df.empty:
                    self.player_transformer.save_to_db(new_players_df, db)

                self.game_transformer.save_to_db(game_df, db)
                self.pitch_transformer.save_to_db(pitches_df, db)
                self.linescore_transformer.save_to_db(linescore_df, db)

                for table_name, df in boxscore_data.items():
                    if not df.empty:
                        stat_type = table_name.split("_")[1]
                        self.boxscore_transformer.save_to_db(df, stat_type, db)

                self._mark_progress(
                    db,
                    pending_game.source_key,
                    pending_game.game_pk,
                    pending_game.season,
                    status="complete",
                    last_error=None,
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, psycopg.Error) as exc:
            self._mark_progress(
                db,
                pending_game.source_key,
                pending_game.game_pk,
                pending_game.season,
                status="failed",
                last_error=str(exc)[:1000],
            )
            raise

        dimension_state.team_ids.update(new_team_ids)
        if should_insert_venue and venue_id is not None:
            dimension_state.venue_ids.add(venue_id)
        dimension_state.player_ids.update(new_player_ids)

    def _delete_existing_game_rows(self, db: PostgresHandler, game_pk: int) -> None:
        for table_name in ("pitches", "linescore", "batting", "pitching", "fielding"):
            db.connection.execute(f"DELETE FROM {table_name} WHERE game_pk = %s", (game_pk,))
        db.connection.execute("DELETE FROM games WHERE game_pk = %s", (game_pk,))

    def _mark_progress(
        self,
        db: PostgresHandler,
        source_key: str,
        game_pk: int,
        season: int,
        *,
        status: str,
        last_error: str | None,
    ) -> None:
        db.connection.execute(
            f"""
            INSERT INTO {BACKFILL_PROGRESS_TABLE} (
                source_key,
                game_pk,
                season,
                status,
                last_error,
                loaded_at,
                updated_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                CASE WHEN %s = 'complete' THEN NOW() ELSE NULL END,
                NOW()
            )
            ON CONFLICT (source_key) DO UPDATE SET
                game_pk = EXCLUDED.game_pk,
                season = EXCLUDED.season,
                status = EXCLUDED.status,
                last_error = EXCLUDED.last_error,
                loaded_at = EXCLUDED.loaded_at,
                updated_at = NOW()
            """,
            (source_key, game_pk, season, status, last_error, status),
        )


class BulkHistoricalBackfill(PostgresBackfill):
    def run(self) -> BulkBackfillSummary:
        self._ensure_schema_ready()

        processed_seasons = 0
        failed_seasons = 0

        with PostgresHandler(self.db_config) as db:
            self._ensure_bulk_progress_table(db)
            dimension_state = self._load_dimension_state(db)
            pending_seasons, skipped_completed = self._build_pending_seasons(db)

            tqdm.write(
                f"Discovered {len(pending_seasons) + skipped_completed} seasons; "
                f"{skipped_completed} already complete."
            )

            for pending_season in tqdm(pending_seasons, desc="Bulk backfilling seasons", unit="season"):
                try:
                    self._process_season(db, pending_season, dimension_state)
                    processed_seasons += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, psycopg.Error) as exc:
                    failed_seasons += 1
                    tqdm.write(f"Failed season {pending_season.season}: {exc}")

        create_indexes(self.db_config)
        with PostgresHandler(self.db_config) as db:
            db.vacuum()

        return BulkBackfillSummary(
            discovered_seasons=len(pending_seasons) + skipped_completed,
            processed_seasons=processed_seasons,
            skipped_completed=skipped_completed,
            failed_seasons=failed_seasons,
        )

    def _ensure_bulk_progress_table(self, db: PostgresHandler) -> None:
        db.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BULK_BACKFILL_PROGRESS_TABLE} (
                season INTEGER PRIMARY KEY,
                total_games INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                last_error VARCHAR,
                loaded_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        db.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{BULK_BACKFILL_PROGRESS_TABLE}_status
            ON {BULK_BACKFILL_PROGRESS_TABLE}(status)
            """
        )

    def _build_pending_seasons(self, db: PostgresHandler) -> tuple[list[PendingSeason], int]:
        existing_game_pks = self._existing_game_pks(db)
        files_by_season: dict[int, list[Path]] = {}
        for file_path in self._discover_game_files():
            season = int(file_path.parent.name)
            files_by_season.setdefault(season, []).append(file_path)

        pending_seasons: list[PendingSeason] = []
        skipped_completed = 0
        for season, files in sorted(files_by_season.items()):
            pending_files = [
                file_path for file_path in sorted(files)
                if int(file_path.stem) not in existing_game_pks
            ]
            if not pending_files:
                skipped_completed += 1
                self._mark_bulk_progress(
                    db,
                    season,
                    total_games=len(files),
                    status="complete",
                    last_error=None,
                )
                continue
            pending_seasons.append(
                PendingSeason(season=season, total_games=len(files), files=pending_files)
            )

        return pending_seasons, skipped_completed

    def _existing_game_pks(self, db: PostgresHandler) -> set[int]:
        games_df = db.query("SELECT game_pk FROM games")
        if games_df.empty:
            return set()

        return {int(game_pk) for game_pk in games_df["game_pk"].tolist() if game_pk is not None}

    def _process_season(
        self,
        db: PostgresHandler,
        pending_season: PendingSeason,
        dimension_state: DimensionState,
    ) -> None:
        self._mark_bulk_progress(
            db,
            pending_season.season,
            total_games=pending_season.total_games,
            status="running",
            last_error=None,
        )
        try:
            for file_chunk in self._iter_file_chunks(pending_season.files):
                self._process_season_chunk(db, pending_season.season, file_chunk, dimension_state)
            self._mark_bulk_progress(
                db,
                pending_season.season,
                total_games=pending_season.total_games,
                status="complete",
                last_error=None,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, psycopg.Error) as exc:
            self._mark_bulk_progress(
                db,
                pending_season.season,
                total_games=pending_season.total_games,
                status="failed",
                last_error=str(exc)[:1000],
            )
            raise

    def _iter_file_chunks(self, files: Sequence[Path]) -> list[list[Path]]:
        return [list(files[idx: idx + BULK_GAME_CHUNK_SIZE]) for idx in range(0, len(files), BULK_GAME_CHUNK_SIZE)]

    def _delete_existing_season_rows(self, db: PostgresHandler, season: int) -> None:
        game_pks_df = db.query(f"SELECT game_pk FROM games WHERE season = '{season}'")
        game_pks = [int(game_pk) for game_pk in game_pks_df["game_pk"].tolist()] if not game_pks_df.empty else []

        with db.connection.transaction():
            if game_pks:
                for table_name in ("pitches", "linescore", "batting", "pitching", "fielding"):
                    db.connection.execute(
                        f"DELETE FROM {table_name} WHERE game_pk = ANY(%s)",
                        (game_pks,),
                    )
            db.connection.execute("DELETE FROM games WHERE season = %s", (str(season),))

    def _process_season_chunk(
        self,
        db: PostgresHandler,
        season: int,
        file_chunk: Sequence[Path],
        dimension_state: DimensionState,
    ) -> None:
        teams_batches: list[pd.DataFrame] = []
        venues_batches: list[pd.DataFrame] = []
        players_batches: list[pd.DataFrame] = []
        games_batches: list[pd.DataFrame] = []
        pitches_batches: list[pd.DataFrame] = []
        linescore_batches: list[pd.DataFrame] = []
        batting_batches: list[pd.DataFrame] = []
        pitching_batches: list[pd.DataFrame] = []
        fielding_batches: list[pd.DataFrame] = []

        new_team_ids: set[int] = set()
        new_venue_ids: set[int] = set()
        new_player_ids: set[int] = set()

        for file_path in file_chunk:
            game_pk = int(file_path.stem)
            with file_path.open("r", encoding="utf-8") as handle:
                game_data = json.load(handle)

            teams_df = self.team_transformer.transform(game_data)
            new_teams_df = teams_df[~teams_df["team_id"].isin(dimension_state.team_ids | new_team_ids)]
            if not new_teams_df.empty:
                teams_batches.append(new_teams_df)
                new_team_ids.update(int(team_id) for team_id in new_teams_df["team_id"].tolist())

            venue_df = self.venue_transformer.transform(game_data)
            if not venue_df.empty:
                venue_id_value = venue_df.iloc[0]["venue_id"]
                venue_id = int(venue_id_value) if pd.notna(venue_id_value) else None
                if venue_id is not None and venue_id not in dimension_state.venue_ids and venue_id not in new_venue_ids:
                    venues_batches.append(venue_df)
                    new_venue_ids.add(venue_id)

            players_df = self.player_transformer.transform(game_data)
            new_players_df = players_df[~players_df["player_id"].isin(dimension_state.player_ids | new_player_ids)]
            if not new_players_df.empty:
                players_batches.append(new_players_df)
                new_player_ids.update(int(player_id) for player_id in new_players_df["player_id"].tolist())

            game_df = self.game_transformer.transform(game_data)
            if not game_df.empty:
                games_batches.append(game_df)

            pitches_df = self.pitch_transformer.transform(game_data, game_id=game_pk, season=season)
            if not pitches_df.empty:
                pitches_batches.append(pitches_df)

            linescore_df = self.linescore_transformer.transform(game_data, game_pk=game_pk)
            if not linescore_df.empty:
                linescore_batches.append(linescore_df)

            boxscore_data = self.boxscore_transformer.transform_all(game_data, game_pk=game_pk)
            for table_name, df in boxscore_data.items():
                if df.empty:
                    continue
                stat_type = table_name.split("_")[1]
                if stat_type == "batting":
                    batting_batches.append(df)
                elif stat_type == "pitching":
                    pitching_batches.append(df)
                elif stat_type == "fielding":
                    fielding_batches.append(df)

        with db.connection.transaction():
            self._save_batch(teams_batches, lambda df: self.team_transformer.save_to_db(df, db))
            self._save_batch(venues_batches, lambda df: self.venue_transformer.save_to_db(df, db))
            self._save_batch(players_batches, lambda df: self.player_transformer.save_to_db(df, db))
            self._save_batch(games_batches, lambda df: self.game_transformer.save_to_db(df, db))
            self._save_batch(pitches_batches, lambda df: self.pitch_transformer.save_to_db(df, db))
            self._save_batch(linescore_batches, lambda df: self.linescore_transformer.save_to_db(df, db))
            self._save_batch(batting_batches, lambda df: self.boxscore_transformer.save_to_db(df, "batting", db))
            self._save_batch(pitching_batches, lambda df: self.boxscore_transformer.save_to_db(df, "pitching", db))
            self._save_batch(fielding_batches, lambda df: self.boxscore_transformer.save_to_db(df, "fielding", db))

        dimension_state.team_ids.update(new_team_ids)
        dimension_state.venue_ids.update(new_venue_ids)
        dimension_state.player_ids.update(new_player_ids)

    def _save_batch(
        self,
        batches: list[pd.DataFrame],
        saver,
    ) -> None:
        if not batches:
            return
        combined_df = pd.concat(batches, ignore_index=True)
        saver(combined_df)

    def _mark_bulk_progress(
        self,
        db: PostgresHandler,
        season: int,
        *,
        total_games: int,
        status: str,
        last_error: str | None,
    ) -> None:
        db.connection.execute(
            f"""
            INSERT INTO {BULK_BACKFILL_PROGRESS_TABLE} (
                season,
                total_games,
                status,
                last_error,
                loaded_at,
                updated_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                CASE WHEN %s = 'complete' THEN NOW() ELSE NULL END,
                NOW()
            )
            ON CONFLICT (season) DO UPDATE SET
                total_games = EXCLUDED.total_games,
                status = EXCLUDED.status,
                last_error = EXCLUDED.last_error,
                loaded_at = EXCLUDED.loaded_at,
                updated_at = NOW()
            """,
            (season, total_games, status, last_error, status),
        )


def default_seasons(
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
) -> list[str]:
    """Return the inclusive season list used for full historical syncs."""

    final_season = datetime.now(tz=UTC).year if end_season is None else end_season
    if final_season < start_season:
        raise ValueError("end_season must be greater than or equal to start_season")
    return [str(season) for season in range(start_season, final_season + 1)]


async def download_missing_schedules_async(
    schedule_dir: Path,
    seasons: Sequence[str],
) -> ScheduleDownloadSummary:
    """Download any missing schedule JSON files for the requested seasons."""

    schedule_dir.mkdir(parents=True, exist_ok=True)

    missing_seasons = [
        season
        for season in seasons
        if not (schedule_dir / f"schedule_{season}.json").exists()
    ]
    skipped_existing = len(seasons) - len(missing_seasons)

    if not missing_seasons:
        return ScheduleDownloadSummary(downloaded=0, skipped_existing=skipped_existing)

    progress_bar = tqdm(total=len(missing_seasons), desc="Downloading schedules", unit="season")

    async def fetch_and_save(schedule_api: Schedule, season: str) -> None:
        output_path = schedule_dir / f"schedule_{season}.json"
        schedule_data = await schedule_api.get_async(sportId=1, season=int(season))
        async with aiofiles.open(output_path, "w") as handle:
            await handle.write(json.dumps(schedule_data, indent=2))
        progress_bar.update(1)

    try:
        async with Schedule(concurrency_limit=min(15, len(missing_seasons))) as schedule_api:
            await asyncio.gather(
                *(fetch_and_save(schedule_api, season) for season in missing_seasons)
            )
    finally:
        progress_bar.close()

    return ScheduleDownloadSummary(
        downloaded=len(missing_seasons),
        skipped_existing=skipped_existing,
    )


def download_missing_schedules(
    schedule_dir: Path,
    seasons: Sequence[str],
) -> ScheduleDownloadSummary:
    return asyncio.run(download_missing_schedules_async(schedule_dir, seasons))


def run_postgres_backfill(db_config: PostgresConfig, raw_data_path: Path) -> BackfillSummary:
    """Backfill raw live feed JSON files into PostgreSQL with resume support."""

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_data_path}")

    return PostgresBackfill(db_config, raw_data_path).run()


def run_postgres_bulk_backfill(db_config: PostgresConfig, raw_data_path: Path) -> BulkBackfillSummary:
    """Backfill raw live feed JSON files into PostgreSQL in season-sized bulk batches."""

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_data_path}")

    return BulkHistoricalBackfill(db_config, raw_data_path).run()


async def download_raw_data_async(
    raw_data_path: Path,
    seasons: Sequence[str],
) -> tuple[ScheduleDownloadSummary, dict[str, dict[str, int]]]:
    """Download missing schedules and live feed JSON files using async APIs."""

    schedule_dir = raw_data_path.parent / "schedules"
    schedule_summary = await download_missing_schedules_async(schedule_dir, seasons)
    live_feed_stats = await live_feed_etl_async(
        skip_existing=True,
        seasons=list(seasons),
        schedule_path=schedule_dir,
        live_feeds_root=raw_data_path,
    )
    return schedule_summary, live_feed_stats


def download_raw_data(
    raw_data_path: Path,
    seasons: Sequence[str],
) -> tuple[ScheduleDownloadSummary, dict[str, dict[str, int]]]:
    return asyncio.run(download_raw_data_async(raw_data_path, seasons))


def sync_and_backfill_postgres(
    db_config: PostgresConfig,
    raw_data_path: Path,
    seasons: Sequence[str] | None = None,
    *,
    bulk_historical: bool = False,
) -> PipelineSummary:
    """Download missing raw data asynchronously and resume the PostgreSQL backfill."""

    selected_seasons = list(seasons) if seasons is not None else default_seasons()

    print("\n[Phase 1] Downloading missing schedule and live feed JSON files...")
    schedule_summary, live_feed_stats = download_raw_data(raw_data_path, selected_seasons)

    print("\n[Phase 2] Backfilling PostgreSQL from raw live feeds...")
    if bulk_historical:
        backfill_summary = run_postgres_bulk_backfill(db_config, raw_data_path)
    else:
        backfill_summary = run_postgres_backfill(db_config, raw_data_path)

    return PipelineSummary(
        schedules=schedule_summary,
        live_feed_stats=live_feed_stats,
        backfill=backfill_summary,
    )
