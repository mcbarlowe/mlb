#!/usr/bin/env python3
"""Fetch historical MLB first-five totals and store open/close rows.

This uses The Odds API historical *event odds* endpoint because period markets such
as ``totals_1st_5_innings`` are not supported by the cheaper all-slate historical
odds endpoint. Writes are idempotent upserts into ``mlb.f5_odds`` keyed by
``(game_pk, bookmaker, line_type)``.

Typical bounded backfill:

    uv run python scripts/fetch_f5_odds_history.py --season 2025 --line-types open,close
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig, PostgresHandler
from src.market_data.f5_odds import F5_TOTALS_MARKET, parse_f5_odds_rows
from src.market_data.f5_odds_store import upsert_f5_odds_rows
from src.market_data.team_mapping import team_abbrev_to_id

HISTORICAL_EVENTS_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"
HISTORICAL_EVENT_ODDS_URL = (
    "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/{event_id}/odds"
)
SOURCE = "the-odds-api-historical"
CSV_FIELDS = (
    "game_pk",
    "season",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "bookmaker",
    "line_type",
    "snapshot_time",
    "h2h_last_update",
    "spreads_last_update",
    "totals_last_update",
    "home_ml",
    "away_ml",
    "home_spread",
    "home_spread_ml",
    "away_spread",
    "away_spread_ml",
    "total_point",
    "over_ml",
    "under_ml",
    "source",
)
NAME_ALIASES = {
    "CLEVELAND GUARDIANS": "CLEVELAND INDIANS",
    "MIAMI MARLINS": "FLORIDA MARLINS",
}


@dataclass(frozen=True)
class HistoricalGame:
    game_pk: int
    season: int
    game_date: str
    game_time: datetime
    away_team: str
    home_team: str
    away_team_id: int
    home_team_id: int


@dataclass(frozen=True)
class ApiResult:
    payload: Mapping[str, Any]
    cost: int
    remaining: str | None
    used: str | None


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_dt(value: object) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _request_json(
    session: requests.Session,
    url: str,
    params: Mapping[str, str | int | float],
    *,
    attempts: int,
) -> ApiResult:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return ApiResult(
                    payload=response.json(),
                    cost=int(response.headers.get("x-requests-last") or 0),
                    remaining=response.headers.get("x-requests-remaining"),
                    used=response.headers.get("x-requests-used"),
                )
            if response.status_code in {401, 402, 422, 429}:
                raise SystemExit(f"Odds API {response.status_code}: {response.text[:500]}")
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2.0 * attempt)
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def _resolve_team(name: object, mapping: Mapping[str, int]) -> int | None:
    if name is None:
        return None
    key = str(name).strip().upper()
    return mapping.get(key) or mapping.get(NAME_ALIASES.get(key, key))


def _event_data(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    return []


def _match_event(
    *,
    events: Sequence[Mapping[str, Any]],
    game: HistoricalGame,
    team_mapping: Mapping[str, int],
    max_match_hours: float,
) -> Mapping[str, Any] | None:
    best: tuple[float, Mapping[str, Any]] | None = None
    for event in events:
        home_id = _resolve_team(event.get("home_team"), team_mapping)
        away_id = _resolve_team(event.get("away_team"), team_mapping)
        if home_id != game.home_team_id or away_id != game.away_team_id:
            continue
        commence = event.get("commence_time")
        if commence is None:
            continue
        diff_hours = abs((_parse_dt(commence) - game.game_time).total_seconds()) / 3600.0
        if diff_hours > max_match_hours:
            continue
        if best is None or diff_hours < best[0]:
            best = (diff_hours, event)
    return None if best is None else best[1]


def _load_games(
    *,
    season: int | None,
    start: str | None,
    end: str | None,
    limit: int,
    db_config: PostgresConfig | None = None,
) -> list[HistoricalGame]:
    filters = ["g.game_type = 'R'", "g.game_datetime IS NOT NULL"]
    params: list[object] = []
    if season is not None:
        filters.append("g.season = %s")
        params.append(season)
    if start is not None:
        filters.append("g.game_date >= %s")
        params.append(start)
    if end is not None:
        filters.append("g.game_date <= %s")
        params.append(end)
    limit_clause = " LIMIT %s" if limit > 0 else ""
    if limit > 0:
        params.append(limit)
    where_clause = " AND ".join(filters)
    query = f"""
        SELECT g.game_pk, g.season, g.game_date, g.game_datetime,
               away.abbreviation AS away_team, home.abbreviation AS home_team,
               g.away_team_id, g.home_team_id
        FROM games g
        JOIN teams away ON away.team_id = g.away_team_id
        JOIN teams home ON home.team_id = g.home_team_id
        WHERE {where_clause}
        ORDER BY g.game_datetime, g.game_pk
        {limit_clause}
        """
    with PostgresHandler(db_config) as db, db.connection.cursor() as cursor:
        cursor.execute(cast(Any, query), tuple(params))
        rows = cursor.fetchall()
    return [
        HistoricalGame(
            game_pk=int(row[0]),
            season=int(row[1]),
            game_date=str(row[2]),
            game_time=_parse_dt(row[3]),
            away_team=str(row[4]),
            home_team=str(row[5]),
            away_team_id=int(row[6]),
            home_team_id=int(row[7]),
        )
        for row in rows
    ]


def _existing_game_line_types(
    games: Sequence[HistoricalGame],
    line_types: Iterable[str],
    db_config: PostgresConfig | None = None,
) -> set[tuple[int, str]]:
    if not games:
        return set()
    game_pks = [game.game_pk for game in games]
    wanted = list(line_types)
    query = """
        SELECT DISTINCT game_pk, line_type
        FROM f5_odds
        WHERE game_pk = ANY(%s)
          AND line_type = ANY(%s)
          AND total_point IS NOT NULL
    """
    with PostgresHandler(db_config) as db, db.connection.cursor() as cursor:
        try:
            cursor.execute(query, (game_pks, wanted))
            rows = cursor.fetchall()
        except Exception:
            return set()
    return {(int(game_pk), str(line_type)) for game_pk, line_type in rows}


def _line_timestamp(game: HistoricalGame, line_type: str, open_hours_before: float, close_minutes_before: float) -> str:
    if line_type == "open":
        return _utc_iso(game.game_time - timedelta(hours=open_hours_before))
    if line_type == "close":
        return _utc_iso(game.game_time - timedelta(minutes=close_minutes_before))
    raise ValueError(f"unsupported line type {line_type!r}")


def _rows_from_event_odds(
    *,
    payload: Mapping[str, Any],
    game: HistoricalGame,
    line_type: str,
    snapshot_time: str,
) -> list[dict[str, object | None]]:
    event_payload = payload.get("data")
    if not isinstance(event_payload, Mapping):
        return []
    rows: list[dict[str, object | None]] = []
    for odds_row in parse_f5_odds_rows(event_payload):
        if odds_row.total_point is None or odds_row.over_ml is None or odds_row.under_ml is None:
            continue
        rows.append(
            {
                "game_pk": game.game_pk,
                "season": game.season,
                "game_date": game.game_date,
                "game_time": game.game_time.isoformat(),
                "away_team": game.away_team,
                "home_team": game.home_team,
                "away_team_id": game.away_team_id,
                "home_team_id": game.home_team_id,
                "bookmaker": odds_row.bookmaker,
                "line_type": line_type,
                "snapshot_time": snapshot_time,
                "h2h_last_update": odds_row.h2h_last_update,
                "spreads_last_update": odds_row.spreads_last_update,
                "totals_last_update": odds_row.totals_last_update,
                "home_ml": odds_row.home_ml,
                "away_ml": odds_row.away_ml,
                "home_spread": odds_row.home_spread,
                "home_spread_ml": odds_row.home_spread_ml,
                "away_spread": odds_row.away_spread,
                "away_spread_ml": odds_row.away_spread_ml,
                "total_point": odds_row.total_point,
                "over_ml": odds_row.over_ml,
                "under_ml": odds_row.under_ml,
                "source": SOURCE,
            }
        )
    return rows


def _append_csv(path: Path, rows: Sequence[Mapping[str, object | None]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _parse_line_types(value: str) -> tuple[str, ...]:
    line_types = tuple(item.strip() for item in value.split(",") if item.strip())
    unsupported = sorted(set(line_types) - {"open", "close"})
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported line types: {unsupported}")
    if not line_types:
        raise argparse.ArgumentTypeError("at least one line type is required")
    return line_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--start", default=None, help="inclusive game_date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="inclusive game_date YYYY-MM-DD")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--line-types", type=_parse_line_types, default=("open", "close"))
    parser.add_argument("--open-hours-before", type=float, default=8.0)
    parser.add_argument("--close-minutes-before", type=float, default=10.0)
    parser.add_argument("--max-match-hours", type=float, default=12.0)
    parser.add_argument("--games", type=int, default=0, help="limit games for smoke runs")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--flush-games", type=int, default=25, help="persist fetched rows every N games")
    parser.add_argument("--refresh", action="store_true", help="refetch even if a game/line_type already has totals rows")
    parser.add_argument("--dry-run", action="store_true", help="fetch and stage rows but do not write Postgres")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--out", default="output/odds/f5_odds_history.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY is required")
    if args.season is None and (args.start is None or args.end is None):
        raise SystemExit("Provide --season or both --start and --end")

    db_config = PostgresConfig.from_env()
    games = _load_games(season=args.season, start=args.start, end=args.end, limit=args.games)
    if not games:
        raise SystemExit("No regular-season games matched the requested scope")
    existing = set() if args.refresh else _existing_game_line_types(games, args.line_types, db_config)

    print(f"DB target: {db_config.describe()}")
    print(
        f"scope: games={len(games)} season={args.season or ''} "
        f"start={args.start or games[0].game_date} end={args.end or games[-1].game_date} "
        f"line_types={','.join(args.line_types)} refresh={args.refresh} dry_run={args.dry_run}"
    )
    print(
        f"timestamps: open={args.open_hours_before:g}h before first pitch; "
        f"close={args.close_minutes_before:g}m before first pitch"
    )

    session = requests.Session()
    team_mapping = team_abbrev_to_id()
    event_cache: dict[str, ApiResult] = {}
    pending_rows: list[dict[str, object | None]] = []
    total_rows = 0
    db_written = 0
    unmatched_events = 0
    no_totals = 0
    skipped_existing = 0
    event_calls = 0
    odds_calls = 0
    cost = 0
    remaining = ""
    used = ""

    total_tasks = len(games) * len(args.line_types)
    completed_tasks = 0

    flush_games = max(1, int(args.flush_games))

    def flush_pending() -> None:
        nonlocal db_written, pending_rows
        if not pending_rows:
            return
        if not args.no_csv:
            _append_csv(Path(args.out), pending_rows)
        if not args.dry_run:
            db_written += upsert_f5_odds_rows(pending_rows, db_config=db_config)
        pending_rows = []
    for index, game in enumerate(games, 1):
        event_by_ts: dict[str, Mapping[str, Any] | None] = {}
        for line_type in args.line_types:
            completed_tasks += 1
            if (game.game_pk, line_type) in existing:
                skipped_existing += 1
                continue
            timestamp = _line_timestamp(
                game,
                line_type,
                open_hours_before=args.open_hours_before,
                close_minutes_before=args.close_minutes_before,
            )
            if timestamp not in event_by_ts:
                result = event_cache.get(timestamp)
                if result is None:
                    result = _request_json(
                        session,
                        HISTORICAL_EVENTS_URL,
                        {"apiKey": api_key, "date": timestamp, "dateFormat": "iso"},
                        attempts=args.attempts,
                    )
                    event_cache[timestamp] = result
                    event_calls += 1
                    cost += result.cost
                    remaining = result.remaining or remaining
                    used = result.used or used
                    time.sleep(args.sleep)
                event_by_ts[timestamp] = _match_event(
                    events=_event_data(result.payload),
                    game=game,
                    team_mapping=team_mapping,
                    max_match_hours=args.max_match_hours,
                )
            event = event_by_ts[timestamp]
            if event is None:
                unmatched_events += 1
                continue
            event_id = event.get("id")
            if event_id is None:
                unmatched_events += 1
                continue
            result = _request_json(
                session,
                HISTORICAL_EVENT_ODDS_URL.format(event_id=event_id),
                {
                    "apiKey": api_key,
                    "regions": args.regions,
                    "markets": F5_TOTALS_MARKET,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                    "date": timestamp,
                },
                attempts=args.attempts,
            )
            odds_calls += 1
            cost += result.cost
            remaining = result.remaining or remaining
            used = result.used or used
            snapshot_time = str(result.payload.get("timestamp") or timestamp)
            fetched = _rows_from_event_odds(
                payload=result.payload,
                game=game,
                line_type=line_type,
                snapshot_time=snapshot_time,
            )
            if not fetched:
                no_totals += 1
            pending_rows.extend(fetched)
            total_rows += len(fetched)
            time.sleep(args.sleep)
        if index % flush_games == 0 or index == len(games):
            flush_pending()
            print(
                f"[{index}/{len(games)} games, {completed_tasks}/{total_tasks} tasks] "
                f"rows={total_rows} db_upserts={db_written} event_calls={event_calls} odds_calls={odds_calls} "
                f"cost={cost} remaining={remaining} skipped_existing={skipped_existing} "
                f"unmatched={unmatched_events} no_totals={no_totals}",
                flush=True,
            )

    flush_pending()
    if args.dry_run:
        print("dry-run: no F5 odds rows written to Postgres")
    print(f"finished: rows={total_rows} db_upserts={db_written} staged_csv={'' if args.no_csv else args.out}")


if __name__ == "__main__":
    main()
