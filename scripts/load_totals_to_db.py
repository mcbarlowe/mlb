"""Load staged Odds API totals (O/U) into Postgres.

Every staged API row is written to ``mlb.odds_totals_raw`` first. Rows that
resolve safely to a regular-season ``game_pk`` are also upserted into
``mlb.odds_totals`` for model/CLV consumers. Idempotent.

    uv run python scripts/load_totals_to_db.py --stage data/odds/odds_2025_totals_stage.parquet --season 2025
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, LiteralString

import polars as pl
import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

TOTALS_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.odds_totals (
    game_pk       integer NOT NULL,
    game_date     date,
    away_team_id  integer,
    home_team_id  integer,
    bookmaker     varchar NOT NULL,
    line_type     varchar NOT NULL DEFAULT 'close',
    total_point   numeric,
    over_ml       integer,
    under_ml      integer,
    snapshot_time timestamptz,
    source        varchar DEFAULT 'the-odds-api',
    ingested_at   timestamptz DEFAULT now(),
    PRIMARY KEY (game_pk, bookmaker, line_type)
);
"""

RAW_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.odds_totals_raw (
    season integer NOT NULL,
    api_game_id text NOT NULL,
    line_type varchar NOT NULL,
    commence_time timestamptz,
    game_date_utc date,
    away_team text,
    home_team text,
    bookmaker varchar NOT NULL,
    snapshot_time timestamptz,
    book_last_update timestamptz,
    total_point numeric,
    over_ml integer,
    under_ml integer,
    source varchar DEFAULT 'the-odds-api',
    ingested_at timestamptz DEFAULT now(),
    PRIMARY KEY (season, api_game_id, bookmaker, line_type)
);
"""

TOTALS_UPSERT = """
INSERT INTO {schema}.odds_totals
    (game_pk, game_date, away_team_id, home_team_id, bookmaker, line_type,
     total_point, over_ml, under_ml, snapshot_time, source)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'the-odds-api')
ON CONFLICT (game_pk, bookmaker, line_type) DO UPDATE SET
    total_point=EXCLUDED.total_point, over_ml=EXCLUDED.over_ml,
    under_ml=EXCLUDED.under_ml, snapshot_time=EXCLUDED.snapshot_time,
    ingested_at=now();
"""

RAW_UPSERT = """
INSERT INTO {schema}.odds_totals_raw
    (season, api_game_id, line_type, commence_time, game_date_utc, away_team,
     home_team, bookmaker, snapshot_time, book_last_update, total_point,
     over_ml, under_ml, source)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'the-odds-api')
ON CONFLICT (season, api_game_id, bookmaker, line_type) DO UPDATE SET
    commence_time=EXCLUDED.commence_time,
    game_date_utc=EXCLUDED.game_date_utc,
    away_team=EXCLUDED.away_team,
    home_team=EXCLUDED.home_team,
    snapshot_time=EXCLUDED.snapshot_time,
    book_last_update=EXCLUDED.book_last_update,
    total_point=EXCLUDED.total_point,
    over_ml=EXCLUDED.over_ml,
    under_ml=EXCLUDED.under_ml,
    source=EXCLUDED.source,
    ingested_at=now();
"""


def _dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _optional_dt(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    return _dt(value)


def _schema_query(template: LiteralString, schema: str) -> sql.Composed:
    return sql.SQL(template).format(schema=sql.Identifier(schema))


def build_raw_totals_rows(
    staged: pl.DataFrame,
    *,
    season: int,
    line_type: str,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for row in staged.iter_rows(named=True):
        rows.append(
            (
                season,
                str(row["game_id"]),
                line_type,
                _optional_dt(row.get("commence_time")),
                row.get("game_date_utc"),
                row.get("away_team"),
                row.get("home_team"),
                row.get("bookmaker"),
                _optional_dt(row.get("snapshot_time")),
                _optional_dt(row.get("book_last_update")),
                row.get("total_point"),
                row.get("over_ml"),
                row.get("under_ml"),
            )
        )
    return rows


def build_resolved_totals_rows(
    staged: pl.DataFrame,
    *,
    gid_to_pk: dict[str, tuple[int, int, int, datetime]],
    line_type: str,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for row in staged.iter_rows(named=True):
        match = gid_to_pk.get(row["game_id"])
        if match is None:
            continue
        game_pk, hid, aid, commence = match
        rows.append(
            (
                game_pk,
                commence.date(),
                aid,
                hid,
                row["bookmaker"],
                line_type,
                float(row["total_point"]),
                int(row["over_ml"]),
                int(row["under_ml"]),
                _optional_dt(row.get("snapshot_time")),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--line-type", choices=("close", "open"), default="close")
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()

    staged = pl.read_parquet(args.stage)
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    raw_params = build_raw_totals_rows(
        staged,
        season=args.season,
        line_type=args.line_type,
    )
    with conn.cursor() as cur:
        cur.execute(_schema_query(TOTALS_DDL, c.schema))
        cur.execute(_schema_query(RAW_DDL, c.schema))
        if raw_params:
            cur.executemany(_schema_query(RAW_UPSERT, c.schema), raw_params)
        cur.execute(
            _schema_query("SELECT team_id, team_name FROM {schema}.teams", c.schema)
        )
        name_to_id = {str(n): int(i) for i, n in cur.fetchall()}
        cur.execute(
            _schema_query(
                """SELECT game_pk, game_datetime, home_team_id, away_team_id
                FROM {schema}.games
                WHERE season::int=%s AND game_type='R' AND game_datetime IS NOT NULL""",
                c.schema,
            ),
            (args.season,),
        )
        pair_games: dict[tuple[int, int], list[tuple[int, datetime]]] = {}
        for game_pk, gdt, hid, aid in cur.fetchall():
            pair_games.setdefault((int(hid), int(aid)), []).append((int(game_pk), _dt(gdt)))

        games = staged.select(
            ["game_id", "commence_time", "home_team", "away_team"]
        ).unique(subset=["game_id"])
        gid_to_pk: dict[str, tuple[int, int, int, datetime]] = {}
        for row in games.iter_rows(named=True):
            hid = name_to_id.get(row["home_team"])
            aid = name_to_id.get(row["away_team"])
            if hid is None or aid is None:
                continue
            commence = _dt(row["commence_time"])
            best = None
            for game_pk, gdt in pair_games.get((hid, aid), []):
                delta = abs((gdt - commence).total_seconds())
                if best is None or delta < best[1]:
                    best = (game_pk, delta)
            if best is None or best[1] > args.max_hours * 3600:
                continue
            gid_to_pk[row["game_id"]] = (best[0], hid, aid, commence)

        params = build_resolved_totals_rows(
            staged,
            gid_to_pk=gid_to_pk,
            line_type=args.line_type,
        )
        if params:
            cur.executemany(_schema_query(TOTALS_UPSERT, c.schema), params)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            _schema_query(
                """SELECT count(DISTINCT o.game_pk), count(*) FROM {schema}.odds_totals o
                JOIN {schema}.games g USING(game_pk)
                WHERE g.season::int=%s AND o.line_type=%s""",
                c.schema,
            ),
            (args.season, args.line_type),
        )
        counts = cur.fetchone()
        if counts is None:
            raise RuntimeError("odds_totals count query returned no row")
        distinct_pk, total_rows = counts
        cur.execute(
            _schema_query(
                "SELECT count(*) FROM {schema}.games WHERE season::int=%s AND game_type='R'",
                c.schema,
            ),
            (args.season,),
        )
        reg_count = cur.fetchone()
        if reg_count is None:
            raise RuntimeError("regular-season game count query returned no row")
        reg = reg_count[0]
    conn.close()
    print(f"raw staged rows upserted into odds_totals_raw: {len(raw_params)}")
    print(f"staged rows: {staged.height}, matched games: {len(gid_to_pk)}, upserts: {len(params)}")
    print(f"mlb.odds_totals ({args.line_type}) {args.season}: {total_rows} rows over "
          f"{distinct_pk} games ({distinct_pk}/{reg} = {distinct_pk / reg:.1%})")


if __name__ == "__main__":
    main()
