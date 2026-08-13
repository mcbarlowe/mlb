"""Load staged Odds API totals (O/U) into ``mlb.odds_totals`` (per book).

Same team/game_pk resolution as the moneyline loader; stores each book's total
line and over/under prices. Idempotent.

    uv run python scripts/load_totals_to_db.py --stage data/odds/odds_2025_totals_stage.parquet --season 2025
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

DDL = """
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

UPSERT = """
INSERT INTO {schema}.odds_totals
    (game_pk, game_date, away_team_id, home_team_id, bookmaker, line_type,
     total_point, over_ml, under_ml, snapshot_time, source)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'the-odds-api')
ON CONFLICT (game_pk, bookmaker, line_type) DO UPDATE SET
    total_point=EXCLUDED.total_point, over_ml=EXCLUDED.over_ml,
    under_ml=EXCLUDED.under_ml, snapshot_time=EXCLUDED.snapshot_time,
    ingested_at=now();
"""


def _dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


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
    with conn.cursor() as cur:
        cur.execute(DDL.format(schema=c.schema))
        cur.execute(f"SELECT team_id, team_name FROM {c.schema}.teams")
        name_to_id = {str(n): int(i) for i, n in cur.fetchall()}
        cur.execute(
            f"""SELECT game_pk, game_datetime, home_team_id, away_team_id
                FROM {c.schema}.games
                WHERE season::int=%s AND game_type='R' AND game_datetime IS NOT NULL""",
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

        params = []
        for row in staged.iter_rows(named=True):
            match = gid_to_pk.get(row["game_id"])
            if match is None:
                continue
            game_pk, hid, aid, commence = match
            params.append((
                game_pk, commence.date(), aid, hid, row["bookmaker"], args.line_type,
                float(row["total_point"]), int(row["over_ml"]), int(row["under_ml"]),
                _dt(row["snapshot_time"]) if row["snapshot_time"] else None,
            ))
        cur.executemany(UPSERT.format(schema=c.schema), params)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(DISTINCT o.game_pk), count(*) FROM {c.schema}.odds_totals o
                JOIN {c.schema}.games g USING(game_pk)
                WHERE g.season::int=%s AND o.line_type=%s""",
            (args.season, args.line_type),
        )
        distinct_pk, total_rows = cur.fetchone()
        cur.execute(
            f"SELECT count(*) FROM {c.schema}.games WHERE season::int=%s AND game_type='R'",
            (args.season,),
        )
        reg = cur.fetchone()[0]
    conn.close()
    print(f"staged rows: {staged.height}, matched games: {len(gid_to_pk)}, upserts: {len(params)}")
    print(f"mlb.odds_totals ({args.line_type}) {args.season}: {total_rows} rows over "
          f"{distinct_pk} games ({distinct_pk}/{reg} = {distinct_pk / reg:.1%})")


if __name__ == "__main__":
    main()
