"""Load staged Odds API history into ``mlb.odds`` (per-book moneylines).

Resolves full team names -> team_id, matches each odds game to a game_pk by the
nearest scheduled game_datetime for that team pair (robust to UTC/ET offset and
doubleheaders), and upserts per-book close lines. Idempotent.

    uv run python scripts/load_odds_to_db.py --stage data/odds/odds_2024_stage.parquet --season 2024
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

# Odds API uses current names; teams table still has some stale names.
NAME_ALIASES = {
    "Cleveland Guardians": "Cleveland Indians",
    "Miami Marlins": "Florida Marlins",
}

DDL = """
CREATE TABLE IF NOT EXISTS {schema}.odds (
    game_pk       integer NOT NULL,
    game_date     date,
    away_team_id  integer,
    home_team_id  integer,
    bookmaker     varchar NOT NULL,
    market        varchar NOT NULL DEFAULT 'h2h',
    line_type     varchar NOT NULL DEFAULT 'close',
    home_ml       integer,
    away_ml       integer,
    snapshot_time timestamptz,
    source        varchar DEFAULT 'the-odds-api',
    ingested_at   timestamptz DEFAULT now(),
    PRIMARY KEY (game_pk, bookmaker, market, line_type)
);
"""

UPSERT = """
INSERT INTO {schema}.odds
    (game_pk, game_date, away_team_id, home_team_id, bookmaker, market,
     line_type, home_ml, away_ml, snapshot_time, source)
VALUES (%s,%s,%s,%s,%s,'h2h',%s,%s,%s,%s,'the-odds-api')
ON CONFLICT (game_pk, bookmaker, market, line_type) DO UPDATE SET
    home_ml=EXCLUDED.home_ml, away_ml=EXCLUDED.away_ml,
    snapshot_time=EXCLUDED.snapshot_time, ingested_at=now();
"""


def _dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="data/odds/odds_2024_stage.parquet")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--line-type", choices=("close", "open"), default="close")
    args = ap.parse_args()

    staged = pl.read_parquet(args.stage)
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(DDL.format(schema=c.schema))
        # full-name -> team_id (restrict to MLB via games participation below)
        cur.execute(f"SELECT team_id, team_name FROM {c.schema}.teams")
        name_to_id = {str(n): int(i) for i, n in cur.fetchall()}
        # 2024 regular-season schedule for pk matching
        cur.execute(
            f"""SELECT game_pk, game_datetime, home_team_id, away_team_id
                FROM {c.schema}.games
                WHERE season::int=%s AND game_type='R' AND game_datetime IS NOT NULL""",
            (args.season,),
        )
        pair_games: dict[tuple[int, int], list[tuple[int, datetime]]] = {}
        for game_pk, gdt, home_id, away_id in cur.fetchall():
            pair_games.setdefault((int(home_id), int(away_id)), []).append(
                (int(game_pk), _dt(gdt))
            )

        # one representative row per (game_id) to resolve pk once
        games = staged.select(
            ["game_id", "commence_time", "home_team", "away_team"]
        ).unique(subset=["game_id"])
        gid_to_pk: dict[str, tuple[int, int, int, datetime]] = {}
        unmatched_names: set[str] = set()
        unmatched_games = 0
        for row in games.iter_rows(named=True):
            hid = name_to_id.get(row["home_team"]) or name_to_id.get(
                NAME_ALIASES.get(row["home_team"], "")
            )
            aid = name_to_id.get(row["away_team"]) or name_to_id.get(
                NAME_ALIASES.get(row["away_team"], "")
            )
            if hid is None:
                unmatched_names.add(row["home_team"])
            if aid is None:
                unmatched_names.add(row["away_team"])
            if hid is None or aid is None:
                continue
            commence = _dt(row["commence_time"])
            candidates = pair_games.get((hid, aid), [])
            best = None
            for game_pk, gdt in candidates:
                delta = abs((gdt - commence).total_seconds())
                if best is None or delta < best[1]:
                    best = (game_pk, delta)
            if best is None or best[1] > args.max_hours * 3600:
                unmatched_games += 1
                continue
            gid_to_pk[row["game_id"]] = (best[0], hid, aid, commence)

        # upsert per-book rows for matched games (batched)
        params = []
        for row in staged.iter_rows(named=True):
            match = gid_to_pk.get(row["game_id"])
            if match is None:
                continue
            game_pk, hid, aid, commence = match
            params.append((
                game_pk, commence.date(), aid, hid, row["bookmaker"], args.line_type,
                int(row["home_ml"]), int(row["away_ml"]),
                _dt(row["snapshot_time"]) if row["snapshot_time"] else None,
            ))
        cur.executemany(UPSERT.format(schema=c.schema), params)
        upserts = len(params)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(DISTINCT o.game_pk), count(*) FROM {c.schema}.odds o
                JOIN {c.schema}.games g USING(game_pk)
                WHERE g.season::int=%s AND o.line_type=%s""",
            (args.season, args.line_type),
        )
        distinct_pk, total_rows = cur.fetchone()
        cur.execute(
            f"SELECT count(*) FROM {c.schema}.games WHERE season::int=%s AND game_type='R'",
            (args.season,),
        )
        reg_games = cur.fetchone()[0]
    conn.close()

    print(f"staged book-rows: {staged.height}, staged games: {games.height}")
    print(f"matched games: {len(gid_to_pk)}, unmatched games: {unmatched_games}")
    if unmatched_names:
        print(f"unmatched team names: {sorted(unmatched_names)}")
    print(f"upserted rows: {upserts}")
    print(
        f"mlb.odds now: {total_rows} rows over {distinct_pk} games "
        f"(coverage {distinct_pk}/{reg_games} = {distinct_pk / reg_games:.1%} of {args.season} regular games)"
    )


if __name__ == "__main__":
    main()
