#!/usr/bin/env python3
"""Fill missing h2h closing lines for dates with paper-trade or open odds rows.

The paper-trade ledger records selected bets, and the opening-line saver records
the matched daily h2h board. CLV/backtests need matching ``mlb.odds`` h2h close
rows, so this script finds prior dates from either source with missing close
odds, fetches that date's near-close historical slate snapshot, and loads the
staged rows into ``mlb.odds``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

DEFAULT_OUT_DIR = Path("data/odds")


@dataclass(frozen=True)
class MissingCloseDate:
    paper_date: date
    paper_games: int
    missing_close_games: int
    open_games: int = 0


def _qualified(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def _table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        return cursor.fetchone()[0] is not None


def missing_close_dates(
    db_config: PostgresConfig,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[MissingCloseDate]:
    """Return prior paper/open dates where any source game lacks h2h close odds."""
    filters = [sql.SQL("source_date < CURRENT_DATE")]
    params: list[object] = []
    if start is not None:
        filters.append(sql.SQL("source_date >= %s"))
        params.append(start)
    if end is not None:
        filters.append(sql.SQL("source_date <= %s"))
        params.append(end)

    where_clause = sql.SQL(" AND ").join(filters)
    with psycopg.connect(
        dbname=db_config.dbname,
        user=db_config.user,
        password=db_config.password,
        host=db_config.host,
        port=db_config.port,
        autocommit=True,
    ) as conn:
        odds_exists = _table_exists(conn, db_config.schema, "odds")
        with conn.cursor() as cursor:
            if odds_exists:
                query = sql.SQL(
                    """
                    WITH source_games AS (
                        SELECT paper_date AS source_date,
                               game_pk,
                               TRUE AS from_paper,
                               FALSE AS from_open
                        FROM {paper_trades}
                        UNION ALL
                        SELECT game_date AS source_date,
                               game_pk,
                               FALSE AS from_paper,
                               TRUE AS from_open
                        FROM {odds}
                        WHERE market = 'h2h'
                          AND line_type = 'open'
                          AND game_date IS NOT NULL
                          AND home_ml IS NOT NULL
                          AND away_ml IS NOT NULL
                    ),
                    source_rollup AS (
                        SELECT source_date,
                               game_pk,
                               BOOL_OR(from_paper) AS from_paper,
                               BOOL_OR(from_open) AS from_open
                        FROM source_games
                        GROUP BY source_date, game_pk
                    ),
                    filtered AS (
                        SELECT *
                        FROM source_rollup
                        WHERE {where_clause}
                    )
                    SELECT f.source_date,
                           COUNT(DISTINCT f.game_pk) FILTER (
                               WHERE f.from_paper
                           ) AS paper_games,
                           COUNT(DISTINCT f.game_pk) FILTER (
                               WHERE f.from_open
                           ) AS open_games,
                           COUNT(DISTINCT f.game_pk) FILTER (
                               WHERE o.game_pk IS NULL
                           ) AS missing_close_games
                    FROM filtered f
                    LEFT JOIN {odds} o
                      ON o.game_pk = f.game_pk
                     AND o.market = 'h2h'
                     AND o.line_type = 'close'
                     AND o.home_ml IS NOT NULL
                     AND o.away_ml IS NOT NULL
                    GROUP BY f.source_date
                    HAVING COUNT(DISTINCT f.game_pk) FILTER (
                        WHERE o.game_pk IS NULL
                    ) > 0
                    ORDER BY f.source_date
                    """
                ).format(
                    paper_trades=_qualified(db_config.schema, "paper_trades"),
                    odds=_qualified(db_config.schema, "odds"),
                    where_clause=where_clause,
                )
            else:
                query = sql.SQL(
                    """
                    WITH source_games AS (
                        SELECT paper_date AS source_date, game_pk
                        FROM {paper_trades}
                    )
                    SELECT source_date,
                           COUNT(DISTINCT game_pk) AS paper_games,
                           0 AS open_games,
                           COUNT(DISTINCT game_pk) AS missing_close_games
                    FROM source_games
                    WHERE {where_clause}
                    GROUP BY source_date
                    ORDER BY source_date
                    """
                ).format(
                    paper_trades=_qualified(db_config.schema, "paper_trades"),
                    where_clause=where_clause,
                )
            cursor.execute(query, params)
            return [
                MissingCloseDate(
                    paper_date=row[0],
                    paper_games=int(row[1]),
                    open_games=int(row[2]),
                    missing_close_games=int(row[3]),
                )
                for row in cursor.fetchall()
            ]


def stage_path(out_dir: Path, paper_date: date) -> Path:
    return out_dir / f"paper_h2h_close_{paper_date.isoformat()}.parquet"


def commands_for_date(
    *,
    paper_date: date,
    stage: Path,
    python: str,
) -> tuple[list[str], list[str]]:
    fetch_cmd = [
        python,
        "scripts/fetch_odds_history.py",
        "--start",
        paper_date.isoformat(),
        "--end",
        paper_date.isoformat(),
        "--markets",
        "h2h",
        "--keep",
        "latest",
        "--out",
        str(stage),
    ]
    load_cmd = [
        python,
        "scripts/load_odds_to_db.py",
        "--stage",
        str(stage),
        "--season",
        str(paper_date.year),
        "--line-type",
        "close",
    ]
    return fetch_cmd, load_cmd


def stage_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    return pl.read_parquet(path).height > 0


def fill_missing_close_dates(
    missing_dates: Sequence[MissingCloseDate],
    *,
    out_dir: Path,
    python: str,
    dry_run: bool,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    has_rows: Callable[[Path], bool] = stage_has_rows,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded_dates = 0
    for item in missing_dates:
        stage = stage_path(out_dir, item.paper_date)
        fetch_cmd, load_cmd = commands_for_date(
            paper_date=item.paper_date,
            stage=stage,
            python=python,
        )
        print(
            f"{item.paper_date}: paper_games={item.paper_games} "
            f"open_games={item.open_games} "
            f"missing_close_games={item.missing_close_games}"
        )
        print("fetch command: " + " ".join(fetch_cmd))
        print("load command: " + " ".join(load_cmd))
        if dry_run:
            continue
        run(fetch_cmd, check=True)
        if not has_rows(stage):
            print(f"no staged close rows found for {item.paper_date}; skipping load")
            continue
        run(load_cmd, check=True)
        loaded_dates += 1
    return loaded_dates


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date is not None and (args.start is not None or args.end is not None):
        raise SystemExit("Use --date or --start/--end, not both")
    start = _parse_date(args.date or args.start)
    end = _parse_date(args.date or args.end)

    db_config = PostgresConfig.from_env()
    missing = missing_close_dates(db_config, start=start, end=end)
    print(f"DB target: {db_config.describe()}")
    if not missing:
        print("No prior paper-trade or open-odds games are missing h2h close odds.")
        return

    loaded_dates = fill_missing_close_dates(
        missing,
        out_dir=args.out_dir,
        python=sys.executable,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("dry-run: no odds fetched or loaded")
    else:
        print(f"loaded close-line stages for {loaded_dates} date(s)")


if __name__ == "__main__":
    main()
