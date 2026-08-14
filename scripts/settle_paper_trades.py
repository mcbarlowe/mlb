#!/usr/bin/env python3
"""Settle moneyline paper-trade CSV or DB rows and print performance."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, decimal_to_american
from src.betting.paper_settlement import (
    settle_paper_trade_row,
    summarize_paper_trade_rows,
)
from src.betting.paper_trade_store import (
    load_paper_trade_rows,
    update_paper_trade_settlement_rows,
)
from src.database import PostgresConfig, PostgresHandler

DEFAULT_PAPER_PATH = Path("output/paper_trades/moneyline_paper_trades.csv")
SETTLEMENT_FIELDS = (
    "status",
    "close_ml",
    "close_fair_prob",
    "clv",
    "result",
    "profit_units",
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise SystemExit(f"Paper-trade file does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _fieldnames(existing: Sequence[str], rows: Sequence[Mapping[str, str]]) -> list[str]:
    ordered = list(dict.fromkeys([*existing, *SETTLEMENT_FIELDS]))
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(field, "") for field in fieldnames])


def _row_game_pk(row: Mapping[str, str]) -> int | None:
    value = row.get("game_pk", "")
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _needs_update(row: Mapping[str, str]) -> bool:
    if row.get("status") != "settled":
        return True
    if not row.get("result") or not row.get("profit_units"):
        return True
    return not row.get("clv")


def _load_final_results(
    game_pks: Sequence[int],
    *,
    db_config: PostgresConfig | None,
) -> dict[int, bool]:
    if not game_pks:
        return {}
    query = """
        WITH scores AS (
            SELECT
                game_pk,
                SUM(runs) FILTER (WHERE team_type = 'away')::int AS away_runs,
                SUM(runs) FILTER (WHERE team_type = 'home')::int AS home_runs
            FROM linescore
            WHERE game_pk = ANY(%s)
            GROUP BY game_pk
        )
        SELECT g.game_pk, scores.home_runs > scores.away_runs AS home_won
        FROM games AS g
        JOIN scores USING (game_pk)
        WHERE g.game_pk = ANY(%s)
          AND g.abstract_game_state = 'Final'
          AND scores.away_runs IS NOT NULL
          AND scores.home_runs IS NOT NULL
          AND scores.away_runs <> scores.home_runs
    """
    pk_list = list(game_pks)
    with PostgresHandler(db_config) as db, db.connection.cursor() as cursor:
        cursor.execute(query, (pk_list, pk_list))
        return {int(game_pk): bool(home_won) for game_pk, home_won in cursor.fetchall()}


def _load_close_prices(
    game_pks: Sequence[int],
    *,
    db_config: PostgresConfig | None,
) -> dict[int, tuple[float, float]]:
    if not game_pks:
        return {}
    query = """
        SELECT game_pk, home_ml, away_ml
        FROM odds
        WHERE game_pk = ANY(%s)
          AND market = 'h2h'
          AND line_type = 'close'
          AND home_ml IS NOT NULL
          AND away_ml IS NOT NULL
    """
    grouped: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    with PostgresHandler(db_config) as db, db.connection.cursor() as cursor:
        cursor.execute(query, (list(game_pks),))
        for game_pk, home_ml, away_ml in cursor.fetchall():
            grouped[int(game_pk)].append((float(home_ml), float(away_ml)))
    close_prices: dict[int, tuple[float, float]] = {}
    for game_pk, lines in grouped.items():
        home_decimal = median(american_to_decimal(home_ml) for home_ml, _ in lines)
        away_decimal = median(american_to_decimal(away_ml) for _, away_ml in lines)
        close_prices[game_pk] = (
            decimal_to_american(home_decimal),
            decimal_to_american(away_decimal),
        )
    return close_prices


def _settle_rows(
    rows: Sequence[dict[str, str]],
    *,
    db_config: PostgresConfig | None,
) -> tuple[list[dict[str, str]], int, int, int]:
    target_pks = sorted(
        {
            game_pk
            for row in rows
            if _needs_update(row) and (game_pk := _row_game_pk(row)) is not None
        }
    )
    finals = _load_final_results(target_pks, db_config=db_config)
    closes = _load_close_prices(target_pks, db_config=db_config)
    settled_rows: list[dict[str, str]] = []
    updated = 0
    missing_final = 0
    missing_close = 0
    for row in rows:
        game_pk = _row_game_pk(row)
        if game_pk is None or not _needs_update(row):
            settled_rows.append(dict(row))
            continue
        home_won = finals.get(game_pk)
        if home_won is None:
            missing_final += 1
            settled_rows.append(dict(row))
            continue
        close_home_ml, close_away_ml = closes.get(game_pk, (None, None))
        if close_home_ml is None or close_away_ml is None:
            missing_close += 1
        new_row = settle_paper_trade_row(
            row,
            home_won=home_won,
            close_home_ml=close_home_ml,
            close_away_ml=close_away_ml,
        )
        updated += int(new_row != row)
        settled_rows.append(new_row)
    return settled_rows, updated, missing_final, missing_close


def _print_report(rows: Sequence[Mapping[str, str]]) -> None:
    summary = summarize_paper_trade_rows(rows)
    print(
        "Paper-trade report: "
        f"rows={summary.rows} open={summary.open_rows} "
        f"settled={summary.settled_rows} clv_rows={summary.clv_rows}"
    )
    print(
        f"Staked {summary.total_staked:.2f}u | "
        f"Profit {summary.profit_units:+.2f}u | "
        f"ROI {summary.roi:+.2%} | "
        f"Win {summary.win_rate:.1%} | "
        f"Avg CLV {summary.avg_clv:+.4f} | "
        f"Beat close {summary.beat_close_rate:.1%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PAPER_PATH)
    parser.add_argument("--db", action="store_true", help="read/update mlb.paper_trades")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_config = PostgresConfig.from_env()
    if args.db:
        rows = load_paper_trade_rows(db_config=db_config)
        existing_fields = list(rows[0].keys()) if rows else []
    else:
        rows, existing_fields = _read_csv(args.path)
    if args.report_only:
        _print_report(rows)
        return

    settled_rows, updated, missing_final, missing_close = _settle_rows(
        rows,
        db_config=db_config,
    )
    print(
        f"Settlement scan: updated={updated} "
        f"missing_final={missing_final} missing_close={missing_close}"
    )
    _print_report(settled_rows)
    if args.dry_run:
        target = "DB" if args.db else "CSV"
        print(f"dry-run: no paper-trade rows written to {target}")
        return
    changed_rows = [
        new
        for old, new in zip(rows, settled_rows, strict=True)
        if new != old
    ]
    if args.db:
        db_updated = update_paper_trade_settlement_rows(
            changed_rows,
            db_config=db_config,
        )
        print(f"updated {db_updated} DB paper_trade rows")
        return
    _write_csv(args.path, settled_rows, _fieldnames(existing_fields, settled_rows))
    print(f"wrote settled rows to {args.path}")


if __name__ == "__main__":
    main()
