#!/usr/bin/env python3
"""Diagnose first-five run-environment bias from F5 CLV report JSON.

This is a research diagnostic, not a calibration gate. It reads the simulated
per-game rows emitted by ``scripts/f5_clv_report.py`` and compares simulated F5
mean runs to actual F5 runs by stable buckets such as opening total, month, and
open-to-close total movement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

DEFAULT_GROUPS = ("overall", "total_point", "month", "close_move")
DEFAULT_MIN_GROUP_ROWS = 10


@dataclass(frozen=True)
class F5RunEnvironmentRow:
    game_pk: int
    season: int
    take_point: float
    sim_mean_total: float
    actual_f5_total: float
    close_point: float | None = None
    game_date: date | None = None


@dataclass(frozen=True)
class F5RunEnvironmentSummary:
    group: str
    value: str
    n: int
    sim_mean_total: float
    actual_mean_total: float
    market_mean_total: float
    actual_minus_sim: float
    actual_minus_market: float
    sim_mae: float
    sim_rmse: float
    close_minus_take: float | None


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return seasons


def _parse_groups(value: str) -> tuple[str, ...]:
    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        raise argparse.ArgumentTypeError("at least one group is required")
    unsupported = sorted(set(groups) - set(DEFAULT_GROUPS))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported groups {unsupported}; expected subset of {DEFAULT_GROUPS}"
        )
    return groups


def _connect(config: PostgresConfig):
    return psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    )


def load_game_dates(seasons: Sequence[int]) -> dict[int, date]:
    """Load MLB game dates for report rows. Read-only SELECT query."""
    config = PostgresConfig.from_env()
    query = sql.SQL(
        """
        SELECT game_pk, game_date
        FROM {}.games
        WHERE season::int = ANY(%s)
          AND game_date IS NOT NULL
        """
    ).format(sql.Identifier(config.schema))
    dates: dict[int, date] = {}
    with _connect(config) as conn, conn.cursor() as cursor:
        cursor.execute(query, (list(seasons),))
        for game_pk, game_date in cursor.fetchall():
            dates[int(game_pk)] = _date_value(game_date)
    return dates


def rows_from_report_payload(
    payload: Mapping[str, Any], *, game_dates: Mapping[int, date] | None = None
) -> list[F5RunEnvironmentRow]:
    games = payload.get("games")
    if not isinstance(games, list):
        raise TypeError("F5 report JSON must contain a games list")
    date_map = game_dates or {}
    rows: list[F5RunEnvironmentRow] = []
    for item in games:
        if not isinstance(item, Mapping):
            continue
        game_pk = int(_required(item, "game_pk"))
        rows.append(
            F5RunEnvironmentRow(
                game_pk=game_pk,
                season=int(_required(item, "season")),
                take_point=float(_required(item, "take_point")),
                close_point=_optional_float(item.get("close_point")),
                sim_mean_total=float(_required(item, "sim_mean_total")),
                actual_f5_total=float(_required(item, "actual_f5_total")),
                game_date=_optional_date(item.get("game_date")) or date_map.get(game_pk),
            )
        )
    return sorted(rows, key=lambda row: (row.game_date or date.min, row.game_pk))


def load_report_rows(
    path: Path, *, game_dates: Mapping[int, date] | None = None
) -> list[F5RunEnvironmentRow]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("F5 report JSON must contain an object")
    return rows_from_report_payload(payload, game_dates=game_dates)


def build_report(
    rows: Sequence[F5RunEnvironmentRow],
    *,
    groups: Sequence[str] = DEFAULT_GROUPS,
    min_group_rows: int = DEFAULT_MIN_GROUP_ROWS,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one F5 run-environment row is required")
    summaries: list[F5RunEnvironmentSummary] = []
    for group in groups:
        buckets: dict[str, list[F5RunEnvironmentRow]] = defaultdict(list)
        for row in rows:
            buckets[_group_value(row, group)].append(row)
        for value, bucket_rows in sorted(buckets.items()):
            if group != "overall" and len(bucket_rows) < min_group_rows:
                continue
            summaries.append(_summarize_bucket(group, value, bucket_rows))

    overall = next(summary for summary in summaries if summary.group == "overall")
    return {
        "report_type": "f5_run_environment_diagnostics",
        "rows": len(rows),
        "groups": list(groups),
        "min_group_rows": min_group_rows,
        "summaries": [asdict(summary) for summary in summaries],
        "calibration_candidate": {
            "status": "closed",
            "reason": "Research diagnostic only; fit and validate on a chronological holdout before applying run calibration.",
            "suggested_additive_runs": overall.actual_minus_sim,
            "checks": {
                "holdout_validated": False,
                "uses_only_report_sample": True,
            },
        },
    }


def _summarize_bucket(
    group: str, value: str, rows: Sequence[F5RunEnvironmentRow]
) -> F5RunEnvironmentSummary:
    errors = [row.actual_f5_total - row.sim_mean_total for row in rows]
    market_errors = [row.actual_f5_total - row.take_point for row in rows]
    close_moves = [
        row.close_point - row.take_point for row in rows if row.close_point is not None
    ]
    return F5RunEnvironmentSummary(
        group=group,
        value=value,
        n=len(rows),
        sim_mean_total=_mean(row.sim_mean_total for row in rows),
        actual_mean_total=_mean(row.actual_f5_total for row in rows),
        market_mean_total=_mean(row.take_point for row in rows),
        actual_minus_sim=_mean(errors),
        actual_minus_market=_mean(market_errors),
        sim_mae=_mean(abs(error) for error in errors),
        sim_rmse=math.sqrt(_mean(error * error for error in errors)),
        close_minus_take=_mean(close_moves) if close_moves else None,
    )


def _group_value(row: F5RunEnvironmentRow, group: str) -> str:
    if group == "overall":
        return "all"
    if group == "total_point":
        return f"{row.take_point:g}"
    if group == "month":
        return row.game_date.strftime("%Y-%m") if row.game_date else "unknown"
    if group == "close_move":
        if row.close_point is None:
            return "none"
        delta = row.close_point - row.take_point
        if math.isclose(delta, 0.0):
            return "flat"
        return "up" if delta > 0.0 else "down"
    raise ValueError(f"unknown group {group!r}")


def _required(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing required field {key!r}")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    return _date_value(value)


def _date_value(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"expected date-compatible value, got {type(value).__name__}")


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot compute mean of an empty sequence")
    return sum(materialized) / len(materialized)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-report-json", type=Path, required=True)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--groups", type=_parse_groups, default=DEFAULT_GROUPS)
    parser.add_argument("--min-group-rows", type=int, default=DEFAULT_MIN_GROUP_ROWS)
    parser.add_argument(
        "--no-db-dates",
        action="store_true",
        help="do not join game dates from Postgres; month groups become unknown unless report rows include game_date",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    game_dates = {} if args.no_db_dates else load_game_dates(args.seasons)
    rows = load_report_rows(args.sim_report_json, game_dates=game_dates)
    report = build_report(
        rows,
        groups=args.groups,
        min_group_rows=args.min_group_rows,
    )
    output = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out_json is None:
        print(output, end="")
        return
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(output)
    print(f"wrote F5 run-environment diagnostics to {args.out_json}")


if __name__ == "__main__":
    main()
