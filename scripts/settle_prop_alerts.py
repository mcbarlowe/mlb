"""Settle prop paper bets against actual results and report ROI.

Reads open rows from ``mlb.prop_paper_bets`` (written by shop_batter_props.py
at alert time: flat 1u at the alerted best price), joins each play to the
player's actual line that day from ``mlb.batting``, and settles:

  - over wins when stat > point, under wins when stat < point (x.5 lines)
  - player never appeared on the game date -> VOID (stake returned)
  - game date not reached / games not final yet -> stays open (PENDING)

Prints the full ledger with per-play P/L and all-time totals. ``--push``
sends a ntfy summary when new plays settle (wired into run_prop_shop.sh so
the morning run settles yesterday's card automatically).

Usage:
  uv run python scripts/settle_prop_alerts.py            # settle + report
  uv run python scripts/settle_prop_alerts.py --push     # + ntfy P&L push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from shop_batter_props import (
    PAPER_DDL,
    SHORT_MARKET,
    STAT_COLUMNS,
    STAT_FNS,
    _norm_name,
)

from src.betting.prop_settlement import (
    kelly_prop_stake_units,
    prop_profit,
    resolve_prop_won,
    summarize_prop_bet_rows,
    summarize_prop_kelly,
)
from src.database import PostgresConfig

ET = ZoneInfo("America/New_York")
DEFAULT_REPORT_STATE_PATH = (
    Path.home() / "Library/Application Support/BarloweAnalytics/settle_prop_alerts_state.json"
)


@dataclass(frozen=True)
class ArbitrageSummary:
    today_bets: int
    today_stake: float
    today_profit: float
    all_bets: int
    all_stake: float
    all_profit: float
    starting_bankroll: float

    @property
    def today_stake_roi(self) -> float | None:
        if self.today_stake == 0.0:
            return None
        return self.today_profit / self.today_stake

    @property
    def all_stake_roi(self) -> float | None:
        if self.all_stake == 0.0:
            return None
        return self.all_profit / self.all_stake

    @property
    def current_bankroll(self) -> float:
        return self.starting_bankroll + self.all_profit

    @property
    def today_starting_bankroll(self) -> float:
        return self.starting_bankroll + self.all_profit - self.today_profit

    @property
    def today_bankroll_return(self) -> float:
        return self.today_profit / self.today_starting_bankroll

    @property
    def all_bankroll_return(self) -> float:
        return self.all_profit / self.starting_bankroll


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _format_money(value: float) -> str:
    sign = "+" if value >= 0.0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _format_profit_roi(profit: float, roi: float | None) -> str:
    profit_text = _format_money(profit)
    if roi is None:
        return f"{profit_text} (n/a stake ROI)"
    return f"{profit_text} ({roi:+.1%} stake ROI)"


def _format_arbitrage_summary(summary: ArbitrageSummary) -> str:
    today = _format_profit_roi(summary.today_profit, summary.today_stake_roi)
    all_time = _format_profit_roi(summary.all_profit, summary.all_stake_roi)
    return (
        f"Arbs expected: today {summary.today_bets} bets {today}; "
        f"all-time {summary.all_bets} bets {all_time}"
    )


def _format_unit_bankroll(
    label: str,
    net: float,
    starting_bankroll: float,
    *,
    today_net: float | None = None,
) -> str:
    current = starting_bankroll + net
    all_time = net / starting_bankroll
    text = f"{label} bankroll {current:+.2f}u ({all_time:+.1%})"
    if today_net is None:
        return text
    today_start = current - today_net
    return f"{text}; today {today_net:+.2f}u ({today_net / today_start:+.1%})"


def _load_arbitrage_summary(
    config: PostgresConfig,
    today: str,
    starting_bankroll: float,
) -> ArbitrageSummary | None:
    conn = _connect(config)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('betting.arbitrage_paper_bets')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                """
                SELECT
                    count(*) FILTER (
                        WHERE (created_at AT TIME ZONE 'America/New_York')::date = %s::date
                    )::int AS today_bets,
                    coalesce(sum(total_stake) FILTER (
                        WHERE (created_at AT TIME ZONE 'America/New_York')::date = %s::date
                    ), 0)::float8 AS today_stake,
                    coalesce(sum(expected_profit) FILTER (
                        WHERE (created_at AT TIME ZONE 'America/New_York')::date = %s::date
                    ), 0)::float8 AS today_profit,
                    count(*)::int AS all_bets,
                    coalesce(sum(total_stake), 0)::float8 AS all_stake,
                    coalesce(sum(expected_profit), 0)::float8 AS all_profit
                FROM betting.arbitrage_paper_bets
                """,
                (today, today, today),
            )
            row = cur.fetchone()
            return ArbitrageSummary(
                today_bets=int(row[0]),
                today_stake=float(row[1]),
                today_profit=float(row[2]),
                all_bets=int(row[3]),
                all_stake=float(row[4]),
                all_profit=float(row[5]),
                starting_bankroll=starting_bankroll,
            )
    finally:
        conn.close()

def _load_arbitrage_bet_rows(config: PostgresConfig) -> list[dict[str, object]]:
    conn = _connect(config)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('betting.arbitrage_paper_bets')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                """
                SELECT
                    (created_at AT TIME ZONE 'America/New_York')::date AS event_date,
                    total_stake::float8 AS total_stake,
                    expected_profit::float8 AS expected_profit
                FROM betting.arbitrage_paper_bets
                WHERE total_stake IS NOT NULL
                  AND expected_profit IS NOT NULL
                ORDER BY event_date, created_at
                """,
            )
            names = [d.name for d in cur.description or []]
            return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _load_report_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _daily_report_due(path: Path, today: str) -> bool:
    return _load_report_state(path).get("last_report_date") != today


def _mark_daily_report_sent(path: Path, today: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_report_date": today}, indent=2) + "\n")


def _build_push_lines(
    today: str,
    newly: list[dict],
    *,
    summary,
    arb_summary: ArbitrageSummary | None,
    prop_bankroll: float,
) -> list[str]:
    graded = [x for x in newly if x["status"] in ("won", "lost")]
    n_w = sum(1 for x in graded if x["status"] == "won")
    n_l = len(graded) - n_w
    n_stake = sum(float(x.get("kelly_stake_units", x.get("stake_units", 0.0))) for x in graded)
    n_net = sum(float(x.get("kelly_profit_units", x.get("profit_units", 0.0))) for x in graded)
    lines = [
        f"Props settled {today}: {n_w}-{n_l}, kelly {n_net:+.2f}u"
        + (f" ({n_net / n_stake:+.0%} stake ROI)" if n_stake else ""),
        _format_unit_bankroll(
            "Props Kelly",
            summary.profit_units,
            prop_bankroll,
            today_net=n_net,
        ),
    ]
    if arb_summary is not None:
        lines.append(_format_arbitrage_summary(arb_summary))
    for x in newly:
        mark = {"won": "W", "lost": "L", "void": "V"}[str(x["status"])]
        value = x["result_value"]
        lines.append(
            f"{mark}: {_play_label(x)} {float(x['price']):+.0f} "
            f"-> {value if value is not None else 'DNP'}"
        )
    if summary.settled_rows:
        lines.append(
            f"All-time: {summary.won}-{summary.lost}, "
            f"{summary.profit_units:+.2f}u ({summary.roi:+.1%} stake ROI)"
        )
    return lines


def _connect(config: PostgresConfig):
    return psycopg.connect(
        dbname=config.dbname, user=config.user, password=config.password,
        host=config.host, port=config.port, connect_timeout=10,
    )


def settle_open_prop_bets(config: PostgresConfig) -> list[dict]:
    """Resolve every open prop bet whose game date has already passed.

    Joins each bet to the player's actual line that day in ``mlb.batting``,
    marks it won/lost (stat vs. point) or void (player never appeared), writes
    ``result_value`` + ``profit_units``, and returns the rows newly settled this
    run. Idempotent: already-graded rows are skipped (``status = 'open'`` gate).
    """
    today = datetime.now(ET).strftime("%Y-%m-%d")
    cols = ", ".join(f"COALESCE(b.{col}, 0)::int AS {col}" for col in STAT_COLUMNS)
    newly: list[dict] = []
    conn = _connect(config)
    try:
        with conn.cursor() as cur:
            cur.execute(PAPER_DDL.format(schema=config.schema))
            cur.execute(
                f"""SELECT alert_date, player, market, point, side, game_date,
                           book, price, decimal_odds, stake_units
                    FROM {config.schema}.prop_paper_bets
                    WHERE status = 'open' AND game_date < %s::date""",
                (today,),
            )
            open_rows = cur.fetchall()

            for (alert_date, player, market, point, side, game_date,
                 book, price, dec, stake) in open_rows:
                cur.execute(
                    f"""
                    SELECT p.full_name, {cols}
                    FROM {config.schema}.batting b
                    JOIN {config.schema}.games g USING (game_pk)
                    JOIN {config.schema}.players p USING (player_id)
                    WHERE g.game_date::date = %s::date
                      AND g.abstract_game_state = 'Final'
                      AND COALESCE(b.plateappearances, 0) > 0
                    ORDER BY COALESCE(g.game_datetime, g.game_date), g.game_pk
                    """,
                    (game_date,),
                )
                col_names = [d.name for d in cur.description or []]
                stats = None
                for row in cur.fetchall():
                    rec = dict(zip(col_names, row))
                    if _norm_name(str(rec["full_name"])) == _norm_name(str(player)):
                        stats = {col: int(rec[col]) for col in STAT_COLUMNS}
                        break
                if stats is None:
                    status, value, profit = "void", None, 0.0
                else:
                    value = int(STAT_FNS[str(market)](stats))
                    won = resolve_prop_won(value=value, point=float(point), side=str(side))
                    profit = prop_profit(
                        won=won, decimal_odds=float(dec), stake_units=float(stake),
                    )
                    status = "won" if won else "lost"
                cur.execute(
                    f"""UPDATE {config.schema}.prop_paper_bets
                        SET status = %s, result_value = %s, profit_units = %s,
                            updated_at = now()
                        WHERE alert_date = %s AND player = %s AND market = %s
                          AND point = %s AND side = %s""",
                    (status, value, profit, alert_date, player, market, point, side),
                )
                newly.append({
                    "alert_date": alert_date, "game_date": game_date,
                    "player": player, "market": market, "point": point,
                    "side": side, "price": price, "book": book,
                    "matchup": None, "decimal_odds": dec,
                    "status": status, "result_value": value,
                    "profit_units": profit, "stake_units": stake,
                })
            conn.commit()
    finally:
        conn.close()
    return newly


def load_prop_bet_rows(config: PostgresConfig) -> list[dict]:
    """Return the full prop ledger as dict rows ordered by game date."""
    conn = _connect(config)
    try:
        with conn.cursor() as cur:
            cur.execute(PAPER_DDL.format(schema=config.schema))
            cur.execute(
                f"""SELECT game_date, player, market, point, side, price, book,
                           status, result_value, profit_units, stake_units,
                           decimal_odds, ev, adj_prob, matchup
                    FROM {config.schema}.prop_paper_bets
                    ORDER BY game_date, player, market""",
            )
            names = [d.name for d in cur.description or []]
            return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _play_label(row: dict) -> str:
    side = str(row["side"])
    line = f"{'o' if side == 'over' else 'u'}{float(row['point']):g}"
    market = SHORT_MARKET.get(str(row["market"]), str(row["market"]))
    return f"{market} {line} {row['player']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="ntfy summary when new plays settle")
    ap.add_argument(
        "--daily-report",
        action="store_true",
        help="With --push, send one daily report even when no props settled.",
    )
    ap.add_argument(
        "--report-state-file",
        type=Path,
        default=DEFAULT_REPORT_STATE_PATH,
        help="State file used to avoid duplicate daily report pushes.",
    )
    ap.add_argument("--ntfy-topic", default="barlowe-props-c47d9e2a51b3")
    ap.add_argument(
        "--prop-bankroll",
        type=float,
        default=_env_float("BARLOWE_PROP_BANKROLL_UNITS", 100.0),
        help="Starting bankroll in units for prop-bet bankroll-return reporting.",
    )
    ap.add_argument(
        "--arb-bankroll",
        type=float,
        default=_env_float("BARLOWE_ARB_BANKROLL", 10_000.0),
        help="Starting bankroll used for arbitrage bankroll-return reporting.",
    )
    args = ap.parse_args()
    if args.prop_bankroll <= 0.0:
        raise SystemExit("--prop-bankroll must be greater than zero")
    if args.arb_bankroll <= 0.0:
        raise SystemExit("--arb-bankroll must be greater than zero")

    config = PostgresConfig.from_env()
    today = datetime.now(ET).strftime("%Y-%m-%d")
    newly = settle_open_prop_bets(config)
    ledger = load_prop_bet_rows(config)
    arb_summary = _load_arbitrage_summary(config, today, args.arb_bankroll)
    kelly_stakes = kelly_prop_stake_units(ledger)
    for row, kelly_stake in zip(ledger, kelly_stakes, strict=True):
        row["kelly_stake_units"] = kelly_stake

    print(f"{'game':<11} {'play':<42} {'price':>6} {'book':<14} {'status':<10} "
          f"{'val':>3} {'P/L':>7} {'kU':>5}")
    print("-" * 106)
    for row in ledger:
        value = row["result_value"]
        profit = row["profit_units"]
        val = "-" if value is None else str(value)
        pl = "-" if profit is None else f"{float(profit):+.2f}u"
        status = str(row["status"])
        shown_status = status if status != "open" else "PENDING"
        print(f"{row['game_date']!s:<11} {_play_label(row):<42.42} "
              f"{float(row['price']):>+6.0f} {row['book']!s:<14.14} "
              f"{shown_status:<10} {val:>3} {pl:>7} "
              f"{float(row['kelly_stake_units']):>5.2f}")
    print("-" * 106)

    flat = summarize_prop_bet_rows(ledger)
    kelly = summarize_prop_kelly(ledger, kelly_stakes)
    open_kelly = sum(
        stake for row, stake in zip(ledger, kelly_stakes, strict=True)
        if str(row["status"]) == "open"
    )
    kelly_by_key = {
        (
            str(row.get("player")),
            str(row.get("market")),
            float(row.get("point")),
            str(row.get("side")),
            str(row.get("game_date")),
        ): stake
        for row, stake in zip(ledger, kelly_stakes, strict=True)
    }
    for row in newly:
        stake = kelly_by_key.get(
            (
                str(row.get("player")),
                str(row.get("market")),
                float(row.get("point")),
                str(row.get("side")),
                str(row.get("game_date")),
            ),
            0.0,
        )
        row["kelly_stake_units"] = stake
        if str(row["status"]) == "won":
            row["kelly_profit_units"] = stake * (float(row["decimal_odds"]) - 1.0)
        elif str(row["status"]) == "lost":
            row["kelly_profit_units"] = -stake
        else:
            row["kelly_profit_units"] = 0.0
    if kelly.settled_rows:
        print(f"settled: {kelly.won}-{kelly.lost} | "
              f"kelly staked {kelly.total_staked:.2f}u | net {kelly.profit_units:+.2f}u | "
              f"stake ROI {kelly.roi:+.1%} | pending {flat.open_rows} | "
              f"void {flat.void_rows} | open {open_kelly:.2f}u")
    else:
        print(f"nothing settled yet | pending {flat.open_rows} | "
              f"void {flat.void_rows} | open {open_kelly:.2f}u")
    print(_format_unit_bankroll("Props Kelly", kelly.profit_units, args.prop_bankroll))
    if arb_summary is not None:
        print(_format_arbitrage_summary(arb_summary))

    should_push = args.push and (
        bool(newly)
        or (
            args.daily_report
            and _daily_report_due(args.report_state_file, today)
        )
    )
    if should_push:
        lines = _build_push_lines(
            today,
            newly,
            summary=kelly,
            arb_summary=arb_summary,
            prop_bankroll=args.prop_bankroll,
        )
        resp = requests.post(
            f"https://ntfy.sh/{args.ntfy_topic}",
            data="\n\n".join(lines).encode(),
            headers={"Title": "MLB Betting P&L", "Tags": "moneybag"},
            timeout=15,
        )
        resp.raise_for_status()
        if args.daily_report:
            _mark_daily_report_sent(args.report_state_file, today)
        print(f"pushed settlement summary ({len(newly)} newly settled)")



if __name__ == "__main__":
    main()
