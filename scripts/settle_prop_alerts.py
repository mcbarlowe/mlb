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


def _load_arbitrage_summary(cur, today, starting_bankroll: float) -> ArbitrageSummary | None:
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


def _format_arbitrage_summary(summary: ArbitrageSummary) -> str:
    today = _format_profit_roi(summary.today_profit, summary.today_stake_roi)
    all_time = _format_profit_roi(summary.all_profit, summary.all_stake_roi)
    return (
        f"Arbs expected: today {summary.today_bets} bets {today} "
        f"({summary.today_bankroll_return:+.1%} bankroll); "
        f"all-time {summary.all_bets} bets {all_time}; "
        f"bankroll {_format_money(summary.current_bankroll)} "
        f"({summary.all_bankroll_return:+.1%})"
    )


def _format_profit_roi(profit: float, roi: float | None) -> str:
    profit_text = _format_money(profit)
    if roi is None:
        return f"{profit_text} (n/a stake ROI)"
    return f"{profit_text} ({roi:+.1%} stake ROI)"


def _format_money(value: float) -> str:
    sign = "+" if value >= 0.0 else "-"
    return f"{sign}${abs(value):,.2f}"


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)

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



def _daily_report_due(path: Path, today) -> bool:
    return _load_report_state(path).get("last_report_date") != str(today)


def _mark_daily_report_sent(path: Path, today) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_report_date": str(today)}, indent=2) + "\n")


def _build_push_lines(
    today,
    newly,
    *,
    wins: int,
    losses: int,
    net: float,
    staked: int,
    arb_summary: ArbitrageSummary | None,
    prop_bankroll: float,
) -> list[str]:
    n_w = sum(1 for x in newly if x[5] == "won")
    n_l = sum(1 for x in newly if x[5] == "lost")
    n_net = sum(x[7] for x in newly if x[5] in ("won", "lost"))
    n_staked = n_w + n_l
    lines = [
        f"Props settled {today}: {n_w}-{n_l}, {n_net:+.2f}u"
        + (f" ({n_net / n_staked:+.0%} stake ROI)" if n_staked else ""),
    ]
    lines.append(
        _format_unit_bankroll(
            "Props",
            net,
            prop_bankroll,
            today_net=n_net,
        )
    )
    if arb_summary is not None:
        lines.append(_format_arbitrage_summary(arb_summary))
    for player, market, point, side, price, status, value, profit in newly:
        mark = {"won": "W", "lost": "L", "void": "V"}[status]
        line = f"{'o' if side == 'over' else 'u'}{float(point):g}"
        lines.append(
            f"{mark}: {SHORT_MARKET.get(str(market), str(market))} {line} "
            f"{player} {float(price):+.0f} -> {value if value is not None else 'DNP'}"
        )
    if staked:
        lines.append(
            f"All-time: {wins}-{losses}, {net:+.2f}u "
            f"({net / staked:+.1%} stake ROI)"
        )
    return lines


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

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    today = datetime.now(ET).date()
    cols = ", ".join(f"COALESCE(b.{col}, 0)::int AS {col}" for col in STAT_COLUMNS)
    newly = []
    with conn.cursor() as cur:
        cur.execute(PAPER_DDL.format(schema=c.schema))
        cur.execute(
            f"""SELECT alert_date, player, market, point, side, game_date,
                       book, price, decimal_odds
                FROM {c.schema}.prop_paper_bets
                WHERE status = 'open' AND game_date < %s::date""",
            (today,),
        )
        open_rows = cur.fetchall()

        for (alert_date, player, market, point, side, game_date,
             book, price, dec) in open_rows:
            cur.execute(
                f"""
                SELECT p.full_name, {cols}
                FROM {c.schema}.batting b
                JOIN {c.schema}.games g USING (game_pk)
                JOIN {c.schema}.players p USING (player_id)
                WHERE g.game_date::date = %s::date
                  AND g.abstract_game_state = 'Final'
                  AND COALESCE(b.plateappearances, 0) > 0
                ORDER BY COALESCE(g.game_datetime, g.game_date), g.game_pk
                """,
                (game_date,),
            )
            col_names = [d.name for d in cur.description]
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
                over_won = value > float(point)
                won = over_won if side == "over" else not over_won
                profit = (float(dec) - 1.0) if won else -1.0
                status = "won" if won else "lost"
            cur.execute(
                f"""UPDATE {c.schema}.prop_paper_bets
                    SET status = %s, result_value = %s, profit_units = %s,
                        updated_at = now()
                    WHERE alert_date = %s AND player = %s AND market = %s
                      AND point = %s AND side = %s""",
                (status, value, profit, alert_date, player, market, point, side),
            )
            newly.append((player, market, point, side, price, status, value, profit))
        conn.commit()

        cur.execute(
            f"""SELECT game_date, player, market, point, side, price, book,
                       status, result_value, profit_units
                FROM {c.schema}.prop_paper_bets
                ORDER BY game_date, player, market""",
        )
        ledger = cur.fetchall()
        arb_summary = _load_arbitrage_summary(cur, today, args.arb_bankroll)
    conn.close()

    print(f"{'game':<11} {'play':<42} {'price':>6} {'book':<14} {'status':<10} "
          f"{'val':>3} {'P/L':>7}")
    print("-" * 100)
    wins = losses = voids = pending = 0
    net = 0.0
    for (game_date, player, market, point, side, price, book,
         status, value, profit) in ledger:
        line = f"{'o' if side == 'over' else 'u'}{float(point):g}"
        play = f"{SHORT_MARKET.get(str(market), str(market))} {line} {player}"
        val = "-" if value is None else str(value)
        pl = "-" if profit is None else f"{float(profit):+.2f}u"
        shown_status = status if status != "open" else "PENDING"
        print(f"{game_date!s:<11} {play:<42.42} {float(price):>+6.0f} "
              f"{book!s:<14.14} {shown_status:<10} {val:>3} {pl:>7}")
        if status == "won":
            wins += 1
            net += float(profit)
        elif status == "lost":
            losses += 1
            net += float(profit)
        elif status == "void":
            voids += 1
        else:
            pending += 1
    print("-" * 100)
    staked = wins + losses
    if staked:
        print(f"settled: {wins}-{losses} | staked {staked}u | net {net:+.2f}u | "
              f"stake ROI {net / staked:+.1%} | pending {pending} | void {voids}")
    else:
        print(f"nothing settled yet | pending {pending} | void {voids}")
    print(_format_unit_bankroll("Props", net, args.prop_bankroll))
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
            wins=wins,
            losses=losses,
            net=net,
            staked=staked,
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
