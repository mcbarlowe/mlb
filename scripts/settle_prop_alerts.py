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
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="ntfy summary when new plays settle")
    ap.add_argument("--ntfy-topic", default="barlowe-props-c47d9e2a51b3")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    today = datetime.now(ET).strftime("%Y-%m-%d")
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
              f"ROI {net / staked:+.1%} | pending {pending} | void {voids}")
    else:
        print(f"nothing settled yet | pending {pending} | void {voids}")

    if args.push and newly:
        n_w = sum(1 for x in newly if x[5] == "won")
        n_l = sum(1 for x in newly if x[5] == "lost")
        n_net = sum(x[7] for x in newly if x[5] in ("won", "lost"))
        n_staked = n_w + n_l
        lines = [
            f"Props settled {today}: {n_w}-{n_l}, {n_net:+.2f}u"
            + (f" ({n_net / n_staked:+.0%} ROI)" if n_staked else ""),
        ]
        for player, market, point, side, price, status, value, profit in newly:
            mark = {"won": "W", "lost": "L", "void": "V"}[status]
            line = f"{'o' if side == 'over' else 'u'}{float(point):g}"
            lines.append(
                f"{mark}: {SHORT_MARKET.get(str(market), str(market))} {line} "
                f"{player} {float(price):+.0f} -> {value if value is not None else 'DNP'}"
            )
        if staked:
            lines.append(f"All-time: {wins}-{losses}, {net:+.2f}u ({net / staked:+.1%})")
        resp = requests.post(
            f"https://ntfy.sh/{args.ntfy_topic}",
            data="\n\n".join(lines).encode(),
            headers={"Title": "MLB Props P&L", "Tags": "moneybag"},
            timeout=15,
        )
        resp.raise_for_status()
        print(f"pushed settlement summary ({len(newly)} newly settled)")


if __name__ == "__main__":
    main()
