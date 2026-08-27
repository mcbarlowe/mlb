#!/usr/bin/env python3
"""Settle previous night's paper trades after backfill completes.

Intended to run after run_daily_postgres_etl.py so game data is fresh.
Usage: uv run python scripts/settle_daily_paper_trades.py [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import UTC, date, datetime
from email.message import EmailMessage
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.bankroll import format_shared_bankroll, summarize_shared_bankroll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Settle paper trades from previous night and show results.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to process in YYYY-MM-DD format. Defaults to today (settling yesterday's bets).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed trade-by-trade results.",
    )
    parser.add_argument(
        "--email-report",
        action="store_true",
        help="Email the settlement summary using Gmail SMTP settings from the environment.",
    )
    parser.add_argument(
        "--email-to",
        default=os.getenv("BARLOWE_PAPER_REPORT_EMAIL_TO") or "mcbarlowe@gmail.com",
        help="Settlement report email recipient.",
    )
    parser.add_argument(
        "--shared-bankroll",
        type=float,
        default=float(os.getenv("BARLOWE_SHARED_BETTING_BANKROLL", "10000")),
        help="Starting shared bankroll in dollars for all paper bets and arbitrage.",
    )
    parser.add_argument(
        "--paper-unit-dollars",
        type=float,
        default=float(os.getenv("BARLOWE_PAPER_UNIT_DOLLARS", "100")),
        help="Dollar value of one paper-bet unit in shared bankroll reporting.",
    )
    return parser.parse_args()


def resolve_target_date(date_arg: str | None) -> date:
    if date_arg is None:
        return datetime.now(tz=UTC).date()
    return datetime.strptime(date_arg, "%Y-%m-%d").replace(tzinfo=UTC).date()


def build_report_text(
    *,
    target_date: date,
    updated: int,
    missing_final: int,
    missing_close: int,
    summary,
    prop_summary=None,
    prop_kelly_summary=None,
    props_newly_settled: int = 0,
    arbitrage_summary_text: str | None = None,
    bankroll_summary=None,
) -> str:
    lines = [
        f"[{target_date.isoformat()}] Settlement scan: updated={updated} missing_final={missing_final} missing_close={missing_close}",
        "",
        "=" * 90,
        "MONEYLINE PAPER TRADING SUMMARY",
        "=" * 90,
        "",
        f"Total Trades:        {summary.rows} ({summary.settled_rows} settled, {summary.open_rows} pending)",
        f"Win Rate:            {summary.win_rate:.1%}",
        f"Stake ROI:           {summary.roi:+.2%}",
        "",
        f"Total Staked:        {summary.total_staked:.2f}u",
        f"Total Profit:        {summary.profit_units:+.2f}u",
        f"Avg CLV:             {summary.avg_clv:+.4f}",
        f"Beat Close Rate:     {summary.beat_close_rate:.1%}",
        "",
    ]
    if prop_summary is not None:
        lines.extend(
            [
                "=" * 90,
                "PLAYER PROPS PAPER TRADING SUMMARY",
                "=" * 90,
                "",
                f"Newly Settled:      {props_newly_settled}",
                f"Total Props:        {prop_summary.rows} ({prop_summary.settled_rows} settled, {prop_summary.open_rows} pending, {prop_summary.void_rows} void)",
                f"Record:             {prop_summary.won}-{prop_summary.lost}",
                f"Win Rate:           {prop_summary.win_rate:.1%}",
                f"Flat Stake ROI:     {prop_summary.roi:+.2%}",
                f"Flat Staked:        {prop_summary.total_staked:.2f}u",
                f"Flat Profit:        {prop_summary.profit_units:+.2f}u",
                "",
            ]
        )
        if prop_kelly_summary is not None:
            lines.extend(
                [
                    f"Kelly Record:       {prop_kelly_summary.won}-{prop_kelly_summary.lost}",
                    f"Kelly Stake ROI:    {prop_kelly_summary.roi:+.2%}",
                    f"Kelly Staked:       {prop_kelly_summary.total_staked:.2f}u",
                    f"Kelly Profit:       {prop_kelly_summary.profit_units:+.2f}u",
                    "",
                ]
            )
    if bankroll_summary is not None:
        lines.extend(
            [
                "=" * 90,
                "SHARED BANKROLL - ALL PAPER BETS + ARBS",
                "=" * 90,
                "",
                *format_shared_bankroll(bankroll_summary),
                "",
            ]
        )
    if arbitrage_summary_text is not None:
        lines.extend(
            [
                "=" * 90,
                "ARBITRAGE PAPER TRADING SUMMARY",
                "=" * 90,
                "",
                arbitrage_summary_text,
                "",
            ]
        )
    lines.append("=" * 90)
    return "\n".join(lines)


def send_email_report(*, recipient: str, subject: str, body: str) -> None:
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        raise RuntimeError("EMAIL_PASSWORD is required for email reports")
    sender = os.getenv("BETTING_ARB_EMAIL_FROM") or "sabresbot@gmail.com"
    username = os.getenv("BETTING_ARB_EMAIL_USERNAME") or sender
    host = os.getenv("BETTING_ARB_SMTP_HOST") or "smtp.gmail.com"
    port = int(os.getenv("BETTING_ARB_SMTP_PORT") or "587")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(host, port, timeout=15.0) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)


def main() -> None:
    from src.betting.paper_settlement import summarize_paper_trade_rows
    from src.betting.paper_trade_store import (
        ensure_paper_trades_table,
        load_paper_trade_rows,
        update_paper_trade_settlement_rows,
    )
    from src.betting.prop_settlement import (
        kelly_prop_stake_units,
        summarize_prop_bet_rows,
        summarize_prop_kelly,
    )
    from src.database import PostgresConfig, PostgresHandler

    args = parse_args()
    if args.shared_bankroll <= 0.0:
        raise SystemExit("--shared-bankroll must be greater than zero")
    if args.paper_unit_dollars <= 0.0:
        raise SystemExit("--paper-unit-dollars must be greater than zero")

    target_date = resolve_target_date(args.date)

    db_config = PostgresConfig()

    # Ensure table exists
    with PostgresHandler(db_config) as db:
        ensure_paper_trades_table(db)

    # Load all paper trades
    rows = load_paper_trade_rows()

    # Use the settlement logic from settle_paper_trades.py
    # Import after sys.path is set
    sys.path.insert(0, str(project_root / "scripts"))
    from settle_paper_trades import _settle_rows
    from settle_prop_alerts import (
        _format_arbitrage_summary,
        _load_arbitrage_bet_rows,
        _load_arbitrage_summary,
        load_prop_bet_rows,
        settle_open_prop_bets,
    )

    settled_rows, updated, missing_final, missing_close = _settle_rows(rows, db_config=db_config)
    newly_settled_props = settle_open_prop_bets(db_config)

    # Update database with settled rows
    update_paper_trade_settlement_rows(settled_rows, db_config=db_config)

    # Reload and show summaries
    rows_fresh = load_paper_trade_rows()
    summary = summarize_paper_trade_rows(rows_fresh)
    prop_rows = load_prop_bet_rows(db_config)
    prop_summary = summarize_prop_bet_rows(prop_rows)
    prop_kelly_stakes = kelly_prop_stake_units(prop_rows)
    prop_kelly_summary = summarize_prop_kelly(prop_rows, prop_kelly_stakes)
    arb_summary = _load_arbitrage_summary(
        db_config,
        target_date.isoformat(),
        args.shared_bankroll,
    )
    arb_rows = _load_arbitrage_bet_rows(db_config)
    bankroll_summary = summarize_shared_bankroll(
        rows_fresh,
        prop_rows,
        arb_rows,
        starting_bankroll=args.shared_bankroll,
        paper_unit_dollars=args.paper_unit_dollars,
    )
    arbitrage_summary_text = (
        _format_arbitrage_summary(arb_summary)
        if arb_summary is not None
        else None
    )
    report_text = build_report_text(
        target_date=target_date,
        updated=updated,
        missing_final=missing_final,
        missing_close=missing_close,
        summary=summary,
        prop_summary=prop_summary,
        prop_kelly_summary=prop_kelly_summary,
        props_newly_settled=len(newly_settled_props),
        arbitrage_summary_text=arbitrage_summary_text,
        bankroll_summary=bankroll_summary,
    )
    print(report_text)
    if args.email_report:
        send_email_report(
            recipient=args.email_to,
            subject=f"Daily paper betting report {target_date.isoformat()}",
            body=report_text,
        )
        print(f"Email report sent to {args.email_to}")

    # Detailed results if requested
    if args.verbose and updated > 0:
        print("Recently Settled Trades:")
        print("-" * 90)
        print(f"{'Date':<12} | {'Team':<6} | {'Side':<6} | {'Odds':<7} | {'Result':<8} | {'Profit'}")
        print("-" * 90)

        with PostgresHandler(db_config) as db:
            query = """
              SELECT paper_date, away_team, home_team, side, best_ml, result, profit_units
              FROM paper_trades
              WHERE status = 'settled'
              ORDER BY paper_date DESC
              LIMIT 20
            """

            with db.connection.cursor() as cursor:
                cursor.execute(query)
                recent = cursor.fetchall()

                for row in recent:
                    paper_date, away, home, side, odds, result, profit_units = row
                    team = home if side == "home" else away
                    result_str = result if result else "—"
                    profit_str = f"{profit_units:+.2f}u" if profit_units is not None else "N/A"

                    print(f"{paper_date!s:<12} | {team:<6} | {side:<6} | {odds:>6.0f} | {result_str:<8} | {profit_str}")

        print("-" * 90)

    print()


if __name__ == "__main__":
    main()
