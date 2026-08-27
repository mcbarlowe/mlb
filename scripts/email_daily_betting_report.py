"""Email a daily betting report: last night's settled bets, all-time moneyline
and player-prop performance, and any outstanding (open) bets.

Reads both paper ledgers (``mlb.paper_trades`` for moneyline, ``mlb.prop_paper_bets``
for props), settles anything now gradable (idempotent, unless ``--no-settle``),
and sends one email. Runs in the morning for the previous night's slate.

Transport is Gmail SMTP over SSL. The sender + app password come from the shell
environment (loaded from ~/.zshrc by the LaunchAgent runner, exactly like
ODDS_API_KEY) so no secret is ever stored in a plist. Shared bankroll reporting
uses one dollar bankroll for moneyline, props, and arbitrage; paper betting
units are converted with BARLOWE_PAPER_UNIT_DOLLARS:

  MLB_REPORT_EMAIL_TO              recipient (default mcbarlowe@gmail.com)
  MLB_REPORT_GMAIL_USER            sender gmail address (default = recipient)
  MLB_REPORT_GMAIL_APP_PASSWORD    16-char Google App Password (required to send)
  BARLOWE_SHARED_BETTING_BANKROLL  shared bankroll dollars (default 10000)
  BARLOWE_PAPER_UNIT_DOLLARS       dollar value of 1 paper unit (default 100)
Usage:
  uv run python scripts/email_daily_betting_report.py            # settle + email yesterday
  uv run python scripts/email_daily_betting_report.py --date 2026-08-18
  uv run python scripts/email_daily_betting_report.py --print    # stdout only, no send
  uv run python scripts/email_daily_betting_report.py --no-settle
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from settle_prop_alerts import (
    _format_arbitrage_summary,
    _load_arbitrage_bet_rows,
    _load_arbitrage_summary,
    load_prop_bet_rows,
    settle_open_prop_bets,
)

from src.betting.bankroll import format_shared_bankroll, summarize_shared_bankroll
from src.betting.paper_settlement import PaperTradeSummary, summarize_paper_trade_rows
from src.betting.paper_trade_store import (
    load_paper_trade_rows,
    update_paper_trade_settlement_rows,
)
from src.betting.prop_settlement import (
    PropBetSummary,
    kelly_prop_stake_units,
    summarize_prop_bet_rows,
    summarize_prop_kelly,
)
from src.database import PostgresConfig

ET = ZoneInfo("America/New_York")
DEFAULT_TO = "mcbarlowe@gmail.com"


# --------------------------------------------------------------------------- #
# Settlement (idempotent)
# --------------------------------------------------------------------------- #
def settle_all(config: PostgresConfig) -> None:
    """Grade any now-final bets in both ledgers. Failures are non-fatal so the
    report still sends against whatever is already settled."""
    try:
        settle_open_prop_bets(config)
    except Exception as exc:
        print(f"warn: prop settlement failed: {exc}", file=sys.stderr)
    try:
        from settle_paper_trades import _settle_rows

        rows = load_paper_trade_rows(db_config=config)
        settled_rows, _, _, _ = _settle_rows(rows, db_config=config)
        update_paper_trade_settlement_rows(settled_rows, db_config=config)
    except Exception as exc:
        print(f"warn: moneyline settlement failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Row selection + formatting
# --------------------------------------------------------------------------- #
def _f(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)  # type: ignore[arg-type]




def _ml_wins(rows: Sequence[Mapping[str, object]]) -> int:
    return sum(1 for r in rows if str(r.get("result")) == "win")


def _roi_str(summary: PaperTradeSummary | PropBetSummary) -> str:
    return f"{summary.roi:+.1%}" if summary.settled_rows else "n/a"




# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def build_report(
    night: date,
    ml_rows: Sequence[Mapping[str, str]],
    prop_rows: list[dict[str, object]],
    *,
    arb_summary,
    arb_rows: Sequence[Mapping[str, object]],
    shared_bankroll: float,
    paper_unit_dollars: float,
) -> tuple[str, str, str]:
    night_str = night.isoformat()
    ml_all = summarize_paper_trade_rows(ml_rows)
    prop_all = summarize_prop_bet_rows(prop_rows)
    kelly_stakes = kelly_prop_stake_units(prop_rows)
    for row, kelly_stake in zip(prop_rows, kelly_stakes, strict=True):
        row["kelly_stake_units"] = kelly_stake
    prop_kelly = summarize_prop_kelly(prop_rows, kelly_stakes)
    shared_summary = summarize_shared_bankroll(
        ml_rows,
        prop_rows,
        arb_rows,
        starting_bankroll=shared_bankroll,
        paper_unit_dollars=paper_unit_dollars,
        prop_stakes=kelly_stakes,
    )

    ml_night = [
        r for r in ml_rows
        if str(r.get("paper_date")) == night_str and str(r.get("status")) == "settled"
    ]
    prop_night = [
        r for r in prop_rows
        if str(r.get("game_date")) == night_str
        and str(r.get("status")) in ("won", "lost", "void")
    ]
    ml_open = [r for r in ml_rows if str(r.get("status")) != "settled"]
    prop_open = [r for r in prop_rows if str(r.get("status")) == "open"]

    lines: list[str] = []
    lines.append(f"MLB Betting Report - night of {night_str}")
    lines.append("=" * 60)

    # --- Last night --------------------------------------------------------- #
    lines.append("")
    lines.append(f"LAST NIGHT ({night_str})")
    ml_night_pl = sum(_f(r.get("profit_units")) for r in ml_night)
    if ml_night:
        w = _ml_wins(ml_night)
        lines.append(f"  Moneyline: {w}-{len(ml_night) - w}, {ml_night_pl:+.2f}u")
    else:
        lines.append("  Moneyline: no bets graded")
    prop_graded = [r for r in prop_night if str(r["status"]) in ("won", "lost")]
    prop_night_kelly_pl = sum(
        _f(r.get("kelly_stake_units")) * (_f(r.get("decimal_odds")) - 1.0)
        if str(r["status"]) == "won"
        else -_f(r.get("kelly_stake_units"))
        for r in prop_graded
    )
    if prop_night:
        pw = sum(1 for r in prop_night if str(r["status"]) == "won")
        pl_ = sum(1 for r in prop_night if str(r["status"]) == "lost")
        pv = sum(1 for r in prop_night if str(r["status"]) == "void")
        lines.append(
            f"  Props: {pw}-{pl_}, kelly {prop_night_kelly_pl:+.2f}u"
            + (f" ({pv} void)" if pv else "")
        )
    else:
        lines.append("  Props: no bets graded")
    lines.append(f"  Night P/L: {ml_night_pl + prop_night_kelly_pl:+.2f}u")
    if arb_summary is not None:
        lines.append(f"  {_format_arbitrage_summary(arb_summary)}")

    # --- Shared bankroll --------------------------------------------------- #
    lines.append("")
    lines.append("SHARED BANKROLL - all paper bets + arbs")
    lines.extend(f"  {line}" for line in format_shared_bankroll(shared_summary))

    # --- All-time moneyline ------------------------------------------------- #
    lines.append("")
    lines.append("MONEYLINE - all time")
    mw = _ml_wins(ml_rows)
    lines.append(
        f"  Record {mw}-{ml_all.settled_rows - mw} ({ml_all.win_rate:.1%}) | "
        f"staked {ml_all.total_staked:.2f}u | profit {ml_all.profit_units:+.2f}u | "
        f"stake ROI {_roi_str(ml_all)}"
    )
    lines.append(
        f"  Avg CLV {ml_all.avg_clv:+.4f} | beat close {ml_all.beat_close_rate:.1%} | "
        f"open {ml_all.open_rows}"
    )

    # --- All-time props ----------------------------------------------------- #
    lines.append("")
    lines.append("PROPS - all time (Kelly)")
    lines.append(
        f"  Record {prop_kelly.won}-{prop_kelly.lost} ({prop_kelly.win_rate:.1%}) | "
        f"staked {prop_kelly.total_staked:.2f}u | "
        f"profit {prop_kelly.profit_units:+.2f}u | "
        f"stake ROI {_roi_str(prop_kelly)}"
    )
    lines.append(f"  void {prop_all.void_rows} | open {prop_all.open_rows}")

    # --- Outstanding -------------------------------------------------------- #
    lines.append("")
    lines.append("OUTSTANDING")
    lines.append(f"  Moneyline pending: {len(ml_open)}")
    lines.append(f"  Props pending: {len(prop_open)}")

    text = "\n".join(lines)
    subject = (
        f"MLB Betting Report {night_str} - "
        f"ML {ml_all.profit_units:+.2f}u / Props Kelly {prop_kelly.profit_units:+.2f}u"
    )
    html = (
        '<pre style="font-family:ui-monospace,Menlo,monospace;font-size:13px;'
        f'line-height:1.4">{escape(text)}</pre>'
    )
    return subject, text, html


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def send_gmail(
    subject: str, text: str, html: str, *, to: str, user: str, app_password: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30) as server:
        server.login(user, app_password)
        server.send_message(msg)


def resolve_night(date_arg: str | None) -> date:
    if date_arg:
        return date.fromisoformat(date_arg)
    return datetime.now(ET).date() - timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="night to report (YYYY-MM-DD); default = yesterday ET")
    ap.add_argument("--to", default=os.environ.get("MLB_REPORT_EMAIL_TO", DEFAULT_TO))
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the report to stdout instead of emailing")
    ap.add_argument("--no-settle", action="store_true",
                    help="report current ledger state without settling first")
    ap.add_argument(
        "--shared-bankroll",
        type=float,
        default=float(os.environ.get("BARLOWE_SHARED_BETTING_BANKROLL", "10000")),
        help="Starting shared bankroll in dollars for all paper bets and arbitrage.",
    )
    ap.add_argument(
        "--paper-unit-dollars",
        type=float,
        default=float(os.environ.get("BARLOWE_PAPER_UNIT_DOLLARS", "100")),
        help="Dollar value of one paper-bet unit in shared bankroll reporting.",
    )
    args = ap.parse_args()
    if args.shared_bankroll <= 0.0:
        raise SystemExit("--shared-bankroll must be greater than zero")
    if args.paper_unit_dollars <= 0.0:
        raise SystemExit("--paper-unit-dollars must be greater than zero")

    config = PostgresConfig.from_env()
    night = resolve_night(args.date)
    if not args.no_settle:
        settle_all(config)

    ml_rows = load_paper_trade_rows(db_config=config)
    prop_rows = load_prop_bet_rows(config)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    arb_summary = _load_arbitrage_summary(config, today, args.shared_bankroll)
    arb_rows = _load_arbitrage_bet_rows(config)
    subject, text, html = build_report(
        night,
        ml_rows,
        prop_rows,
        arb_summary=arb_summary,
        arb_rows=arb_rows,
        shared_bankroll=args.shared_bankroll,
        paper_unit_dollars=args.paper_unit_dollars,
    )

    if args.print_only:
        print(subject)
        print(text)
        return

    app_password = os.environ.get("MLB_REPORT_GMAIL_APP_PASSWORD", "")
    user = os.environ.get("MLB_REPORT_GMAIL_USER") or str(args.to)
    if not app_password:
        raise SystemExit(
            "MLB_REPORT_GMAIL_APP_PASSWORD is not set. Create a Google App Password "
            "(https://myaccount.google.com/apppasswords) for the sender account and "
            "export it (e.g. in ~/.zshrc). Use --print to preview without sending."
        )
    send_gmail(subject, text, html, to=args.to, user=user, app_password=app_password)
    print(f"sent report for {night.isoformat()} to {args.to}")


if __name__ == "__main__":
    main()
