from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from scripts.email_daily_betting_report import (
    build_report as build_daily_betting_report,
)
from scripts.settle_daily_paper_trades import build_report_text, send_email_report
from src.betting.bankroll import summarize_shared_bankroll


def test_build_report_text_contains_daily_summary() -> None:
    summary = SimpleNamespace(
        rows=12,
        settled_rows=10,
        open_rows=2,
        win_rate=0.6,
        roi=0.1234,
        total_staked=31.95,
        profit_units=3.94,
        avg_clv=0.013,
        beat_close_rate=0.9,
    )
    prop_summary = SimpleNamespace(
        rows=8,
        settled_rows=6,
        open_rows=1,
        void_rows=1,
        won=4,
        lost=2,
        win_rate=4 / 6,
        roi=0.1842,
        total_staked=6.0,
        profit_units=1.105,
    )
    prop_kelly_summary = SimpleNamespace(
        won=4,
        lost=2,
        roi=0.2211,
        total_staked=12.5,
        profit_units=2.76375,
    )

    text = build_report_text(
        target_date=date(2026, 8, 24),
        updated=3,
        missing_final=1,
        missing_close=0,
        summary=summary,
        prop_summary=prop_summary,
        prop_kelly_summary=prop_kelly_summary,
        props_newly_settled=2,
    )

    assert "[2026-08-24] Settlement scan: updated=3 missing_final=1 missing_close=0" in text
    assert "Stake ROI:           +12.34%" in text
    assert "PLAYER PROPS PAPER TRADING SUMMARY" in text
    assert "Newly Settled:      2" in text
    assert "Total Props:        8 (6 settled, 1 pending, 1 void)" in text
    assert "Record:             4-2" in text
    assert "Flat Stake ROI:     +18.42%" in text
    assert "Kelly Stake ROI:    +22.11%" in text


def test_build_report_text_includes_arbitrage_summary() -> None:
    summary = SimpleNamespace(
        rows=0,
        settled_rows=0,
        open_rows=0,
        win_rate=0.0,
        roi=0.0,
        total_staked=0.0,
        profit_units=0.0,
        avg_clv=0.0,
        beat_close_rate=0.0,
    )

    text = build_report_text(
        target_date=date(2026, 8, 24),
        updated=0,
        missing_final=0,
        missing_close=0,
        summary=summary,
        arbitrage_summary_text="Arbs expected: today 2 bets +$4.20 (+1.2% stake ROI)",
    )

    assert "ARBITRAGE PAPER TRADING SUMMARY" in text
    assert "Arbs expected: today 2 bets +$4.20 (+1.2% stake ROI)" in text


def test_shared_bankroll_summary_uses_net_profit_over_total_staked() -> None:
    summary = summarize_shared_bankroll(
        [
            {
                "paper_date": "2026-08-24",
                "status": "settled",
                "stake_units": 2.0,
                "profit_units": 1.5,
            }
        ],
        [
            {
                "game_date": "2026-08-25",
                "status": "won",
                "stake_units": 1.0,
                "profit_units": 1.2,
            },
            {
                "game_date": "2026-08-26",
                "status": "lost",
                "stake_units": 1.0,
                "profit_units": -1.0,
            },
            {
                "game_date": "2026-08-26",
                "status": "void",
                "stake_units": 1.0,
                "profit_units": 0.0,
            },
        ],
        [{"event_date": "2026-08-25", "total_stake": 500.0, "expected_profit": 25.0}],
        starting_bankroll=10_000.0,
        paper_unit_dollars=100.0,
    )

    assert summary.total_bets == 4
    assert summary.total_staked == 900.0
    assert summary.net_profit == 195.0
    assert summary.roi == 195.0 / 900.0
    assert summary.current_bankroll == 10_195.0
    assert [point.bankroll for point in summary.daily_points] == [
        10_150.0,
        10_295.0,
        10_195.0,
    ]
    assert summary.max_drawdown == -100.0


def test_daily_betting_report_includes_shared_bankroll_curve() -> None:
    subject, text, _ = build_daily_betting_report(
        date(2026, 8, 25),
        [
            {
                "paper_date": "2026-08-24",
                "status": "settled",
                "side": "home",
                "away_team": "NYY",
                "home_team": "BAL",
                "best_ml": "+150",
                "best_decimal": "2.5",
                "stake_units": "2",
                "profit_units": "3",
                "result": "win",
            }
        ],
        [
            {
                "game_date": "2026-08-25",
                "status": "lost",
                "player": "A Batter",
                "market": "batter_hits",
                "point": 1.5,
                "side": "over",
                "price": "+100",
                "decimal_odds": "2.0",
                "stake_units": "1",
                "profit_units": "-1",
                "book": "book",
                "matchup": "NYY@BAL",
            },
            {
                "game_date": "2026-08-26",
                "status": "open",
                "player": "Busy Detail",
                "market": "batter_home_runs",
                "point": 0.5,
                "side": "over",
                "price": "+900",
                "decimal_odds": "10.0",
                "stake_units": "1",
                "book": "book",
                "matchup": "NYY@BAL",
            },
        ],
        arb_summary=None,
        arb_rows=[
            {"event_date": "2026-08-25", "total_stake": 500.0, "expected_profit": 25.0}
        ],
        shared_bankroll=10_000.0,
        paper_unit_dollars=100.0,
    )

    assert subject == "MLB Betting Report 2026-08-25 - ML +3.00u / Props -1.00u"
    assert "SHARED BANKROLL - all paper bets + arbs" in text
    assert "Current +$10,225.00" in text
    assert "Staked +$800.00 | Net +$225.00 | Stake ROI +28.1%" in text
    assert "OUTSTANDING" in text
    assert "Props pending: 1" in text
    assert "A Batter" not in text
    assert "Busy Detail" not in text
    assert "NYY@BAL" not in text


def test_send_email_report_uses_gmail_smtp_env() -> None:
    with patch.dict(
        "os.environ",
        {
            "EMAIL_PASSWORD": "secret",
            "BETTING_ARB_EMAIL_FROM": "sabresbot@gmail.com",
            "BETTING_ARB_EMAIL_USERNAME": "sabresbot@gmail.com",
            "BETTING_ARB_SMTP_HOST": "smtp.gmail.com",
            "BETTING_ARB_SMTP_PORT": "587",
        },
        clear=True,
    ), patch("scripts.settle_daily_paper_trades.smtplib.SMTP") as smtp:
        client = smtp.return_value.__enter__.return_value

        send_email_report(
            recipient="mcbarlowe@gmail.com",
            subject="Daily paper betting report 2026-08-24",
            body="report body",
        )

    smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=15.0)
    client.starttls.assert_called_once()
    client.login.assert_called_once_with("sabresbot@gmail.com", "secret")
    message = client.send_message.call_args.args[0]
    assert message["From"] == "sabresbot@gmail.com"
    assert message["To"] == "mcbarlowe@gmail.com"
    assert message["Subject"] == "Daily paper betting report 2026-08-24"
    assert "report body" in message.get_content()
