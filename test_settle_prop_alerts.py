from __future__ import annotations

from datetime import date

from scripts.settle_prop_alerts import (
    ArbitrageSummary,
    _build_push_lines,
    _daily_report_due,
    _format_arbitrage_summary,
    _load_arbitrage_summary,
    _mark_daily_report_sent,
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchone(self):
        return self.rows.pop(0)


def test_format_arbitrage_summary_includes_staked_roi_and_bankroll_return() -> None:
    summary = ArbitrageSummary(
        today_bets=4,
        today_stake=400.0,
        today_profit=21.45,
        all_bets=43,
        all_stake=4300.0,
        all_profit=106.96,
        starting_bankroll=10_000.0,
    )

    assert _format_arbitrage_summary(summary) == (
        "Arbs expected: today 4 bets +$21.45 (+5.4% stake ROI) "
        "(+0.2% bankroll); all-time 43 bets +$106.96 (+2.5% stake ROI); "
        "bankroll +$10,106.96 (+1.1%)"
    )

def test_push_lines_include_arbitrage_bankroll_summary() -> None:
    summary = ArbitrageSummary(
        today_bets=1,
        today_stake=100.0,
        today_profit=2.50,
        all_bets=2,
        all_stake=200.0,
        all_profit=5.00,
        starting_bankroll=1_000.0,
    )

    lines = _build_push_lines(
        date(2026, 8, 27),
        [("Player", "batter_hits", 0.5, "over", 120, "won", 1, 1.2)],
        wins=10,
        losses=5,
        net=3.5,
        staked=15,
        arb_summary=summary,
        prop_bankroll=100.0,
    )

    assert lines[0] == "Props settled 2026-08-27: 1-0, +1.20u (+120% stake ROI)"
    assert lines[1] == "Props bankroll +103.50u (+3.5%); today +1.20u (+1.2%)"
    assert lines[2] == (
        "Arbs expected: today 1 bets +$2.50 (+2.5% stake ROI) "
        "(+0.2% bankroll); all-time 2 bets +$5.00 (+2.5% stake ROI); "
        "bankroll +$1,005.00 (+0.5%)"
    )


def test_load_arbitrage_summary_returns_none_when_table_missing() -> None:
    cur = FakeCursor([(None,)])

    assert _load_arbitrage_summary(cur, date(2026, 8, 27), 10_000.0) is None


def test_load_arbitrage_summary_reads_betting_schema_totals() -> None:
    today = date(2026, 8, 27)
    cur = FakeCursor(
        [
            ("betting.arbitrage_paper_bets",),
            (4, 400.0, 21.45, 43, 4300.0, 106.96),
        ]
    )

    summary = _load_arbitrage_summary(cur, today, 10_000.0)

    assert summary == ArbitrageSummary(
        today_bets=4,
        today_stake=400.0,
        today_profit=21.45,
        all_bets=43,
        all_stake=4300.0,
        all_profit=106.96,
        starting_bankroll=10_000.0,
    )
    assert "FROM betting.arbitrage_paper_bets" in cur.queries[1][0]
    assert cur.queries[1][1] == (today, today, today)

def test_daily_report_state_tracks_one_push_per_day(tmp_path) -> None:
    path = tmp_path / "settle_prop_alerts_state.json"
    today = date(2026, 8, 27)

    assert _daily_report_due(path, today) is True

    _mark_daily_report_sent(path, today)

    assert _daily_report_due(path, today) is False
    assert _daily_report_due(path, date(2026, 8, 28)) is True
