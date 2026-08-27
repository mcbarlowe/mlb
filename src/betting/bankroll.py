"""Shared bankroll and stake-ROI aggregation for paper betting ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class BankrollEvent:
    source: str
    event_date: date
    stake: float
    profit: float


@dataclass(frozen=True)
class DailyBankrollPoint:
    event_date: date
    bets: int
    staked: float
    profit: float
    bankroll: float


@dataclass(frozen=True)
class SharedBankrollSummary:
    starting_bankroll: float
    current_bankroll: float
    total_bets: int
    total_staked: float
    net_profit: float
    roi: float
    bankroll_return: float
    peak_bankroll: float
    max_drawdown: float
    max_drawdown_pct: float
    daily_points: tuple[DailyBankrollPoint, ...]


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)  # type: ignore[arg-type]


def _as_date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return datetime.fromisoformat(text).date()


def _moneyline_events(
    rows: Sequence[Mapping[str, object]],
    *,
    paper_unit_dollars: float,
) -> list[BankrollEvent]:
    events: list[BankrollEvent] = []
    for row in rows:
        if str(row.get("status", "")).lower() != "settled":
            continue
        event_date = _as_date(row.get("paper_date"))
        if event_date is None:
            continue
        stake_units = _as_float(row.get("stake_units"))
        profit_units = _as_float(row.get("profit_units"))
        if stake_units <= 0.0:
            continue
        events.append(
            BankrollEvent(
                source="moneyline",
                event_date=event_date,
                stake=stake_units * paper_unit_dollars,
                profit=profit_units * paper_unit_dollars,
            )
        )
    return events


def _prop_events(
    rows: Sequence[Mapping[str, object]],
    *,
    paper_unit_dollars: float,
    prop_stakes: Sequence[float] | None,
) -> list[BankrollEvent]:
    events: list[BankrollEvent] = []
    if prop_stakes is not None and len(prop_stakes) != len(rows):
        raise ValueError("prop_stakes must match prop rows length")
    for index, row in enumerate(rows):
        if str(row.get("status", "")).lower() not in {"won", "lost"}:
            continue
        event_date = _as_date(row.get("game_date") or row.get("alert_date"))
        if event_date is None:
            continue
        if prop_stakes is None:
            stake_units = _as_float(row.get("stake_units"), 1.0)
            profit_units = _as_float(row.get("profit_units"))
        else:
            stake_units = prop_stakes[index]
            decimal_odds = _as_float(row.get("decimal_odds"))
            profit_units = (
                stake_units * (decimal_odds - 1.0)
                if str(row.get("status", "")).lower() == "won"
                else -stake_units
            )
        if stake_units <= 0.0:
            continue
        events.append(
            BankrollEvent(
                source="props",
                event_date=event_date,
                stake=stake_units * paper_unit_dollars,
                profit=profit_units * paper_unit_dollars,
            )
        )
    return events


def _arbitrage_events(rows: Sequence[Mapping[str, object]]) -> list[BankrollEvent]:
    events: list[BankrollEvent] = []
    for row in rows:
        event_date = _as_date(
            row.get("event_date") or row.get("created_date") or row.get("created_at")
        )
        if event_date is None:
            continue
        stake = _as_float(row.get("total_stake"))
        profit = _as_float(row.get("expected_profit"))
        if stake <= 0.0:
            continue
        events.append(
            BankrollEvent(
                source="arbitrage",
                event_date=event_date,
                stake=stake,
                profit=profit,
            )
        )
    return events


def summarize_shared_bankroll(
    moneyline_rows: Sequence[Mapping[str, object]],
    prop_rows: Sequence[Mapping[str, object]],
    arbitrage_rows: Sequence[Mapping[str, object]],
    *,
    starting_bankroll: float,
    paper_unit_dollars: float,
    prop_stakes: Sequence[float] | None = None,
) -> SharedBankrollSummary:
    """Aggregate all paper ledgers into one realized bankroll curve.

    ROI is always net profit divided by total money staked. Moneyline rows are
    stored in units; prop rows use ``prop_stakes`` when supplied so reports can
    aggregate Kelly-sized prop bets instead of the flat stake persisted in the
    ledger. ``paper_unit_dollars`` converts those units into the same dollar
    bankroll used by arbitrage rows.
    """
    if starting_bankroll <= 0.0:
        raise ValueError("starting_bankroll must be greater than zero")
    if paper_unit_dollars <= 0.0:
        raise ValueError("paper_unit_dollars must be greater than zero")

    events = sorted(
        [
            *_moneyline_events(moneyline_rows, paper_unit_dollars=paper_unit_dollars),
            *_prop_events(
                prop_rows,
                paper_unit_dollars=paper_unit_dollars,
                prop_stakes=prop_stakes,
            ),
            *_arbitrage_events(arbitrage_rows),
        ],
        key=lambda event: event.event_date,
    )

    current = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    points: list[DailyBankrollPoint] = []
    for event_date in sorted({event.event_date for event in events}):
        day_events = [event for event in events if event.event_date == event_date]
        day_profit = sum(event.profit for event in day_events)
        current += day_profit
        peak = max(peak, current)
        drawdown = current - peak
        max_drawdown = min(max_drawdown, drawdown)
        points.append(
            DailyBankrollPoint(
                event_date=event_date,
                bets=len(day_events),
                staked=sum(event.stake for event in day_events),
                profit=day_profit,
                bankroll=current,
            )
        )

    total_staked = sum(event.stake for event in events)
    net_profit = current - starting_bankroll
    return SharedBankrollSummary(
        starting_bankroll=starting_bankroll,
        current_bankroll=current,
        total_bets=len(events),
        total_staked=total_staked,
        net_profit=net_profit,
        roi=net_profit / total_staked if total_staked else 0.0,
        bankroll_return=net_profit / starting_bankroll,
        peak_bankroll=peak,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown / peak if peak else 0.0,
        daily_points=tuple(points),
    )


def format_money(value: float) -> str:
    sign = "+" if value >= 0.0 else "-"
    return f"{sign}${abs(value):,.2f}"


def format_shared_bankroll(
    summary: SharedBankrollSummary,
    *,
    recent_days: int = 7,
) -> list[str]:
    lines = [
        (
            f"Start {format_money(summary.starting_bankroll)} | "
            f"Current {format_money(summary.current_bankroll)} | "
            f"Return {summary.bankroll_return:+.1%}"
        ),
        (
            f"Bets {summary.total_bets} | "
            f"Staked {format_money(summary.total_staked)} | "
            f"Net {format_money(summary.net_profit)} | Stake ROI {summary.roi:+.1%}"
        ),
        (
            f"Peak {format_money(summary.peak_bankroll)} | "
            f"Max drawdown {format_money(summary.max_drawdown)} "
            f"({summary.max_drawdown_pct:+.1%})"
        ),
    ]
    if recent_days > 0 and summary.daily_points:
        lines.append("Recent daily bankroll:")
        for point in summary.daily_points[-recent_days:]:
            lines.append(
                f"  {point.event_date.isoformat()}: {point.bets} bets | "
                f"staked {format_money(point.staked)} | "
                f"net {format_money(point.profit)} | "
                f"bankroll {format_money(point.bankroll)}"
            )
    return lines
