"""Settlement and reporting helpers for moneyline paper trades."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.betting.odds import no_vig_two_way

__all__ = [
    "PaperTradeSummary",
    "moneyline_profit",
    "settle_paper_trade_row",
    "summarize_paper_trade_rows",
]


@dataclass(frozen=True)
class PaperTradeSummary:
    rows: int
    open_rows: int
    settled_rows: int
    clv_rows: int
    total_staked: float
    profit_units: float
    roi: float
    avg_clv: float
    beat_close_rate: float
    win_rate: float


def _float_value(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing required numeric field {key!r}")
    return float(value)


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def _selected_won(side: str, home_won: bool) -> bool:
    if side == "home":
        return home_won
    if side == "away":
        return not home_won
    raise ValueError(f"Unknown side {side!r}")


def moneyline_profit(
    *,
    side: str,
    best_decimal: float,
    stake_units: float,
    home_won: bool,
) -> float:
    """Return settled profit in stake units for a selected moneyline side."""
    if stake_units < 0:
        raise ValueError("stake_units must be non-negative")
    if best_decimal <= 1.0:
        raise ValueError("best_decimal must be greater than 1")
    if _selected_won(side, home_won):
        return stake_units * (best_decimal - 1.0)
    return -stake_units


def settle_paper_trade_row(
    row: Mapping[str, str],
    *,
    home_won: bool,
    close_home_ml: float | None = None,
    close_away_ml: float | None = None,
    devig_method: str = "proportional",
) -> dict[str, str]:
    """Return a paper-trade CSV row with result, profit, and optional CLV filled."""
    side = row.get("side", "")
    best_decimal = _float_value(row, "best_decimal")
    stake_units = _float_value(row, "stake_units")
    profit = moneyline_profit(
        side=side,
        best_decimal=best_decimal,
        stake_units=stake_units,
        home_won=home_won,
    )
    settled = dict(row)
    won = _selected_won(side, home_won)
    settled["status"] = "settled"
    settled["result"] = "win" if won else "loss"
    settled["profit_units"] = f"{profit:.4f}"

    if close_home_ml is None or close_away_ml is None:
        return settled

    close_home_prob, close_away_prob = no_vig_two_way(
        close_home_ml,
        close_away_ml,
        method=devig_method,
    )
    close_prob = close_home_prob if side == "home" else close_away_prob
    selected_close_ml = close_home_ml if side == "home" else close_away_ml
    open_fair_prob = _float_value(row, "best_fair_prob")
    settled["close_ml"] = f"{selected_close_ml:.1f}"
    settled["close_fair_prob"] = f"{close_prob:.6f}"
    settled["clv"] = f"{close_prob - open_fair_prob:.6f}"
    return settled


def summarize_paper_trade_rows(
    rows: Sequence[Mapping[str, str]],
) -> PaperTradeSummary:
    settled = [row for row in rows if row.get("status") == "settled"]
    open_rows = len(rows) - len(settled)
    total_staked = sum(_optional_float(row, "stake_units") or 0.0 for row in settled)
    profit_units = sum(_optional_float(row, "profit_units") or 0.0 for row in settled)
    clvs = [value for row in settled if (value := _optional_float(row, "clv")) is not None]
    wins = [row for row in settled if row.get("result") == "win"]
    return PaperTradeSummary(
        rows=len(rows),
        open_rows=open_rows,
        settled_rows=len(settled),
        clv_rows=len(clvs),
        total_staked=total_staked,
        profit_units=profit_units,
        roi=profit_units / total_staked if total_staked else 0.0,
        avg_clv=sum(clvs) / len(clvs) if clvs else 0.0,
        beat_close_rate=(sum(1 for clv in clvs if clv > 0.0) / len(clvs))
        if clvs
        else 0.0,
        win_rate=len(wins) / len(settled) if settled else 0.0,
    )
