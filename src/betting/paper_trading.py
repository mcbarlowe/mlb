"""Moneyline paper-trade selection helpers.

The selector is intentionally pure: callers provide model probability and the
available book prices, and it returns the paper-trade pick that should be logged.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from src.betting.backtest import kelly_fraction
from src.betting.odds import american_to_decimal, decimal_to_american, no_vig_two_way

__all__ = [
    "PaperOddsLine",
    "PaperTradePick",
    "select_moneyline_paper_trade",
]


@dataclass(frozen=True)
class PaperOddsLine:
    bookmaker: str
    home_ml: float
    away_ml: float
    last_update: str | None = None


@dataclass(frozen=True)
class PaperTradePick:
    side: str
    edge: float
    model_prob: float
    consensus_market_prob: float
    consensus_home_prob: float
    consensus_away_prob: float
    consensus_home_ml: float
    consensus_away_ml: float
    best_books: tuple[str, ...]
    best_ml: float
    best_decimal: float
    best_fair_prob: float
    stake_fraction: float
    stake_units: float


@dataclass(frozen=True)
class _ConsensusPrice:
    home_ml: float
    away_ml: float
    home_prob: float
    away_prob: float


def _consensus_price(
    lines: Sequence[PaperOddsLine], *, devig_method: str
) -> _ConsensusPrice:
    if not lines:
        raise ValueError("At least one odds line is required")
    home_decimal = statistics.median(american_to_decimal(line.home_ml) for line in lines)
    away_decimal = statistics.median(american_to_decimal(line.away_ml) for line in lines)
    home_ml = decimal_to_american(home_decimal)
    away_ml = decimal_to_american(away_decimal)
    home_prob, away_prob = no_vig_two_way(home_ml, away_ml, method=devig_method)
    return _ConsensusPrice(
        home_ml=home_ml,
        away_ml=away_ml,
        home_prob=home_prob,
        away_prob=away_prob,
    )


def _line_decimal(line: PaperOddsLine, side: str) -> float:
    return american_to_decimal(line.home_ml if side == "home" else line.away_ml)


def _line_ml(line: PaperOddsLine, side: str) -> float:
    return line.home_ml if side == "home" else line.away_ml


def _side_fair_prob(line: PaperOddsLine, side: str, devig_method: str) -> float:
    home_prob, away_prob = no_vig_two_way(
        line.home_ml,
        line.away_ml,
        method=devig_method,
    )
    return home_prob if side == "home" else away_prob


def _stake_fraction(
    *,
    staking: str,
    model_prob: float,
    decimal_odds: float,
    flat_stake_units: float,
    bankroll_units: float,
    kelly_multiplier: float,
    kelly_cap: float,
) -> float:
    if staking == "flat":
        if bankroll_units <= 0:
            raise ValueError("bankroll_units must be positive")
        return flat_stake_units / bankroll_units
    if staking == "kelly":
        full_kelly = kelly_fraction(model_prob, decimal_odds)
        return min(full_kelly * kelly_multiplier, kelly_cap)
    raise ValueError(f"Unknown staking plan {staking!r}")


def select_moneyline_paper_trade(
    *,
    model_prob_home: float,
    odds_lines: Sequence[PaperOddsLine],
    edge_threshold: float = 0.05,
    staking: str = "kelly",
    bankroll_units: float = 100.0,
    flat_stake_units: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    devig_method: str = "proportional",
) -> PaperTradePick | None:
    """Return the selected paper trade, or None when no side clears the edge.

    The side and edge are computed against the consensus no-vig opening market.
    Execution uses the best available book price for that side.
    """
    consensus = _consensus_price(odds_lines, devig_method=devig_method)
    home_edge = model_prob_home - consensus.home_prob
    away_model_prob = 1.0 - model_prob_home
    away_edge = away_model_prob - consensus.away_prob
    if home_edge >= away_edge:
        side = "home"
        edge = home_edge
        model_prob = model_prob_home
        market_prob = consensus.home_prob
    else:
        side = "away"
        edge = away_edge
        model_prob = away_model_prob
        market_prob = consensus.away_prob
    if edge < edge_threshold:
        return None

    best_decimal = max(_line_decimal(line, side) for line in odds_lines)
    best_lines = tuple(
        sorted(
            (
                line
                for line in odds_lines
                if _line_decimal(line, side) == best_decimal
            ),
            key=lambda line: line.bookmaker,
        )
    )
    primary = best_lines[0]
    stake_fraction = _stake_fraction(
        staking=staking,
        model_prob=model_prob,
        decimal_odds=best_decimal,
        flat_stake_units=flat_stake_units,
        bankroll_units=bankroll_units,
        kelly_multiplier=kelly_multiplier,
        kelly_cap=kelly_cap,
    )
    stake_units = bankroll_units * stake_fraction
    if stake_units <= 0.0:
        return None

    return PaperTradePick(
        side=side,
        edge=edge,
        model_prob=model_prob,
        consensus_market_prob=market_prob,
        consensus_home_prob=consensus.home_prob,
        consensus_away_prob=consensus.away_prob,
        consensus_home_ml=consensus.home_ml,
        consensus_away_ml=consensus.away_ml,
        best_books=tuple(line.bookmaker for line in best_lines),
        best_ml=_line_ml(primary, side),
        best_decimal=best_decimal,
        best_fair_prob=_side_fair_prob(primary, side, devig_method),
        stake_fraction=stake_fraction,
        stake_units=stake_units,
    )
