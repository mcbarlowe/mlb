"""Selection-fixed moneyline line-shopping simulation.

The model side and edge come from the consensus opening market. Execution then
uses the best available sportsbook opening price for that same side, which
isolates price-shopping lift from model-selection changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.betting.backtest import kelly_fraction
from src.betting.odds import american_to_decimal, no_vig_two_way

__all__ = [
    "BookLine",
    "LineShoppingBet",
    "LineShoppingGame",
    "LineShoppingSummary",
    "line_shop_moneyline",
    "summarize_line_shopping",
]


@dataclass(frozen=True)
class BookLine:
    bookmaker: str
    home_ml: float
    away_ml: float


@dataclass(frozen=True)
class LineShoppingGame:
    game_pk: int
    season: int
    model_prob_home: float
    consensus_open_home: float
    consensus_open_away: float
    consensus_close_home: float
    consensus_close_away: float
    open_lines: Sequence[BookLine]
    close_lines: Sequence[BookLine]
    home_won: bool


@dataclass(frozen=True)
class LineShoppingBet:
    game_pk: int
    season: int
    side: str
    edge: float
    source_book: str
    source_books: tuple[str, ...]
    consensus_decimal: float
    best_decimal: float
    consensus_stake: float
    best_stake: float
    consensus_profit: float
    best_profit: float
    consensus_clv: float
    best_clv_vs_consensus_close: float
    best_clv_vs_source_close: float | None
    consensus_beat_close: bool
    best_beat_consensus_close: bool
    best_beat_source_close: bool | None
    won: bool


@dataclass(frozen=True)
class LineShoppingSummary:
    n_games: int
    n_bets: int
    consensus_total_staked: float
    best_total_staked: float
    consensus_net_profit: float
    best_net_profit: float
    consensus_roi: float
    best_roi: float
    roi_lift: float
    consensus_avg_clv: float
    best_avg_clv_vs_consensus_close: float
    clv_lift_vs_consensus_close: float
    consensus_pct_beat_close: float
    best_pct_beat_consensus_close: float
    avg_decimal_lift: float
    avg_relative_decimal_lift: float
    source_close_n: int
    best_avg_clv_vs_source_close: float
    best_pct_beat_source_close: float
    settings: dict[str, object] = field(default_factory=dict)


def _side_fair_prob(
    *, home_ml: float, away_ml: float, side: str, devig_method: str
) -> float:
    home_p, away_p = no_vig_two_way(home_ml, away_ml, method=devig_method)
    return home_p if side == "home" else away_p


def _side_decimal(line: BookLine, side: str) -> float:
    return american_to_decimal(line.home_ml if side == "home" else line.away_ml)


def _stake(
    *,
    staking: str,
    model_prob: float,
    decimal_odds: float,
    flat_stake: float,
    kelly_multiplier: float,
    kelly_cap: float,
) -> float:
    if staking == "flat":
        return flat_stake
    if staking == "kelly":
        full_kelly = kelly_fraction(model_prob, decimal_odds)
        return min(full_kelly * kelly_multiplier, kelly_cap)
    raise ValueError(f"Unknown staking plan {staking!r}")


def _profit(decimal_odds: float, won: bool, stake: float) -> float:
    return stake * (decimal_odds - 1.0) if won else -stake


def summarize_line_shopping(
    *,
    n_games: int,
    bets: Sequence[LineShoppingBet],
    settings: dict[str, object] | None = None,
) -> LineShoppingSummary:
    n_bets = len(bets)
    consensus_staked = sum(b.consensus_stake for b in bets)
    best_staked = sum(b.best_stake for b in bets)
    consensus_net = sum(b.consensus_profit for b in bets)
    best_net = sum(b.best_profit for b in bets)
    consensus_clv = sum(b.consensus_clv for b in bets) / n_bets if n_bets else 0.0
    best_clv = (
        sum(b.best_clv_vs_consensus_close for b in bets) / n_bets if n_bets else 0.0
    )
    source_clvs = [
        b.best_clv_vs_source_close
        for b in bets
        if b.best_clv_vs_source_close is not None
    ]
    source_beats = [
        b.best_beat_source_close
        for b in bets
        if b.best_beat_source_close is not None
    ]
    return LineShoppingSummary(
        n_games=n_games,
        n_bets=n_bets,
        consensus_total_staked=consensus_staked,
        best_total_staked=best_staked,
        consensus_net_profit=consensus_net,
        best_net_profit=best_net,
        consensus_roi=consensus_net / consensus_staked if consensus_staked else 0.0,
        best_roi=best_net / best_staked if best_staked else 0.0,
        roi_lift=(best_net / best_staked - consensus_net / consensus_staked)
        if best_staked and consensus_staked
        else 0.0,
        consensus_avg_clv=consensus_clv,
        best_avg_clv_vs_consensus_close=best_clv,
        clv_lift_vs_consensus_close=best_clv - consensus_clv,
        consensus_pct_beat_close=(
            sum(b.consensus_beat_close for b in bets) / n_bets if n_bets else 0.0
        ),
        best_pct_beat_consensus_close=(
            sum(b.best_beat_consensus_close for b in bets) / n_bets if n_bets else 0.0
        ),
        avg_decimal_lift=(
            sum(b.best_decimal - b.consensus_decimal for b in bets) / n_bets
            if n_bets
            else 0.0
        ),
        avg_relative_decimal_lift=(
            sum((b.best_decimal / b.consensus_decimal) - 1.0 for b in bets) / n_bets
            if n_bets
            else 0.0
        ),
        source_close_n=len(source_clvs),
        best_avg_clv_vs_source_close=(
            sum(source_clvs) / len(source_clvs) if source_clvs else 0.0
        ),
        best_pct_beat_source_close=(
            sum(bool(v) for v in source_beats) / len(source_beats)
            if source_beats
            else 0.0
        ),
        settings=settings or {},
    )


def line_shop_moneyline(
    games: Sequence[LineShoppingGame],
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.02,
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
) -> tuple[LineShoppingSummary, list[LineShoppingBet]]:
    bets: list[LineShoppingBet] = []
    if staking not in {"flat", "kelly"}:
        raise ValueError(f"Unknown staking plan {staking!r}")

    for game in games:
        if not game.open_lines:
            continue

        open_home_p, open_away_p = no_vig_two_way(
            game.consensus_open_home,
            game.consensus_open_away,
            method=devig_method,
        )
        close_home_p, close_away_p = no_vig_two_way(
            game.consensus_close_home,
            game.consensus_close_away,
            method=devig_method,
        )
        model_home = game.model_prob_home
        model_away = 1.0 - model_home
        home_edge = model_home - open_home_p
        away_edge = model_away - open_away_p
        if home_edge >= away_edge:
            side = "home"
            edge = home_edge
            model_prob = model_home
            consensus_decimal = american_to_decimal(game.consensus_open_home)
            consensus_take_p = open_home_p
            consensus_close_p = close_home_p
        else:
            side = "away"
            edge = away_edge
            model_prob = model_away
            consensus_decimal = american_to_decimal(game.consensus_open_away)
            consensus_take_p = open_away_p
            consensus_close_p = close_away_p
        if edge < edge_threshold:
            continue

        best_decimal = max(_side_decimal(line, side) for line in game.open_lines)
        best_lines = tuple(
            sorted(
                (
                    line
                    for line in game.open_lines
                    if math.isclose(
                        _side_decimal(line, side),
                        best_decimal,
                        abs_tol=1e-12,
                    )
                ),
                key=lambda line: line.bookmaker,
            )
        )
        close_by_book = {line.bookmaker: line for line in game.close_lines}
        best_line = next(
            (line for line in best_lines if line.bookmaker in close_by_book),
            best_lines[0],
        )
        consensus_stake = _stake(
            staking=staking,
            model_prob=model_prob,
            decimal_odds=consensus_decimal,
            flat_stake=flat_stake,
            kelly_multiplier=kelly_multiplier,
            kelly_cap=kelly_cap,
        )
        best_stake = _stake(
            staking=staking,
            model_prob=model_prob,
            decimal_odds=best_decimal,
            flat_stake=flat_stake,
            kelly_multiplier=kelly_multiplier,
            kelly_cap=kelly_cap,
        )
        if consensus_stake <= 0.0 and best_stake <= 0.0:
            continue

        best_take_p = _side_fair_prob(
            home_ml=best_line.home_ml,
            away_ml=best_line.away_ml,
            side=side,
            devig_method=devig_method,
        )
        source_close = close_by_book.get(best_line.bookmaker)
        if source_close is None:
            best_source_clv = None
            best_source_beat = None
        else:
            source_close_p = _side_fair_prob(
                home_ml=source_close.home_ml,
                away_ml=source_close.away_ml,
                side=side,
                devig_method=devig_method,
            )
            best_source_clv = source_close_p - best_take_p
            best_source_beat = best_take_p < source_close_p

        won = (side == "home") == game.home_won
        bets.append(
            LineShoppingBet(
                game_pk=game.game_pk,
                season=game.season,
                side=side,
                edge=edge,
                source_book=best_line.bookmaker,
                source_books=tuple(line.bookmaker for line in best_lines),
                consensus_decimal=consensus_decimal,
                best_decimal=best_decimal,
                consensus_stake=consensus_stake,
                best_stake=best_stake,
                consensus_profit=_profit(consensus_decimal, won, consensus_stake),
                best_profit=_profit(best_decimal, won, best_stake),
                consensus_clv=consensus_close_p - consensus_take_p,
                best_clv_vs_consensus_close=consensus_close_p - best_take_p,
                best_clv_vs_source_close=best_source_clv,
                consensus_beat_close=consensus_take_p < consensus_close_p,
                best_beat_consensus_close=best_take_p < consensus_close_p,
                best_beat_source_close=best_source_beat,
                won=won,
            )
        )

    summary = summarize_line_shopping(
        n_games=len(games),
        bets=bets,
        settings={
            "devig_method": devig_method,
            "edge_threshold": edge_threshold,
            "staking": staking,
            "flat_stake": flat_stake,
            "kelly_multiplier": kelly_multiplier,
            "kelly_cap": kelly_cap,
        },
    )
    return summary, bets
