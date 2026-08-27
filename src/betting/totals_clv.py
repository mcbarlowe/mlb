"""Totals open-to-close CLV and betting-summary helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from src.betting.backtest import kelly_fraction
from src.betting.odds import american_to_decimal, decimal_to_american, no_vig_two_way

__all__ = [
    "TotalsBookLine",
    "TotalsClvBet",
    "TotalsClvGame",
    "TotalsClvSummary",
    "TotalsConsensusLine",
    "TotalsExecution",
    "consensus_totals_line",
    "select_totals_clv_bet",
    "summarize_totals_clv",
]

TotalsExecution = Literal["consensus", "best"]


@dataclass(frozen=True)
class TotalsBookLine:
    bookmaker: str
    point: float
    over_ml: float
    under_ml: float


@dataclass(frozen=True)
class TotalsConsensusLine:
    point: float
    over_ml: float
    under_ml: float
    prob_over: float
    prob_under: float


@dataclass(frozen=True)
class _ExecutedTotalsLine:
    point: float
    ml: float
    decimal: float
    fair_prob: float
    bookmakers: tuple[str, ...]


@dataclass(frozen=True)
class TotalsClvGame:
    game_pk: int
    season: int
    open_lines: Sequence[TotalsBookLine]
    close_lines: Sequence[TotalsBookLine]
    simulated_totals: Sequence[float]
    actual_total: float


@dataclass(frozen=True)
class TotalsClvBet:
    game_pk: int
    season: int
    side: str
    edge: float
    model_prob: float
    open_point: float
    close_point: float
    open_market_prob: float
    close_market_prob: float
    open_ml: float
    open_decimal: float
    stake: float
    result: str
    profit: float
    point_clv: float
    prob_clv: float
    beat_close: bool
    execution: TotalsExecution = "consensus"
    execution_bookmakers: tuple[str, ...] = ("consensus",)
    execution_fair_prob: float | None = None
    execution_model_prob: float | None = None


@dataclass(frozen=True)
class TotalsClvSummary:
    n_games: int
    n_bets: int
    pushes: int
    wins: int
    total_staked: float
    net_profit: float
    roi: float
    win_rate: float
    avg_edge: float
    avg_point_clv: float
    avg_prob_clv: float
    beat_close_rate: float
    settings: dict[str, object] = field(default_factory=dict)


def consensus_totals_line(
    lines: Sequence[TotalsBookLine], *, devig_method: str = "proportional"
) -> TotalsConsensusLine | None:
    if not lines:
        return None
    point = statistics.median(line.point for line in lines)
    over_decimal = statistics.median(american_to_decimal(line.over_ml) for line in lines)
    under_decimal = statistics.median(american_to_decimal(line.under_ml) for line in lines)
    over_ml = decimal_to_american(over_decimal)
    under_ml = decimal_to_american(under_decimal)
    prob_over, prob_under = no_vig_two_way(over_ml, under_ml, method=devig_method)
    return TotalsConsensusLine(
        point=point,
        over_ml=over_ml,
        under_ml=under_ml,
        prob_over=prob_over,
        prob_under=prob_under,
    )


def _sim_probability(
    simulated_totals: Sequence[float], *, point: float, side: str
) -> float:
    if not simulated_totals:
        raise ValueError("simulated_totals must not be empty")
    pushes = sum(1 for total in simulated_totals if math.isclose(total, point))
    if side == "over":
        wins = sum(1 for total in simulated_totals if total > point)
    elif side == "under":
        wins = sum(1 for total in simulated_totals if total < point)
    else:
        raise ValueError(f"unknown totals side {side!r}")
    return (wins + 0.5 * pushes) / len(simulated_totals)


def _side_prob(line: TotalsConsensusLine, side: str) -> float:
    if side == "over":
        return line.prob_over
    if side == "under":
        return line.prob_under
    raise ValueError(f"unknown totals side {side!r}")


def _side_ml(line: TotalsConsensusLine | TotalsBookLine, side: str) -> float:
    if side == "over":
        return line.over_ml
    if side == "under":
        return line.under_ml
    raise ValueError(f"unknown totals side {side!r}")


def _book_side_prob(
    line: TotalsBookLine, side: str, *, devig_method: str
) -> float:
    prob_over, prob_under = no_vig_two_way(
        line.over_ml, line.under_ml, method=devig_method
    )
    if side == "over":
        return prob_over
    if side == "under":
        return prob_under
    raise ValueError(f"unknown totals side {side!r}")


def _normalize_execution(execution: str) -> TotalsExecution:
    if execution == "consensus" or execution == "best":
        return execution
    raise ValueError(f"unknown totals execution mode {execution!r}")


def _consensus_execution_line(
    line: TotalsConsensusLine, side: str
) -> _ExecutedTotalsLine:
    ml = _side_ml(line, side)
    return _ExecutedTotalsLine(
        point=line.point,
        ml=ml,
        decimal=american_to_decimal(ml),
        fair_prob=_side_prob(line, side),
        bookmakers=("consensus",),
    )


def _best_execution_line(
    lines: Sequence[TotalsBookLine], side: str, *, devig_method: str
) -> _ExecutedTotalsLine | None:
    if not lines:
        return None
    if side not in {"over", "under"}:
        raise ValueError(f"unknown totals side {side!r}")

    def key(line: TotalsBookLine) -> tuple[float, float]:
        point_value = -line.point if side == "over" else line.point
        return point_value, american_to_decimal(_side_ml(line, side))

    best_point_value, best_decimal = max(key(line) for line in lines)
    candidates = [
        line
        for line in lines
        if key(line) == (best_point_value, best_decimal)
    ]
    fair_prob = min(
        _book_side_prob(line, side, devig_method=devig_method) for line in candidates
    )
    best_lines = [
        line
        for line in candidates
        if math.isclose(
            _book_side_prob(line, side, devig_method=devig_method),
            fair_prob,
            abs_tol=1e-12,
        )
    ]
    selected = min(best_lines, key=lambda line: line.bookmaker)
    return _ExecutedTotalsLine(
        point=selected.point,
        ml=_side_ml(selected, side),
        decimal=best_decimal,
        fair_prob=fair_prob,
        bookmakers=tuple(sorted(line.bookmaker for line in best_lines)),
    )


def _execution_line(
    lines: Sequence[TotalsBookLine],
    consensus_line: TotalsConsensusLine,
    side: str,
    *,
    execution: TotalsExecution,
    devig_method: str,
) -> _ExecutedTotalsLine | None:
    if execution == "consensus":
        return _consensus_execution_line(consensus_line, side)
    return _best_execution_line(lines, side, devig_method=devig_method)


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
        return min(kelly_fraction(model_prob, decimal_odds) * kelly_multiplier, kelly_cap)
    raise ValueError(f"Unknown staking plan {staking!r}")


def _settlement(side: str, *, actual_total: float, point: float) -> str:
    if math.isclose(actual_total, point):
        return "push"
    if side == "over":
        return "win" if actual_total > point else "loss"
    if side == "under":
        return "win" if actual_total < point else "loss"
    raise ValueError(f"unknown totals side {side!r}")


def _profit(decimal_odds: float, result: str, stake: float) -> float:
    if result == "win":
        return stake * (decimal_odds - 1.0)
    if result == "loss":
        return -stake
    if result == "push":
        return 0.0
    raise ValueError(f"unknown result {result!r}")


def _point_clv(side: str, *, open_point: float, close_point: float) -> float:
    if side == "over":
        return close_point - open_point
    if side == "under":
        return open_point - close_point
    raise ValueError(f"unknown totals side {side!r}")


def _beat_close(*, point_clv: float, prob_clv: float) -> bool:
    return point_clv > 0.0 or (
        math.isclose(point_clv, 0.0, abs_tol=1e-12) and prob_clv > 0.0
    )


def select_totals_clv_bet(
    game: TotalsClvGame,
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.03,
    execution: str = "consensus",
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
) -> TotalsClvBet | None:
    open_line = consensus_totals_line(game.open_lines, devig_method=devig_method)
    execution_mode = _normalize_execution(execution)
    close_line = consensus_totals_line(game.close_lines, devig_method=devig_method)
    if open_line is None or close_line is None:
        return None

    model_over = _sim_probability(
        game.simulated_totals, point=open_line.point, side="over"
    )
    model_under = _sim_probability(
        game.simulated_totals, point=open_line.point, side="under"
    )
    over_edge = model_over - open_line.prob_over
    under_edge = model_under - open_line.prob_under
    if over_edge >= under_edge:
        side = "over"
        edge = over_edge
        model_prob = model_over
    else:
        side = "under"
        edge = under_edge
        model_prob = model_under
    if edge < edge_threshold:
        return None

    executed_line = _execution_line(
        game.open_lines,
        open_line,
        side,
        execution=execution_mode,
        devig_method=devig_method,
    )
    if executed_line is None:
        return None

    close_market_prob = _side_prob(close_line, side)
    execution_model_prob = _sim_probability(
        game.simulated_totals, point=executed_line.point, side=side
    )
    stake = _stake(
        staking=staking,
        model_prob=execution_model_prob,
        decimal_odds=executed_line.decimal,
        flat_stake=flat_stake,
        kelly_multiplier=kelly_multiplier,
        kelly_cap=kelly_cap,
    )
    if stake <= 0.0:
        return None

    result = _settlement(side, actual_total=game.actual_total, point=executed_line.point)
    point_clv = _point_clv(
        side, open_point=executed_line.point, close_point=close_line.point
    )
    prob_clv = close_market_prob - executed_line.fair_prob
    return TotalsClvBet(
        game_pk=game.game_pk,
        season=game.season,
        side=side,
        edge=edge,
        model_prob=model_prob,
        open_point=executed_line.point,
        close_point=close_line.point,
        open_market_prob=executed_line.fair_prob,
        close_market_prob=close_market_prob,
        open_ml=executed_line.ml,
        open_decimal=executed_line.decimal,
        stake=stake,
        result=result,
        profit=_profit(executed_line.decimal, result, stake),
        point_clv=point_clv,
        prob_clv=prob_clv,
        beat_close=_beat_close(point_clv=point_clv, prob_clv=prob_clv),
        execution=execution_mode,
        execution_bookmakers=executed_line.bookmakers,
        execution_fair_prob=executed_line.fair_prob,
        execution_model_prob=execution_model_prob,
    )


def summarize_totals_clv(
    games: Sequence[TotalsClvGame],
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.03,
    execution: str = "consensus",
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
) -> tuple[TotalsClvSummary, list[TotalsClvBet]]:
    execution_mode = _normalize_execution(execution)
    bets = [
        bet
        for game in games
        if (
            bet := select_totals_clv_bet(
                game,
                devig_method=devig_method,
                edge_threshold=edge_threshold,
                execution=execution_mode,
                staking=staking,
                flat_stake=flat_stake,
                kelly_multiplier=kelly_multiplier,
                kelly_cap=kelly_cap,
            )
        )
        is not None
    ]
    total_staked = sum(bet.stake for bet in bets)
    net_profit = sum(bet.profit for bet in bets)
    non_push = [bet for bet in bets if bet.result != "push"]
    wins = sum(1 for bet in non_push if bet.result == "win")
    return (
        TotalsClvSummary(
            n_games=len(games),
            n_bets=len(bets),
            pushes=sum(1 for bet in bets if bet.result == "push"),
            wins=wins,
            total_staked=total_staked,
            net_profit=net_profit,
            roi=net_profit / total_staked if total_staked else 0.0,
            win_rate=wins / len(non_push) if non_push else 0.0,
            avg_edge=sum(bet.edge for bet in bets) / len(bets) if bets else 0.0,
            avg_point_clv=sum(bet.point_clv for bet in bets) / len(bets)
            if bets
            else 0.0,
            avg_prob_clv=sum(bet.prob_clv for bet in bets) / len(bets)
            if bets
            else 0.0,
            beat_close_rate=sum(bet.beat_close for bet in bets) / len(bets)
            if bets
            else 0.0,
            settings={
                "devig_method": devig_method,
                "edge_threshold": edge_threshold,
                "execution": execution_mode,
                "staking": staking,
                "flat_stake": flat_stake,
                "kelly_multiplier": kelly_multiplier,
                "kelly_cap": kelly_cap,
            },
        ),
        bets,
    )
