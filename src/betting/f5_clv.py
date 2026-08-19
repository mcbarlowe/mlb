"""First-five totals CLV and betting-evaluation helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from src.betting.backtest import kelly_fraction
from src.betting.odds import american_to_decimal, decimal_to_american, no_vig_two_way

__all__ = [
    "F5BookLine",
    "F5ClvBet",
    "F5ClvGame",
    "F5ClvSummary",
    "F5ConsensusLine",
    "F5Execution",
    "F5LineShoppingBet",
    "F5LineShoppingSummary",
    "F5ModelProbability",
    "best_f5_execution_line",
    "compare_f5_line_shopping",
    "consensus_f5_totals_line",
    "select_f5_clv_bet",
    "summarize_f5_clv",
]

EPS = 1e-6
F5Execution = Literal["best", "consensus"]



@dataclass(frozen=True)
class F5BookLine:
    bookmaker: str
    point: float
    over_ml: float
    under_ml: float


@dataclass(frozen=True)
class F5ConsensusLine:
    point: float
    over_ml: float
    under_ml: float
    prob_over: float
    prob_under: float

@dataclass(frozen=True)
class _ExecutedF5Line:
    point: float
    ml: float
    decimal: float
    fair_prob: float
    bookmakers: tuple[str, ...]


@dataclass(frozen=True)
class F5ClvGame:
    game_pk: int
    season: int
    open_lines: Sequence[F5BookLine]
    simulated_totals: Sequence[float]
    actual_total: float | None = None
    close_lines: Sequence[F5BookLine] = ()
    take_line_type: str = "open"

F5ModelProbability = Callable[[F5ClvGame, float, str, float], float]



@dataclass(frozen=True)
class F5ClvBet:
    game_pk: int
    season: int
    side: str
    edge: float
    model_prob: float
    execution_model_prob: float
    open_point: float
    take_point: float
    close_point: float | None
    open_market_prob: float
    take_market_prob: float
    close_market_prob: float | None
    take_bookmaker: str
    take_ml: float
    take_decimal: float
    stake: float
    result: str
    profit: float
    point_clv: float | None
    prob_clv: float | None
    beat_close: bool | None
    execution: F5Execution = "best"
    execution_bookmakers: tuple[str, ...] = ()
    execution_fair_prob: float | None = None


@dataclass(frozen=True)
class F5LineShoppingBet:
    game_pk: int
    season: int
    side: str
    edge: float
    model_prob: float
    consensus_point: float
    consensus_ml: float
    consensus_decimal: float
    consensus_market_prob: float
    consensus_model_prob: float
    consensus_stake: float
    consensus_result: str
    consensus_profit: float
    best_bookmaker: str
    best_bookmakers: tuple[str, ...]
    best_point: float
    best_ml: float
    best_decimal: float
    best_market_prob: float
    best_model_prob: float
    best_stake: float
    best_result: str
    best_profit: float
    point_lift: float
    ml_lift: float
    decimal_lift: float
    relative_decimal_lift: float
    close_point: float | None
    close_market_prob: float | None
    consensus_point_clv: float | None
    best_point_clv: float | None
    point_clv_lift: float | None
    consensus_prob_clv: float | None
    best_prob_clv: float | None
    prob_clv_lift: float | None
    consensus_beat_close: bool | None
    best_beat_close: bool | None


@dataclass(frozen=True)
class F5LineShoppingSummary:
    n_games: int
    n_bets: int
    consensus_total_staked: float
    best_total_staked: float
    consensus_net_profit: float
    best_net_profit: float
    consensus_roi: float
    best_roi: float
    roi_lift: float
    avg_point_lift: float
    avg_ml_lift: float
    avg_decimal_lift: float
    avg_relative_decimal_lift: float
    close_n: int
    consensus_avg_point_clv: float | None
    best_avg_point_clv: float | None
    point_clv_lift: float | None
    consensus_avg_prob_clv: float | None
    best_avg_prob_clv: float | None
    prob_clv_lift: float | None
    consensus_beat_close_rate: float | None
    best_beat_close_rate: float | None
    settings: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class F5ClvSummary:
    n_games: int
    n_scored_games: int
    n_bets: int
    n_settled_bets: int
    pushes: int
    wins: int
    total_staked: float
    net_profit: float
    roi: float
    win_rate: float
    avg_edge: float
    avg_point_clv: float | None
    avg_prob_clv: float | None
    beat_close_rate: float | None
    model_brier_all: float | None
    market_brier_all: float | None
    model_log_loss_all: float | None
    market_log_loss_all: float | None
    settings: dict[str, object] = field(default_factory=dict)


def consensus_f5_totals_line(
    lines: Sequence[F5BookLine], *, devig_method: str = "proportional"
) -> F5ConsensusLine | None:
    if not lines:
        return None
    point = statistics.median(line.point for line in lines)
    over_decimal = statistics.median(american_to_decimal(line.over_ml) for line in lines)
    under_decimal = statistics.median(american_to_decimal(line.under_ml) for line in lines)
    over_ml = decimal_to_american(over_decimal)
    under_ml = decimal_to_american(under_decimal)
    prob_over, prob_under = no_vig_two_way(over_ml, under_ml, method=devig_method)
    return F5ConsensusLine(
        point=point,
        over_ml=over_ml,
        under_ml=under_ml,
        prob_over=prob_over,
        prob_under=prob_under,
    )


def best_f5_execution_line(lines: Sequence[F5BookLine], side: str) -> F5BookLine | None:
    if not lines:
        return None
    if side == "over":
        return min(lines, key=lambda line: (line.point, -american_to_decimal(line.over_ml)))
    if side == "under":
        return max(lines, key=lambda line: (line.point, american_to_decimal(line.under_ml)))
    raise ValueError(f"unknown F5 totals side {side!r}")


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
        raise ValueError(f"unknown F5 totals side {side!r}")
    return (wins + 0.5 * pushes) / len(simulated_totals)

def _model_probability(
    game: F5ClvGame,
    *,
    point: float,
    side: str,
    market_prob: float,
    model_probability: F5ModelProbability | None,
) -> float:
    if model_probability is None:
        return _sim_probability(game.simulated_totals, point=point, side=side)
    probability = model_probability(game, point, side, market_prob)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"model probability must be in [0, 1], got {probability!r}")
    return probability


def _side_prob(line: F5ConsensusLine, side: str) -> float:
    if side == "over":
        return line.prob_over
    if side == "under":
        return line.prob_under
    raise ValueError(f"unknown F5 totals side {side!r}")


def _side_ml(line: F5BookLine | F5ConsensusLine, side: str) -> float:
    if side == "over":
        return line.over_ml
    if side == "under":
        return line.under_ml
    raise ValueError(f"unknown F5 totals side {side!r}")


def _book_side_prob(
    line: F5BookLine, side: str, *, devig_method: str = "proportional"
) -> float:
    prob_over, prob_under = no_vig_two_way(
        line.over_ml,
        line.under_ml,
        method=devig_method,
    )
    if side == "over":
        return prob_over
    if side == "under":
        return prob_under
    raise ValueError(f"unknown F5 totals side {side!r}")

def _normalize_execution(execution: str) -> F5Execution:
    if execution == "best":
        return "best"
    if execution == "consensus":
        return "consensus"
    raise ValueError(f"unknown F5 totals execution mode {execution!r}")


def _consensus_execution_line(
    line: F5ConsensusLine, side: str
) -> _ExecutedF5Line:
    ml = _side_ml(line, side)
    return _ExecutedF5Line(
        point=line.point,
        ml=ml,
        decimal=american_to_decimal(ml),
        fair_prob=_side_prob(line, side),
        bookmakers=("consensus",),
    )


def _execution_quality(line: F5BookLine, side: str) -> tuple[float, float]:
    point_value = -line.point if side == "over" else line.point
    return point_value, american_to_decimal(_side_ml(line, side))


def _best_execution_line(
    lines: Sequence[F5BookLine], side: str, *, devig_method: str
) -> _ExecutedF5Line | None:
    if not lines:
        return None
    if side not in {"over", "under"}:
        raise ValueError(f"unknown F5 totals side {side!r}")

    best_point_value, best_decimal = max(_execution_quality(line, side) for line in lines)
    candidates = [
        line
        for line in lines
        if _execution_quality(line, side) == (best_point_value, best_decimal)
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
    return _ExecutedF5Line(
        point=selected.point,
        ml=_side_ml(selected, side),
        decimal=best_decimal,
        fair_prob=fair_prob,
        bookmakers=tuple(sorted(line.bookmaker for line in best_lines)),
    )


def _execution_line(
    lines: Sequence[F5BookLine],
    consensus_line: F5ConsensusLine,
    side: str,
    *,
    execution: F5Execution,
    devig_method: str,
) -> _ExecutedF5Line | None:
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


def _settlement(side: str, *, actual_total: float | None, point: float) -> str:
    if actual_total is None:
        return "pending"
    if math.isclose(actual_total, point):
        return "push"
    if side == "over":
        return "win" if actual_total > point else "loss"
    if side == "under":
        return "win" if actual_total < point else "loss"
    raise ValueError(f"unknown F5 totals side {side!r}")


def _profit(decimal_odds: float, result: str, stake: float) -> float:
    if result == "win":
        return stake * (decimal_odds - 1.0)
    if result == "loss":
        return -stake
    if result in {"push", "pending"}:
        return 0.0
    raise ValueError(f"unknown result {result!r}")


def _point_clv(side: str, *, take_point: float, close_point: float) -> float:
    if side == "over":
        return close_point - take_point
    if side == "under":
        return take_point - close_point
    raise ValueError(f"unknown F5 totals side {side!r}")


def _beat_close(*, point_clv: float, prob_clv: float) -> bool:
    return point_clv > 0.0 or (
        math.isclose(point_clv, 0.0, abs_tol=1e-12) and prob_clv > 0.0
    )


def _actual_over(actual_total: float | None, point: float) -> float | None:
    if actual_total is None or math.isclose(actual_total, point):
        return None
    return 1.0 if actual_total > point else 0.0


def _brier(probability: float, actual: float) -> float:
    return (probability - actual) ** 2


def _log_loss(probability: float, actual: float) -> float:
    probability = min(max(probability, EPS), 1.0 - EPS)
    return -(
        actual * math.log(probability)
        + (1.0 - actual) * math.log(1.0 - probability)
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def select_f5_clv_bet(
    game: F5ClvGame,
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.03,
    execution: str = "best",
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    model_probability: F5ModelProbability | None = None,
) -> F5ClvBet | None:
    open_line = consensus_f5_totals_line(game.open_lines, devig_method=devig_method)
    execution_mode = _normalize_execution(execution)
    if open_line is None:
        return None

    model_over = _model_probability(
        game,
        point=open_line.point,
        side="over",
        market_prob=open_line.prob_over,
        model_probability=model_probability,
    )
    model_under = _model_probability(
        game,
        point=open_line.point,
        side="under",
        market_prob=open_line.prob_under,
        model_probability=model_probability,
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
    execution_model_prob = _model_probability(
        game,
        point=executed_line.point,
        side=side,
        market_prob=executed_line.fair_prob,
        model_probability=model_probability,
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

    open_market_prob = _side_prob(open_line, side)
    close_line = consensus_f5_totals_line(game.close_lines, devig_method=devig_method)
    close_point = close_line.point if close_line else None
    close_market_prob = _side_prob(close_line, side) if close_line else None
    point_clv = (
        _point_clv(side, take_point=executed_line.point, close_point=close_line.point)
        if close_line
        else None
    )
    prob_clv = (
        close_market_prob - executed_line.fair_prob
        if close_market_prob is not None
        else None
    )
    beat_close = (
        _beat_close(point_clv=point_clv, prob_clv=prob_clv)
        if point_clv is not None and prob_clv is not None
        else None
    )
    result = _settlement(
        side, actual_total=game.actual_total, point=executed_line.point
    )
    return F5ClvBet(
        game_pk=game.game_pk,
        season=game.season,
        side=side,
        edge=edge,
        model_prob=model_prob,
        execution_model_prob=execution_model_prob,
        open_point=open_line.point,
        take_point=executed_line.point,
        close_point=close_point,
        open_market_prob=open_market_prob,
        take_market_prob=executed_line.fair_prob,
        close_market_prob=close_market_prob,
        take_bookmaker="|".join(executed_line.bookmakers),
        take_ml=executed_line.ml,
        take_decimal=executed_line.decimal,
        stake=stake,
        result=result,
        profit=_profit(executed_line.decimal, result, stake),
        point_clv=point_clv,
        prob_clv=prob_clv,
        beat_close=beat_close,
        execution=execution_mode,
        execution_bookmakers=executed_line.bookmakers,
        execution_fair_prob=executed_line.fair_prob,
    )


def summarize_f5_clv(
    games: Sequence[F5ClvGame],
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.03,
    execution: str = "best",
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    model_probability: F5ModelProbability | None = None,
) -> tuple[F5ClvSummary, list[F5ClvBet]]:
    execution_mode = _normalize_execution(execution)
    bets = [
        bet
        for game in games
        if (
            bet := select_f5_clv_bet(
                game,
                devig_method=devig_method,
                edge_threshold=edge_threshold,
                execution=execution_mode,
                staking=staking,
                flat_stake=flat_stake,
                kelly_multiplier=kelly_multiplier,
                kelly_cap=kelly_cap,
                model_probability=model_probability,
            )
        )
        is not None
    ]

    settled = [bet for bet in bets if bet.result != "pending"]
    total_staked = sum(bet.stake for bet in settled)
    net_profit = sum(bet.profit for bet in settled)
    non_push = [bet for bet in settled if bet.result != "push"]
    wins = sum(1 for bet in non_push if bet.result == "win")

    scored_games = 0
    model_briers: list[float] = []
    market_briers: list[float] = []
    model_log_losses: list[float] = []
    market_log_losses: list[float] = []
    for game in games:
        open_line = consensus_f5_totals_line(game.open_lines, devig_method=devig_method)
        if open_line is None:
            continue
        actual_over = _actual_over(game.actual_total, open_line.point)
        if actual_over is None:
            continue
        scored_games += 1
        model_over = _model_probability(
            game,
            point=open_line.point,
            side="over",
            market_prob=open_line.prob_over,
            model_probability=model_probability,
        )
        market_over = open_line.prob_over
        model_briers.append(_brier(model_over, actual_over))
        market_briers.append(_brier(market_over, actual_over))
        model_log_losses.append(_log_loss(model_over, actual_over))
        market_log_losses.append(_log_loss(market_over, actual_over))

    clv_bets = [bet for bet in bets if bet.beat_close is not None]
    point_clvs = [bet.point_clv for bet in clv_bets if bet.point_clv is not None]
    prob_clvs = [bet.prob_clv for bet in clv_bets if bet.prob_clv is not None]
    return (
        F5ClvSummary(
            n_games=len(games),
            n_scored_games=scored_games,
            n_bets=len(bets),
            n_settled_bets=len(settled),
            pushes=sum(1 for bet in settled if bet.result == "push"),
            wins=wins,
            total_staked=total_staked,
            net_profit=net_profit,
            roi=net_profit / total_staked if total_staked else 0.0,
            win_rate=wins / len(non_push) if non_push else 0.0,
            avg_edge=sum(bet.edge for bet in bets) / len(bets) if bets else 0.0,
            avg_point_clv=_mean(point_clvs),
            avg_prob_clv=_mean(prob_clvs),
            beat_close_rate=(
                sum(1 for bet in clv_bets if bet.beat_close) / len(clv_bets)
                if clv_bets
                else None
            ),
            model_brier_all=_mean(model_briers),
            market_brier_all=_mean(market_briers),
            model_log_loss_all=_mean(model_log_losses),
            market_log_loss_all=_mean(market_log_losses),
            settings={
                "devig_method": devig_method,
                "edge_threshold": edge_threshold,
                "execution": execution_mode,
                "staking": staking,
                "flat_stake": flat_stake,
                "kelly_multiplier": kelly_multiplier,
                "kelly_cap": kelly_cap,
                "probability_source": "sim" if model_probability is None else "custom",
            },
        ),
        bets,
    )


def compare_f5_line_shopping(
    games: Sequence[F5ClvGame],
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.03,
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    model_probability: F5ModelProbability | None = None,
) -> tuple[F5LineShoppingSummary, list[F5LineShoppingBet]]:
    bets: list[F5LineShoppingBet] = []
    for game in games:
        open_line = consensus_f5_totals_line(game.open_lines, devig_method=devig_method)
        if open_line is None:
            continue

        model_over = _model_probability(
            game,
            point=open_line.point,
            side="over",
            market_prob=open_line.prob_over,
            model_probability=model_probability,
        )
        model_under = _model_probability(
            game,
            point=open_line.point,
            side="under",
            market_prob=open_line.prob_under,
            model_probability=model_probability,
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
            continue

        consensus_line = _consensus_execution_line(open_line, side)
        best_line = _best_execution_line(
            game.open_lines, side, devig_method=devig_method
        )
        if best_line is None:
            continue

        consensus_model_prob = _model_probability(
            game,
            point=consensus_line.point,
            side=side,
            market_prob=consensus_line.fair_prob,
            model_probability=model_probability,
        )
        best_model_prob = _model_probability(
            game,
            point=best_line.point,
            side=side,
            market_prob=best_line.fair_prob,
            model_probability=model_probability,
        )
        consensus_stake = _stake(
            staking=staking,
            model_prob=consensus_model_prob,
            decimal_odds=consensus_line.decimal,
            flat_stake=flat_stake,
            kelly_multiplier=kelly_multiplier,
            kelly_cap=kelly_cap,
        )
        best_stake = _stake(
            staking=staking,
            model_prob=best_model_prob,
            decimal_odds=best_line.decimal,
            flat_stake=flat_stake,
            kelly_multiplier=kelly_multiplier,
            kelly_cap=kelly_cap,
        )
        if consensus_stake <= 0.0 and best_stake <= 0.0:
            continue

        close_line = consensus_f5_totals_line(
            game.close_lines, devig_method=devig_method
        )
        close_market_prob = _side_prob(close_line, side) if close_line else None
        if close_line is None or close_market_prob is None:
            consensus_point_clv = None
            best_point_clv = None
            point_clv_lift = None
            consensus_prob_clv = None
            best_prob_clv = None
            prob_clv_lift = None
            consensus_beat_close = None
            best_beat_close = None
        else:
            consensus_point_clv = _point_clv(
                side, take_point=consensus_line.point, close_point=close_line.point
            )
            best_point_clv = _point_clv(
                side, take_point=best_line.point, close_point=close_line.point
            )
            point_clv_lift = best_point_clv - consensus_point_clv
            consensus_prob_clv = close_market_prob - consensus_line.fair_prob
            best_prob_clv = close_market_prob - best_line.fair_prob
            prob_clv_lift = best_prob_clv - consensus_prob_clv
            consensus_beat_close = _beat_close(
                point_clv=consensus_point_clv, prob_clv=consensus_prob_clv
            )
            best_beat_close = _beat_close(
                point_clv=best_point_clv, prob_clv=best_prob_clv
            )

        consensus_result = _settlement(
            side, actual_total=game.actual_total, point=consensus_line.point
        )
        best_result = _settlement(
            side, actual_total=game.actual_total, point=best_line.point
        )
        bets.append(
            F5LineShoppingBet(
                game_pk=game.game_pk,
                season=game.season,
                side=side,
                edge=edge,
                model_prob=model_prob,
                consensus_point=consensus_line.point,
                consensus_ml=consensus_line.ml,
                consensus_decimal=consensus_line.decimal,
                consensus_market_prob=consensus_line.fair_prob,
                consensus_model_prob=consensus_model_prob,
                consensus_stake=consensus_stake,
                consensus_result=consensus_result,
                consensus_profit=_profit(
                    consensus_line.decimal, consensus_result, consensus_stake
                ),
                best_bookmaker=best_line.bookmakers[0],
                best_bookmakers=best_line.bookmakers,
                best_point=best_line.point,
                best_ml=best_line.ml,
                best_decimal=best_line.decimal,
                best_market_prob=best_line.fair_prob,
                best_model_prob=best_model_prob,
                best_stake=best_stake,
                best_result=best_result,
                best_profit=_profit(best_line.decimal, best_result, best_stake),
                point_lift=_point_clv(
                    side,
                    take_point=best_line.point,
                    close_point=consensus_line.point,
                ),
                ml_lift=best_line.ml - consensus_line.ml,
                decimal_lift=best_line.decimal - consensus_line.decimal,
                relative_decimal_lift=best_line.decimal / consensus_line.decimal - 1.0,
                close_point=close_line.point if close_line else None,
                close_market_prob=close_market_prob,
                consensus_point_clv=consensus_point_clv,
                best_point_clv=best_point_clv,
                point_clv_lift=point_clv_lift,
                consensus_prob_clv=consensus_prob_clv,
                best_prob_clv=best_prob_clv,
                prob_clv_lift=prob_clv_lift,
                consensus_beat_close=consensus_beat_close,
                best_beat_close=best_beat_close,
            )
        )

    consensus_settled = [bet for bet in bets if bet.consensus_result != "pending"]
    best_settled = [bet for bet in bets if bet.best_result != "pending"]
    consensus_staked = sum(bet.consensus_stake for bet in consensus_settled)
    best_staked = sum(bet.best_stake for bet in best_settled)
    consensus_profit = sum(bet.consensus_profit for bet in consensus_settled)
    best_profit = sum(bet.best_profit for bet in best_settled)
    close_bets = [bet for bet in bets if bet.point_clv_lift is not None]
    consensus_point_clvs = [
        bet.consensus_point_clv
        for bet in close_bets
        if bet.consensus_point_clv is not None
    ]
    best_point_clvs = [
        bet.best_point_clv for bet in close_bets if bet.best_point_clv is not None
    ]
    consensus_prob_clvs = [
        bet.consensus_prob_clv
        for bet in close_bets
        if bet.consensus_prob_clv is not None
    ]
    best_prob_clvs = [
        bet.best_prob_clv for bet in close_bets if bet.best_prob_clv is not None
    ]
    consensus_roi = consensus_profit / consensus_staked if consensus_staked else 0.0
    best_roi = best_profit / best_staked if best_staked else 0.0
    consensus_avg_point_clv = _mean(consensus_point_clvs)
    best_avg_point_clv = _mean(best_point_clvs)
    consensus_avg_prob_clv = _mean(consensus_prob_clvs)
    best_avg_prob_clv = _mean(best_prob_clvs)
    return (
        F5LineShoppingSummary(
            n_games=len(games),
            n_bets=len(bets),
            consensus_total_staked=consensus_staked,
            best_total_staked=best_staked,
            consensus_net_profit=consensus_profit,
            best_net_profit=best_profit,
            consensus_roi=consensus_roi,
            best_roi=best_roi,
            roi_lift=best_roi - consensus_roi,
            avg_point_lift=(
                sum(bet.point_lift for bet in bets) / len(bets) if bets else 0.0
            ),
            avg_ml_lift=sum(bet.ml_lift for bet in bets) / len(bets)
            if bets
            else 0.0,
            avg_decimal_lift=(
                sum(bet.decimal_lift for bet in bets) / len(bets) if bets else 0.0
            ),
            avg_relative_decimal_lift=(
                sum(bet.relative_decimal_lift for bet in bets) / len(bets)
                if bets
                else 0.0
            ),
            close_n=len(close_bets),
            consensus_avg_point_clv=consensus_avg_point_clv,
            best_avg_point_clv=best_avg_point_clv,
            point_clv_lift=(
                best_avg_point_clv - consensus_avg_point_clv
                if best_avg_point_clv is not None
                and consensus_avg_point_clv is not None
                else None
            ),
            consensus_avg_prob_clv=consensus_avg_prob_clv,
            best_avg_prob_clv=best_avg_prob_clv,
            prob_clv_lift=(
                best_avg_prob_clv - consensus_avg_prob_clv
                if best_avg_prob_clv is not None
                and consensus_avg_prob_clv is not None
                else None
            ),
            consensus_beat_close_rate=(
                sum(1 for bet in close_bets if bet.consensus_beat_close)
                / len(close_bets)
                if close_bets
                else None
            ),
            best_beat_close_rate=(
                sum(1 for bet in close_bets if bet.best_beat_close)
                / len(close_bets)
                if close_bets
                else None
            ),
            settings={
                "devig_method": devig_method,
                "edge_threshold": edge_threshold,
                "staking": staking,
                "flat_stake": flat_stake,
                "kelly_multiplier": kelly_multiplier,
                "kelly_cap": kelly_cap,
                "probability_source": "sim" if model_probability is None else "custom",
            },
        ),
        bets,
    )
