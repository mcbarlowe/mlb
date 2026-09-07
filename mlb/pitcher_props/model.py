"""Hierarchical empirical-Bayes models for starting-pitcher counting props.

The suite models opportunity before outcomes:

* outs use a discrete per-out removal hazard with league priors and pitcher
  partial pooling;
* batters faced use a beta-negative-binomial exposure model conditional on
  simulated outs;
* strikeouts, hits allowed, and walks use beta-binomial rate models conditional
  on the same simulated batters faced.

All posterior state is chronological and exponentially decayed.  The model has
no sportsbook, price, staking, or ledger dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MODEL_FAMILY = "pitcher-prop-bayes"
MODEL_CONTRACT_VERSION = "pitcher-prop-bayes-v1"
MODEL_COLLECTION = "pitcher_prop_models"

OUTS_MARKET = "pitcher_outs"
BATTERS_FACED_MARKET = "pitcher_batters_faced"
STRIKEOUT_MARKET = "pitcher_strikeouts"
HITS_ALLOWED_MARKET = "pitcher_hits_allowed"
WALKS_MARKET = "pitcher_walks"
RATE_MARKETS = (STRIKEOUT_MARKET, HITS_ALLOWED_MARKET, WALKS_MARKET)
SUPPORTED_MARKETS = frozenset((OUTS_MARKET, BATTERS_FACED_MARKET, *RATE_MARKETS))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def _logit(value: float) -> float:
    probability = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return math.log(probability / (1.0 - probability))


def _logistic(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _decay_factor(
    updated_at: datetime,
    as_of: datetime,
    half_life_days: float,
) -> float:
    current = _as_utc(as_of)
    previous = _as_utc(updated_at)
    elapsed_days = (current - previous).total_seconds() / 86_400.0
    if elapsed_days < -1e-9:
        raise ValueError("posterior state cannot move backwards in time")
    if half_life_days <= 0.0 or elapsed_days <= 0.0:
        return 1.0
    return 0.5 ** (elapsed_days / half_life_days)


def _rest_bucket(rest_days: float | None) -> str:
    if rest_days is None or not math.isfinite(rest_days):
        return "unknown"
    if rest_days < 5.0:
        return "short"
    if rest_days <= 6.0:
        return "normal"
    return "long"


def _workload_context(*, is_home: bool, rest_days: float | None) -> str:
    venue = "home" if is_home else "away"
    return f"{venue}:{_rest_bucket(rest_days)}"


@dataclass(frozen=True)
class PitcherStart:
    """One completed regular-season start used as a chronological observation."""

    game_pk: int
    game_time: datetime
    season: int
    pitcher_id: int
    opponent_team_id: int
    is_home: bool
    outs: int
    batters_faced: int
    strikeouts: int
    hits_allowed: int
    walks: int
    pitch_count: int
    rest_days: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_time", _as_utc(self.game_time))
        if self.game_pk <= 0 or self.pitcher_id <= 0 or self.opponent_team_id <= 0:
            raise ValueError("game, pitcher, and opponent identifiers must be positive")
        if self.outs < 0 or self.batters_faced < self.outs:
            raise ValueError("batters faced must be at least recorded outs")
        for label, value in (
            ("strikeouts", self.strikeouts),
            ("hits_allowed", self.hits_allowed),
            ("walks", self.walks),
        ):
            if value < 0 or value > self.batters_faced:
                raise ValueError(f"{label} must be between zero and batters faced")
        if self.pitch_count < 0:
            raise ValueError("pitch_count cannot be negative")
        if self.rest_days is not None and self.rest_days < 0.0:
            raise ValueError("rest_days cannot be negative")

    def actual(self, market: str) -> int:
        values = {
            OUTS_MARKET: self.outs,
            BATTERS_FACED_MARKET: self.batters_faced,
            STRIKEOUT_MARKET: self.strikeouts,
            HITS_ALLOWED_MARKET: self.hits_allowed,
            WALKS_MARKET: self.walks,
        }
        try:
            return values[market]
        except KeyError as exc:
            raise ValueError(f"unsupported pitcher prop market {market!r}") from exc


@dataclass(frozen=True)
class PitcherPropConfig:
    max_outs: int = 27
    max_batters_faced: int = 50
    workload_prior_starts: float = 25.0
    workload_context_prior_starts: float = 100.0
    batters_faced_prior: float = 200.0
    rate_prior_batters: float = 250.0
    opponent_prior_batters: float = 1_000.0
    opponent_weight: float = 0.5
    recency_half_life_days: float = 365.0
    min_draws: int = 4_000
    max_draws: int = 40_000
    draw_batch: int = 4_000
    mc_tolerance: float = 0.005
    seed: int = 17

    def __post_init__(self) -> None:
        if self.max_outs < 1 or self.max_batters_faced <= self.max_outs:
            raise ValueError("invalid outs or batters-faced bounds")
        for name in (
            "workload_prior_starts",
            "workload_context_prior_starts",
            "batters_faced_prior",
            "rate_prior_batters",
            "opponent_prior_batters",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.opponent_weight <= 1.0:
            raise ValueError("opponent_weight must be between zero and one")
        if self.recency_half_life_days <= 0.0:
            raise ValueError("recency_half_life_days must be positive")
        if not 0 < self.min_draws <= self.max_draws:
            raise ValueError("draw bounds must satisfy 0 < min_draws <= max_draws")
        if self.draw_batch < 1 or self.mc_tolerance <= 0.0:
            raise ValueError("draw_batch and mc_tolerance must be positive")


@dataclass
class _PitcherState:
    workload_survives: list[float]
    workload_stops: list[float]
    outs: float = 0.0
    non_outs: float = 0.0
    rate_successes: dict[str, float] = field(
        default_factory=lambda: {market: 0.0 for market in RATE_MARKETS}
    )
    rate_failures: dict[str, float] = field(
        default_factory=lambda: {market: 0.0 for market in RATE_MARKETS}
    )
    updated_at: datetime | None = None

    @classmethod
    def empty(cls, max_outs: int) -> _PitcherState:
        return cls([0.0] * max_outs, [0.0] * max_outs)

    def decay_to(self, as_of: datetime, half_life_days: float) -> None:
        current = _as_utc(as_of)
        if self.updated_at is None:
            self.updated_at = current
            return
        factor = _decay_factor(self.updated_at, current, half_life_days)
        if factor != 1.0:
            self.workload_survives = [value * factor for value in self.workload_survives]
            self.workload_stops = [value * factor for value in self.workload_stops]
            self.outs *= factor
            self.non_outs *= factor
            for market in RATE_MARKETS:
                self.rate_successes[market] *= factor
                self.rate_failures[market] *= factor
        self.updated_at = current

    def scaled(self, as_of: datetime, half_life_days: float) -> _PitcherState:
        if self.updated_at is None:
            return _PitcherState.empty(len(self.workload_survives))
        factor = _decay_factor(self.updated_at, as_of, half_life_days)
        return _PitcherState(
            workload_survives=[value * factor for value in self.workload_survives],
            workload_stops=[value * factor for value in self.workload_stops],
            outs=self.outs * factor,
            non_outs=self.non_outs * factor,
            rate_successes={
                market: self.rate_successes[market] * factor for market in RATE_MARKETS
            },
            rate_failures={
                market: self.rate_failures[market] * factor for market in RATE_MARKETS
            },
            updated_at=_as_utc(as_of),
        )


@dataclass
class _OpponentState:
    outs: float = 0.0
    non_outs: float = 0.0
    rate_successes: dict[str, float] = field(
        default_factory=lambda: {market: 0.0 for market in RATE_MARKETS}
    )
    rate_failures: dict[str, float] = field(
        default_factory=lambda: {market: 0.0 for market in RATE_MARKETS}
    )
    updated_at: datetime | None = None

    def decay_to(self, as_of: datetime, half_life_days: float) -> None:
        current = _as_utc(as_of)
        if self.updated_at is None:
            self.updated_at = current
            return
        factor = _decay_factor(self.updated_at, current, half_life_days)
        if factor != 1.0:
            self.outs *= factor
            self.non_outs *= factor
            for market in RATE_MARKETS:
                self.rate_successes[market] *= factor
                self.rate_failures[market] *= factor
        self.updated_at = current

    def scaled(self, as_of: datetime, half_life_days: float) -> _OpponentState:
        if self.updated_at is None:
            return _OpponentState(updated_at=_as_utc(as_of))
        factor = _decay_factor(self.updated_at, as_of, half_life_days)
        return _OpponentState(
            outs=self.outs * factor,
            non_outs=self.non_outs * factor,
            rate_successes={
                market: self.rate_successes[market] * factor for market in RATE_MARKETS
            },
            rate_failures={
                market: self.rate_failures[market] * factor for market in RATE_MARKETS
            },
            updated_at=_as_utc(as_of),
        )


@dataclass(frozen=True)
class _LeagueState:
    workload_survives: tuple[float, ...]
    workload_stops: tuple[float, ...]
    workload_contexts: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    outs: float
    non_outs: float
    rate_successes: dict[str, float]
    rate_failures: dict[str, float]

    def rate(self, market: str) -> float:
        if market == BATTERS_FACED_MARKET:
            successes, failures = self.outs, self.non_outs
        else:
            successes = self.rate_successes[market]
            failures = self.rate_failures[market]
        return (successes + 0.5) / (successes + failures + 1.0)

    def workload_rates(
        self,
        context: str,
        context_prior_starts: float,
    ) -> tuple[float, ...]:
        contextual = self.workload_contexts.get(context)
        rates: list[float] = []
        for index, (survives, stops) in enumerate(
            zip(self.workload_survives, self.workload_stops, strict=True)
        ):
            overall = (survives + 0.5) / (survives + stops + 1.0)
            if contextual is None:
                rates.append(overall)
                continue
            context_survives = contextual[0][index]
            context_stops = contextual[1][index]
            rates.append(
                (context_survives + context_prior_starts * overall)
                / (context_survives + context_stops + context_prior_starts)
            )
        return tuple(rates)


@dataclass(frozen=True)
class PosteriorPrediction:
    market: str
    point: float
    mean: float
    standard_deviation: float
    q05: float
    q50: float
    q95: float
    probability_over: float
    probability_push: float
    probability_under: float
    samples: int
    mc_standard_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PitcherPropModel:
    """Fitted hierarchical posterior state and posterior-predictive sampler."""

    def __init__(
        self,
        *,
        config: PitcherPropConfig,
        league: _LeagueState,
        pitchers: dict[int, _PitcherState] | None = None,
        opponents: dict[int, _OpponentState] | None = None,
        trained_from: datetime | None = None,
        trained_through: datetime | None = None,
        training_starts: int = 0,
    ) -> None:
        self.config = config
        self.league = league
        self.pitchers = pitchers or {}
        self.opponents = opponents or {}
        self.trained_from = trained_from
        self.trained_through = trained_through
        self.training_starts = training_starts

    @classmethod
    def fit(
        cls,
        starts: list[PitcherStart],
        *,
        config: PitcherPropConfig | None = None,
    ) -> PitcherPropModel:
        if not starts:
            raise ValueError("at least one completed pitcher start is required")
        resolved_config = config or PitcherPropConfig()
        ordered = sorted(starts, key=lambda row: (row.game_time, row.game_pk, row.pitcher_id))
        league = cls._fit_league(ordered, resolved_config)
        model = cls(config=resolved_config, league=league)
        for start in ordered:
            model.observe(start)
        model.trained_from = ordered[0].game_time
        model.trained_through = ordered[-1].game_time
        model.training_starts = len(ordered)
        return model

    @staticmethod
    def _fit_league(
        starts: list[PitcherStart],
        config: PitcherPropConfig,
    ) -> _LeagueState:
        survives = [0.0] * config.max_outs
        stops = [0.0] * config.max_outs
        contexts: dict[str, tuple[list[float], list[float]]] = {}
        outs = non_outs = 0.0
        rate_successes = {market: 0.0 for market in RATE_MARKETS}
        rate_failures = {market: 0.0 for market in RATE_MARKETS}
        for start in starts:
            context = _workload_context(is_home=start.is_home, rest_days=start.rest_days)
            context_counts = contexts.setdefault(
                context,
                ([0.0] * config.max_outs, [0.0] * config.max_outs),
            )
            capped_outs = min(start.outs, config.max_outs)
            for threshold in range(capped_outs):
                survives[threshold] += 1.0
                context_counts[0][threshold] += 1.0
            if start.outs < config.max_outs:
                stops[start.outs] += 1.0
                context_counts[1][start.outs] += 1.0
            outs += start.outs
            non_outs += start.batters_faced - start.outs
            values = {
                STRIKEOUT_MARKET: start.strikeouts,
                HITS_ALLOWED_MARKET: start.hits_allowed,
                WALKS_MARKET: start.walks,
            }
            for market, value in values.items():
                rate_successes[market] += value
                rate_failures[market] += start.batters_faced - value
        frozen_contexts = {
            key: (tuple(value[0]), tuple(value[1])) for key, value in contexts.items()
        }
        return _LeagueState(
            workload_survives=tuple(survives),
            workload_stops=tuple(stops),
            workload_contexts=frozen_contexts,
            outs=outs,
            non_outs=non_outs,
            rate_successes=rate_successes,
            rate_failures=rate_failures,
        )

    def observe(self, start: PitcherStart) -> None:
        """Apply a completed start after any prediction for that start was emitted."""

        state = self.pitchers.setdefault(
            start.pitcher_id,
            _PitcherState.empty(self.config.max_outs),
        )
        state.decay_to(start.game_time, self.config.recency_half_life_days)
        capped_outs = min(start.outs, self.config.max_outs)
        for threshold in range(capped_outs):
            state.workload_survives[threshold] += 1.0
        if start.outs < self.config.max_outs:
            state.workload_stops[start.outs] += 1.0
        state.outs += start.outs
        state.non_outs += start.batters_faced - start.outs

        opponent = self.opponents.setdefault(start.opponent_team_id, _OpponentState())
        opponent.decay_to(start.game_time, self.config.recency_half_life_days)
        opponent.outs += start.outs
        opponent.non_outs += start.batters_faced - start.outs

        values = {
            STRIKEOUT_MARKET: start.strikeouts,
            HITS_ALLOWED_MARKET: start.hits_allowed,
            WALKS_MARKET: start.walks,
        }
        for market, value in values.items():
            failures = start.batters_faced - value
            state.rate_successes[market] += value
            state.rate_failures[market] += failures
            opponent.rate_successes[market] += value
            opponent.rate_failures[market] += failures

    def clone(self) -> PitcherPropModel:
        return self.from_dict(self.to_dict())

    def _pitcher_state(self, pitcher_id: int, as_of: datetime) -> _PitcherState:
        state = self.pitchers.get(pitcher_id)
        if state is None:
            return _PitcherState.empty(self.config.max_outs)
        return state.scaled(as_of, self.config.recency_half_life_days)

    def _opponent_state(
        self,
        opponent_team_id: int | None,
        as_of: datetime,
    ) -> _OpponentState:
        state = self.opponents.get(opponent_team_id) if opponent_team_id else None
        if state is None:
            return _OpponentState(updated_at=_as_utc(as_of))
        return state.scaled(as_of, self.config.recency_half_life_days)

    def workload_posterior(
        self,
        *,
        pitcher_id: int,
        as_of: datetime,
        is_home: bool,
        rest_days: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self._pitcher_state(pitcher_id, as_of)
        context = _workload_context(is_home=is_home, rest_days=rest_days)
        league_rates = self.league.workload_rates(
            context,
            self.config.workload_context_prior_starts,
        )
        alpha = np.asarray(
            [
                rate * self.config.workload_prior_starts
                + state.workload_survives[index]
                for index, rate in enumerate(league_rates)
            ],
            dtype=float,
        )
        beta = np.asarray(
            [
                (1.0 - rate) * self.config.workload_prior_starts
                + state.workload_stops[index]
                for index, rate in enumerate(league_rates)
            ],
            dtype=float,
        )
        return np.maximum(alpha, 1e-6), np.maximum(beta, 1e-6)

    def rate_posterior(
        self,
        *,
        market: str,
        pitcher_id: int,
        opponent_team_id: int | None,
        as_of: datetime,
    ) -> tuple[float, float]:
        if market not in {BATTERS_FACED_MARKET, *RATE_MARKETS}:
            raise ValueError(f"market {market!r} has no beta rate posterior")
        pitcher = self._pitcher_state(pitcher_id, as_of)
        opponent = self._opponent_state(opponent_team_id, as_of)
        league_rate = self.league.rate(market)
        if market == BATTERS_FACED_MARKET:
            prior_strength = self.config.batters_faced_prior
            successes, failures = pitcher.outs, pitcher.non_outs
            opponent_successes, opponent_failures = opponent.outs, opponent.non_outs
        else:
            prior_strength = self.config.rate_prior_batters
            successes = pitcher.rate_successes[market]
            failures = pitcher.rate_failures[market]
            opponent_successes = opponent.rate_successes[market]
            opponent_failures = opponent.rate_failures[market]
        alpha = league_rate * prior_strength + successes
        beta = (1.0 - league_rate) * prior_strength + failures
        concentration = alpha + beta
        pitcher_mean = alpha / concentration

        opponent_alpha = (
            league_rate * self.config.opponent_prior_batters + opponent_successes
        )
        opponent_beta = (
            (1.0 - league_rate) * self.config.opponent_prior_batters
            + opponent_failures
        )
        opponent_mean = opponent_alpha / (opponent_alpha + opponent_beta)
        adjusted_mean = _logistic(
            _logit(pitcher_mean)
            + self.config.opponent_weight * (_logit(opponent_mean) - _logit(league_rate))
        )
        return (
            max(adjusted_mean * concentration, 1e-6),
            max((1.0 - adjusted_mean) * concentration, 1e-6),
        )

    def _simulate_with_rng(
        self,
        *,
        pitcher_id: int,
        opponent_team_id: int | None,
        as_of: datetime,
        is_home: bool,
        rest_days: float | None,
        samples: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        workload_alpha, workload_beta = self.workload_posterior(
            pitcher_id=pitcher_id,
            as_of=as_of,
            is_home=is_home,
            rest_days=rest_days,
        )
        outs = np.zeros(samples, dtype=np.int16)
        active = np.ones(samples, dtype=bool)
        for threshold in range(self.config.max_outs):
            if not np.any(active):
                break
            indexes = np.flatnonzero(active)
            survival = rng.beta(
                workload_alpha[threshold],
                workload_beta[threshold],
                size=len(indexes),
            )
            survived = rng.random(len(indexes)) < survival
            outs[indexes[survived]] += 1
            active[indexes[~survived]] = False

        exposure_alpha, exposure_beta = self.rate_posterior(
            market=BATTERS_FACED_MARKET,
            pitcher_id=pitcher_id,
            opponent_team_id=opponent_team_id,
            as_of=as_of,
        )
        out_probability = np.clip(
            rng.beta(exposure_alpha, exposure_beta, size=samples),
            1e-6,
            1.0 - 1e-6,
        )
        non_outs = np.empty(samples, dtype=np.int16)
        recorded_out = outs > 0
        non_outs[recorded_out] = rng.negative_binomial(
            outs[recorded_out],
            out_probability[recorded_out],
        ).astype(np.int16)
        zero_out = ~recorded_out
        non_outs[zero_out] = (
            1
            + rng.negative_binomial(1, out_probability[zero_out]).astype(np.int16)
        )
        batters_faced = np.minimum(
            outs.astype(np.int32) + non_outs.astype(np.int32),
            self.config.max_batters_faced,
        ).astype(np.int16)

        simulations: dict[str, np.ndarray] = {
            OUTS_MARKET: outs,
            BATTERS_FACED_MARKET: batters_faced,
        }
        for market in RATE_MARKETS:
            alpha, beta = self.rate_posterior(
                market=market,
                pitcher_id=pitcher_id,
                opponent_team_id=opponent_team_id,
                as_of=as_of,
            )
            probability = np.clip(
                rng.beta(alpha, beta, size=samples),
                1e-6,
                1.0 - 1e-6,
            )
            simulations[market] = rng.binomial(batters_faced, probability).astype(
                np.int16
            )
        return simulations

    def simulate_components(
        self,
        *,
        pitcher_id: int,
        opponent_team_id: int | None,
        as_of: datetime,
        is_home: bool = False,
        rest_days: float | None = None,
        samples: int = 4_000,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        if samples < 1:
            raise ValueError("samples must be positive")
        return self._simulate_with_rng(
            pitcher_id=pitcher_id,
            opponent_team_id=opponent_team_id,
            as_of=_as_utc(as_of),
            is_home=is_home,
            rest_days=rest_days,
            samples=samples,
            rng=np.random.default_rng(self.config.seed if seed is None else seed),
        )

    def predict(
        self,
        *,
        pitcher_id: int,
        opponent_team_id: int | None,
        market: str,
        point: float,
        as_of: datetime,
        is_home: bool = False,
        rest_days: float | None = None,
        seed: int | None = None,
        min_draws: int | None = None,
        max_draws: int | None = None,
        mc_tolerance: float | None = None,
    ) -> PosteriorPrediction:
        if market not in SUPPORTED_MARKETS:
            raise ValueError(f"unsupported pitcher prop market {market!r}")
        if not math.isfinite(point) or point < 0.0:
            raise ValueError("point must be a non-negative finite number")
        minimum = self.config.min_draws if min_draws is None else min_draws
        maximum = self.config.max_draws if max_draws is None else max_draws
        tolerance = self.config.mc_tolerance if mc_tolerance is None else mc_tolerance
        if not 0 < minimum <= maximum or tolerance <= 0.0:
            raise ValueError("invalid posterior simulation bounds")

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        chunks: list[np.ndarray] = []
        total = 0
        standard_error = math.inf
        while total < maximum:
            batch = min(self.config.draw_batch, maximum - total)
            simulated = self._simulate_with_rng(
                pitcher_id=pitcher_id,
                opponent_team_id=opponent_team_id,
                as_of=_as_utc(as_of),
                is_home=is_home,
                rest_days=rest_days,
                samples=batch,
                rng=rng,
            )[market]
            chunks.append(simulated)
            total += batch
            values = np.concatenate(chunks)
            probability_over = float(np.mean(values > point))
            standard_error = math.sqrt(
                max(probability_over * (1.0 - probability_over), 0.0) / total
            )
            if total >= minimum and standard_error <= tolerance:
                break

        values = np.concatenate(chunks).astype(float)
        probability_over = float(np.mean(values > point))
        probability_push = float(np.mean(values == point))
        probability_under = float(np.mean(values < point))
        return PosteriorPrediction(
            market=market,
            point=float(point),
            mean=float(np.mean(values)),
            standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            q05=float(np.quantile(values, 0.05)),
            q50=float(np.quantile(values, 0.50)),
            q95=float(np.quantile(values, 0.95)),
            probability_over=probability_over,
            probability_push=probability_push,
            probability_under=probability_under,
            samples=total,
            mc_standard_error=standard_error,
        )

    def to_dict(self) -> dict[str, object]:
        def pitcher_state(state: _PitcherState) -> dict[str, object]:
            return {
                "workload_survives": state.workload_survives,
                "workload_stops": state.workload_stops,
                "outs": state.outs,
                "non_outs": state.non_outs,
                "rate_successes": state.rate_successes,
                "rate_failures": state.rate_failures,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            }

        def opponent_state(state: _OpponentState) -> dict[str, object]:
            return {
                "outs": state.outs,
                "non_outs": state.non_outs,
                "rate_successes": state.rate_successes,
                "rate_failures": state.rate_failures,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            }

        return {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "model_family": MODEL_FAMILY,
            "config": asdict(self.config),
            "league": {
                "workload_survives": self.league.workload_survives,
                "workload_stops": self.league.workload_stops,
                "workload_contexts": self.league.workload_contexts,
                "outs": self.league.outs,
                "non_outs": self.league.non_outs,
                "rate_successes": self.league.rate_successes,
                "rate_failures": self.league.rate_failures,
            },
            "pitchers": {
                str(identifier): pitcher_state(state)
                for identifier, state in sorted(self.pitchers.items())
            },
            "opponents": {
                str(identifier): opponent_state(state)
                for identifier, state in sorted(self.opponents.items())
            },
            "trained_from": self.trained_from.isoformat() if self.trained_from else None,
            "trained_through": (
                self.trained_through.isoformat() if self.trained_through else None
            ),
            "training_starts": self.training_starts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PitcherPropModel:
        if payload.get("model_contract_version") != MODEL_CONTRACT_VERSION:
            raise ValueError("unsupported pitcher prop model contract")
        league_payload = payload["league"]
        contexts = {
            str(key): (tuple(value[0]), tuple(value[1]))
            for key, value in league_payload["workload_contexts"].items()
        }
        league = _LeagueState(
            workload_survives=tuple(league_payload["workload_survives"]),
            workload_stops=tuple(league_payload["workload_stops"]),
            workload_contexts=contexts,
            outs=float(league_payload["outs"]),
            non_outs=float(league_payload["non_outs"]),
            rate_successes={
                str(key): float(value)
                for key, value in league_payload["rate_successes"].items()
            },
            rate_failures={
                str(key): float(value)
                for key, value in league_payload["rate_failures"].items()
            },
        )
        pitchers: dict[int, _PitcherState] = {}
        for identifier, state in payload["pitchers"].items():
            pitchers[int(identifier)] = _PitcherState(
                workload_survives=[float(value) for value in state["workload_survives"]],
                workload_stops=[float(value) for value in state["workload_stops"]],
                outs=float(state["outs"]),
                non_outs=float(state["non_outs"]),
                rate_successes={
                    str(key): float(value)
                    for key, value in state["rate_successes"].items()
                },
                rate_failures={
                    str(key): float(value)
                    for key, value in state["rate_failures"].items()
                },
                updated_at=(
                    datetime.fromisoformat(state["updated_at"])
                    if state["updated_at"]
                    else None
                ),
            )
        opponents: dict[int, _OpponentState] = {}
        for identifier, state in payload["opponents"].items():
            opponents[int(identifier)] = _OpponentState(
                outs=float(state["outs"]),
                non_outs=float(state["non_outs"]),
                rate_successes={
                    str(key): float(value)
                    for key, value in state["rate_successes"].items()
                },
                rate_failures={
                    str(key): float(value)
                    for key, value in state["rate_failures"].items()
                },
                updated_at=(
                    datetime.fromisoformat(state["updated_at"])
                    if state["updated_at"]
                    else None
                ),
            )
        return cls(
            config=PitcherPropConfig(**payload["config"]),
            league=league,
            pitchers=pitchers,
            opponents=opponents,
            trained_from=(
                datetime.fromisoformat(payload["trained_from"])
                if payload.get("trained_from")
                else None
            ),
            trained_through=(
                datetime.fromisoformat(payload["trained_through"])
                if payload.get("trained_through")
                else None
            ),
            training_starts=int(payload.get("training_starts", 0)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PitcherPropModel:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "BATTERS_FACED_MARKET",
    "HITS_ALLOWED_MARKET",
    "MODEL_COLLECTION",
    "MODEL_CONTRACT_VERSION",
    "MODEL_FAMILY",
    "OUTS_MARKET",
    "RATE_MARKETS",
    "STRIKEOUT_MARKET",
    "SUPPORTED_MARKETS",
    "WALKS_MARKET",
    "PitcherPropConfig",
    "PitcherPropModel",
    "PitcherStart",
    "PosteriorPrediction",
]
