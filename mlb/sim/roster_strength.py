"""Leak-free player-lineup and bullpen projections for pregame win models."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

WOBA_EDGE_UNIT = 0.010


@dataclass(frozen=True)
class RosterStrengthConfig:
    league_woba: float = 0.320
    batter_prior_pa: float = 200.0
    batter_recency_half_life_days: float = 180.0
    batter_season_decay: float = 0.60
    batter_peak_age: float = 27.0
    batter_growth_per_year: float = 0.0015
    batter_decline_per_year: float = 0.0030
    bullpen_prior_ip: float = 60.0
    league_fip: float = 4.20
    fip_constant: float = 3.10
    reliever_season_decay: float = 0.50
    reliever_active_days: int = 210
    bullpen_size: int = 8
    workload_pitch_limit: float = 70.0
    workload_fip_penalty: float = 0.50

    def __post_init__(self) -> None:
        if self.batter_prior_pa <= 0.0:
            raise ValueError("batter_prior_pa must be positive")
        if self.batter_recency_half_life_days <= 0.0:
            raise ValueError("batter_recency_half_life_days must be positive")
        if self.bullpen_prior_ip <= 0.0:
            raise ValueError("bullpen_prior_ip must be positive")
        if self.bullpen_size <= 0:
            raise ValueError("bullpen_size must be positive")
        if self.workload_pitch_limit <= 0.0:
            raise ValueError("workload_pitch_limit must be positive")


DEFAULT_ROSTER_STRENGTH_CONFIG = RosterStrengthConfig()


@dataclass(frozen=True)
class BatterGameLine:
    player_id: int
    batting_order: int
    at_bats: int
    hits: int
    doubles: int
    triples: int
    home_runs: int
    walks: int
    intentional_walks: int
    hit_by_pitch: int
    sacrifice_flies: int
    birth_date: str | None = None

    @property
    def started(self) -> bool:
        return self.batting_order > 0 and self.batting_order % 100 == 0


@dataclass(frozen=True)
class RelieverGameLine:
    player_id: int
    outs: int
    strikeouts: int
    walks: int
    home_runs: int
    hit_batters: int
    pitches: int


@dataclass(frozen=True)
class RosterMatchupFeatures:
    lineup_woba_edge: float
    bullpen_fip_edge: float
    bullpen_availability_edge: float


@dataclass
class _BatterState:
    numerator: float = 0.0
    plate_appearances: float = 0.0
    birth_date: date | None = None
    age_adjustment: float = 0.0
    last_seen: date | None = None

    def decay(self, factor: float) -> None:
        self.numerator *= factor
        self.plate_appearances *= factor

    def update(
        self,
        line: BatterGameLine,
        game_date: date,
        config: RosterStrengthConfig,
    ) -> None:
        if self.last_seen is not None:
            days = max((game_date - self.last_seen).days, 0)
            factor = 0.5 ** (days / config.batter_recency_half_life_days)
            self.numerator *= factor
            self.plate_appearances *= factor
            if self.birth_date is not None:
                elapsed_years = days / 365.25
                age_at_last_game = (self.last_seen - self.birth_date).days / 365.25
                slope = (
                    config.batter_growth_per_year
                    if age_at_last_game < config.batter_peak_age
                    else -config.batter_decline_per_year
                )
                self.age_adjustment += slope * elapsed_years
        singles = max(
            line.hits - line.doubles - line.triples - line.home_runs,
            0,
        )
        unintentional_walks = max(line.walks - line.intentional_walks, 0)
        self.numerator += (
            0.69 * unintentional_walks
            + 0.72 * line.hit_by_pitch
            + 0.88 * singles
            + 1.247 * line.doubles
            + 1.578 * line.triples
            + 2.031 * line.home_runs
        )
        self.plate_appearances += (
            line.at_bats
            + unintentional_walks
            + line.hit_by_pitch
            + line.sacrifice_flies
        )
        if line.birth_date:
            self.birth_date = date.fromisoformat(line.birth_date)
        self.last_seen = game_date


@dataclass
class _RelieverState:
    outs: float = 0.0
    component_numerator: float = 0.0
    appearances: float = 0.0
    last_seen: date | None = None
    recent_pitches: deque[tuple[date, int]] = field(default_factory=deque)

    def decay(self, factor: float) -> None:
        self.outs *= factor
        self.component_numerator *= factor
        self.appearances *= factor

    def update(self, line: RelieverGameLine, game_date: date) -> None:
        self.outs += line.outs
        self.component_numerator += (
            13.0 * line.home_runs
            + 3.0 * (line.walks + line.hit_batters)
            - 2.0 * line.strikeouts
        )
        self.appearances += 1.0
        self.last_seen = game_date
        self.recent_pitches.append((game_date, line.pitches))
        self._prune_workload(game_date)

    def workload(self, as_of: date) -> float:
        self._prune_workload(as_of)
        total = 0.0
        for pitched_on, pitches in self.recent_pitches:
            days_ago = (as_of - pitched_on).days
            if days_ago <= 1:
                weight = 1.0
            elif days_ago == 2:
                weight = 0.5
            else:
                weight = 0.25
            total += pitches * weight
        return total

    def _prune_workload(self, as_of: date) -> None:
        while self.recent_pitches and (as_of - self.recent_pitches[0][0]).days > 3:
            self.recent_pitches.popleft()


class RosterFeatureBuilder:
    """Maintain prior-only player projections and current roster state."""

    def __init__(
        self,
        config: RosterStrengthConfig = DEFAULT_ROSTER_STRENGTH_CONFIG,
    ) -> None:
        self.config = config
        self._batters: defaultdict[int, _BatterState] = defaultdict(_BatterState)
        self._relievers: defaultdict[int, _RelieverState] = defaultdict(_RelieverState)
        self._last_lineups: dict[int, tuple[int, ...]] = {}
        self._team_relievers: defaultdict[int, dict[int, date]] = defaultdict(dict)
        self._season: int | None = None
        self._reliever_teams: dict[int, int] = {}
        self._starter_last_seen: dict[int, date] = {}

    def advance_to_season(self, season: int) -> None:
        if self._season is None:
            self._season = season
            return
        if season < self._season:
            raise ValueError(
                f"Cannot move roster state backward from {self._season} to {season}"
            )
        while self._season < season:
            for state in self._batters.values():
                state.decay(self.config.batter_season_decay)
            for state in self._relievers.values():
                state.decay(self.config.reliever_season_decay)
            self._last_lineups.clear()
            self._season += 1

    def matchup_features(
        self,
        *,
        season: int,
        prediction_date: date,
        away_team_id: int,
        home_team_id: int,
        away_starter_id: int,
        home_starter_id: int,
        away_batter_ids: tuple[int, ...] | None = None,
        home_batter_ids: tuple[int, ...] | None = None,
        away_active_batter_ids: tuple[int, ...] | None = None,
        home_active_batter_ids: tuple[int, ...] | None = None,
        away_reliever_ids: tuple[int, ...] | None = None,
        home_reliever_ids: tuple[int, ...] | None = None,
    ) -> RosterMatchupFeatures:
        self.advance_to_season(season)
        away_lineup = self._resolve_lineup(
            away_team_id,
            prediction_date,
            away_batter_ids,
            away_active_batter_ids,
        )
        home_lineup = self._resolve_lineup(
            home_team_id,
            prediction_date,
            home_batter_ids,
            home_active_batter_ids,
        )
        away_bullpen = self._bullpen_projection(
            away_team_id,
            prediction_date,
            away_reliever_ids,
            away_starter_id,
        )
        home_bullpen = self._bullpen_projection(
            home_team_id,
            prediction_date,
            home_reliever_ids,
            home_starter_id,
        )
        return RosterMatchupFeatures(
            lineup_woba_edge=(home_lineup - away_lineup) / WOBA_EDGE_UNIT,
            bullpen_fip_edge=away_bullpen[0] - home_bullpen[0],
            bullpen_availability_edge=(away_bullpen[1] - away_bullpen[0])
            - (home_bullpen[1] - home_bullpen[0]),
        )

    def observe(
        self,
        *,
        season: int,
        game_date: date,
        away_team_id: int,
        home_team_id: int,
        away_starter_id: int,
        home_starter_id: int,
        away_batters: tuple[BatterGameLine, ...],
        home_batters: tuple[BatterGameLine, ...],
        away_relievers: tuple[RelieverGameLine, ...],
        home_relievers: tuple[RelieverGameLine, ...],
    ) -> RosterMatchupFeatures:
        features = self.matchup_features(
            season=season,
            prediction_date=game_date,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            away_starter_id=away_starter_id,
            home_starter_id=home_starter_id,
        )
        self.update(
            season=season,
            game_date=game_date,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            away_batters=away_batters,
            home_batters=home_batters,
            away_relievers=away_relievers,
            home_relievers=home_relievers,
            away_starter_id=away_starter_id,
            home_starter_id=home_starter_id,
        )
        return features

    def update(
        self,
        *,
        season: int,
        game_date: date,
        away_team_id: int,
        home_team_id: int,
        away_batters: tuple[BatterGameLine, ...],
        home_batters: tuple[BatterGameLine, ...],
        away_relievers: tuple[RelieverGameLine, ...],
        home_relievers: tuple[RelieverGameLine, ...],
        away_starter_id: int | None = None,
        home_starter_id: int | None = None,
    ) -> None:
        self.advance_to_season(season)
        self._observe_side(away_team_id, game_date, away_batters, away_relievers)
        self._observe_side(home_team_id, game_date, home_batters, home_relievers)
        if away_starter_id is not None:
            self._starter_last_seen[away_starter_id] = game_date
        if home_starter_id is not None:
            self._starter_last_seen[home_starter_id] = game_date

    def _observe_side(
        self,
        team_id: int,
        game_date: date,
        batters: tuple[BatterGameLine, ...],
        relievers: tuple[RelieverGameLine, ...],
    ) -> None:
        starters = tuple(line.player_id for line in batters if line.started)
        if len(starters) >= 8:
            self._last_lineups[team_id] = starters[:9]
        for line in batters:
            self._batters[line.player_id].update(
                line,
                game_date,
                self.config,
            )
        for line in relievers:
            previous_team = self._reliever_teams.get(line.player_id)
            if previous_team is not None and previous_team != team_id:
                self._team_relievers[previous_team].pop(line.player_id, None)
            self._reliever_teams[line.player_id] = team_id
            self._relievers[line.player_id].update(line, game_date)
            self._team_relievers[team_id][line.player_id] = game_date

    def _resolve_lineup(
        self,
        team_id: int,
        as_of: date,
        requested: tuple[int, ...] | None,
        active: tuple[int, ...] | None,
    ) -> float:
        player_ids = list(
            dict.fromkeys(requested or self._last_lineups.get(team_id, ()))
        )
        active_ids = tuple(dict.fromkeys(active or ()))
        if active_ids:
            active_set = set(active_ids)
            player_ids = [
                player_id for player_id in player_ids if player_id in active_set
            ]
            remaining = sorted(
                (player_id for player_id in active_ids if player_id not in player_ids),
                key=lambda player_id: self._projected_woba(player_id, as_of),
                reverse=True,
            )
            player_ids.extend(remaining[: max(9 - len(player_ids), 0)])
        player_ids = player_ids[:9]
        if not player_ids:
            return self.config.league_woba
        return sum(
            self._projected_woba(player_id, as_of) for player_id in player_ids
        ) / len(player_ids)

    def _projected_woba(self, player_id: int, as_of: date) -> float:
        state = self._batters[player_id]
        config = self.config
        days_since_seen = (
            max((as_of - state.last_seen).days, 0) if state.last_seen is not None else 0
        )
        recency = 0.5 ** (days_since_seen / config.batter_recency_half_life_days)
        plate_appearances = state.plate_appearances * recency
        denominator = config.batter_prior_pa + plate_appearances
        projection = (
            config.league_woba * config.batter_prior_pa + state.numerator * recency
        ) / denominator
        projection += state.age_adjustment
        if state.birth_date is not None and state.last_seen is not None:
            elapsed_years = days_since_seen / 365.25
            age_at_last_game = (state.last_seen - state.birth_date).days / 365.25
            slope = (
                config.batter_growth_per_year
                if age_at_last_game < config.batter_peak_age
                else -config.batter_decline_per_year
            )
            projection += slope * elapsed_years
        return min(max(projection, 0.240), 0.430)

    def _has_current_reliever_role(self, player_id: int) -> bool:
        last_start = self._starter_last_seen.get(player_id)
        state = self._relievers.get(player_id)
        last_relief = state.last_seen if state is not None else None
        return last_start is None or (
            last_relief is not None and last_relief > last_start
        )

    def _bullpen_projection(
        self,
        team_id: int,
        as_of: date,
        requested: tuple[int, ...] | None,
        starter_id: int,
    ) -> tuple[float, float]:
        candidates = [
            player_id
            for player_id in dict.fromkeys(requested or ())
            if player_id != starter_id and self._has_current_reliever_role(player_id)
        ]
        if not candidates:
            candidates = [
                player_id
                for player_id, last_seen in self._team_relievers[team_id].items()
                if player_id != starter_id
                and 0 <= (as_of - last_seen).days <= self.config.reliever_active_days
                and self._has_current_reliever_role(player_id)
            ]
        candidates.sort(
            key=lambda player_id: (
                self._relievers[player_id].last_seen or date.min,
                self._relievers[player_id].appearances,
            ),
            reverse=True,
        )
        candidates = candidates[: self.config.bullpen_size]
        if not candidates:
            return self.config.league_fip, self.config.league_fip

        quality_weights: list[float] = []
        fips: list[float] = []
        availability: list[float] = []
        for player_id in candidates:
            state = self._relievers[player_id]
            quality_weights.append(1.0 + math.log1p(state.appearances))
            fips.append(self._projected_fip(state))
            availability.append(
                max(
                    0.05,
                    1.0 - state.workload(as_of) / self.config.workload_pitch_limit,
                )
            )
        baseline = _weighted_average(fips, quality_weights)
        available_weights = [
            weight * available
            for weight, available in zip(quality_weights, availability, strict=True)
        ]
        adjusted_quality = _weighted_average(fips, available_weights)
        unavailable_share = 1.0 - _weighted_average(availability, quality_weights)
        effective = (
            adjusted_quality + self.config.workload_fip_penalty * unavailable_share
        )
        return baseline, effective

    def _projected_fip(self, state: _RelieverState) -> float:
        config = self.config
        innings = state.outs / 3.0
        prior_numerator = config.bullpen_prior_ip * (
            config.league_fip - config.fip_constant
        )
        return config.fip_constant + (prior_numerator + state.component_numerator) / (
            config.bullpen_prior_ip + innings
        )


def _weighted_average(values: list[float], weights: list[float]) -> float:
    denominator = sum(weights)
    if denominator <= 0.0:
        raise ValueError("Projection weights must have positive mass")
    return (
        sum(value * weight for value, weight in zip(values, weights, strict=True))
        / denominator
    )
