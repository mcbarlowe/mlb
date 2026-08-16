"""Preseason team prior offsets from prior regular-season results.

The priors are odds-free and use only completed seasons before the projection
season. They are expressed on the game logit scale so callers can add
``home_offset - away_offset`` to a base home-win logit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "DEFAULT_PRIOR_WEIGHTS",
    "TeamSeasonResult",
    "load_team_prior_offsets",
    "team_prior_offsets_from_results",
]

DEFAULT_PRIOR_WEIGHTS = (0.55, 0.30, 0.15)
_PYTHAGOREAN_EXPONENT = 1.83


@dataclass(frozen=True)
class TeamSeasonResult:
    season: int
    team_id: int
    games: int
    wins: int
    runs_for: int
    runs_against: int

    @property
    def win_pct(self) -> float:
        if self.games <= 0:
            raise ValueError("games must be positive")
        return self.wins / self.games

    @property
    def pythagorean_win_pct(self) -> float:
        if self.runs_for < 0 or self.runs_against < 0:
            raise ValueError("runs must be non-negative")
        if self.runs_for == 0 and self.runs_against == 0:
            return self.win_pct
        runs_for = float(self.runs_for) ** _PYTHAGOREAN_EXPONENT
        runs_against = float(self.runs_against) ** _PYTHAGOREAN_EXPONENT
        return runs_for / (runs_for + runs_against)


def load_team_prior_offsets(
    prediction_season: int,
    *,
    lookback: int = 3,
    weights: Sequence[float] = DEFAULT_PRIOR_WEIGHTS,
    win_pct_weight: float = 0.35,
    db_config: PostgresConfig | None = None,
) -> dict[int, float]:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    start_season = int(prediction_season) - lookback
    end_season = int(prediction_season) - 1
    query = f"""
        WITH scores AS (
            SELECT
                game_pk,
                SUM(runs) FILTER (WHERE team_type = 'away')::int AS away_runs,
                SUM(runs) FILTER (WHERE team_type = 'home')::int AS home_runs
            FROM linescore
            GROUP BY game_pk
        ),
        team_games AS (
            SELECT
                g.season::int AS season,
                g.away_team_id AS team_id,
                CASE WHEN s.away_runs > s.home_runs THEN 1 ELSE 0 END AS wins,
                s.away_runs AS runs_for,
                s.home_runs AS runs_against
            FROM games AS g
            JOIN scores AS s USING (game_pk)
            WHERE g.game_type = 'R'
              AND g.abstract_game_state = 'Final'
              AND g.season::int BETWEEN {start_season} AND {end_season}
              AND s.away_runs <> s.home_runs
            UNION ALL
            SELECT
                g.season::int AS season,
                g.home_team_id AS team_id,
                CASE WHEN s.home_runs > s.away_runs THEN 1 ELSE 0 END AS wins,
                s.home_runs AS runs_for,
                s.away_runs AS runs_against
            FROM games AS g
            JOIN scores AS s USING (game_pk)
            WHERE g.game_type = 'R'
              AND g.abstract_game_state = 'Final'
              AND g.season::int BETWEEN {start_season} AND {end_season}
              AND s.away_runs <> s.home_runs
        )
        SELECT
            season,
            team_id,
            COUNT(*)::int AS games,
            SUM(wins)::int AS wins,
            SUM(runs_for)::int AS runs_for,
            SUM(runs_against)::int AS runs_against
        FROM team_games
        GROUP BY season, team_id
        ORDER BY season, team_id
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)

    results = [
        TeamSeasonResult(
            season=int(row.season),
            team_id=int(row.team_id),
            games=int(row.games),
            wins=int(row.wins),
            runs_for=int(row.runs_for),
            runs_against=int(row.runs_against),
        )
        for row in frame.itertuples(index=False)
    ]
    return team_prior_offsets_from_results(
        results,
        prediction_season=prediction_season,
        lookback=lookback,
        weights=weights,
        win_pct_weight=win_pct_weight,
    )


def team_prior_offsets_from_results(
    results: Iterable[TeamSeasonResult],
    *,
    prediction_season: int,
    lookback: int = 3,
    weights: Sequence[float] = DEFAULT_PRIOR_WEIGHTS,
    win_pct_weight: float = 0.35,
) -> dict[int, float]:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if not 0.0 <= win_pct_weight <= 1.0:
        raise ValueError("win_pct_weight must be between zero and one")
    resolved_weights = _weights_for_lookback(lookback, weights)
    by_season_team: dict[tuple[int, int], TeamSeasonResult] = {
        (result.season, result.team_id): result for result in results
    }
    team_ids = {
        team_id
        for season, team_id in by_season_team
        if prediction_season - lookback <= season < prediction_season
    }

    offsets: dict[int, float] = {}
    for team_id in team_ids:
        weighted_win_pct = 0.0
        weighted_pythagorean = 0.0
        total_weight = 0.0
        for seasons_ago, weight in enumerate(resolved_weights, start=1):
            result = by_season_team.get((prediction_season - seasons_ago, team_id))
            if result is None:
                continue
            weighted_win_pct += weight * result.win_pct
            weighted_pythagorean += weight * result.pythagorean_win_pct
            total_weight += weight
        if total_weight <= 0.0:
            continue
        weighted_win_pct /= total_weight
        weighted_pythagorean /= total_weight
        blended_true_talent = (
            win_pct_weight * weighted_win_pct
            + (1.0 - win_pct_weight) * weighted_pythagorean
        )
        offsets[team_id] = _probability_to_logit(
            _shrink_probability(blended_true_talent)
        )
    return offsets


def _weights_for_lookback(lookback: int, weights: Sequence[float]) -> tuple[float, ...]:
    if not weights:
        raise ValueError("weights must not be empty")
    if any(weight <= 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be positive finite values")
    if len(weights) >= lookback:
        return tuple(float(weight) for weight in weights[:lookback])
    tail_weight = float(weights[-1])
    return tuple(float(weight) for weight in weights) + (tail_weight,) * (
        lookback - len(weights)
    )


def _shrink_probability(probability: float, shrinkage: float = 0.65) -> float:
    if not math.isfinite(probability):
        raise ValueError("probability must be finite")
    return 0.5 + shrinkage * (min(max(probability, 0.0), 1.0) - 0.5)


def _probability_to_logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))
