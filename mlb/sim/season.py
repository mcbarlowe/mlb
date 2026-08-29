"""Season-level standings simulation from game win probabilities.

This module intentionally has no odds-market dependency. It projects division and
playoff probabilities from the existing team-strength home-win model plus the MLB
regular-season schedule.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from mlb.database import PostgresConfig, PostgresHandler

__all__ = [
    "HomeWinPredictor",
    "SeasonEvaluation",
    "SeasonProjection",
    "SeasonScheduleGame",
    "TeamInfo",
    "TeamProjection",
    "TeamSeasonOutcome",
    "actual_outcomes",
    "build_baseline_projection",
    "evaluate_projection",
    "first_regular_season_date",
    "load_season_schedule",
    "load_team_info",
    "schedule_strength_offsets_from_games",
    "simulate_season",
]


class HomeWinPredictor(Protocol):
    def predict_home_probability(
        self,
        *,
        season: int,
        away_team_id: int,
        home_team_id: int,
        away_starter_id: int,
        home_starter_id: int,
        prediction_date: date | None = None,
        away_batter_ids: Sequence[int] | None = None,
        home_batter_ids: Sequence[int] | None = None,
        away_active_batter_ids: Sequence[int] | None = None,
        home_active_batter_ids: Sequence[int] | None = None,
        away_reliever_ids: Sequence[int] | None = None,
        home_reliever_ids: Sequence[int] | None = None,
    ) -> float: ...


@dataclass(frozen=True)
class TeamInfo:
    team_id: int
    abbreviation: str
    team_name: str
    league_name: str
    division_name: str


@dataclass(frozen=True)
class SeasonScheduleGame:
    game_pk: int
    season: int
    game_date: date
    game_datetime: str | None
    status: str
    away_team_id: int
    home_team_id: int
    away_probable_pitcher_id: int | None = None
    home_probable_pitcher_id: int | None = None
    away_runs: int | None = None
    home_runs: int | None = None

    @property
    def is_final(self) -> bool:
        return (
            self.status == "Final"
            and self.away_runs is not None
            and self.home_runs is not None
        )

    @property
    def winning_team_id(self) -> int | None:
        if not self.is_final or self.away_runs == self.home_runs:
            return None
        return (
            self.home_team_id if self.home_runs > self.away_runs else self.away_team_id
        )


@dataclass(frozen=True)
class TeamProjection:
    team_id: int
    actual_wins_as_of: int
    expected_wins: float
    division_win_prob: float
    playoff_prob: float
    division_series_prob: float = 0.0
    league_championship_prob: float = 0.0
    world_series_prob: float = 0.0
    championship_prob: float = 0.0
    team_prior_offset: float = 0.0
    team_prior_weight: float = 1.0
    market_prior_offset: float = 0.0
    market_prior_weight: float = 1.0
    roster_prior_offset: float = 0.0
    roster_prior_weight: float = 1.0
    combined_prior_offset: float = 0.0


@dataclass(frozen=True)
class SeasonProjection:
    season: int
    as_of_date: date
    trials: int
    wild_cards_per_league: int
    teams: tuple[TeamProjection, ...]
    probability_logit_scale: float = 1.0
    team_strength_sd: float = 0.0
    team_prior_scale: float = 0.0
    market_prior_scale: float = 0.0
    schedule_strength_scale: float = 0.0
    playoff_calibration_slope: float | None = None
    division_calibration_slope: float | None = None
    division_series_calibration_slope: float | None = None
    league_championship_calibration_slope: float | None = None
    world_series_calibration_slope: float | None = None
    championship_calibration_slope: float | None = None
    team_prior_decay_games: float | None = None
    market_prior_decay_games: float | None = None
    roster_prior_scale: float = 0.0
    roster_prior_decay_games: float | None = None
    input_market_sources: str = ""

    def by_team_id(self) -> dict[int, TeamProjection]:
        return {team.team_id: team for team in self.teams}


@dataclass(frozen=True)
class TeamSeasonOutcome:
    team_id: int
    wins: int
    division_winner: bool
    playoff_team: bool


@dataclass(frozen=True)
class PostseasonStages:
    division_series_teams: frozenset[int]
    league_championship_teams: frozenset[int]
    world_series_teams: frozenset[int]
    champion: int | None


POSTSEASON_WIN_GAP_LOGIT = 0.04


@dataclass(frozen=True)
class SeasonEvaluation:
    season: int
    teams: int
    actual_wins_mae: float
    actual_wins_rmse: float
    division_brier: float
    division_log_loss: float
    playoff_brier: float
    playoff_log_loss: float


def load_team_info(db_config: PostgresConfig | None = None) -> dict[int, TeamInfo]:
    query = """
        SELECT DISTINCT ON (team_id)
            team_id,
            abbreviation,
            team_name,
            league_name,
            division_name
        FROM teams
        WHERE team_id IS NOT NULL
        ORDER BY team_id, active DESC NULLS LAST
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)

    teams: dict[int, TeamInfo] = {}
    for row in frame.itertuples(index=False):
        team_id = _required_int(row.team_id, "team_id")
        teams[team_id] = TeamInfo(
            team_id=team_id,
            abbreviation=_text(row.abbreviation, str(team_id)),
            team_name=_text(row.team_name, str(team_id)),
            league_name=_text(row.league_name, "Unknown League"),
            division_name=_text(row.division_name, "Unknown Division"),
        )
    return teams


def load_season_schedule(
    season: int,
    *,
    db_config: PostgresConfig | None = None,
) -> list[SeasonScheduleGame]:
    season = int(season)
    query = f"""
        WITH scores AS (
            SELECT
                game_pk,
                SUM(runs) FILTER (WHERE team_type = 'away')::int AS away_runs,
                SUM(runs) FILTER (WHERE team_type = 'home')::int AS home_runs
            FROM linescore
            GROUP BY game_pk
        )
        SELECT
            g.game_pk,
            g.season::int AS season,
            g.game_date,
            g.game_datetime,
            g.abstract_game_state AS status,
            g.away_team_id,
            g.home_team_id,
            g.away_probable_pitcher_id,
            g.home_probable_pitcher_id,
            s.away_runs,
            s.home_runs
        FROM games AS g
        LEFT JOIN scores AS s USING (game_pk)
        WHERE g.game_type = 'R'
          AND g.season::int = {season}
        ORDER BY COALESCE(g.game_datetime, g.game_date::timestamptz), g.game_pk
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)

    schedule: list[SeasonScheduleGame] = []
    for row in frame.itertuples(index=False):
        schedule.append(
            SeasonScheduleGame(
                game_pk=_required_int(row.game_pk, "game_pk"),
                season=_required_int(row.season, "season"),
                game_date=_coerce_date(row.game_date),
                game_datetime=_optional_text(row.game_datetime),
                status=_text(row.status, ""),
                away_team_id=_required_int(row.away_team_id, "away_team_id"),
                home_team_id=_required_int(row.home_team_id, "home_team_id"),
                away_probable_pitcher_id=_optional_int(row.away_probable_pitcher_id),
                home_probable_pitcher_id=_optional_int(row.home_probable_pitcher_id),
                away_runs=_optional_int(row.away_runs),
                home_runs=_optional_int(row.home_runs),
            )
        )
    return schedule


def first_regular_season_date(games: Sequence[SeasonScheduleGame]) -> date:
    if not games:
        raise ValueError(
            "Cannot resolve first regular-season date from an empty schedule"
        )
    return min(game.game_date for game in games)


def schedule_strength_offsets_from_games(
    games: Sequence[SeasonScheduleGame],
    *,
    as_of_date: date,
    team_prior_offsets: Mapping[int, float],
) -> dict[int, float]:
    """Return centered remaining-schedule offsets; positive means easier path."""
    if not games:
        raise ValueError("games must not be empty")

    team_ids = _schedule_team_ids(games)
    opponent_strength_total = {team_id: 0.0 for team_id in team_ids}
    remaining_games = {team_id: 0 for team_id in team_ids}
    for game in games:
        if game.game_date < as_of_date:
            continue
        away_strength = _finite_float(
            team_prior_offsets.get(game.away_team_id, 0.0),
            f"team_prior_offsets[{game.away_team_id}]",
        )
        home_strength = _finite_float(
            team_prior_offsets.get(game.home_team_id, 0.0),
            f"team_prior_offsets[{game.home_team_id}]",
        )
        opponent_strength_total[game.away_team_id] += home_strength
        opponent_strength_total[game.home_team_id] += away_strength
        remaining_games[game.away_team_id] += 1
        remaining_games[game.home_team_id] += 1

    active_team_ids = tuple(
        team_id for team_id in team_ids if remaining_games[team_id] > 0
    )
    if not active_team_ids:
        return {team_id: 0.0 for team_id in team_ids}

    average_opponent_strength = {
        team_id: opponent_strength_total[team_id] / remaining_games[team_id]
        for team_id in active_team_ids
    }
    league_average = sum(average_opponent_strength.values()) / len(
        average_opponent_strength
    )
    return {
        team_id: (
            league_average - average_opponent_strength[team_id]
            if team_id in average_opponent_strength
            else 0.0
        )
        for team_id in team_ids
    }

def _games_played_from_finals(
    games: Sequence[SeasonScheduleGame],
    *,
    team_ids: Iterable[int],
    before_date: date,
) -> dict[int, int]:
    played = {team_id: 0 for team_id in team_ids}
    for game in games:
        if game.game_date >= before_date or not game.is_final:
            continue
        played[game.away_team_id] += 1
        played[game.home_team_id] += 1
    return played


def _validate_decay_games(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _decayed_prior_weights(
    *,
    team_ids: Iterable[int],
    games_played: Mapping[int, int],
    decay_games: float | None,
) -> dict[int, float]:
    if decay_games is None or decay_games == 0.0:
        return {team_id: 1.0 for team_id in team_ids}
    return {
        team_id: decay_games / (decay_games + games_played.get(team_id, 0))
        for team_id in team_ids
    }


def _team_ids_for(
    mapping: Mapping[int, Sequence[int]] | None,
    team_id: int,
) -> tuple[int, ...] | None:
    if mapping is None:
        return None
    return tuple(mapping.get(team_id, ()))


def _prediction_kwargs(
    game: SeasonScheduleGame,
    *,
    lineup_batter_ids_by_team: Mapping[int, Sequence[int]] | None,
    active_batter_ids_by_team: Mapping[int, Sequence[int]] | None,
    reliever_ids_by_team: Mapping[int, Sequence[int]] | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "season": game.season,
        "away_team_id": game.away_team_id,
        "home_team_id": game.home_team_id,
        "away_starter_id": game.away_probable_pitcher_id or 0,
        "home_starter_id": game.home_probable_pitcher_id or 0,
        "prediction_date": game.game_date,
    }
    away_lineup = _team_ids_for(lineup_batter_ids_by_team, game.away_team_id)
    home_lineup = _team_ids_for(lineup_batter_ids_by_team, game.home_team_id)
    away_active = _team_ids_for(active_batter_ids_by_team, game.away_team_id)
    home_active = _team_ids_for(active_batter_ids_by_team, game.home_team_id)
    away_relievers = _team_ids_for(reliever_ids_by_team, game.away_team_id)
    home_relievers = _team_ids_for(reliever_ids_by_team, game.home_team_id)
    if away_lineup is not None:
        kwargs["away_batter_ids"] = away_lineup
    if home_lineup is not None:
        kwargs["home_batter_ids"] = home_lineup
    if away_active is not None:
        kwargs["away_active_batter_ids"] = away_active
    if home_active is not None:
        kwargs["home_active_batter_ids"] = home_active
    if away_relievers is not None:
        kwargs["away_reliever_ids"] = away_relievers
    if home_relievers is not None:
        kwargs["home_reliever_ids"] = home_relievers
    return kwargs



def simulate_season(
    *,
    games: Sequence[SeasonScheduleGame],
    teams: Mapping[int, TeamInfo],
    as_of_date: date,
    trials: int,
    predictor: HomeWinPredictor,
    seed: int = 42,
    wild_cards_per_league: int = 3,
    probability_logit_scale: float = 1.0,
    team_strength_sd: float = 0.0,
    team_prior_offsets: Mapping[int, float] | None = None,
    team_prior_scale: float = 0.0,
    team_prior_decay_games: float | None = None,
    market_prior_offsets: Mapping[int, float] | None = None,
    market_prior_scale: float = 0.0,
    market_prior_decay_games: float | None = None,
    roster_prior_offsets: Mapping[int, float] | None = None,
    roster_prior_scale: float = 0.0,
    roster_prior_decay_games: float | None = None,
    schedule_strength_offsets: Mapping[int, float] | None = None,
    schedule_strength_scale: float = 0.0,
    lineup_batter_ids_by_team: Mapping[int, Sequence[int]] | None = None,
    active_batter_ids_by_team: Mapping[int, Sequence[int]] | None = None,
    reliever_ids_by_team: Mapping[int, Sequence[int]] | None = None,
    input_market_sources: str = "",
) -> SeasonProjection:
    if trials < 1:
        raise ValueError("trials must be positive")
    if wild_cards_per_league < 0:
        raise ValueError("wild_cards_per_league must be non-negative")
    if not math.isfinite(probability_logit_scale) or probability_logit_scale <= 0.0:
        raise ValueError("probability_logit_scale must be positive")
    if not math.isfinite(team_strength_sd) or team_strength_sd < 0.0:
        raise ValueError("team_strength_sd must be non-negative")
    if not math.isfinite(team_prior_scale) or team_prior_scale < 0.0:
        raise ValueError("team_prior_scale must be non-negative")
    if not math.isfinite(market_prior_scale) or market_prior_scale < 0.0:
        raise ValueError("market_prior_scale must be non-negative")
    if not math.isfinite(roster_prior_scale) or roster_prior_scale < 0.0:
        raise ValueError("roster_prior_scale must be non-negative")
    if not math.isfinite(schedule_strength_scale) or schedule_strength_scale < 0.0:
        raise ValueError("schedule_strength_scale must be non-negative")
    resolved_team_decay = _validate_decay_games(
        team_prior_decay_games,
        "team_prior_decay_games",
    )
    resolved_market_decay = _validate_decay_games(
        market_prior_decay_games,
        "market_prior_decay_games",
    )
    resolved_roster_decay = _validate_decay_games(
        roster_prior_decay_games,
        "roster_prior_decay_games",
    )
    if not games:
        raise ValueError("games must not be empty")

    season = games[0].season
    if any(game.season != season for game in games):
        raise ValueError("All games must be from one season")

    team_ids = _schedule_team_ids(games)
    _require_team_info(team_ids, teams)

    wins_as_of = _wins_from_finals(
        games,
        team_ids=team_ids,
        before_date=as_of_date,
        require_complete=False,
    )
    games_played_as_of = _games_played_from_finals(
        games,
        team_ids=team_ids,
        before_date=as_of_date,
    )
    future_games = tuple(game for game in games if game.game_date >= as_of_date)
    future_probabilities = tuple(
        _clamp_probability(
            predictor.predict_home_probability(
                **_prediction_kwargs(
                    game,
                    lineup_batter_ids_by_team=lineup_batter_ids_by_team,
                    active_batter_ids_by_team=active_batter_ids_by_team,
                    reliever_ids_by_team=reliever_ids_by_team,
                )
            )
        )
        for game in future_games
    )
    use_probability_adjustments = (
        probability_logit_scale != 1.0
        or team_strength_sd > 0.0
        or team_prior_scale > 0.0
        or market_prior_scale > 0.0
        or roster_prior_scale > 0.0
        or schedule_strength_scale > 0.0
    )
    future_logits = (
        tuple(
            _probability_to_logit(probability) * probability_logit_scale
            for probability in future_probabilities
        )
        if use_probability_adjustments
        else ()
    )
    prior_offsets = {
        team_id: _finite_float(
            (team_prior_offsets or {}).get(team_id, 0.0),
            f"team_prior_offsets[{team_id}]",
        )
        for team_id in team_ids
    }
    market_offsets = {
        team_id: _finite_float(
            (market_prior_offsets or {}).get(team_id, 0.0),
            f"market_prior_offsets[{team_id}]",
        )
        for team_id in team_ids
    }
    roster_offsets = {
        team_id: _finite_float(
            (roster_prior_offsets or {}).get(team_id, 0.0),
            f"roster_prior_offsets[{team_id}]",
        )
        for team_id in team_ids
    }
    schedule_offsets = {
        team_id: _finite_float(
            (schedule_strength_offsets or {}).get(team_id, 0.0),
            f"schedule_strength_offsets[{team_id}]",
        )
        for team_id in team_ids
    }
    team_prior_weights = _decayed_prior_weights(
        team_ids=team_ids,
        games_played=games_played_as_of,
        decay_games=resolved_team_decay,
    )
    market_prior_weights = _decayed_prior_weights(
        team_ids=team_ids,
        games_played=games_played_as_of,
        decay_games=resolved_market_decay,
    )
    roster_prior_weights = _decayed_prior_weights(
        team_ids=team_ids,
        games_played=games_played_as_of,
        decay_games=resolved_roster_decay,
    )
    team_prior_effects = {
        team_id: team_prior_scale * prior_offsets[team_id] * team_prior_weights[team_id]
        for team_id in team_ids
    }
    market_prior_effects = {
        team_id: market_prior_scale
        * market_offsets[team_id]
        * market_prior_weights[team_id]
        for team_id in team_ids
    }
    roster_prior_effects = {
        team_id: roster_prior_scale
        * roster_offsets[team_id]
        * roster_prior_weights[team_id]
        for team_id in team_ids
    }
    combined_prior_effects = {
        team_id: (
            team_prior_effects[team_id]
            + market_prior_effects[team_id]
            + roster_prior_effects[team_id]
        )
        for team_id in team_ids
    }

    rng = random.Random(seed)
    wins_total = defaultdict(float)
    division_wins = defaultdict(int)
    playoff_berths = defaultdict(int)
    division_series_berths = defaultdict(int)
    league_championship_berths = defaultdict(int)
    world_series_berths = defaultdict(int)
    championships = defaultdict(int)

    for _ in range(trials):
        wins = dict(wins_as_of)
        team_effects = (
            {team_id: rng.gauss(0.0, team_strength_sd) for team_id in team_ids}
            if team_strength_sd > 0.0
            else {}
        )
        for game_index, (game, base_home_win_probability) in enumerate(
            zip(
                future_games,
                future_probabilities,
                strict=True,
            )
        ):
            home_win_probability = (
                _logistic(
                    future_logits[game_index]
                    + combined_prior_effects.get(game.home_team_id, 0.0)
                    - combined_prior_effects.get(game.away_team_id, 0.0)
                    + schedule_strength_scale
                    * (
                        schedule_offsets.get(game.home_team_id, 0.0)
                        - schedule_offsets.get(game.away_team_id, 0.0)
                    )
                    + team_effects.get(game.home_team_id, 0.0)
                    - team_effects.get(game.away_team_id, 0.0)
                )
                if use_probability_adjustments
                else base_home_win_probability
            )
            winner = (
                game.home_team_id
                if rng.random() < home_win_probability
                else game.away_team_id
            )
            wins[winner] += 1

        division_winners, playoff_teams = _qualifiers_from_wins(
            wins,
            team_ids=team_ids,
            teams=teams,
            wild_cards_per_league=wild_cards_per_league,
            rng=rng,
        )
        for team_id, wins_count in wins.items():
            wins_total[team_id] += wins_count
        for team_id in division_winners:
            division_wins[team_id] += 1
        for team_id in playoff_teams:
            playoff_berths[team_id] += 1
        stages = _postseason_stages(
            wins,
            team_ids=team_ids,
            teams=teams,
            division_winners=division_winners,
            playoff_teams=playoff_teams,
            team_effects=team_effects,
            rng=rng,
        )
        for team_id in stages.division_series_teams:
            division_series_berths[team_id] += 1
        for team_id in stages.league_championship_teams:
            league_championship_berths[team_id] += 1
        for team_id in stages.world_series_teams:
            world_series_berths[team_id] += 1
        if stages.champion is not None:
            championships[stages.champion] += 1

    projections = tuple(
        TeamProjection(
            team_id=team_id,
            actual_wins_as_of=wins_as_of[team_id],
            expected_wins=wins_total[team_id] / trials,
            division_win_prob=division_wins[team_id] / trials,
            playoff_prob=playoff_berths[team_id] / trials,
            division_series_prob=division_series_berths[team_id] / trials,
            league_championship_prob=league_championship_berths[team_id] / trials,
            world_series_prob=world_series_berths[team_id] / trials,
            championship_prob=championships[team_id] / trials,
            team_prior_offset=prior_offsets[team_id],
            team_prior_weight=team_prior_weights[team_id],
            market_prior_offset=market_offsets[team_id],
            market_prior_weight=market_prior_weights[team_id],
            roster_prior_offset=roster_offsets[team_id],
            roster_prior_weight=roster_prior_weights[team_id],
            combined_prior_offset=combined_prior_effects[team_id],
        )
        for team_id in _sort_team_ids(team_ids, teams)
    )
    return SeasonProjection(
        season=season,
        as_of_date=as_of_date,
        trials=trials,
        wild_cards_per_league=wild_cards_per_league,
        teams=projections,
        probability_logit_scale=probability_logit_scale,
        team_strength_sd=team_strength_sd,
        team_prior_scale=team_prior_scale,
        market_prior_scale=market_prior_scale,
        schedule_strength_scale=schedule_strength_scale,
        team_prior_decay_games=resolved_team_decay,
        market_prior_decay_games=resolved_market_decay,
        roster_prior_scale=roster_prior_scale,
        roster_prior_decay_games=resolved_roster_decay,
        input_market_sources=input_market_sources,
    )


def build_baseline_projection(
    *,
    games: Sequence[SeasonScheduleGame],
    teams: Mapping[int, TeamInfo],
    as_of_date: date,
    wild_cards_per_league: int = 3,
) -> SeasonProjection:
    if wild_cards_per_league < 0:
        raise ValueError("wild_cards_per_league must be non-negative")
    if not games:
        raise ValueError("games must not be empty")

    season = games[0].season
    if any(game.season != season for game in games):
        raise ValueError("All games must be from one season")

    team_ids = _schedule_team_ids(games)
    _require_team_info(team_ids, teams)
    wins_as_of = _wins_from_finals(
        games,
        team_ids=team_ids,
        before_date=as_of_date,
        require_complete=False,
    )
    remaining_games = {team_id: 0 for team_id in team_ids}
    for game in games:
        if game.game_date >= as_of_date:
            remaining_games[game.away_team_id] += 1
            remaining_games[game.home_team_id] += 1

    division_probabilities: dict[int, float] = {}
    division_counts_by_league: defaultdict[str, int] = defaultdict(int)
    for (league_name, _division_name), division_team_ids in _division_groups(
        team_ids,
        teams,
    ).items():
        probability = 1.0 / len(division_team_ids)
        division_counts_by_league[league_name] += 1
        for team_id in division_team_ids:
            division_probabilities[team_id] = probability

    playoff_probabilities: dict[int, float] = {}
    division_series_probabilities: dict[int, float] = {}
    league_championship_probabilities: dict[int, float] = {}
    world_series_probabilities: dict[int, float] = {}
    championship_probabilities: dict[int, float] = {}
    league_groups = _league_groups(team_ids, teams)
    championship_probability = 1.0 / len(team_ids)
    for league_name, league_team_ids in league_groups.items():
        playoff_slots = division_counts_by_league[league_name] + wild_cards_per_league
        playoff_probability = min(playoff_slots / len(league_team_ids), 1.0)
        division_series_probability = min(4, playoff_slots) / len(league_team_ids)
        league_championship_probability = min(2, playoff_slots) / len(league_team_ids)
        world_series_probability = (
            1.0 / len(league_team_ids) if len(league_groups) > 1 else 0.0
        )
        for team_id in league_team_ids:
            playoff_probabilities[team_id] = playoff_probability
            division_series_probabilities[team_id] = division_series_probability
            league_championship_probabilities[team_id] = league_championship_probability
            world_series_probabilities[team_id] = world_series_probability
            championship_probabilities[team_id] = championship_probability

    projections = tuple(
        TeamProjection(
            team_id=team_id,
            actual_wins_as_of=wins_as_of[team_id],
            expected_wins=wins_as_of[team_id] + 0.5 * remaining_games[team_id],
            division_win_prob=division_probabilities[team_id],
            playoff_prob=playoff_probabilities[team_id],
            division_series_prob=division_series_probabilities[team_id],
            league_championship_prob=league_championship_probabilities[team_id],
            world_series_prob=world_series_probabilities[team_id],
            championship_prob=championship_probabilities[team_id],
        )
        for team_id in _sort_team_ids(team_ids, teams)
    )
    return SeasonProjection(
        season=season,
        as_of_date=as_of_date,
        trials=0,
        wild_cards_per_league=wild_cards_per_league,
        teams=projections,
    )


def actual_outcomes(
    games: Sequence[SeasonScheduleGame],
    teams: Mapping[int, TeamInfo],
    *,
    wild_cards_per_league: int = 3,
) -> dict[int, TeamSeasonOutcome]:
    team_ids = _schedule_team_ids(games)
    _require_team_info(team_ids, teams)
    wins = _wins_from_finals(
        games,
        team_ids=team_ids,
        before_date=None,
        require_complete=True,
    )
    division_winners, playoff_teams = _qualifiers_from_wins(
        wins,
        team_ids=team_ids,
        teams=teams,
        wild_cards_per_league=wild_cards_per_league,
        rng=None,
    )
    return {
        team_id: TeamSeasonOutcome(
            team_id=team_id,
            wins=wins[team_id],
            division_winner=team_id in division_winners,
            playoff_team=team_id in playoff_teams,
        )
        for team_id in team_ids
    }


def evaluate_projection(
    projection: SeasonProjection,
    games: Sequence[SeasonScheduleGame],
    teams: Mapping[int, TeamInfo],
) -> SeasonEvaluation:
    projected = projection.by_team_id()
    actual = actual_outcomes(
        games,
        teams,
        wild_cards_per_league=projection.wild_cards_per_league,
    )
    missing = sorted(set(actual) - set(projected))
    if missing:
        raise ValueError(f"Projection missing teams: {missing}")

    win_errors: list[float] = []
    division_losses: list[float] = []
    division_log_losses: list[float] = []
    playoff_losses: list[float] = []
    playoff_log_losses: list[float] = []
    for team_id, outcome in actual.items():
        team_projection = projected[team_id]
        win_errors.append(team_projection.expected_wins - outcome.wins)
        division_target = 1.0 if outcome.division_winner else 0.0
        playoff_target = 1.0 if outcome.playoff_team else 0.0
        division_losses.append(
            (team_projection.division_win_prob - division_target) ** 2
        )
        playoff_losses.append((team_projection.playoff_prob - playoff_target) ** 2)
        division_log_losses.append(
            _binary_log_loss(team_projection.division_win_prob, division_target)
        )
        playoff_log_losses.append(
            _binary_log_loss(team_projection.playoff_prob, playoff_target)
        )

    return SeasonEvaluation(
        season=projection.season,
        teams=len(actual),
        actual_wins_mae=sum(abs(error) for error in win_errors) / len(win_errors),
        actual_wins_rmse=math.sqrt(
            sum(error * error for error in win_errors) / len(win_errors)
        ),
        division_brier=sum(division_losses) / len(division_losses),
        division_log_loss=sum(division_log_losses) / len(division_log_losses),
        playoff_brier=sum(playoff_losses) / len(playoff_losses),
        playoff_log_loss=sum(playoff_log_losses) / len(playoff_log_losses),
    )


def _wins_from_finals(
    games: Sequence[SeasonScheduleGame],
    *,
    team_ids: Iterable[int],
    before_date: date | None,
    require_complete: bool,
) -> dict[int, int]:
    wins = {team_id: 0 for team_id in team_ids}
    for game in games:
        if before_date is not None and game.game_date >= before_date:
            continue
        if not game.is_final:
            if require_complete:
                raise ValueError(
                    f"Cannot evaluate incomplete season; game {game.game_pk} is not final"
                )
            continue
        winner = game.winning_team_id
        if winner is not None:
            wins[winner] += 1
    return wins


def _qualifiers_from_wins(
    wins: Mapping[int, int],
    *,
    team_ids: Iterable[int],
    teams: Mapping[int, TeamInfo],
    wild_cards_per_league: int,
    rng: random.Random | None,
) -> tuple[set[int], set[int]]:
    division_winners: set[int] = set()
    for division_team_ids in _division_groups(team_ids, teams).values():
        division_winners.add(_pick_highest(wins, division_team_ids, rng))

    playoff_teams = set(division_winners)
    for league_team_ids in _league_groups(team_ids, teams).values():
        candidates = [
            team_id for team_id in league_team_ids if team_id not in division_winners
        ]
        for _ in range(min(wild_cards_per_league, len(candidates))):
            wild_card = _pick_highest(wins, candidates, rng)
            playoff_teams.add(wild_card)
            candidates.remove(wild_card)
    return division_winners, playoff_teams


def _postseason_stages(
    wins: Mapping[int, int],
    *,
    team_ids: Iterable[int],
    teams: Mapping[int, TeamInfo],
    division_winners: set[int],
    playoff_teams: set[int],
    team_effects: Mapping[int, float],
    rng: random.Random,
) -> PostseasonStages:
    division_series_teams: set[int] = set()
    league_championship_teams: set[int] = set()
    league_champions: list[int] = []

    for league_team_ids in _league_groups(team_ids, teams).values():
        seeds = _playoff_seeds(
            wins,
            league_team_ids=league_team_ids,
            division_winners=division_winners,
            playoff_teams=playoff_teams,
            rng=rng,
        )
        if not seeds:
            continue
        if len(seeds) == 6:
            league_champion = _mlb_six_team_league_champion(
                seeds,
                wins=wins,
                team_effects=team_effects,
                rng=rng,
                division_series_teams=division_series_teams,
                league_championship_teams=league_championship_teams,
            )
        else:
            league_champion = _generic_league_champion(
                seeds,
                wins=wins,
                team_effects=team_effects,
                rng=rng,
                division_series_teams=division_series_teams,
                league_championship_teams=league_championship_teams,
            )
        league_champions.append(league_champion)

    world_series_teams = set(league_champions) if len(league_champions) > 1 else set()
    champion = _champion_from_league_champions(
        league_champions,
        wins=wins,
        team_effects=team_effects,
        rng=rng,
    )
    return PostseasonStages(
        division_series_teams=frozenset(division_series_teams),
        league_championship_teams=frozenset(league_championship_teams),
        world_series_teams=frozenset(world_series_teams),
        champion=champion,
    )


def _playoff_seeds(
    wins: Mapping[int, int],
    *,
    league_team_ids: Sequence[int],
    division_winners: set[int],
    playoff_teams: set[int],
    rng: random.Random,
) -> list[int]:
    division_seeds = _rank_by_wins(
        wins,
        [team_id for team_id in league_team_ids if team_id in division_winners],
        rng,
    )
    wild_card_seeds = _rank_by_wins(
        wins,
        [
            team_id
            for team_id in league_team_ids
            if team_id in playoff_teams and team_id not in division_winners
        ],
        rng,
    )
    return division_seeds + wild_card_seeds


def _mlb_six_team_league_champion(
    seeds: Sequence[int],
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
    rng: random.Random,
    division_series_teams: set[int],
    league_championship_teams: set[int],
) -> int:
    three_six_winner = _play_series(
        seeds[2],
        seeds[5],
        wins=wins,
        team_effects=team_effects,
        rng=rng,
        games_to_win=2,
    )
    four_five_winner = _play_series(
        seeds[3],
        seeds[4],
        wins=wins,
        team_effects=team_effects,
        rng=rng,
        games_to_win=2,
    )
    division_series_teams.update(
        (seeds[0], seeds[1], three_six_winner, four_five_winner)
    )
    one_side_winner = _play_series(
        seeds[0],
        four_five_winner,
        wins=wins,
        team_effects=team_effects,
        rng=rng,
        games_to_win=3,
    )
    two_side_winner = _play_series(
        seeds[1],
        three_six_winner,
        wins=wins,
        team_effects=team_effects,
        rng=rng,
        games_to_win=3,
    )
    league_championship_teams.update((one_side_winner, two_side_winner))
    return _play_series(
        one_side_winner,
        two_side_winner,
        wins=wins,
        team_effects=team_effects,
        rng=rng,
        games_to_win=4,
    )


def _generic_league_champion(
    seeds: Sequence[int],
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
    rng: random.Random,
    division_series_teams: set[int],
    league_championship_teams: set[int],
) -> int:
    contenders = list(seeds)
    while len(contenders) > 1:
        if len(contenders) <= 4:
            division_series_teams.update(contenders)
        if len(contenders) <= 2:
            league_championship_teams.update(contenders)
        contenders = _play_seeded_round(
            contenders,
            wins=wins,
            team_effects=team_effects,
            rng=rng,
            games_to_win=4 if len(contenders) <= 2 else 3,
        )
    return contenders[0]


def _champion_from_league_champions(
    league_champions: Sequence[int],
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
    rng: random.Random,
) -> int | None:
    if not league_champions:
        return None
    contenders = list(league_champions)
    while len(contenders) > 1:
        contenders = _play_seeded_round(
            _rank_by_wins(wins, contenders, rng),
            wins=wins,
            team_effects=team_effects,
            rng=rng,
            games_to_win=4,
        )
    return contenders[0]


def _play_seeded_round(
    seeds: Sequence[int],
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
    rng: random.Random,
    games_to_win: int,
) -> list[int]:
    winners: list[int] = []
    low_seed_index = 0
    high_seed_index = len(seeds) - 1
    while low_seed_index <= high_seed_index:
        if low_seed_index == high_seed_index:
            winners.append(seeds[low_seed_index])
        else:
            winners.append(
                _play_series(
                    seeds[low_seed_index],
                    seeds[high_seed_index],
                    wins=wins,
                    team_effects=team_effects,
                    rng=rng,
                    games_to_win=games_to_win,
                )
            )
        low_seed_index += 1
        high_seed_index -= 1
    return _rank_by_wins(wins, winners, rng)


def _play_series(
    team_id: int,
    opponent_id: int,
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
    rng: random.Random,
    games_to_win: int,
) -> int:
    team_wins = 0
    opponent_wins = 0
    team_probability = _series_game_probability(
        team_id,
        opponent_id,
        wins=wins,
        team_effects=team_effects,
    )
    while team_wins < games_to_win and opponent_wins < games_to_win:
        if rng.random() < team_probability:
            team_wins += 1
        else:
            opponent_wins += 1
    return team_id if team_wins > opponent_wins else opponent_id


def _series_game_probability(
    team_id: int,
    opponent_id: int,
    *,
    wins: Mapping[int, int],
    team_effects: Mapping[int, float],
) -> float:
    rating_gap = (
        POSTSEASON_WIN_GAP_LOGIT * (wins[team_id] - wins[opponent_id])
        + team_effects.get(team_id, 0.0)
        - team_effects.get(opponent_id, 0.0)
    )
    return _logistic(rating_gap)


def _rank_by_wins(
    wins: Mapping[int, int],
    candidates: Sequence[int],
    rng: random.Random,
) -> list[int]:
    ranked: list[int] = []
    remaining = sorted(candidates)
    while remaining:
        high = max(wins[team_id] for team_id in remaining)
        tied = [team_id for team_id in remaining if wins[team_id] == high]
        rng.shuffle(tied)
        ranked.extend(tied)
        remaining = [team_id for team_id in remaining if wins[team_id] != high]
    return ranked


def _pick_highest(
    wins: Mapping[int, int],
    candidates: Sequence[int],
    rng: random.Random | None,
) -> int:
    if not candidates:
        raise ValueError("Cannot pick from an empty candidate set")
    high = max(wins[team_id] for team_id in candidates)
    tied = [team_id for team_id in candidates if wins[team_id] == high]
    if rng is None:
        return min(tied)
    return rng.choice(tied)


def _division_groups(
    team_ids: Iterable[int],
    teams: Mapping[int, TeamInfo],
) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for team_id in team_ids:
        team = teams[team_id]
        groups[(team.league_name, team.division_name)].append(team_id)
    return groups


def _league_groups(
    team_ids: Iterable[int],
    teams: Mapping[int, TeamInfo],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for team_id in team_ids:
        groups[teams[team_id].league_name].append(team_id)
    return groups


def _schedule_team_ids(games: Sequence[SeasonScheduleGame]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                team_id
                for game in games
                for team_id in (game.away_team_id, game.home_team_id)
            }
        )
    )


def _sort_team_ids(
    team_ids: Iterable[int], teams: Mapping[int, TeamInfo]
) -> tuple[int, ...]:
    return tuple(
        sorted(
            team_ids,
            key=lambda team_id: (
                teams[team_id].league_name,
                teams[team_id].division_name,
                teams[team_id].abbreviation,
                team_id,
            ),
        )
    )


def _require_team_info(team_ids: Iterable[int], teams: Mapping[int, TeamInfo]) -> None:
    missing = sorted(team_id for team_id in team_ids if team_id not in teams)
    if missing:
        raise ValueError(f"Missing team metadata for team ids: {missing}")


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Invalid win probability: {value!r}")
    return min(max(float(value), 0.0), 1.0)


def _finite_float(value: object, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _probability_to_logit(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def _logistic(logit: float) -> float:
    if logit >= 0.0:
        denominator = 1.0 + math.exp(-logit)
        return 1.0 / denominator
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _binary_log_loss(probability: float, target: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _required_int(value: object, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"Missing required integer: {label}")
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "<NA>", "NaT", "None"} or text.lower() == "nan":
        return None
    return int(value)


def _text(value: object, default: str) -> str:
    text = _optional_text(value)
    return text if text is not None else default


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in {"", "<NA>", "NaT", "None"} or text.lower() == "nan":
        return None
    return text
