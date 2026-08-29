"""Leak-free pregame team, starter, lineup, and bullpen strength model.

The pitch-by-pitch simulator is useful for score distributions, but its raw
home-win spread has not reliably beaten simple home-team baselines. This module
builds an independent pregame probability from information available before
each game:

- slowly updating team Elo,
- exponentially weighted team runs scored/allowed,
- Bayesian-shrunk starting-pitcher FIP and expected length,
- recency- and age-adjusted player lineup projections,
- individual bullpen quality and recent workload availability.

Every feature row is emitted before that game's result updates the state. Model
coefficients are fit only on seasons before the prediction season.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from mlb.database import PostgresConfig, PostgresHandler
from mlb.sim.roster_strength import (
    BatterGameLine,
    RelieverGameLine,
    RosterFeatureBuilder,
    RosterStrengthConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sklearn.linear_model import LogisticRegression

LEGACY_FEATURE_NAMES = (
    "elo_diff",
    "run_edge",
    "starter_era_edge",
    "starter_fip_edge",
    "starter_length_edge",
)
FEATURE_NAMES = (
    *LEGACY_FEATURE_NAMES,
    "lineup_woba_edge",
    "bullpen_fip_edge",
    "bullpen_availability_edge",
)

STRENGTH_MODEL_FAMILY = "team_strength_win"
LEGACY_STRENGTH_MODEL_CONTRACT_VERSION = 1
STRENGTH_MODEL_CONTRACT_VERSION = 2
DEFAULT_REGISTERED_STRENGTH_MODEL = "mlb-team-strength-win"
WIN_PROBABILITY_MODEL_TYPE = "win_probability_model"
WIN_PROBABILITY_MODEL_COLLECTION = "win_probability_models"


@dataclass(frozen=True)
class StrengthConfig:
    """State-update constants selected on seasons through 2024."""

    initial_elo: float = 1500.0
    elo_k: float = 4.0
    elo_home_advantage: float = 15.0
    elo_season_regression: float = 0.25
    initial_runs_per_game: float = 4.5
    run_alpha: float = 0.01
    run_season_regression: float = 0.25
    starter_prior_ip: float = 120.0
    starter_season_decay: float = 0.4
    roster: RosterStrengthConfig = field(default_factory=RosterStrengthConfig)


DEFAULT_STRENGTH_CONFIG = StrengthConfig()


@dataclass(frozen=True)
class StarterLine:
    player_id: int
    outs: int
    earned_runs: int
    strikeouts: int
    walks: int
    home_runs: int
    hit_batters: int


@dataclass(frozen=True)
class CompletedGame:
    game_pk: int
    season: int
    game_datetime: str
    away_team_id: int
    home_team_id: int
    away_runs: int
    home_runs: int
    away_starter: StarterLine
    home_starter: StarterLine
    away_batters: tuple[BatterGameLine, ...] = ()
    home_batters: tuple[BatterGameLine, ...] = ()
    away_relievers: tuple[RelieverGameLine, ...] = ()
    home_relievers: tuple[RelieverGameLine, ...] = ()

    @property
    def home_won(self) -> bool:
        return self.home_runs > self.away_runs


@dataclass(frozen=True)
class PregameFeatures:
    elo_diff: float
    run_edge: float
    starter_era_edge: float
    starter_fip_edge: float
    starter_length_edge: float
    lineup_woba_edge: float
    bullpen_fip_edge: float
    bullpen_availability_edge: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "elo_diff": self.elo_diff,
            "run_edge": self.run_edge,
            "starter_era_edge": self.starter_era_edge,
            "starter_fip_edge": self.starter_fip_edge,
            "starter_length_edge": self.starter_length_edge,
            "lineup_woba_edge": self.lineup_woba_edge,
            "bullpen_fip_edge": self.bullpen_fip_edge,
            "bullpen_availability_edge": self.bullpen_availability_edge,
        }

    def as_list(self, feature_names: tuple[str, ...] = FEATURE_NAMES) -> list[float]:
        values = self.as_mapping()
        return [values[name] for name in feature_names]


@dataclass
class _PitcherState:
    outs: float = 0.0
    earned_runs: float = 0.0
    strikeouts: float = 0.0
    walks: float = 0.0
    home_runs: float = 0.0
    hit_batters: float = 0.0
    starts: float = 0.0

    def decay(self, factor: float) -> None:
        self.outs *= factor
        self.earned_runs *= factor
        self.strikeouts *= factor
        self.walks *= factor
        self.home_runs *= factor
        self.hit_batters *= factor
        self.starts *= factor

    def update(self, line: StarterLine) -> None:
        self.outs += line.outs
        self.earned_runs += line.earned_runs
        self.strikeouts += line.strikeouts
        self.walks += line.walks
        self.home_runs += line.home_runs
        self.hit_batters += line.hit_batters
        self.starts += 1.0


class StrengthFeatureBuilder:
    """Maintain chronological state and emit pregame feature vectors."""

    def __init__(self, config: StrengthConfig = DEFAULT_STRENGTH_CONFIG) -> None:
        self.config = config
        self._elo: defaultdict[int, float] = defaultdict(
            lambda: self.config.initial_elo
        )
        self._offense: defaultdict[int, float] = defaultdict(
            lambda: self.config.initial_runs_per_game
        )
        self._defense: defaultdict[int, float] = defaultdict(
            lambda: self.config.initial_runs_per_game
        )
        self._pitchers: defaultdict[int, _PitcherState] = defaultdict(_PitcherState)
        self._season: int | None = None
        self._rosters = RosterFeatureBuilder(config.roster)
        self._latest_date: date | None = None

    def advance_to_season(self, season: int) -> None:
        self._rosters.advance_to_season(season)
        if self._season is None:
            self._season = season
            return
        if season < self._season:
            raise ValueError(
                f"Cannot move strength state backward from {self._season} to {season}"
            )
        while self._season < season:
            self._regress_for_new_season()
            self._season += 1

    def matchup_features(
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
    ) -> PregameFeatures:
        self.advance_to_season(season)
        as_of = prediction_date or self._latest_date or date(season, 4, 1)
        away_era, away_fip, away_length = self._starter_features(away_starter_id)
        home_era, home_fip, home_length = self._starter_features(home_starter_id)
        roster = self._rosters.matchup_features(
            season=season,
            prediction_date=as_of,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            away_starter_id=away_starter_id,
            home_starter_id=home_starter_id,
            away_batter_ids=tuple(away_batter_ids) if away_batter_ids else None,
            home_batter_ids=tuple(home_batter_ids) if home_batter_ids else None,
            away_active_batter_ids=(
                tuple(away_active_batter_ids) if away_active_batter_ids else None
            ),
            home_active_batter_ids=(
                tuple(home_active_batter_ids) if home_active_batter_ids else None
            ),
            away_reliever_ids=tuple(away_reliever_ids) if away_reliever_ids else None,
            home_reliever_ids=tuple(home_reliever_ids) if home_reliever_ids else None,
        )
        return PregameFeatures(
            elo_diff=(self._elo[home_team_id] - self._elo[away_team_id]) / 100.0,
            run_edge=(self._offense[home_team_id] - self._offense[away_team_id])
            + (self._defense[away_team_id] - self._defense[home_team_id]),
            starter_era_edge=away_era - home_era,
            starter_fip_edge=away_fip - home_fip,
            starter_length_edge=(home_length - away_length) / 3.0,
            lineup_woba_edge=roster.lineup_woba_edge,
            bullpen_fip_edge=roster.bullpen_fip_edge,
            bullpen_availability_edge=roster.bullpen_availability_edge,
        )

    def observe(self, game: CompletedGame) -> PregameFeatures:
        """Return pregame features, then update state with the final result."""
        game_date = date.fromisoformat(game.game_datetime[:10])
        features = self.matchup_features(
            season=game.season,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            away_starter_id=game.away_starter.player_id,
            home_starter_id=game.home_starter.player_id,
            prediction_date=game_date,
        )
        self._update_teams(game)
        self._pitchers[game.away_starter.player_id].update(game.away_starter)
        self._pitchers[game.home_starter.player_id].update(game.home_starter)
        self._rosters.update(
            season=game.season,
            game_date=game_date,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            away_batters=game.away_batters,
            home_batters=game.home_batters,
            away_relievers=game.away_relievers,
            home_relievers=game.home_relievers,
            away_starter_id=game.away_starter.player_id,
            home_starter_id=game.home_starter.player_id,
        )
        self._latest_date = game_date
        return features

    def _regress_for_new_season(self) -> None:
        c = self.config
        for team_id, rating in self._elo.items():
            self._elo[team_id] = c.initial_elo + (rating - c.initial_elo) * (
                1.0 - c.elo_season_regression
            )
        for state in (self._offense, self._defense):
            for team_id, value in state.items():
                state[team_id] = c.initial_runs_per_game + (
                    value - c.initial_runs_per_game
                ) * (1.0 - c.run_season_regression)
        for pitcher in self._pitchers.values():
            pitcher.decay(c.starter_season_decay)

    def _starter_features(self, player_id: int) -> tuple[float, float, float]:
        state = self._pitchers[player_id]
        prior_ip = self.config.starter_prior_ip
        innings = state.outs / 3.0
        denominator = innings + prior_ip
        era = 9.0 * (state.earned_runs + prior_ip * 0.50) / denominator
        fip = (
            13.0 * (state.home_runs + prior_ip * 0.12)
            + 3.0 * (state.walks + state.hit_batters + prior_ip * 0.405)
            - 2.0 * (state.strikeouts + prior_ip * 1.0)
        ) / denominator
        prior_starts = prior_ip / 25.0
        expected_outs = (state.outs + prior_starts * 15.0) / (
            state.starts + prior_starts
        )
        return era, fip, expected_outs

    def _update_teams(self, game: CompletedGame) -> None:
        c = self.config
        home_rating = self._elo[game.home_team_id]
        away_rating = self._elo[game.away_team_id]
        expected_home = 1.0 / (
            1.0 + 10.0 ** (-(home_rating + c.elo_home_advantage - away_rating) / 400.0)
        )
        delta = c.elo_k * (float(game.home_won) - expected_home)
        self._elo[game.home_team_id] += delta
        self._elo[game.away_team_id] -= delta

        alpha = c.run_alpha
        self._offense[game.home_team_id] = (1.0 - alpha) * self._offense[
            game.home_team_id
        ] + alpha * game.home_runs
        self._defense[game.home_team_id] = (1.0 - alpha) * self._defense[
            game.home_team_id
        ] + alpha * game.away_runs
        self._offense[game.away_team_id] = (1.0 - alpha) * self._offense[
            game.away_team_id
        ] + alpha * game.away_runs
        self._defense[game.away_team_id] = (1.0 - alpha) * self._defense[
            game.away_team_id
        ] + alpha * game.home_runs


@dataclass(frozen=True)
class StrengthModelSource:
    registered_model_name: str
    version: str
    run_id: str


@dataclass(frozen=True)
class TeamStrengthPredictor:
    coefficients: tuple[float, ...]
    intercept: float
    feature_builder: StrengthFeatureBuilder
    feature_names: tuple[str, ...] = FEATURE_NAMES
    source: StrengthModelSource | None = None

    def __post_init__(self) -> None:
        if self.feature_names == FEATURE_NAMES and len(self.coefficients) == len(
            LEGACY_FEATURE_NAMES
        ):
            object.__setattr__(self, "feature_names", LEGACY_FEATURE_NAMES)
        if len(self.coefficients) != len(self.feature_names):
            raise ValueError("Coefficient count does not match the feature contract")

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
    ) -> float:
        features = self.feature_builder.matchup_features(
            season=season,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            away_starter_id=away_starter_id,
            home_starter_id=home_starter_id,
            prediction_date=prediction_date,
            away_batter_ids=away_batter_ids,
            home_batter_ids=home_batter_ids,
            away_active_batter_ids=away_active_batter_ids,
            home_active_batter_ids=home_active_batter_ids,
            away_reliever_ids=away_reliever_ids,
            home_reliever_ids=home_reliever_ids,
        )
        log_odds = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(
                self.coefficients,
                features.as_list(self.feature_names),
                strict=True,
            )
        )
        return 1.0 / (1.0 + math.exp(-log_odds))


@dataclass(frozen=True)
class StrengthModelFit:
    """Fitted estimator plus the exact feature frame used to train it."""

    estimator: LogisticRegression
    predictor: TeamStrengthPredictor
    feature_frame: pd.DataFrame
    train_seasons: tuple[int, ...]
    config: StrengthConfig


@dataclass(frozen=True)
class _LoadedStrengthModel:
    estimator: LogisticRegression
    config: StrengthConfig
    feature_names: tuple[str, ...]
    start_season: int
    source: StrengthModelSource


def _db_int(value: object | None) -> int:
    """Normalize integer-valued PostgreSQL/Pandas scalars."""
    if value is None:
        return 0
    return int(str(value))


def _db_optional_date(value: object | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return None if text in {"", "NaT", "<NA>"} else text[:10]


def load_completed_games(
    *,
    start_season: int = 2015,
    end_season: int,
    include_rosters: bool = True,
    game_types: tuple[str, ...] = ("R",),
    db_config: PostgresConfig | None = None,
) -> list[CompletedGame]:
    """Load finals (regular season by default) with each starter's game line."""
    start_season = int(start_season)
    end_season = int(end_season)
    if end_season < start_season:
        raise ValueError("end_season must be at least start_season")
    if not game_types or any(len(t) != 1 or not t.isalpha() for t in game_types):
        raise ValueError("game_types must be single-letter MLB game type codes")
    game_type_list = ", ".join(f"'{t}'" for t in game_types)
    query = f"""
        WITH scores AS (
            SELECT
                game_pk,
                SUM(runs) FILTER (WHERE team_type = 'away')::int AS away_runs,
                SUM(runs) FILTER (WHERE team_type = 'home')::int AS home_runs
            FROM linescore
            GROUP BY game_pk
        ),
        starters AS (
            SELECT
                game_pk,
                MAX(player_id) FILTER (WHERE team_type = 'away')::int AS away_id,
                MAX(outs) FILTER (WHERE team_type = 'away')::int AS away_outs,
                MAX(earnedruns) FILTER (WHERE team_type = 'away')::int AS away_er,
                MAX(strikeouts) FILTER (WHERE team_type = 'away')::int AS away_k,
                MAX(baseonballs) FILTER (WHERE team_type = 'away')::int AS away_bb,
                MAX(homeruns) FILTER (WHERE team_type = 'away')::int AS away_hr,
                MAX(hitbypitch) FILTER (WHERE team_type = 'away')::int AS away_hbp,
                MAX(player_id) FILTER (WHERE team_type = 'home')::int AS home_id,
                MAX(outs) FILTER (WHERE team_type = 'home')::int AS home_outs,
                MAX(earnedruns) FILTER (WHERE team_type = 'home')::int AS home_er,
                MAX(strikeouts) FILTER (WHERE team_type = 'home')::int AS home_k,
                MAX(baseonballs) FILTER (WHERE team_type = 'home')::int AS home_bb,
                MAX(homeruns) FILTER (WHERE team_type = 'home')::int AS home_hr,
                MAX(hitbypitch) FILTER (WHERE team_type = 'home')::int AS home_hbp
            FROM pitching
            WHERE gamesstarted = 1
            GROUP BY game_pk
        )
        SELECT
            g.game_pk,
            g.season::int AS season,
            COALESCE(g.game_datetime, g.game_date) AS game_datetime,
            g.away_team_id,
            g.home_team_id,
            s.away_runs,
            s.home_runs,
            p.away_id,
            p.away_outs,
            p.away_er,
            p.away_k,
            p.away_bb,
            p.away_hr,
            p.away_hbp,
            p.home_id,
            p.home_outs,
            p.home_er,
            p.home_k,
            p.home_bb,
            p.home_hr,
            p.home_hbp
        FROM games AS g
        JOIN scores AS s USING (game_pk)
        JOIN starters AS p USING (game_pk)
        WHERE g.game_type IN ({game_type_list})
          AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN {start_season} AND {end_season}
          AND s.away_runs <> s.home_runs
          AND p.away_id IS NOT NULL
          AND p.home_id IS NOT NULL
        ORDER BY game_datetime, g.game_pk
    """
    batting_query = f"""
        SELECT
            b.game_pk,
            b.team_type,
            b.player_id,
            b.batting_order,
            COALESCE(b.atbats, 0)::int AS at_bats,
            COALESCE(b.hits, 0)::int AS hits,
            COALESCE(b.doubles, 0)::int AS doubles,
            COALESCE(b.triples, 0)::int AS triples,
            COALESCE(b.homeruns, 0)::int AS home_runs,
            COALESCE(b.baseonballs, 0)::int AS walks,
            COALESCE(b.intentionalwalks, 0)::int AS intentional_walks,
            COALESCE(b.hitbypitch, 0)::int AS hit_by_pitch,
            COALESCE(b.sacflies, 0)::int AS sacrifice_flies,
            players.birth_date
        FROM batting AS b
        JOIN games AS g USING (game_pk)
        LEFT JOIN players USING (player_id)
        WHERE g.game_type IN ({game_type_list})
          AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN {start_season} AND {end_season}
          AND b.player_id IS NOT NULL
          AND NULLIF(b.batting_order, '') IS NOT NULL
        ORDER BY b.game_pk, b.team_type, b.batting_order, b.player_id
    """
    reliever_query = f"""
        SELECT
            p.game_pk,
            p.team_type,
            p.player_id,
            COALESCE(p.outs, 0)::int AS outs,
            COALESCE(p.strikeouts, 0)::int AS strikeouts,
            COALESCE(p.baseonballs, 0)::int AS walks,
            COALESCE(p.homeruns, 0)::int AS home_runs,
            COALESCE(p.hitbypitch, 0)::int AS hit_batters,
            COALESCE(p.numberofpitches, 0)::int AS pitches
        FROM pitching AS p
        JOIN games AS g USING (game_pk)
        WHERE g.game_type IN ({game_type_list})
          AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN {start_season} AND {end_season}
          AND COALESCE(p.gamesstarted, 0) = 0
          AND p.player_id IS NOT NULL
        ORDER BY p.game_pk, p.team_type, p.player_id
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)
        batting_frame = db.query(batting_query) if include_rosters else pd.DataFrame()
        reliever_frame = db.query(reliever_query) if include_rosters else pd.DataFrame()

    batters: defaultdict[tuple[int, str], list[BatterGameLine]] = defaultdict(list)
    for row in batting_frame.itertuples(index=False):
        batters[(_db_int(row.game_pk), str(row.team_type))].append(
            BatterGameLine(
                player_id=_db_int(row.player_id),
                batting_order=_db_int(row.batting_order),
                at_bats=_db_int(row.at_bats),
                hits=_db_int(row.hits),
                doubles=_db_int(row.doubles),
                triples=_db_int(row.triples),
                home_runs=_db_int(row.home_runs),
                walks=_db_int(row.walks),
                intentional_walks=_db_int(row.intentional_walks),
                hit_by_pitch=_db_int(row.hit_by_pitch),
                sacrifice_flies=_db_int(row.sacrifice_flies),
                birth_date=_db_optional_date(row.birth_date),
            )
        )
    relievers: defaultdict[tuple[int, str], list[RelieverGameLine]] = defaultdict(list)
    for row in reliever_frame.itertuples(index=False):
        relievers[(_db_int(row.game_pk), str(row.team_type))].append(
            RelieverGameLine(
                player_id=_db_int(row.player_id),
                outs=_db_int(row.outs),
                strikeouts=_db_int(row.strikeouts),
                walks=_db_int(row.walks),
                home_runs=_db_int(row.home_runs),
                hit_batters=_db_int(row.hit_batters),
                pitches=_db_int(row.pitches),
            )
        )

    games: list[CompletedGame] = []
    for row in frame.itertuples(index=False):
        game_pk = _db_int(row.game_pk)
        games.append(
            CompletedGame(
                game_pk=game_pk,
                season=_db_int(row.season),
                game_datetime=str(row.game_datetime),
                away_team_id=_db_int(row.away_team_id),
                home_team_id=_db_int(row.home_team_id),
                away_runs=_db_int(row.away_runs),
                home_runs=_db_int(row.home_runs),
                away_starter=StarterLine(
                    player_id=_db_int(row.away_id),
                    outs=_db_int(row.away_outs),
                    earned_runs=_db_int(row.away_er),
                    strikeouts=_db_int(row.away_k),
                    walks=_db_int(row.away_bb),
                    home_runs=_db_int(row.away_hr),
                    hit_batters=_db_int(row.away_hbp),
                ),
                home_starter=StarterLine(
                    player_id=_db_int(row.home_id),
                    outs=_db_int(row.home_outs),
                    earned_runs=_db_int(row.home_er),
                    strikeouts=_db_int(row.home_k),
                    walks=_db_int(row.home_bb),
                    home_runs=_db_int(row.home_hr),
                    hit_batters=_db_int(row.home_hbp),
                ),
                away_batters=tuple(batters[(game_pk, "away")]),
                home_batters=tuple(batters[(game_pk, "home")]),
                away_relievers=tuple(relievers[(game_pk, "away")]),
                home_relievers=tuple(relievers[(game_pk, "home")]),
            )
        )
    return games


def build_feature_frame(
    games: Iterable[CompletedGame],
    config: StrengthConfig = DEFAULT_STRENGTH_CONFIG,
) -> tuple[pd.DataFrame, StrengthFeatureBuilder]:
    """Build chronological pregame features and return the final live state."""
    builder = StrengthFeatureBuilder(config)
    rows: list[dict[str, float | int | str]] = []
    for game in sorted(games, key=lambda item: (item.game_datetime, item.game_pk)):
        features = builder.observe(game)
        rows.append(
            {
                "game_pk": game.game_pk,
                "season": game.season,
                "game_date": game.game_datetime[:10],
                **dict(zip(FEATURE_NAMES, features.as_list(), strict=True)),
                "home_won": int(game.home_won),
            }
        )
    return pd.DataFrame(rows), builder


def train_strength_model(
    games: Sequence[CompletedGame],
    *,
    prediction_season: int,
    train_seasons: Sequence[int] | None = None,
    config: StrengthConfig = DEFAULT_STRENGTH_CONFIG,
) -> StrengthModelFit:
    """Fit a comparable estimator and retain chronological inference state."""
    from sklearn.linear_model import LogisticRegression

    feature_frame, builder = build_feature_frame(games, config)
    resolved_train_seasons = tuple(
        train_seasons
        if train_seasons is not None
        else range(prediction_season - 4, prediction_season)
    )
    train = feature_frame[feature_frame["season"].isin(resolved_train_seasons)]
    if train.empty:
        raise ValueError(
            f"No training games for seasons {list(resolved_train_seasons)}"
        )
    estimator = LogisticRegression(C=1.0, max_iter=1000)
    estimator.fit(train[list(FEATURE_NAMES)], train["home_won"])
    predictor = TeamStrengthPredictor(
        coefficients=tuple(
            float(value) for value in estimator.coef_.reshape(-1).tolist()
        ),
        intercept=float(np.asarray(estimator.intercept_).reshape(-1)[0]),
        feature_builder=builder,
    )
    return StrengthModelFit(
        estimator=estimator,
        predictor=predictor,
        feature_frame=feature_frame,
        train_seasons=resolved_train_seasons,
        config=config,
    )


def fit_strength_predictor(
    games: Sequence[CompletedGame],
    *,
    prediction_season: int,
    train_seasons: Sequence[int] | None = None,
    config: StrengthConfig = DEFAULT_STRENGTH_CONFIG,
) -> tuple[TeamStrengthPredictor, pd.DataFrame]:
    """Fit on prior seasons and retain state through all completed games."""
    fitted = train_strength_model(
        games,
        prediction_season=prediction_season,
        train_seasons=train_seasons,
        config=config,
    )
    return fitted.predictor, fitted.feature_frame


def _object_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Invalid {label} in MLflow model contract")
    return cast(dict[str, object], value)


def _contract_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Invalid {key!r} in MLflow model contract")
    return float(value)


def _load_champion_strength_model(
    *,
    tracking_uri: str | None,
    registered_model_name: str,
) -> _LoadedStrengthModel:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.sklearn import load_model
    from sklearn.linear_model import LogisticRegression

    from mlb.ml.mlflow_utils import resolve_mlflow_tracking_uri

    resolved_uri = resolve_mlflow_tracking_uri(tracking_uri)
    if not resolved_uri:
        raise RuntimeError("A shared MLflow tracking URI is required")
    mlflow.set_tracking_uri(resolved_uri)
    client = MlflowClient(tracking_uri=resolved_uri)
    registered_model = client.get_registered_model(registered_model_name)
    selected = client.get_model_version_by_alias(
        registered_model_name,
        "champion",
    )
    version = str(selected.version)
    run_id = selected.run_id
    if selected.status != "READY" or not run_id:
        raise RuntimeError(
            f"Champion version for {registered_model_name!r} is not ready"
        )
    if (
        registered_model.tags.get("champion_version") != version
        or registered_model.tags.get("champion_run_id") != run_id
    ):
        raise RuntimeError(
            f"Champion metadata for {registered_model_name!r} is inconsistent"
        )
    if selected.tags.get("promotion_gate") != "passed":
        raise RuntimeError(
            f"Champion version {version!r} did not pass the promotion gate"
        )
    if selected.tags.get("model_type") != WIN_PROBABILITY_MODEL_TYPE:
        raise ValueError("Registered MLflow model has an invalid model type")
    run = client.get_run(run_id)
    if run.data.tags.get("promotion_gate") != "passed":
        raise RuntimeError(f"Champion run {run_id!r} did not pass the promotion gate")

    contract_path = Path(client.download_artifacts(run_id, "model_contract.json"))
    contract = _object_mapping(
        json.loads(contract_path.read_text()),
        "model contract",
    )
    contract_version = contract.get("contract_version")
    if contract_version == LEGACY_STRENGTH_MODEL_CONTRACT_VERSION:
        feature_names = LEGACY_FEATURE_NAMES
    elif contract_version == STRENGTH_MODEL_CONTRACT_VERSION:
        feature_names = FEATURE_NAMES
    else:
        raise ValueError("Unsupported win-model contract version")
    if contract.get("model_family") != STRENGTH_MODEL_FAMILY:
        raise ValueError("Registered MLflow model has the wrong model family")
    if contract.get("features") != list(feature_names):
        raise ValueError("Registered MLflow model has incompatible features")

    training = _object_mapping(
        contract.get("training"),
        "training metadata",
    )
    start_season = training.get("start_season")
    if not isinstance(start_season, int):
        raise TypeError("Invalid training start season in MLflow contract")
    config_data = _object_mapping(
        contract.get("strength_config"),
        "strength configuration",
    )
    roster_config = RosterStrengthConfig()
    if contract_version == STRENGTH_MODEL_CONTRACT_VERSION:
        roster_data = _object_mapping(
            config_data.get("roster"),
            "roster strength configuration",
        )
        roster_config = RosterStrengthConfig(
            league_woba=_contract_float(roster_data, "league_woba"),
            batter_prior_pa=_contract_float(roster_data, "batter_prior_pa"),
            batter_recency_half_life_days=_contract_float(
                roster_data, "batter_recency_half_life_days"
            ),
            batter_season_decay=_contract_float(roster_data, "batter_season_decay"),
            batter_peak_age=_contract_float(roster_data, "batter_peak_age"),
            batter_growth_per_year=_contract_float(
                roster_data, "batter_growth_per_year"
            ),
            batter_decline_per_year=_contract_float(
                roster_data, "batter_decline_per_year"
            ),
            bullpen_prior_ip=_contract_float(roster_data, "bullpen_prior_ip"),
            league_fip=_contract_float(roster_data, "league_fip"),
            fip_constant=_contract_float(roster_data, "fip_constant"),
            reliever_season_decay=_contract_float(roster_data, "reliever_season_decay"),
            reliever_active_days=int(
                _contract_float(roster_data, "reliever_active_days")
            ),
            bullpen_size=int(_contract_float(roster_data, "bullpen_size")),
            workload_pitch_limit=_contract_float(roster_data, "workload_pitch_limit"),
            workload_fip_penalty=_contract_float(roster_data, "workload_fip_penalty"),
        )
    config = StrengthConfig(
        initial_elo=_contract_float(config_data, "initial_elo"),
        elo_k=_contract_float(config_data, "elo_k"),
        elo_home_advantage=_contract_float(config_data, "elo_home_advantage"),
        elo_season_regression=_contract_float(config_data, "elo_season_regression"),
        initial_runs_per_game=_contract_float(config_data, "initial_runs_per_game"),
        run_alpha=_contract_float(config_data, "run_alpha"),
        run_season_regression=_contract_float(config_data, "run_season_regression"),
        starter_prior_ip=_contract_float(config_data, "starter_prior_ip"),
        starter_season_decay=_contract_float(config_data, "starter_season_decay"),
        roster=roster_config,
    )

    estimator = load_model(f"models:/{registered_model_name}/{version}")
    if not isinstance(estimator, LogisticRegression):
        raise TypeError("Registered win model is not logistic regression")
    if estimator.coef_.size != len(feature_names):
        raise ValueError("Registered win model has incompatible coefficients")
    return _LoadedStrengthModel(
        estimator=estimator,
        config=config,
        feature_names=feature_names,
        start_season=start_season,
        source=StrengthModelSource(
            registered_model_name=registered_model_name,
            version=version,
            run_id=run_id,
        ),
    )


def build_live_strength_predictor(
    prediction_date: date,
    *,
    start_season: int | None = None,
    db_config: PostgresConfig | None = None,
    tracking_uri: str | None = None,
    registered_model_name: str = DEFAULT_REGISTERED_STRENGTH_MODEL,
) -> TeamStrengthPredictor:
    """Load champion coefficients and build state strictly before a slate."""
    loaded = _load_champion_strength_model(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if start_season is not None and start_season != loaded.start_season:
        raise ValueError(
            "start_season must match the registered model contract "
            f"({loaded.start_season})"
        )

    prediction_season = prediction_date.year
    completed = load_completed_games(
        include_rosters=loaded.feature_names == FEATURE_NAMES,
        start_season=loaded.start_season,
        end_season=prediction_season,
        db_config=db_config,
    )
    cutoff = prediction_date.isoformat()
    builder = StrengthFeatureBuilder(loaded.config)
    for game in completed:
        if game.game_datetime[:10] < cutoff:
            builder.observe(game)
    builder.advance_to_season(prediction_season)
    return TeamStrengthPredictor(
        coefficients=tuple(
            float(value) for value in loaded.estimator.coef_.reshape(-1).tolist()
        ),
        intercept=float(np.asarray(loaded.estimator.intercept_).reshape(-1)[0]),
        feature_builder=builder,
        feature_names=loaded.feature_names,
        source=loaded.source,
    )
