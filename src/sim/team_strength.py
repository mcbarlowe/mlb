"""Leak-free pregame team and starting-pitcher strength model.

The pitch-by-pitch simulator is useful for score distributions, but its raw
home-win spread has not reliably beaten simple home-team baselines. This module
builds an independent pregame probability from information available before
each game:

- slowly updating team Elo,
- exponentially weighted team runs scored/allowed,
- Bayesian-shrunk starting-pitcher FIP and expected length.

Every feature row is emitted before that game's result updates the state. Model
coefficients are fit only on seasons before the prediction season.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from src.database import PostgresConfig, PostgresHandler

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FEATURE_NAMES = (
    "elo_diff",
    "run_edge",
    "starter_era_edge",
    "starter_fip_edge",
    "starter_length_edge",
)


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

    def as_list(self) -> list[float]:
        return [
            self.elo_diff,
            self.run_edge,
            self.starter_era_edge,
            self.starter_fip_edge,
            self.starter_length_edge,
        ]


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

    def advance_to_season(self, season: int) -> None:
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
    ) -> PregameFeatures:
        self.advance_to_season(season)
        away_era, away_fip, away_length = self._starter_features(away_starter_id)
        home_era, home_fip, home_length = self._starter_features(home_starter_id)
        return PregameFeatures(
            elo_diff=(self._elo[home_team_id] - self._elo[away_team_id]) / 100.0,
            run_edge=(self._offense[home_team_id] - self._offense[away_team_id])
            + (self._defense[away_team_id] - self._defense[home_team_id]),
            starter_era_edge=away_era - home_era,
            starter_fip_edge=away_fip - home_fip,
            starter_length_edge=(home_length - away_length) / 3.0,
        )

    def observe(self, game: CompletedGame) -> PregameFeatures:
        """Return pregame features, then update state with the final result."""
        features = self.matchup_features(
            season=game.season,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            away_starter_id=game.away_starter.player_id,
            home_starter_id=game.home_starter.player_id,
        )
        self._update_teams(game)
        self._pitchers[game.away_starter.player_id].update(game.away_starter)
        self._pitchers[game.home_starter.player_id].update(game.home_starter)
        return features

    def _regress_for_new_season(self) -> None:
        c = self.config
        for team_id, rating in self._elo.items():
            self._elo[team_id] = c.initial_elo + (
                rating - c.initial_elo
            ) * (1.0 - c.elo_season_regression)
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
            + 3.0
            * (
                state.walks
                + state.hit_batters
                + prior_ip * 0.405
            )
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
            1.0
            + 10.0
            ** (
                -(
                    home_rating
                    + c.elo_home_advantage
                    - away_rating
                )
                / 400.0
            )
        )
        delta = c.elo_k * (float(game.home_won) - expected_home)
        self._elo[game.home_team_id] += delta
        self._elo[game.away_team_id] -= delta

        alpha = c.run_alpha
        self._offense[game.home_team_id] = (
            (1.0 - alpha) * self._offense[game.home_team_id]
            + alpha * game.home_runs
        )
        self._defense[game.home_team_id] = (
            (1.0 - alpha) * self._defense[game.home_team_id]
            + alpha * game.away_runs
        )
        self._offense[game.away_team_id] = (
            (1.0 - alpha) * self._offense[game.away_team_id]
            + alpha * game.away_runs
        )
        self._defense[game.away_team_id] = (
            (1.0 - alpha) * self._defense[game.away_team_id]
            + alpha * game.home_runs
        )


@dataclass(frozen=True)
class TeamStrengthPredictor:
    coefficients: tuple[float, ...]
    intercept: float
    feature_builder: StrengthFeatureBuilder

    def predict_home_probability(
        self,
        *,
        season: int,
        away_team_id: int,
        home_team_id: int,
        away_starter_id: int,
        home_starter_id: int,
    ) -> float:
        features = self.feature_builder.matchup_features(
            season=season,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            away_starter_id=away_starter_id,
            home_starter_id=home_starter_id,
        )
        log_odds = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, features.as_list())
        )
        return 1.0 / (1.0 + math.exp(-log_odds))


def _db_int(value: object | None) -> int:
    """Normalize integer-valued PostgreSQL/Pandas scalars."""
    if value is None:
        return 0
    return int(str(value))


def load_completed_games(
    *,
    start_season: int = 2015,
    end_season: int,
    db_config: PostgresConfig | None = None,
) -> list[CompletedGame]:
    """Load regular-season finals with each starter's game line."""
    start_season = int(start_season)
    end_season = int(end_season)
    if end_season < start_season:
        raise ValueError("end_season must be at least start_season")
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
            COALESCE(NULLIF(g.game_datetime, ''), g.game_date) AS game_datetime,
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
        WHERE g.game_type = 'R'
          AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN {start_season} AND {end_season}
          AND s.away_runs <> s.home_runs
          AND p.away_id IS NOT NULL
          AND p.home_id IS NOT NULL
        ORDER BY game_datetime, g.game_pk
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)
    games: list[CompletedGame] = []
    for row in frame.itertuples(index=False):
        games.append(
            CompletedGame(
                game_pk=_db_int(row.game_pk),
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
            )
        )
    return games


def build_feature_frame(
    games: Iterable[CompletedGame],
    config: StrengthConfig = DEFAULT_STRENGTH_CONFIG,
) -> tuple[pd.DataFrame, StrengthFeatureBuilder]:
    """Build chronological pregame features and return the final live state."""
    builder = StrengthFeatureBuilder(config)
    rows: list[dict[str, float | int]] = []
    for game in games:
        features = builder.observe(game)
        rows.append(
            {
                "game_pk": game.game_pk,
                "season": game.season,
                **dict(zip(FEATURE_NAMES, features.as_list())),
                "home_won": int(game.home_won),
            }
        )
    return pd.DataFrame(rows), builder


def fit_strength_predictor(
    games: Sequence[CompletedGame],
    *,
    prediction_season: int,
    train_seasons: Sequence[int] | None = None,
    config: StrengthConfig = DEFAULT_STRENGTH_CONFIG,
) -> tuple[TeamStrengthPredictor, pd.DataFrame]:
    """Fit on prior seasons and retain state through all completed games."""
    from sklearn.linear_model import LogisticRegression

    feature_frame, builder = build_feature_frame(games, config)
    if train_seasons is None:
        train_seasons = list(range(prediction_season - 4, prediction_season))
    train = feature_frame[feature_frame["season"].isin(train_seasons)]
    if train.empty:
        raise ValueError(f"No training games for seasons {list(train_seasons)}")
    estimator = LogisticRegression(C=1.0, max_iter=1000)
    estimator.fit(train[list(FEATURE_NAMES)], train["home_won"])
    predictor = TeamStrengthPredictor(
        coefficients=tuple(
            float(value) for value in estimator.coef_.reshape(-1).tolist()
        ),
        intercept=float(estimator.intercept_),
        feature_builder=builder,
    )
    return predictor, feature_frame


def build_live_strength_predictor(
    prediction_date: date,
    *,
    start_season: int = 2015,
    db_config: PostgresConfig | None = None,
) -> TeamStrengthPredictor:
    """Fit prior-season coefficients and build state strictly before a slate."""
    prediction_season = prediction_date.year
    completed = load_completed_games(
        start_season=start_season,
        end_season=prediction_season,
        db_config=db_config,
    )
    cutoff = prediction_date.isoformat()
    available = [game for game in completed if game.game_datetime[:10] < cutoff]
    predictor, _ = fit_strength_predictor(
        available,
        prediction_season=prediction_season,
    )
    predictor.feature_builder.advance_to_season(prediction_season)
    return predictor
