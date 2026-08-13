"""Join the win model, historical odds, and final results into backtest rows.

Three inputs, one output:
  * ``load_finals`` / ``champion_home_probs`` come from Postgres + the MLflow
    champion contract and are fully reproducible here.
  * historical prices are provider-specific and are read through
    ``load_odds_csv`` against a small normalized schema, so any odds feed
    (scraped, purchased, or hand-collected) plugs in by conforming to it.
  * ``build_moneyline_games`` inner-joins the three on ``game_pk`` and emits the
    ``MoneylineGame`` rows the backtest consumes.

Normalized odds CSV schema (header row, case-insensitive), one row per game:
  date            game date, YYYY-MM-DD (local game date, matches mlb.games)
  away_team       team abbreviation (e.g. NYY) or full/location name
  home_team       team abbreviation or full/location name
  home_ml_close   home closing moneyline, American odds (required)
  away_ml_close   away closing moneyline, American odds (required)
  home_ml_open    home opening moneyline, American odds (optional, for CLV)
  away_ml_open    away opening moneyline, American odds (optional, for CLV)
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.betting.backtest import MoneylineGame

MLFLOW_HTTP_URI = "http://10.0.0.171:5001"
CHAMPION_MODEL = "mlb-team-strength-win"

REQUIRED_ODDS_COLUMNS = ("date", "away_team", "home_team", "home_ml_close", "away_ml_close")
OPTIONAL_ODDS_COLUMNS = ("home_ml_open", "away_ml_open")


def load_finals(seasons: Sequence[int]) -> pd.DataFrame:
    """Regular-season finals keyed for the odds join.

    Sourced from ``load_completed_games`` (the same universe the win model
    trains and predicts on) so a game present in the model is present here.
    Returns: game_pk, game_date, season, away_team_id, home_team_id, home_won.
    """
    from src.sim.team_strength import load_completed_games

    wanted = {int(s) for s in seasons}
    games = load_completed_games(
        start_season=min(wanted), end_season=max(wanted), include_rosters=False
    )
    rows = [
        {
            "game_pk": g.game_pk,
            "game_date": g.game_datetime[:10],
            "season": g.season,
            "away_team_id": g.away_team_id,
            "home_team_id": g.home_team_id,
            "home_won": g.home_won,
        }
        for g in games
        if g.season in wanted and g.home_runs != g.away_runs
    ]
    return pd.DataFrame(rows)


def team_abbrev_to_id() -> dict[str, int]:
    """Map every alias (abbreviation, name, short name, location) to team_id.

    Restricted to the 30 MLB clubs (sport_id = 1) so minor/spring rows in
    ``mlb.teams`` never shadow a big-league abbreviation.
    """
    from src.database.postgres_handler import PostgresHandler

    handler = PostgresHandler()
    with handler.connection.cursor() as cur:
        cur.execute(
            """
            SELECT team_id, abbreviation, team_name, team_name_short, location_name
            FROM mlb.teams
            WHERE sport_id = 1
            """
        )
        rows = cur.fetchall()
    mapping: dict[str, int] = {}
    for team_id, abbrev, name, short, location in rows:
        for alias in (abbrev, name, short, location):
            if alias:
                mapping[str(alias).strip().upper()] = int(team_id)
    return mapping


def _resolve_team(value: object, mapping: dict[str, int]) -> int | None:
    if value is None:
        return None
    return mapping.get(str(value).strip().upper())


def champion_home_probs(seasons: Sequence[int]) -> pd.DataFrame:
    """Champion team-strength home-win probabilities, reproduced from the contract.

    Downloads the registered champion's coefficients + strength config from the
    MLflow HTTP server, rebuilds the leak-free feature frame from Postgres, and
    applies the exact logistic model. Returns ``game_pk, model_prob_home``.
    """
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    from src.sim.team_strength import (
        LEGACY_FEATURE_NAMES,
        StrengthConfig,
        build_feature_frame,
        load_completed_games,
    )

    client = MlflowClient(tracking_uri=MLFLOW_HTTP_URI)
    version = client.get_model_version_by_alias(CHAMPION_MODEL, "champion")
    contract_path = download_artifacts(
        run_id=version.run_id,
        artifact_path="model_contract.json",
        tracking_uri=MLFLOW_HTTP_URI,
    )
    contract = json.loads(Path(contract_path).read_text())
    features = tuple(contract["features"])
    coefficients = contract["coefficients"]
    intercept = float(contract["intercept"])
    sc = contract["strength_config"]
    config = StrengthConfig(
        initial_elo=sc["initial_elo"],
        elo_k=sc["elo_k"],
        elo_home_advantage=sc["elo_home_advantage"],
        elo_season_regression=sc["elo_season_regression"],
        initial_runs_per_game=sc["initial_runs_per_game"],
        run_alpha=sc["run_alpha"],
        run_season_regression=sc["run_season_regression"],
        starter_prior_ip=sc["starter_prior_ip"],
        starter_season_decay=sc["starter_season_decay"],
    )
    if set(features) - set(LEGACY_FEATURE_NAMES) and features != LEGACY_FEATURE_NAMES:
        # Non-legacy contract: features must still all exist in the frame.
        pass
    start_season = int(contract["training"]["start_season"])
    games = load_completed_games(start_season=start_season, end_season=max(seasons))
    frame, _ = build_feature_frame(games, config)
    log_odds = intercept + sum(
        float(coefficients[name]) * frame[name] for name in features
    )
    frame = frame.assign(model_prob_home=1.0 / (1.0 + (-log_odds).map(math.exp)))
    keep = frame[frame["season"].isin([int(s) for s in seasons])]
    return keep[["game_pk", "model_prob_home"]].reset_index(drop=True)


def load_odds_csv(path: str | Path) -> pd.DataFrame:
    """Read a bring-your-own odds CSV into the normalized in-memory schema.

    Returns columns: date, away_team, home_team, home_ml_close, away_ml_close,
    and (when present) home_ml_open, away_ml_open. Raises on missing required
    columns so a malformed feed fails loudly rather than silently dropping games.
    """
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED_ODDS_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Odds CSV missing required columns: {missing}")
    keep_cols = list(REQUIRED_ODDS_COLUMNS) + [
        c for c in OPTIONAL_ODDS_COLUMNS if c in raw.columns
    ]
    out = raw[keep_cols].copy()
    out["date"] = out["date"].astype(str).str.slice(0, 10)
    for col in ("home_ml_close", "away_ml_close", *OPTIONAL_ODDS_COLUMNS):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["home_ml_close", "away_ml_close"]).reset_index(drop=True)


def build_moneyline_games(
    *,
    finals: pd.DataFrame,
    model_probs: pd.DataFrame,
    odds: pd.DataFrame,
    team_map: dict[str, int] | None = None,
) -> list[MoneylineGame]:
    """Inner-join finals, model probs, and odds into backtest rows.

    Odds team strings resolve to team_ids and match a final on
    (date, home_team_id, away_team_id). Games without a matching price or an
    unresolved team are skipped; the caller reports the coverage.
    """
    mapping = team_map if team_map is not None else team_abbrev_to_id()
    odds = odds.copy()
    odds["home_team_id"] = odds["home_team"].map(lambda v: _resolve_team(v, mapping))
    odds["away_team_id"] = odds["away_team"].map(lambda v: _resolve_team(v, mapping))
    odds = odds.dropna(subset=["home_team_id", "away_team_id"])
    odds["home_team_id"] = odds["home_team_id"].astype(int)
    odds["away_team_id"] = odds["away_team_id"].astype(int)

    keyed = finals.merge(
        odds,
        left_on=["game_date", "home_team_id", "away_team_id"],
        right_on=["date", "home_team_id", "away_team_id"],
        how="inner",
    ).merge(model_probs, on="game_pk", how="inner")

    has_open = "home_ml_open" in keyed.columns and "away_ml_open" in keyed.columns
    games: list[MoneylineGame] = []
    for row in keyed.itertuples(index=False):
        home_close = float(row.home_ml_close)
        away_close = float(row.away_ml_close)
        if has_open and not (
            math.isnan(getattr(row, "home_ml_open", math.nan))
            or math.isnan(getattr(row, "away_ml_open", math.nan))
        ):
            home_take = float(row.home_ml_open)
            away_take = float(row.away_ml_open)
        else:
            home_take, away_take = home_close, away_close
        games.append(
            MoneylineGame(
                game_pk=int(row.game_pk),
                model_prob_home=float(row.model_prob_home),
                home_take=home_take,
                away_take=away_take,
                home_close=home_close,
                away_close=away_close,
                home_won=bool(row.home_won),
            )
        )
    return games
