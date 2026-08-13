"""Day-ahead slate simulation helpers.

Shared by the ad-hoc `scripts/simulate_game.py --slate` entrypoint and the
scheduled daily all-games simulation workflow.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import requests

from src.ml.mlflow_utils import resolve_mlflow_tracking_uri
from src.outcome.inference import PitchOutcomePredictor
from src.outcome.mlflow_artifacts import (
    OutcomeProductionSelection,
    resolve_outcome_artifact_dirs,
    resolve_production_outcome_selection,
)
from src.sim.base_out import BaseOutEngine
from src.sim.calibration import load_win_calibration
from src.sim.game import (
    BULLPEN_ARM,
    Batter,
    GameResult,
    GameSimulator,
    Lineup,
    Pitcher,
    summarize,
)
from src.sim.lineups import lineup_from_feed, starting_batters_from_feed
from src.sim.matchup import MatchupProviderFactory
from src.sim.pitch_mix import PitchMixProfiles
from src.sim.team_strength import TeamStrengthPredictor

LIVEFEED_ROOT = Path("data/raw/livefeeds")
STATS_API = "https://statsapi.mlb.com/api/v1"
_DEFAULT_STARTER_LABEL = "TBD (league-average arm)"


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return cast(Mapping[str, object], value)


def _as_int(value: object | None, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{context} must be an int-like value")


def _as_str(value: object | None) -> str | None:
    return str(value) if value is not None else None


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [_as_mapping(item, "list item") for item in value]


@dataclass(frozen=True)
class ProbablePitcher:
    player_id: int | None
    full_name: str | None

    @property
    def display_name(self) -> str:
        return self.full_name or _DEFAULT_STARTER_LABEL

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProbablePitcher:
        return cls(
            player_id=_as_int(payload.get("player_id"), "player_id"),
            full_name=_as_str(payload.get("full_name")),
        )


@dataclass(frozen=True)
class SlateGame:
    game_pk: int
    slate_date: str
    game_datetime: str | None
    status: str
    away_team_id: int
    home_team_id: int
    away_abbrev: str
    home_abbrev: str
    venue: str | None
    away_probable: ProbablePitcher
    home_probable: ProbablePitcher

    @property
    def label(self) -> str:
        return f"{self.away_abbrev} @ {self.home_abbrev}"

    def probable_for(self, side: str) -> ProbablePitcher:
        if side == "away":
            return self.away_probable
        if side == "home":
            return self.home_probable
        raise KeyError(f"Unknown side {side!r}")

    def team_id_for(self, side: str) -> int:
        if side == "away":
            return self.away_team_id
        if side == "home":
            return self.home_team_id
        raise KeyError(f"Unknown side {side!r}")

    def abbrev_for(self, side: str) -> str:
        if side == "away":
            return self.away_abbrev
        if side == "home":
            return self.home_abbrev
        raise KeyError(f"Unknown side {side!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SlateGame:
        return cls(
            game_pk=_as_int(payload.get("game_pk"), "game_pk") or 0,
            slate_date=str(payload["slate_date"]),
            game_datetime=_as_str(payload.get("game_datetime")),
            status=str(payload["status"]),
            away_team_id=_as_int(payload.get("away_team_id"), "away_team_id") or 0,
            home_team_id=_as_int(payload.get("home_team_id"), "home_team_id") or 0,
            away_abbrev=str(payload["away_abbrev"]),
            home_abbrev=str(payload["home_abbrev"]),
            venue=_as_str(payload.get("venue")),
            away_probable=ProbablePitcher.from_dict(
                _as_mapping(payload["away_probable"], "away_probable")
            ),
            home_probable=ProbablePitcher.from_dict(
                _as_mapping(payload["home_probable"], "home_probable")
            ),
        )


@dataclass(frozen=True)
class StarterChange:
    side: str
    previous: str
    current: str


@dataclass(frozen=True)
class SlatePrediction:
    game: SlateGame
    results: list[GameResult]
    away_starter: str
    home_starter: str
    stats: dict[str, float]


@dataclass(frozen=True)
class DailySlateState:
    slate_date: str
    saved_at: str
    board_path: str
    board_post_id: str | None
    games: list[SlateGame]

    def by_game_pk(self) -> dict[int, SlateGame]:
        return {game.game_pk: game for game in self.games}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DailySlateState:
        games = [
            SlateGame.from_dict(item) for item in _mapping_list(payload.get("games"))
        ]
        return cls(
            slate_date=str(payload["slate_date"]),
            saved_at=str(payload["saved_at"]),
            board_path=str(payload["board_path"]),
            board_post_id=(
                str(payload["board_post_id"])
                if payload.get("board_post_id") is not None
                else None
            ),
            games=games,
        )


class SlateSimulationError(RuntimeError):
    """Raised when the day-ahead simulator cannot resolve its model inputs."""


def _fetch_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _is_final(feed: dict) -> bool:
    return (
        feed.get("gameData", {}).get("status", {}).get("abstractGameState") == "Final"
    )


def load_feed(game_pk: int, season: int | None) -> dict:
    """Archived feed for a game; refresh from the API when the local copy is stale."""
    seasons = (
        [season] if season else [path.name for path in sorted(LIVEFEED_ROOT.iterdir())]
    )
    local: dict | None = None
    for candidate in seasons:
        path = LIVEFEED_ROOT / str(candidate) / f"{game_pk}.json"
        if path.exists():
            local = json.loads(path.read_text())
            break
    if local is not None and _is_final(local):
        return local
    try:
        return _fetch_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        )
    except Exception:
        if local is not None:
            return local
        raise SystemExit(
            f"No archived feed for game {game_pk}; pass --season or backfill"
        )


def resolve_outcome_model_dirs(
    run_dir_arg: str = "auto",
    *,
    tracking_uri: str | None = None,
    selection: OutcomeProductionSelection | None = None,
) -> tuple[Path, Path]:
    resolved = resolve_outcome_artifact_dirs(
        run_dir_arg,
        tracking_uri=resolve_mlflow_tracking_uri(tracking_uri),
        selection=selection,
    )
    if resolved is None:
        raise SlateSimulationError(
            "Outcome models not found locally or in shared MLflow"
        )
    return resolved


def build_day_ahead_simulator(
    *,
    season: int,
    seed: int = 42,
    outcome_run_dir: str = "auto",
    tracking_uri: str | None = None,
) -> tuple[GameSimulator, Path]:
    """Load the outcome-model chain, pitch-mix tables, and base-out engine."""
    from src.sim.artifacts import ensure_sim_artifacts
    from src.sim.calibration import (
        DEFAULT_CALIBRATION_PATH,
        SimCalibration,
        load_pa_outcome_calibration,
    )

    resolved_tracking_uri = resolve_mlflow_tracking_uri(tracking_uri)
    selection = (
        resolve_production_outcome_selection(resolved_tracking_uri)
        if outcome_run_dir == "auto" and resolved_tracking_uri
        else None
    )
    ensure_sim_artifacts(
        resolved_tracking_uri,
        run_id=selection.sim_inputs_run_id if selection else None,
    )
    run_dir, profiles_dir = resolve_outcome_model_dirs(
        outcome_run_dir,
        tracking_uri=resolved_tracking_uri,
        selection=selection,
    )
    predictor = PitchOutcomePredictor(run_dir, profiles_dir=profiles_dir)
    mix = PitchMixProfiles.load(seed=seed)
    calibration = SimCalibration.load() if DEFAULT_CALIBRATION_PATH.exists() else None
    pa_calibration = load_pa_outcome_calibration()
    factory = MatchupProviderFactory(
        predictor,
        mix,
        season=season,
        seed=seed,
        calibration=calibration,
        pa_outcome_calibration=pa_calibration,
    )
    engine = BaseOutEngine.load(seed=seed)
    return GameSimulator(factory, engine, rng=random.Random(seed)), run_dir


def _team_abbrev(team: Mapping[str, object]) -> str:
    return str(team.get("abbreviation") or team.get("name") or "?")


def _probable_pitcher(team_entry: Mapping[str, object]) -> ProbablePitcher:
    probable = team_entry.get("probablePitcher")
    if not isinstance(probable, Mapping):
        return ProbablePitcher(player_id=None, full_name=None)
    player_id = probable.get("id")
    return ProbablePitcher(
        player_id=int(player_id) if player_id is not None else None,
        full_name=(
            str(probable.get("fullName"))
            if probable.get("fullName") is not None
            else None
        ),
    )


def slate_game_from_schedule_entry(
    game: Mapping[str, object], slate_date: str
) -> SlateGame:
    teams = _as_mapping(game["teams"], "teams")
    away = _as_mapping(teams["away"], "away team entry")
    home = _as_mapping(teams["home"], "home team entry")
    away_team = _as_mapping(away["team"], "away team")
    home_team = _as_mapping(home["team"], "home team")
    status = _as_mapping(game.get("status", {}), "status")
    venue_payload = game.get("venue")
    venue = None
    if venue_payload is not None:
        venue = _as_str(_as_mapping(venue_payload, "venue").get("name"))
    return SlateGame(
        game_pk=_as_int(game.get("gamePk"), "gamePk") or 0,
        slate_date=slate_date,
        game_datetime=_as_str(game.get("gameDate")),
        status=str(status.get("abstractGameState", "")),
        away_team_id=_as_int(away_team.get("id"), "away team id") or 0,
        home_team_id=_as_int(home_team.get("id"), "home team id") or 0,
        away_abbrev=_team_abbrev(away_team),
        home_abbrev=_team_abbrev(home_team),
        venue=venue,
        away_probable=_probable_pitcher(away),
        home_probable=_probable_pitcher(home),
    )


def fetch_slate_games(
    target_date: date,
    *,
    abstract_states: set[str] | None = None,
) -> list[SlateGame]:
    slate_date = target_date.isoformat()
    schedule = _fetch_json(
        f"{STATS_API}/schedule",
        {"sportId": 1, "date": slate_date, "hydrate": "probablePitcher,team"},
    )
    games: list[SlateGame] = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            state = str(game.get("status", {}).get("abstractGameState", ""))
            if abstract_states is not None and state not in abstract_states:
                continue
            games.append(slate_game_from_schedule_entry(game, slate_date))
    return sorted(games, key=lambda game: (game.game_datetime or "", game.game_pk))


def _feed_for_game(game_pk: int, season: int) -> dict:
    path = LIVEFEED_ROOT / str(season) / f"{game_pk}.json"
    if path.exists():
        feed = json.loads(path.read_text())
        if _is_final(feed):
            return feed
    return _fetch_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")


def _pitch_hand(player_id: int) -> str:
    data = _fetch_json(f"{STATS_API}/people/{player_id}")
    people = data.get("people", [])
    return people[0].get("pitchHand", {}).get("code", "R") if people else "R"


def active_roster_ids(
    team_id: int,
    slate_date: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        payload = _fetch_json(
            f"{STATS_API}/teams/{team_id}/roster",
            {
                "rosterType": "active",
                "date": slate_date,
            },
        )
    except (requests.RequestException, ValueError):
        return (), ()
    batter_ids: list[int] = []
    pitcher_ids: list[int] = []
    try:
        roster = _mapping_list(payload.get("roster"))
    except (TypeError, ValueError):
        return (), ()
    for entry in roster:
        try:
            person = _as_mapping(entry.get("person", {}), "roster person")
            player_id = _as_int(person.get("id"), "roster player id")
            if player_id is None:
                continue
            position = _as_mapping(entry.get("position", {}), "roster position")
            position_type = _as_str(position.get("type")) or ""
            abbreviation = _as_str(position.get("abbreviation")) or ""
        except (TypeError, ValueError):
            continue
        is_pitcher = position_type in {"Pitcher", "Two-Way Player"} or abbreviation in {
            "P",
            "TWP",
        }
        if is_pitcher:
            pitcher_ids.append(player_id)
        if position_type != "Pitcher" or abbreviation == "TWP":
            batter_ids.append(player_id)
    return tuple(batter_ids), tuple(pitcher_ids)


def _projected_batters(team_id: int, slate_date: str, season: int) -> list:
    from datetime import date as date_cls
    from datetime import timedelta

    end = date_cls.fromisoformat(slate_date)
    schedule = _fetch_json(
        f"{STATS_API}/schedule",
        {
            "sportId": 1,
            "teamId": team_id,
            "startDate": (end - timedelta(days=10)).isoformat(),
            "endDate": (end - timedelta(days=1)).isoformat(),
        },
    )
    finals: list[tuple[str, int, str]] = []
    for day in _mapping_list(schedule.get("dates")):
        for game in _mapping_list(day.get("games")):
            status = _as_mapping(game.get("status", {}), "status")
            if status.get("abstractGameState") != "Final":
                continue
            teams = _as_mapping(game["teams"], "teams")
            home_team = _as_mapping(
                _as_mapping(teams["home"], "home team entry")["team"],
                "home team",
            )
            side = (
                "home"
                if _as_int(home_team.get("id"), "home team id") == team_id
                else "away"
            )
            game_pk = _as_int(game.get("gamePk"), "gamePk")
            game_date = _as_str(game.get("gameDate"))
            if game_pk is None or game_date is None:
                continue
            finals.append((game_date, game_pk, side))
    for _, game_pk, side in sorted(finals, reverse=True):
        try:
            return lineup_from_feed(_feed_for_game(game_pk, season), side).batters
        except (ValueError, KeyError):
            continue
    raise ValueError(f"No recent lineup found for team {team_id}")


def _announced_batters(game_pk: int) -> dict[str, list[Batter]]:
    try:
        feed = _fetch_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        )
    except (requests.RequestException, ValueError):
        return {}
    announced: dict[str, list[Batter]] = {}
    for side in ("away", "home"):
        try:
            batters = starting_batters_from_feed(feed, side)
        except (KeyError, TypeError, ValueError):
            continue
        if len(batters) == 9 and len({batter.player_id for batter in batters}) == 9:
            announced[side] = batters
    return announced


def build_projected_lineups(
    game: SlateGame,
    *,
    season: int,
    announced_lineups: dict[str, list[Batter]] | None = None,
) -> tuple[dict[str, Lineup], dict[str, str]]:
    lineups: dict[str, Lineup] = {}
    starters: dict[str, str] = {}
    announced = (
        _announced_batters(game.game_pk)
        if announced_lineups is None
        else announced_lineups
    )
    for side in ("away", "home"):
        from src.sim.bullpen import bullpen_for_team
        from src.sim.db_games import trailing_reliever_pool

        probable = game.probable_for(side)
        if probable.player_id is not None:
            starter = Pitcher(probable.player_id, _pitch_hand(probable.player_id))
        else:
            starter = BULLPEN_ARM
        starters[side] = probable.display_name
        team_id = game.team_id_for(side)
        relievers = trailing_reliever_pool(team_id, game.slate_date, season)
        lineups[side] = Lineup(
            batters=announced.get(side)
            or _projected_batters(team_id, game.slate_date, season),
            starter=starter,
            bullpen=bullpen_for_team(team_id),
            relievers=relievers,
        )
    return lineups, starters


def simulate_slate_game(
    game: SlateGame,
    simulator: GameSimulator,
    *,
    season: int,
    n_sims: int,
    win_predictor: TeamStrengthPredictor,
) -> SlatePrediction:
    announced = _announced_batters(game.game_pk)
    lineups, starters = build_projected_lineups(
        game,
        season=season,
        announced_lineups=announced,
    )
    results = simulator.simulate_many(lineups["away"], lineups["home"], n_sims)
    stats = summarize(results)
    sim_probability = stats["home_win_probability"]
    calibration = load_win_calibration()
    stats["home_win_probability_raw"] = sim_probability
    stats["home_win_probability_sim"] = (
        calibration.apply(sim_probability)
        if calibration is not None
        else sim_probability
    )
    if "lineup_woba_edge" in win_predictor.feature_names:
        away_active_batters, away_relievers = active_roster_ids(
            game.away_team_id, game.slate_date
        )
        home_active_batters, home_relievers = active_roster_ids(
            game.home_team_id, game.slate_date
        )
        if "away" in announced:
            away_active_batters = tuple(
                dict.fromkeys(
                    (
                        *away_active_batters,
                        *(batter.player_id for batter in announced["away"]),
                    )
                )
            )
        if "home" in announced:
            home_active_batters = tuple(
                dict.fromkeys(
                    (
                        *home_active_batters,
                        *(batter.player_id for batter in announced["home"]),
                    )
                )
            )
        stats["home_win_probability"] = win_predictor.predict_home_probability(
            season=season,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            away_starter_id=lineups["away"].starter.player_id,
            home_starter_id=lineups["home"].starter.player_id,
            prediction_date=date.fromisoformat(game.slate_date),
            away_batter_ids=tuple(
                batter.player_id for batter in lineups["away"].batters
            ),
            home_batter_ids=tuple(
                batter.player_id for batter in lineups["home"].batters
            ),
            away_active_batter_ids=away_active_batters,
            home_active_batter_ids=home_active_batters,
            away_reliever_ids=away_relievers,
            home_reliever_ids=home_relievers,
        )
    else:
        stats["home_win_probability"] = win_predictor.predict_home_probability(
            season=season,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            away_starter_id=lineups["away"].starter.player_id,
            home_starter_id=lineups["home"].starter.player_id,
        )
    return SlatePrediction(
        game=game,
        results=results,
        away_starter=starters["away"],
        home_starter=starters["home"],
        stats=stats,
    )


def render_prediction_card(prediction: SlatePrediction, out_path: Path) -> Path:
    from src.live.game_sim_card import card_data_from_results, render_game_sim_card

    data = card_data_from_results(
        prediction.results,
        away_abbrev=prediction.game.away_abbrev,
        home_abbrev=prediction.game.home_abbrev,
        away_team_id=prediction.game.away_team_id,
        home_team_id=prediction.game.home_team_id,
        away_starter=prediction.away_starter,
        home_starter=prediction.home_starter,
        game_date=prediction.game.slate_date,
        venue=prediction.game.venue,
        home_win_probability=prediction.stats.get("home_win_probability"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_game_sim_card(data, out_path)


def starter_changes(previous: SlateGame, current: SlateGame) -> list[StarterChange]:
    changes: list[StarterChange] = []
    for side in ("away", "home"):
        before = previous.probable_for(side)
        after = current.probable_for(side)
        if (before.player_id, before.full_name or "") == (
            after.player_id,
            after.full_name or "",
        ):
            continue
        changes.append(
            StarterChange(
                side=side,
                previous=before.display_name,
                current=after.display_name,
            )
        )
    return changes


def changed_games(
    previous_games: Mapping[int, SlateGame],
    current_games: Sequence[SlateGame],
) -> list[tuple[SlateGame, list[StarterChange]]]:
    updates: list[tuple[SlateGame, list[StarterChange]]] = []
    for game in current_games:
        previous = previous_games.get(game.game_pk)
        if previous is None:
            continue
        changes = starter_changes(previous, game)
        if changes:
            updates.append((game, changes))
    return updates


def build_daily_board_caption(
    slate_date: str,
    *,
    games_summary: str,
    include_update_note: bool,
) -> str:
    suffix = (
        " Updates will follow if probable starters change."
        if include_update_note
        else ""
    )
    return (
        f"MLB daily sim board for {slate_date}. "
        f"{games_summary} with win odds and projected scores.{suffix}"
    )[:300]


def build_update_caption(
    prediction: SlatePrediction,
    changes: Sequence[StarterChange],
) -> str:
    change_bits = [
        f"{prediction.game.abbrev_for(change.side)} probable starter {change.previous} → {change.current}"
        for change in changes
    ]
    p_home = prediction.stats["home_win_probability"]
    favorite, favorite_prob = (
        (prediction.game.home_abbrev, p_home)
        if p_home >= 0.5
        else (prediction.game.away_abbrev, 1 - p_home)
    )
    text = (
        f"Updated {prediction.game.label}: {'; '.join(change_bits)}. "
        f"New model: {favorite} {favorite_prob:.0%} to win. "
        f"Projected score {prediction.game.away_abbrev} {prediction.stats['mean_away_runs']:.1f} - "
        f"{prediction.game.home_abbrev} {prediction.stats['mean_home_runs']:.1f}."
    )
    return text[:300]


def save_daily_slate_state(path: Path, state: DailySlateState) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    return path


def load_daily_slate_state(path: Path) -> DailySlateState | None:
    if not path.exists():
        return None
    return DailySlateState.from_dict(json.loads(path.read_text()))


def snapshot_state(
    slate_date: str,
    *,
    board_path: Path,
    board_post_id: str | None,
    games: Sequence[SlateGame],
) -> DailySlateState:
    return DailySlateState(
        slate_date=slate_date,
        saved_at=datetime.now(tz=UTC).isoformat(),
        board_path=str(board_path),
        board_post_id=board_post_id,
        games=list(games),
    )
