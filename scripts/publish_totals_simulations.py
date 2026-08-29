#!/usr/bin/env python3
"""Publish model-only MLB totals simulation histograms as JSON."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig
from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.contact_environment import ContactEnvironment, GameWeather, parse_weather
from src.sim.db_games import GameDataStore
from src.sim.slate import build_day_ahead_simulator
from src.sim.totals_artifact import (
    TotalsSimulationArtifact,
    game_totals_from_simulations,
    write_totals_artifact,
)

DEFAULT_MODEL_NAME = "mlb-game-simulator-totals"


@dataclass(frozen=True)
class TotalsSimulationRequest:
    season: int
    games: int
    sims_per_game: int
    seed: int
    pa_calibration: Path | None
    mlflow_tracking_uri: str | None
    model_name: str
    model_version: str | None


GameEnvironmentFacts = Mapping[int, tuple[int | None, GameWeather]]
TotalsProducer = Callable[[TotalsSimulationRequest], TotalsSimulationArtifact]


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_game_environment_facts(season: int) -> dict[int, tuple[int | None, GameWeather]]:
    """Load venue and weather facts used by the contact environment."""
    import psycopg
    from psycopg import sql

    config = PostgresConfig.from_env()
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                SELECT game_pk, venue_id, weather_temp, weather_wind,
                       weather_condition
                FROM {schema}.games
                WHERE season::int = %s
                  AND game_type = 'R'
                  AND abstract_game_state = 'Final'
                """
            ).format(schema=sql.Identifier(config.schema)),
            (season,),
        )
        rows = cursor.fetchall()
    return {
        int(game_pk): (
            int(venue_id) if venue_id is not None else None,
            parse_weather(temperature, wind, condition),
        )
        for game_pk, venue_id, temperature, wind, condition in rows
    }


def _environment_for_game(
    game_pk: int,
    *,
    contact_environment: ContactEnvironment | None,
    environment_facts: GameEnvironmentFacts,
) -> dict[str, float] | None:
    if contact_environment is None:
        return None
    venue_id, weather = environment_facts.get(game_pk, (None, None))
    return contact_environment.multipliers(venue_id, weather)


def _resolved_model_version(
    requested_version: str | None,
    outcome_run_dir: Path,
) -> str:
    if requested_version is not None:
        return requested_version
    return outcome_run_dir.name


def produce_totals_simulations(
    request: TotalsSimulationRequest,
) -> TotalsSimulationArtifact:
    store = GameDataStore.load(request.season)
    simulator, outcome_run_dir = build_day_ahead_simulator(
        season=request.season,
        seed=request.seed,
        tracking_uri=request.mlflow_tracking_uri,
        pa_calibration_path=request.pa_calibration,
    )
    contact_environment = ContactEnvironment.load(request.season)
    environment_facts = (
        load_game_environment_facts(request.season) if contact_environment else {}
    )

    candidates = store.final_game_pks(
        seed=request.seed,
        limit=max(request.games, 10_000),
    )
    games = []
    for game_pk in candidates:
        if len(games) >= request.games:
            break
        try:
            away = store.lineup(game_pk, "away", individual_bullpen=True)
            home = store.lineup(game_pk, "home", individual_bullpen=True)
        except (KeyError, ValueError):
            continue
        environment = _environment_for_game(
            game_pk,
            contact_environment=contact_environment,
            environment_facts=environment_facts,
        )
        results = simulator.simulate_many(
            away,
            home,
            request.sims_per_game,
            environment=environment,
        )
        games.append(
            game_totals_from_simulations(
                game_pk,
                (result.away_runs + result.home_runs for result in results),
            )
        )
        print(
            f"simulated game_pk={game_pk} "
            f"sims={request.sims_per_game} "
            f"mean_total={games[-1].sim_mean_total:.2f}",
            flush=True,
        )

    if len(games) < request.games:
        raise RuntimeError(
            f"Only simulated {len(games)} usable games; requested {request.games}"
        )

    return TotalsSimulationArtifact(
        generated_at=datetime.now(UTC).isoformat(),
        season=request.season,
        sims_per_game=request.sims_per_game,
        model_name=request.model_name,
        model_version=_resolved_model_version(request.model_version, outcome_run_dir),
        games=tuple(games),
    )


def publish_totals_simulations(
    request: TotalsSimulationRequest,
    output_json: Path,
    *,
    producer: TotalsProducer = produce_totals_simulations,
) -> TotalsSimulationArtifact:
    artifact = producer(request)
    write_totals_artifact(output_json, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=_positive_int_arg, required=True)
    parser.add_argument("--games", type=_positive_int_arg, required=True)
    parser.add_argument("--sims", dest="sims_per_game", type=_positive_int_arg, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pa-calibration", type=Path, default=None)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking server used to resolve the simulator model inputs.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-version",
        default=None,
        help="Artifact model version label; defaults to the resolved outcome run.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    producer: TotalsProducer = produce_totals_simulations,
) -> TotalsSimulationArtifact:
    args = parse_args(argv)
    request = TotalsSimulationRequest(
        season=args.season,
        games=args.games,
        sims_per_game=args.sims_per_game,
        seed=args.seed,
        pa_calibration=args.pa_calibration,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        model_name=args.model_name,
        model_version=args.model_version,
    )
    artifact = publish_totals_simulations(
        request,
        args.output_json,
        producer=producer,
    )
    print(
        f"published {len(artifact.games)} totals simulations for season "
        f"{artifact.season} to {args.output_json}"
    )
    return artifact


if __name__ == "__main__":
    main()
