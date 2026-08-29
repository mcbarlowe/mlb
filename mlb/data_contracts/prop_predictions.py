"""Versioned, odds-free MLB batter-prop probability production.

This module owns historical-stat loading and the calibrated rate estimator that
previously lived inside ``scripts/shop_batter_props.py``.  It deliberately has
no sportsbook, price, EV, ledger, notification, or reporting dependencies.
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import psycopg

from mlb.database import PostgresConfig

CONTRACT_VERSION = "v1"
MODEL_NAME = "mlb-prop-rate-estimator"
MODEL_VERSION = "prop-rate-cond-v3"
LEGACY_MODEL_VERSION_TAGS = {"props-cond-v3": MODEL_VERSION}
RECENT_SEASONS = 3
START_PA = 3
EXP_PA_WINDOW = 30
DEFAULT_SHRINK_K = 50.0
DEFAULT_RECENCY_HALF_LIFE = 400.0
ET = ZoneInfo("America/New_York")

STAT_FNS: dict[str, Callable[[Mapping[str, int]], int]] = {
    "batter_home_runs": lambda row: row["homeruns"],
    "batter_hits": lambda row: row["hits"],
    "batter_total_bases": lambda row: row["totalbases"],
    "batter_rbis": lambda row: row["rbi"],
    "batter_runs_scored": lambda row: row["runs"],
    "batter_walks": lambda row: row["baseonballs"],
    "batter_stolen_bases": lambda row: row["stolenbases"],
    "batter_strikeouts": lambda row: row["strikeouts"],
    "batter_doubles": lambda row: row["doubles"],
    "batter_singles": lambda row: (
        row["hits"] - row["doubles"] - row["triples"] - row["homeruns"]
    ),
    "batter_hits_runs_rbis": lambda row: row["hits"] + row["runs"] + row["rbi"],
}
STAT_COLUMNS = (
    "hits",
    "doubles",
    "triples",
    "homeruns",
    "totalbases",
    "rbi",
    "runs",
    "baseonballs",
    "stolenbases",
    "strikeouts",
)
CONDITIONED_MARKETS = frozenset(STAT_FNS) - {"batter_home_runs"}
REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "event_id",
        "game_pk",
        "game_time",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "player",
        "player_id",
        "market",
        "point",
    }
)


class PlayerLines(NamedTuple):
    """Chronological per-game rows and a player's age on prediction day."""

    lines: list[tuple[int, float, int, dict[str, int]]]
    age_now: float


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    letters = "".join(character for character in text.lower() if character.isalpha() or character.isspace())
    return " ".join(letters.split())


def validate_request(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate and normalize the versioned, price-free batch request."""

    version = payload.get("contract_version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise ValueError(f"Unsupported prop prediction contract {version!r}")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("Prop prediction request must contain a requests list")

    requests: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, Mapping):
            raise ValueError(f"requests[{index}] must be an object")
        unknown = set(raw) - REQUEST_FIELDS
        if unknown:
            raise ValueError(
                f"requests[{index}] contains non-model fields: {sorted(unknown)!r}"
            )
        request = dict(raw)
        for field in ("player", "market", "point"):
            if request.get(field) in (None, ""):
                raise ValueError(f"requests[{index}] is missing {field!r}")
        market = str(request["market"])
        if market not in STAT_FNS:
            raise ValueError(f"Unsupported prop market {market!r}")
        try:
            point = float(request["point"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"requests[{index}].point must be numeric") from exc
        if not math.isfinite(point):
            raise ValueError(f"requests[{index}].point must be finite")
        request["player"] = str(request["player"])
        request["market"] = market
        request["point"] = point
        request_id = str(request.get("request_id") or index)
        if request_id in seen_ids:
            raise ValueError(f"Duplicate request_id {request_id!r}")
        seen_ids.add(request_id)
        request["request_id"] = request_id
        requests.append(request)
    return requests


def load_game_lines(
    requests: Sequence[Mapping[str, object]], prediction_date: date
) -> dict[str, PlayerLines]:
    """Load each requested player's prior regular-season batting history."""

    config = PostgresConfig.from_env()
    connection = psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=10,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT player_id, full_name FROM {config.schema}.players")
            by_norm: dict[str, list[int]] = {}
            for player_id, full_name in cursor.fetchall():
                by_norm.setdefault(normalize_name(str(full_name)), []).append(int(player_id))

            player_id_to_norm: dict[int, str] = {}
            for request in requests:
                normalized = normalize_name(str(request["player"]))
                explicit_id = request.get("player_id")
                ids = [int(explicit_id)] if explicit_id not in (None, "") else by_norm.get(normalized, [])
                for player_id in ids:
                    player_id_to_norm[player_id] = normalized
            if not player_id_to_norm:
                return {}

            columns = ", ".join(
                f"COALESCE(b.{column}, 0)::int AS {column}"
                for column in STAT_COLUMNS
            )
            cursor.execute(
                f"""
                SELECT b.player_id, g.season::int AS season,
                       g.game_date::date AS game_date, p.birth_date::date AS birth_date,
                       COALESCE(b.plateappearances, 0)::int AS pa,
                       {columns}
                FROM {config.schema}.batting b
                JOIN {config.schema}.games g USING (game_pk)
                LEFT JOIN {config.schema}.players p USING (player_id)
                WHERE b.player_id = ANY(%s)
                  AND g.game_type = 'R'
                  AND g.abstract_game_state = 'Final'
                  AND g.season::int BETWEEN %s AND %s
                  AND g.game_date::date < %s
                  AND COALESCE(b.plateappearances, 0) > 0
                ORDER BY COALESCE(g.game_datetime, g.game_date), g.game_pk
                """,
                (
                    list(player_id_to_norm),
                    prediction_date.year - RECENT_SEASONS + 1,
                    prediction_date.year,
                    prediction_date,
                ),
            )
            column_names = [description.name for description in cursor.description]
            per_player: dict[int, list[tuple[int, float, int, dict[str, int]]]] = {}
            births: dict[int, date | None] = {}
            for row in cursor.fetchall():
                record = dict(zip(column_names, row, strict=True))
                player_id = int(record["player_id"])
                birth_date = record["birth_date"]
                births.setdefault(player_id, birth_date)
                age = (
                    (record["game_date"] - birth_date).days / 365.25
                    if birth_date is not None
                    else float("nan")
                )
                per_player.setdefault(player_id, []).append(
                    (
                        int(record["season"]),
                        age,
                        int(record["pa"]),
                        {column: int(record[column]) for column in STAT_COLUMNS},
                    )
                )
    finally:
        connection.close()

    best_for_name: dict[str, int] = {}
    for player_id, normalized in player_id_to_norm.items():
        incumbent = best_for_name.get(normalized)
        if incumbent is None or len(per_player.get(player_id, [])) > len(
            per_player.get(incumbent, [])
        ):
            best_for_name[normalized] = player_id

    result: dict[str, PlayerLines] = {}
    for normalized, player_id in best_for_name.items():
        birth_date = births.get(player_id)
        result[normalized] = PlayerLines(
            lines=per_player.get(player_id, []),
            age_now=(
                (prediction_date - birth_date).days / 365.25
                if birth_date is not None
                else float("nan")
            ),
        )
    return result


def load_team_ids() -> dict[str, int]:
    """Return MLB full team name to stable MLB team id."""

    config = PostgresConfig.from_env()
    connection = psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=10,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT team_name, team_id FROM {config.schema}.teams WHERE sport_id = 1"
            )
            return {str(name): int(team_id) for name, team_id in cursor.fetchall()}
    finally:
        connection.close()


def load_game_ids(
    requests: Sequence[Mapping[str, object]],
    prediction_date: date,
    *,
    max_match_hours: float = 12.0,
) -> dict[str, int]:
    """Resolve caller-supplied event metadata to stable MLB game IDs, including twin bills."""
    config = PostgresConfig.from_env()
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=10,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT g.game_pk, g.game_datetime, g.home_team_id, g.away_team_id,
                   home.team_name, away.team_name
            FROM {config.schema}.games AS g
            JOIN {config.schema}.teams AS home ON home.team_id = g.home_team_id
            JOIN {config.schema}.teams AS away ON away.team_id = g.away_team_id
            WHERE g.game_date::date = %s
            """,
            (prediction_date,),
        )
        games = cursor.fetchall()
    resolved: dict[str, int] = {}
    for request in requests:
        request_id = str(request["request_id"])
        explicit = request.get("game_pk")
        if explicit not in (None, ""):
            resolved[request_id] = int(explicit)
            continue
        home_id = request.get("home_team_id")
        away_id = request.get("away_team_id")
        home_name = normalize_name(str(request.get("home_team") or ""))
        away_name = normalize_name(str(request.get("away_team") or ""))
        candidates = [
            row
            for row in games
            if (
                home_id not in (None, "")
                and away_id not in (None, "")
                and int(row[2]) == int(home_id)
                and int(row[3]) == int(away_id)
            )
            or (
                home_id in (None, "")
                and away_id in (None, "")
                and normalize_name(str(row[4])) == home_name
                and normalize_name(str(row[5])) == away_name
            )
        ]
        game_time = request.get("game_time")
        if not candidates or game_time in (None, ""):
            if len(candidates) == 1:
                resolved[request_id] = int(candidates[0][0])
            continue
        requested_time = datetime.fromisoformat(str(game_time))
        if requested_time.tzinfo is None:
            requested_time = requested_time.replace(tzinfo=UTC)
        requested_time = requested_time.astimezone(UTC)
        timed = [row for row in candidates if row[1] is not None]
        if not timed:
            continue
        closest = min(
            timed,
            key=lambda row: abs(
                (row[1].astimezone(UTC) - requested_time).total_seconds()
            ),
        )
        difference = abs(
            (closest[1].astimezone(UTC) - requested_time).total_seconds()
        )
        if difference <= max_match_hours * 3600:
            resolved[request_id] = int(closest[0])
    return resolved


def rate_over(
    lines: Sequence[tuple[int, float, int, Mapping[str, int]]],
    stat_fn: Callable[[Mapping[str, int]], int],
    point: float,
    season: int | None,
    last_n: int | None = None,
) -> tuple[float, int]:
    rows = [stats for year, _age, _pa, stats in lines if season is None or year == season]
    if last_n:
        rows = rows[-last_n:]
    if not rows:
        return float("nan"), 0
    successes = sum(1 for stats in rows if stat_fn(stats) > point)
    return successes / len(rows), len(rows)


def decayed_over(
    lines: Sequence[tuple[int, float, int, Mapping[str, int]]],
    stat_fn: Callable[[Mapping[str, int]], int],
    point: float,
    decay: float,
) -> tuple[float, float, float, float]:
    successes = weight = weighted_age = weighted_pa = 0.0
    has_age = False
    for _year, age, plate_appearances, stats in lines:
        successes = decay * successes + (1.0 if stat_fn(stats) > point else 0.0)
        weight = decay * weight + 1.0
        weighted_age = decay * weighted_age + (0.0 if math.isnan(age) else age)
        weighted_pa = decay * weighted_pa + plate_appearances
        has_age = has_age or not math.isnan(age)
    if weight <= 0.0:
        return 0.0, 0.0, float("nan"), float("nan")
    return (
        successes,
        weight,
        weighted_age / weight if has_age else float("nan"),
        weighted_pa / weight,
    )


def expected_pa(lines: Sequence[tuple[int, float, int, Mapping[str, int]]]) -> float:
    recent = [plate_appearances for _year, _age, plate_appearances, _stats in lines[-EXP_PA_WINDOW:]]
    return sum(recent) / len(recent) if recent else float("nan")


def curve_at(curve: Mapping[int, float], age: float) -> float:
    ages = sorted(curve)
    bounded_age = min(max(age, ages[0]), ages[-1])
    lower = math.floor(bounded_age)
    upper = min(lower + 1, ages[-1])
    fraction = bounded_age - lower
    return curve[lower] * (1.0 - fraction) + curve[upper] * fraction


def load_aging_curves(path: Path) -> dict[str, dict[int, float]]:
    return _load_adjustments(path, collection="curves")


def load_park_factors(path: Path) -> dict[str, dict[int, float]]:
    return _load_adjustments(path, collection="factors")


def _load_adjustments(path: Path, *, collection: str) -> dict[str, dict[int, float]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    raw = payload.get(collection, {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): {int(member): float(value) for member, value in values.items()}
        for key, values in raw.items()
        if isinstance(values, Mapping)
    }


def default_tracking_uri() -> str:
    configured = os.environ.get("MLFLOW_TRACKING_URI", "")
    return configured if configured.startswith(("http://", "https://")) else "http://10.0.0.171:5001"


def resolve_registry_artifacts(
    tracking_uri: str, *, repo: Path
) -> tuple[Path | None, Path | None, str]:
    """Resolve matching champion artifacts with the legacy offline cache fallback."""
    cache_root = repo / "models/mlflow_cache/prop_rate_estimator"
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        from mlflow.artifacts import download_artifacts
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        model_version = version.tags.get("model_contract_version")
        if model_version is None:
            model_version = LEGACY_MODEL_VERSION_TAGS.get(
                version.tags.get("strategy_version", "")
            )
        if model_version != MODEL_VERSION:
            return None, None, (
                f"incompatible champion v{version.version}: {model_version}"
            )
        cache = cache_root / f"v{version.version}"
        curves = cache / "estimator/aging_curves.json"
        parks = cache / "estimator/park_factors.json"
        if not (curves.exists() and parks.exists()):
            cache.mkdir(parents=True, exist_ok=True)
            download_artifacts(
                run_id=version.run_id,
                artifact_path="estimator",
                dst_path=str(cache),
                tracking_uri=tracking_uri,
            )
        return (
            curves if curves.exists() else None,
            parks if parks.exists() else None,
            f"mlflow champion v{version.version} ({model_version})",
        )
    except Exception as exc:
        if cache_root.exists():
            cache_directories = sorted(
                cache_root.glob("v*"),
                key=lambda path: int(path.name[1:]),
                reverse=True,
            )
            for cache in cache_directories:
                curves = cache / "estimator/aging_curves.json"
                parks = cache / "estimator/park_factors.json"
                if curves.exists() and parks.exists():
                    return curves, parks, f"mlflow cache {cache.name}"
        return None, None, f"registry unavailable: {type(exc).__name__}"


def resolve_adjustments(
    *,
    repo: Path,
    model_source: str = "registry",
    tracking_uri: str | None = None,
    aging_curves_path: Path | None = None,
    park_factors_path: Path | None = None,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]], str]:
    curves_path = aging_curves_path or repo / "models/props/aging_curves.json"
    parks_path = park_factors_path or repo / "models/props/park_factors.json"
    provenance = "repo-local artifacts"
    if model_source == "registry" and aging_curves_path is None and park_factors_path is None:
        registry_curves, registry_parks, provenance = resolve_registry_artifacts(
            tracking_uri or default_tracking_uri(), repo=repo
        )
        if registry_curves is not None:
            curves_path = registry_curves
        if registry_parks is not None:
            parks_path = registry_parks
    elif model_source != "local":
        raise ValueError(f"Unknown model source {model_source!r}")
    return load_aging_curves(curves_path), load_park_factors(parks_path), provenance


def build_prop_prediction_artifact(
    request_payload: Mapping[str, object],
    *,
    prediction_date: date,
    predicted_at: datetime | None = None,
    lines_by_name: Mapping[str, PlayerLines] | None = None,
    aging_curves: Mapping[str, Mapping[int, float]] | None = None,
    park_factors: Mapping[str, Mapping[int, float]] | None = None,
    team_ids: Mapping[str, int] | None = None,
    game_ids: Mapping[str, int] | None = None,
    shrink_k: float = DEFAULT_SHRINK_K,
    recency_half_life: float = DEFAULT_RECENCY_HALF_LIFE,
    model_provenance: str = "repo-local artifacts",
) -> dict[str, object]:
    """Score every requested player/market/point without observing any price."""

    requests = validate_request(request_payload)
    if shrink_k < 0:
        raise ValueError("shrink_k cannot be negative")
    timestamp = (predicted_at or datetime.now(UTC)).astimezone(UTC)
    histories = dict(lines_by_name) if lines_by_name is not None else load_game_lines(requests, prediction_date)
    curves = aging_curves or {}
    parks = park_factors or {}
    resolved_team_ids = dict(team_ids) if team_ids is not None else load_team_ids()
    resolved_game_ids = (
        dict(game_ids)
        if game_ids is not None
        else (
            load_game_ids(requests, prediction_date)
            if lines_by_name is None
            else {}
        )
    )
    starts = {
        name: PlayerLines(
            [line for line in entry.lines if line[2] >= START_PA], entry.age_now
        )
        for name, entry in histories.items()
    }
    all_pool = [stats for entry in histories.values() for _year, _age, _pa, stats in entry.lines]
    starts_pool = [stats for entry in starts.values() for _year, _age, _pa, stats in entry.lines]
    league_rates: dict[tuple[str, float], float] = {}
    for market, point in {
        (str(request["market"]), float(request["point"])) for request in requests
    }:
        stat_fn = STAT_FNS[market]
        pool = starts_pool if market in CONDITIONED_MARKETS else all_pool
        if pool:
            league_rates[(market, point)] = sum(
                1 for stats in pool if stat_fn(stats) > point
            ) / len(pool)

    decay = 1.0 if recency_half_life <= 0 else 0.5 ** (1.0 / recency_half_life)
    predictions: list[dict[str, object]] = []
    for request in requests:
        market = str(request["market"])
        point = float(request["point"])
        conditioned = market in CONDITIONED_MARKETS
        history = (starts if conditioned else histories).get(
            normalize_name(str(request["player"]))
        )
        lines = history.lines if history else []
        age_now = history.age_now if history else float("nan")
        probability: float | None = None
        sample_games = len(lines)
        league_rate = league_rates.get((market, point))
        if sample_games and league_rate is not None:
            successes, weight, mean_age, mean_pa = decayed_over(
                lines, STAT_FNS[market], point, decay
            )
            smoothed = (successes + 0.5) / (weight + 1.0)
            curve = curves.get(f"{market}|{point:g}")
            age_delta = 0.0
            if curve and not math.isnan(age_now) and not math.isnan(mean_age):
                age_delta = curve_at(curve, age_now) - curve_at(curve, mean_age)
                age_delta = max(-0.75, min(0.75, age_delta))
            aged = 1.0 / (
                1.0
                + math.exp(
                    -(math.log(smoothed / (1.0 - smoothed)) + age_delta)
                )
            )
            probability = (
                (weight * aged + shrink_k * league_rate) / (weight + shrink_k)
                if shrink_k > 0
                else aged
            )
            if conditioned:
                projected_pa = expected_pa(lines)
                if point == 0.5 and not math.isnan(projected_pa) and mean_pa >= 1.0:
                    per_pa = 1.0 - (1.0 - probability) ** (1.0 / mean_pa)
                    probability = 1.0 - (1.0 - per_pa) ** min(
                        max(projected_pa, 1.0), 6.0
                    )
                park_map = parks.get(f"{market}|{point:g}")
                home_team_id = request.get("home_team_id")
                if home_team_id in (None, ""):
                    home_team_id = resolved_team_ids.get(str(request.get("home_team", "")))
                if park_map and home_team_id not in (None, ""):
                    park_offset = park_map.get(int(home_team_id))
                    if park_offset is not None:
                        clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
                        probability = 1.0 / (
                            1.0
                            + math.exp(
                                -(
                                    math.log(clipped / (1.0 - clipped))
                                    + park_offset
                                )
                            )
                        )
        prediction = dict(request)
        game_pk = request.get("game_pk") or resolved_game_ids.get(
            str(request["request_id"])
        )
        if game_pk not in (None, ""):
            prediction["game_pk"] = int(game_pk)
        prediction.update(
            {
                "probability_over": probability,
                "sample_games": sample_games,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "predicted_at": timestamp.isoformat().replace("+00:00", "Z"),
            }
        )
        predictions.append(prediction)

    return {
        "contract_version": CONTRACT_VERSION,
        "prediction_date": prediction_date.isoformat(),
        "predicted_at": timestamp.isoformat().replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_provenance": model_provenance,
        "predictions": predictions,
    }


__all__ = [
    "CONDITIONED_MARKETS",
    "CONTRACT_VERSION",
    "DEFAULT_RECENCY_HALF_LIFE",
    "DEFAULT_SHRINK_K",
    "MODEL_NAME",
    "MODEL_VERSION",
    "STAT_FNS",
    "PlayerLines",
    "build_prop_prediction_artifact",
    "load_game_ids",
    "normalize_name",
    "resolve_adjustments",
    "validate_request",
]
