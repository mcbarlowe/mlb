"""Resolve and cache champion pitch prediction models from MLflow.

The registry ``champion`` alias is the single serving lever for the live
pitch prediction pipeline, mirroring ``src/outcome/mlflow_artifacts.py``
for the simulation stack: promote a version in MLflow and every consumer
picks it up at its next launch, no file moves.

Champion versions are downloaded once into a version-keyed cache under
``models/mlflow_cache/pitch``. If the tracking server is unreachable the
newest cached champion serves as a loud fallback so a scheduled launch
never dies on a registry blip.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlb.ml.mlflow_registry import PITCH_LOCATION_SPEC, PITCH_TYPE_SPEC

if TYPE_CHECKING:
    from mlb.ml.pitch_predictor import PitchPredictor
    from mlb.ml.pitch_type_location_model import PitchTypeConditionedMDN

PITCH_MODEL_CACHE_ROOT = Path("models/mlflow_cache/pitch")
CHAMPION_ALIAS = "champion"

# Registered pitch models are pickle-serialized torch modules.
os.environ.setdefault("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")


@dataclass(frozen=True)
class ChampionModelSource:
    registered_model_name: str
    version: str
    run_id: str
    model_root: Path
    from_cache_fallback: bool = False

    def describe(self) -> str:
        suffix = " (stale cache fallback)" if self.from_cache_fallback else ""
        return f"{self.registered_model_name} v{self.version}{suffix}"


def _find_model_root(directory: Path) -> Path | None:
    """Locate the MLflow model root (dir containing MLmodel) under a cache dir."""
    if (directory / "MLmodel").is_file():
        return directory
    candidates = sorted(directory.rglob("MLmodel"))
    return candidates[0].parent if candidates else None


def _newest_cached(name: str) -> tuple[str, Path] | None:
    root = PITCH_MODEL_CACHE_ROOT / name
    if not root.is_dir():
        return None
    best: tuple[int, str, Path] | None = None
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("v"):
            continue
        model_root = _find_model_root(entry)
        if model_root is None:
            continue
        try:
            ordinal = int(entry.name[1:])
        except ValueError:
            continue
        if best is None or ordinal > best[0]:
            best = (ordinal, entry.name[1:], model_root)
    return (best[1], best[2]) if best else None


def resolve_champion_artifacts(
    name: str,
    tracking_uri: str | None = None,
) -> ChampionModelSource:
    """Resolve a champion version and materialize its artifacts locally."""
    try:
        from mlflow.artifacts import download_artifacts
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        cache_dir = PITCH_MODEL_CACHE_ROOT / name / f"v{version.version}"
        model_root = _find_model_root(cache_dir) if cache_dir.is_dir() else None
        if model_root is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            downloaded = Path(
                download_artifacts(
                    artifact_uri=f"models:/{name}/{version.version}",
                    dst_path=str(cache_dir),
                    tracking_uri=tracking_uri,
                )
            )
            model_root = _find_model_root(downloaded) or _find_model_root(cache_dir)
        if model_root is None:
            raise RuntimeError(f"Downloaded artifacts for {name} contain no MLmodel")
        return ChampionModelSource(
            registered_model_name=name,
            version=str(version.version),
            run_id=version.run_id or "",
            model_root=model_root,
        )
    except Exception as exc:
        cached = _newest_cached(name)
        if cached is None:
            raise RuntimeError(
                f"Cannot resolve {name}@{CHAMPION_ALIAS} and no cached version "
                f"exists under {PITCH_MODEL_CACHE_ROOT / name}: {exc}"
            ) from exc
        version, model_root = cached
        print(
            f"[mlflow] {name}@{CHAMPION_ALIAS} unavailable ({exc}); "
            f"serving cached v{version}"
        )
        return ChampionModelSource(
            registered_model_name=name,
            version=version,
            run_id="",
            model_root=model_root,
            from_cache_fallback=True,
        )


def load_champion_pitch_type_predictor(
    device: str = "cpu",
    tracking_uri: str | None = None,
) -> tuple[PitchPredictor, ChampionModelSource]:
    """Load the champion pitch type model as a ready PitchPredictor."""
    from mlflow.pytorch import load_model

    from mlb.ml.features import PitchFeatureEngine
    from mlb.ml.pitch_predictor import PitchPredictor

    source = resolve_champion_artifacts(
        PITCH_TYPE_SPEC.registered_model_name, tracking_uri
    )
    model = load_model(str(source.model_root), map_location="cpu")
    engine_path = source.model_root / "extra_files" / "feature_engine.json"
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"{source.describe()} has no extra_files/feature_engine.json"
        )
    engine = PitchFeatureEngine.load(engine_path)
    predictor = PitchPredictor(
        lstm_model=model,
        feature_engine=engine,
        feature_columns=engine.get_feature_columns(),
        pitcher_to_idx=engine.pitcher_to_idx,
        batter_to_idx=engine.batter_to_idx,
        device=device,
        model_type="lstm",
    )
    return predictor, source


def load_champion_location_model(
    device: str = "cpu",
    tracking_uri: str | None = None,
) -> tuple[PitchTypeConditionedMDN, list[str], ChampionModelSource]:
    """Load the champion location model plus its feature-column contract."""
    from mlflow.pytorch import load_model

    from mlb.ml.pitch_type_location_model import PitchTypeConditionedMDN

    source = resolve_champion_artifacts(
        PITCH_LOCATION_SPEC.registered_model_name, tracking_uri
    )
    model = load_model(str(source.model_root), map_location="cpu")
    if not isinstance(model, PitchTypeConditionedMDN):
        raise TypeError(
            f"{source.describe()} is not a PitchTypeConditionedMDN "
            f"(got {type(model).__name__})"
        )
    model = model.to(device)
    model.eval()
    config_path = source.model_root / "extra_files" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{source.describe()} has no extra_files/config.json")
    config = json.loads(config_path.read_text())
    feature_columns = [
        column
        for column in config.get("feature_columns", [])
        if column != "pitch_type_idx"
    ]
    if not feature_columns:
        raise ValueError(f"{source.describe()} has an empty feature contract")
    return model, feature_columns, source
