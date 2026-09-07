"""Portable architecture specs for the registered pitch prediction models.

``mlflow.pytorch.log_model(model, serialization_format="pickle")`` serializes
the module *object*, so the artifact records the defining class's import path.
That makes every already-registered version depend on the repository layout at
training time: renaming ``src`` to ``mlb`` in ``9f4943c`` made every champion
unloadable with ``ModuleNotFoundError: No module named 'src'``, and the live
pitch pipeline died at 09:00 for nine days before anyone noticed.

A spec here is a plain JSON object naming a stable *builder token* plus the
keyword arguments that reconstruct the module. Paired with a bare
``state_dict`` it carries no import paths at all, so a model registered today
still loads after the defining module is renamed, moved, or split. The builder
tokens are part of the artifact contract: rename a class freely, but never a
token.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import nn

PITCH_TYPE_BUILDER = "pitch_type_sequence"
PITCH_LOCATION_BUILDER = "pitch_type_conditioned_location"

SPEC_FILENAME = "architecture.json"
STATE_DICT_FILENAME = "state_dict.pt"

__all__ = [
    "PITCH_LOCATION_BUILDER",
    "PITCH_TYPE_BUILDER",
    "SPEC_FILENAME",
    "STATE_DICT_FILENAME",
    "build_model",
    "load_model_from_spec",
    "pitch_location_spec",
    "pitch_type_spec",
]


def pitch_type_spec(
    *,
    n_pitch_types: int,
    n_pitchers: int,
    n_batters: int,
    n_features: int,
    model_type: str,
    feature_indices: dict[str, Any],
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    embedding_dim: int,
    n_location_components: int,
    n_attention_heads: int | None = None,
    n_attention_layers: int | None = None,
) -> dict[str, Any]:
    """Spec for a pitch type sequence model built by ``create_model``."""
    kwargs: dict[str, Any] = {
        "n_pitch_types": n_pitch_types,
        "n_pitchers": n_pitchers,
        "n_batters": n_batters,
        "n_features": n_features,
        "model_type": model_type,
        "feature_indices": feature_indices,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "embedding_dim": embedding_dim,
        "n_location_components": n_location_components,
    }
    if n_attention_heads is not None:
        kwargs["n_attention_heads"] = n_attention_heads
    if n_attention_layers is not None:
        kwargs["n_attention_layers"] = n_attention_layers
    return {"builder": PITCH_TYPE_BUILDER, "kwargs": kwargs}


def pitch_location_spec(
    *,
    n_features: int,
    n_pitch_types: int,
    hidden_dims: list[int],
    n_components: int,
    dropout: float,
) -> dict[str, Any]:
    """Spec for a ``PitchTypeConditionedMDN``."""
    return {
        "builder": PITCH_LOCATION_BUILDER,
        "kwargs": {
            "n_features": n_features,
            "n_pitch_types": n_pitch_types,
            "hidden_dims": list(hidden_dims),
            "n_components": n_components,
            "dropout": dropout,
        },
    }


def build_model(spec: Mapping[str, Any]) -> nn.Module:
    """Reconstruct an untrained module from a spec, ready for ``load_state_dict``."""
    builder = spec.get("builder")
    kwargs = spec.get("kwargs")
    if not isinstance(kwargs, dict):
        raise TypeError(f"Architecture spec for {builder!r} has no kwargs object")

    if builder == PITCH_TYPE_BUILDER:
        from mlb.ml.model import create_model

        return create_model(**kwargs)
    if builder == PITCH_LOCATION_BUILDER:
        from mlb.ml.pitch_type_location_model import PitchTypeConditionedMDN

        return PitchTypeConditionedMDN(**kwargs)
    raise ValueError(
        f"Unknown architecture builder {builder!r}; expected one of "
        f"{PITCH_TYPE_BUILDER!r} or {PITCH_LOCATION_BUILDER!r}"
    )


def load_model_from_spec(
    spec_path: Path,
    state_dict_path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Rebuild a trained module from its spec plus a bare ``state_dict``."""
    import torch

    for path in (spec_path, state_dict_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec, dict):
        raise TypeError(f"Expected a JSON object in {spec_path}")
    model = build_model(spec)
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.eval()
    return model.to(device)
