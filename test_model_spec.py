"""Guards on the portable architecture-spec contract for registered pitch models.

The pitch champions used to be served by unpickling ``mlflow.pytorch``'s
whole-model artifact, which records the defining class's import path. Renaming
``src`` to ``mlb`` therefore broke every registered version at load time and the
live pitch pipeline exited 1 every morning for nine days. These tests pin the
property that prevents a repeat: what serving reads carries no import paths.
"""

from __future__ import annotations

import json
import re
import zipfile

import pytest
import torch

from mlb.ml.mlflow_artifacts import ChampionModelSource, _newest_cached, _spec_paths
from mlb.ml.model_spec import (
    PITCH_LOCATION_BUILDER,
    PITCH_TYPE_BUILDER,
    SPEC_FILENAME,
    STATE_DICT_FILENAME,
    build_model,
    load_model_from_spec,
    pitch_location_spec,
    pitch_type_spec,
)

# No \b prefix: the pickle GLOBAL opcode glues a "c" onto the module name.
REPO_MODULE_REFERENCE = re.compile(rb"mlb\.ml\.[\w.]+")


def _tiny_pitch_type_spec() -> dict:
    return pitch_type_spec(
        n_pitch_types=4,
        n_pitchers=6,
        n_batters=5,
        n_features=3,
        model_type="lstm_attention",
        feature_indices={
            "embedding_indices": {
                "pitcher_idx": 0,
                "batter_idx": 1,
                "prev_pitch_type_idx": 2,
            },
            "continuous_indices": [3, 4, 5],
        },
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        embedding_dim=4,
        n_location_components=2,
        n_attention_heads=2,
        n_attention_layers=1,
    )


def _tiny_location_spec() -> dict:
    return pitch_location_spec(
        n_features=5,
        n_pitch_types=3,
        hidden_dims=[8, 4],
        n_components=2,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    ("spec_factory", "builder"),
    [
        (_tiny_pitch_type_spec, PITCH_TYPE_BUILDER),
        (_tiny_location_spec, PITCH_LOCATION_BUILDER),
    ],
)
def test_a_spec_round_trips_through_json_and_rebuilds_the_same_weights(
    tmp_path, spec_factory, builder
):
    spec = spec_factory()
    assert spec["builder"] == builder

    trained = build_model(spec)
    for parameter in trained.parameters():
        torch.nn.init.normal_(parameter, std=0.5)

    spec_path = tmp_path / SPEC_FILENAME
    state_dict_path = tmp_path / STATE_DICT_FILENAME
    spec_path.write_text(json.dumps(spec))
    torch.save(trained.state_dict(), state_dict_path)

    restored = load_model_from_spec(spec_path, state_dict_path)

    expected = trained.state_dict()
    actual = restored.state_dict()
    assert sorted(actual) == sorted(expected)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert not restored.training


def _pickle_payload(path) -> bytes:
    """``torch.save`` writes a zip; the class references live in its data.pkl."""
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        return archive.read(member)


def test_the_serving_payload_records_no_repository_import_paths(tmp_path):
    """The defect: a whole-model pickle names its class's module, a state_dict does not."""
    model = build_model(_tiny_pitch_type_spec())

    portable = tmp_path / STATE_DICT_FILENAME
    torch.save(model.state_dict(), portable)
    whole_model = tmp_path / "whole_model.pt"
    torch.save(model, whole_model)

    assert REPO_MODULE_REFERENCE.search(_pickle_payload(portable)) is None
    # Control: the payload serving used to read does embed the import path, which
    # is what renaming the package broke. Without this the guard above is vacuous.
    assert REPO_MODULE_REFERENCE.search(_pickle_payload(whole_model)) is not None

    spec_path = tmp_path / SPEC_FILENAME
    spec_path.write_text(json.dumps(_tiny_pitch_type_spec()))
    assert REPO_MODULE_REFERENCE.search(spec_path.read_bytes()) is None


def test_build_model_rejects_an_unknown_builder_token():
    with pytest.raises(ValueError, match="Unknown architecture builder"):
        build_model({"builder": "mlb.ml.model.PitchPredictor", "kwargs": {}})


def _cache_version(root, version: str, *, portable: bool):
    model_root = root / f"v{version}"
    (model_root / "extra_files").mkdir(parents=True)
    (model_root / "MLmodel").write_text("flavors: {}\n")
    if portable:
        (model_root / "extra_files" / SPEC_FILENAME).write_text("{}")
        (model_root / "extra_files" / STATE_DICT_FILENAME).write_bytes(b"")
    return model_root


def test_a_version_without_the_portable_payload_fails_with_a_repair_instruction(tmp_path):
    model_root = _cache_version(tmp_path, "3", portable=False)
    source = ChampionModelSource(
        registered_model_name="mlb-pitch-type-lstm-attention",
        version="3",
        run_id="",
        model_root=model_root,
    )

    with pytest.raises(FileNotFoundError, match="import_pitch_models_to_mlflow"):
        _spec_paths(source)


def test_the_offline_fallback_skips_cached_versions_it_cannot_rebuild(tmp_path, monkeypatch):
    """A registry blip must not hand serving an unloadable pickle-only version."""
    root = tmp_path / "mlb-pitch-type-lstm-attention"
    _cache_version(root, "5", portable=False)
    servable = _cache_version(root, "4", portable=True)
    monkeypatch.setattr("mlb.ml.mlflow_artifacts.PITCH_MODEL_CACHE_ROOT", tmp_path)

    assert _newest_cached("mlb-pitch-type-lstm-attention") == ("4", servable)

    _cache_version(root, "6", portable=True)
    version, _ = _newest_cached("mlb-pitch-type-lstm-attention")
    assert version == "6"


def test_no_cached_version_is_servable_returns_none(tmp_path, monkeypatch):
    root = tmp_path / "mlb-pitch-type-lstm-attention"
    _cache_version(root, "3", portable=False)
    monkeypatch.setattr("mlb.ml.mlflow_artifacts.PITCH_MODEL_CACHE_ROOT", tmp_path)

    assert _newest_cached("mlb-pitch-type-lstm-attention") is None
