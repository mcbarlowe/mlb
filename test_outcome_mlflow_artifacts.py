from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlb.outcome.mlflow_artifacts import (
    CachedOutcomeArtifacts,
    OutcomeProductionSelection,
    resolve_outcome_artifact_dirs,
    resolve_production_outcome_selection,
)
from mlb.outcome.mlflow_registry import (
    OUTCOME_CONTRACT_VERSION,
    OUTCOME_MODEL_COLLECTION,
    OUTCOME_RELEASE_TAG,
    OUTCOME_STAGE_SPECS,
    SIM_INPUTS_RUN_TAG,
)
from mlb.sim.artifacts import SIM_INPUT_FILES, ensure_sim_artifacts


class _StubClient:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri

    def get_model_version_by_alias(self, name: str, alias: str):
        assert alias == "champion"
        for stage, spec in OUTCOME_STAGE_SPECS.items():
            if name == spec.registered_model_name:
                return SimpleNamespace(
                    version="7" if stage == "a" else "9",
                    run_id="release-run",
                    status="READY",
                    tags={
                        "contract_version": str(OUTCOME_CONTRACT_VERSION),
                        "model_collection": OUTCOME_MODEL_COLLECTION,
                        "model_family": spec.model_family,
                        "outcome_stage": stage,
                        "promotion_gate": "passed",
                        OUTCOME_RELEASE_TAG: "release-1234567890abcdef",
                        SIM_INPUTS_RUN_TAG: "sim-inputs-run",
                    },
                )
        raise AssertionError(name)


class _MismatchedStubClient(_StubClient):
    def get_model_version_by_alias(self, name: str, alias: str):
        version = super().get_model_version_by_alias(name, alias)
        if name == OUTCOME_STAGE_SPECS["b"].registered_model_name:
            version.tags[OUTCOME_RELEASE_TAG] = "other-release"
        return version


def test_resolve_production_outcome_selection(monkeypatch):
    monkeypatch.setattr("mlb.outcome.mlflow_artifacts.MlflowClient", _StubClient)
    selection = resolve_production_outcome_selection("postgresql://shared")
    assert selection.release_id == "release-1234567890abcdef"
    assert selection.sim_inputs_run_id == "sim-inputs-run"
    assert selection.stage_a_run_id == "release-run"
    assert selection.stage_a_version == "7"
    assert selection.stage_b_run_id == "release-run"
    assert selection.stage_b_version == "9"
    assert selection.cache_key == "release-12345678_a7_b9"


def test_resolve_production_outcome_selection_rejects_mismatched_release(
    monkeypatch,
):
    monkeypatch.setattr(
        "mlb.outcome.mlflow_artifacts.MlflowClient",
        _MismatchedStubClient,
    )
    with pytest.raises(ValueError, match="one paired release"):
        resolve_production_outcome_selection("postgresql://shared")


def test_resolve_outcome_artifact_dirs_prefers_mlflow_cache(
    monkeypatch, tmp_path: Path
):
    cached = CachedOutcomeArtifacts(
        cache_dir=tmp_path / "cache",
        selection=OutcomeProductionSelection(
            release_id="release",
            sim_inputs_run_id="sim",
            stage_a_run_id="run",
            stage_a_version="1",
            stage_b_run_id="run",
            stage_b_version="2",
        ),
    )
    (cached.run_dir).mkdir(parents=True)
    (cached.profiles_dir).mkdir(parents=True)
    monkeypatch.setattr(
        "mlb.outcome.mlflow_artifacts.cache_production_outcome_artifacts",
        lambda tracking_uri, cache_root, selection=None: cached,
    )
    resolved = resolve_outcome_artifact_dirs(
        "auto", tracking_uri="postgresql://shared", cache_root=tmp_path
    )
    assert resolved == (cached.run_dir, cached.profiles_dir)


def test_resolve_outcome_artifact_dirs_falls_back_to_latest_run(
    tmp_path: Path, monkeypatch
):
    base = tmp_path / "models" / "outcome"
    run = base / "run_20260809_065054"
    run.mkdir(parents=True)
    (base / "latest_run.txt").write_text("run_20260809_065054\n")
    monkeypatch.chdir(tmp_path)
    resolved = resolve_outcome_artifact_dirs("auto", tracking_uri=None)
    assert resolved == (run, base)


def test_resolve_outcome_artifact_dirs_handles_explicit_path(tmp_path: Path):
    run = tmp_path / "run_local"
    run.mkdir()
    resolved = resolve_outcome_artifact_dirs(str(run), tracking_uri=None)
    assert resolved == (run, run.parent)


def test_ensure_sim_artifacts_honors_pinned_release(
    monkeypatch,
    tmp_path: Path,
):
    sim_dir = tmp_path / "models" / "sim"
    sim_dir.mkdir(parents=True)
    for name in SIM_INPUT_FILES:
        (sim_dir / name).write_text("stale")

    class StubSimClient:
        def __init__(self, tracking_uri: str | None = None):
            self.tracking_uri = tracking_uri

        def get_run(self, run_id: str):
            assert run_id == "pinned-run"
            return SimpleNamespace(
                info=SimpleNamespace(status="FINISHED"),
                data=SimpleNamespace(tags={"artifact_kind": "sim-inputs"}),
            )

    download_calls: list[str] = []

    def fake_download_artifacts(
        *,
        run_id: str,
        artifact_path: str,
        dst_path: str,
        tracking_uri: str | None,
    ) -> str:
        assert artifact_path == "sim"
        assert tracking_uri == "http://shared"
        download_calls.append(run_id)
        downloaded = Path(dst_path) / "sim"
        downloaded.mkdir()
        for name in SIM_INPUT_FILES:
            (downloaded / name).write_text(f"{run_id}:{name}")
        return str(downloaded)

    monkeypatch.setattr("mlflow.tracking.MlflowClient", StubSimClient)
    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        fake_download_artifacts,
    )

    ensure_sim_artifacts(
        "http://shared",
        sim_dir,
        run_id="pinned-run",
    )
    assert download_calls == ["pinned-run"]
    assert (sim_dir / "pitch_mix.parquet").read_text().startswith("pinned-run:")
    manifest = json.loads((sim_dir / ".mlflow_sim_inputs.json").read_text())
    assert manifest == {"run_id": "pinned-run"}

    ensure_sim_artifacts(
        "http://shared",
        sim_dir,
        run_id="pinned-run",
    )
    assert download_calls == ["pinned-run"]
