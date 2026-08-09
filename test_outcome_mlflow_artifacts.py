from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.outcome.mlflow_artifacts import (
    CachedOutcomeArtifacts,
    OutcomeProductionSelection,
    resolve_outcome_artifact_dirs,
    resolve_production_outcome_selection,
)


class _StubRun:
    def __init__(self, run_id: str):
        self.info = SimpleNamespace(run_id=run_id)


class _StubClient:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, name: str):
        if name != "mlb-model-training":
            return None
        return SimpleNamespace(experiment_id="1", name=name)

    def search_runs(self, *, experiment_ids, filter_string, order_by, max_results):
        assert experiment_ids == ["1"]
        assert order_by == ["attributes.start_time DESC"]
        assert max_results == 1
        if "pitch_result_stage_a" in filter_string:
            return [_StubRun("run-stage-a")]
        if "in_play_event_stage_b" in filter_string:
            return [_StubRun("run-stage-b")]
        return []


def test_resolve_production_outcome_selection(monkeypatch):
    monkeypatch.setattr("src.outcome.mlflow_artifacts.MlflowClient", _StubClient)
    selection = resolve_production_outcome_selection("postgresql://shared")
    assert selection.experiment_id == "1"
    assert selection.stage_a_run_id == "run-stage-a"
    assert selection.stage_b_run_id == "run-stage-b"
    assert selection.cache_key == "run-stage-a_run-stage-b"


def test_resolve_outcome_artifact_dirs_prefers_mlflow_cache(monkeypatch, tmp_path: Path):
    cached = CachedOutcomeArtifacts(
        cache_dir=tmp_path / "cache",
        selection=OutcomeProductionSelection(
            experiment_id="1",
            experiment_name="mlb-model-training",
            stage_a_run_id="a",
            stage_b_run_id="b",
        ),
    )
    (cached.run_dir).mkdir(parents=True)
    (cached.profiles_dir).mkdir(parents=True)
    monkeypatch.setattr(
        "src.outcome.mlflow_artifacts.cache_production_outcome_artifacts",
        lambda tracking_uri, cache_root: cached,
    )
    resolved = resolve_outcome_artifact_dirs(
        "auto", tracking_uri="postgresql://shared", cache_root=tmp_path
    )
    assert resolved == (cached.run_dir, cached.profiles_dir)


def test_resolve_outcome_artifact_dirs_falls_back_to_latest_run(tmp_path: Path, monkeypatch):
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
