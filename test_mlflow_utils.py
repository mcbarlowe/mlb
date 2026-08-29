from pathlib import Path

from mlb.ml.mlflow_utils import (
    DEFAULT_MLFLOW_TRACKING_URI,
    build_metric_dict,
    build_param_dict,
    resolve_mlflow_tracking_uri,
)


def test_build_param_dict_flattens_nested_values():
    params = build_param_dict(
        {
            "alpha": 1,
            "nested": {"beta": 2.5, "path": Path("models")},
            "flags": {"enabled": True},
            "items": ["a", "b"],
            "skip": None,
        }
    )

    assert params == {
        "alpha": 1,
        "nested.beta": 2.5,
        "nested.path": "models",
        "flags.enabled": True,
        "items": '["a", "b"]',
    }



def test_build_metric_dict_keeps_numeric_scalars_only():
    metrics = build_metric_dict(
        {
            "accuracy": 0.9,
            "nested": {"loss": 1.2},
            "enabled": True,
            "labels": [1, 2, 3],
        }
    )

    assert metrics == {
        "accuracy": 0.9,
        "nested.loss": 1.2,
    }


def test_resolve_mlflow_tracking_uri_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-server:5001")
    assert resolve_mlflow_tracking_uri("http://explicit:5001") == "http://explicit:5001"


def test_resolve_mlflow_tracking_uri_uses_shared_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_SHARED_TRACKING_URI", raising=False)
    assert resolve_mlflow_tracking_uri() == DEFAULT_MLFLOW_TRACKING_URI
