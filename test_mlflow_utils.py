from pathlib import Path

from src.ml.mlflow_utils import build_metric_dict, build_param_dict


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
