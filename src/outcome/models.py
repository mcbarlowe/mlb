"""CatBoost training and evaluation for the pitch outcome models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.outcome.dataset import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def _to_catboost_frame(frame: pl.DataFrame, label_column: str):
    """Feature matrix (pandas) + labels for CatBoost consumption."""
    features = frame.select(FEATURE_COLUMNS).to_pandas()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("unknown").astype(str)
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
    features[numeric_columns] = features[numeric_columns].astype("float64")
    labels = frame.get_column(label_column).to_list()
    return features, labels


def train_outcome_model(
    train: pl.DataFrame,
    val: pl.DataFrame,
    label_column: str,
    iterations: int = 1000,
    depth: int = 8,
    learning_rate: float | None = None,
    random_seed: int = 42,
):
    """Train a multiclass CatBoost model with early stopping on ``val``."""
    from catboost import CatBoostClassifier, Pool

    train_features, train_labels = _to_catboost_frame(train, label_column)
    val_features, val_labels = _to_catboost_frame(val, label_column)

    train_pool = Pool(train_features, train_labels, cat_features=CATEGORICAL_FEATURES)
    val_pool = Pool(val_features, val_labels, cat_features=CATEGORICAL_FEATURES)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=random_seed,
        early_stopping_rounds=50,
        verbose=100,
        allow_writing_files=False,
        # One-hot every categorical (max cardinality here is 11 pitch types):
        # CTR target statistics are banned — their ordered-boosting train/apply
        # asymmetry systematically shifted predicted marginals off-window.
        one_hot_max_size=16,
    )
    model.fit(train_pool, eval_set=val_pool)
    return model


def evaluate_model(model, frame: pl.DataFrame, label_column: str) -> dict:
    """Log loss / accuracy for a trained model on a labeled frame."""
    from sklearn.metrics import accuracy_score, log_loss

    features, labels = _to_catboost_frame(frame, label_column)
    probabilities = model.predict_proba(features)
    classes = [str(c) for c in model.classes_]
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "accuracy": float(
            accuracy_score(labels, np.asarray(classes)[probabilities.argmax(axis=1)])
        ),
        "n_rows": len(labels),
    }


def conditional_baseline_log_loss(
    train: pl.DataFrame,
    eval_frame: pl.DataFrame,
    label_column: str,
    condition_columns: list[str],
    smoothing: float = 1.0,
) -> float:
    """Log loss of the empirical P(label | conditions) baseline.

    Stage A should be compared against a count-conditioned baseline
    (``balls_before``/``strikes_before``); Stage B against pitch type +
    platoon. The models must beat these to justify existing.
    """
    from sklearn.metrics import log_loss

    classes = sorted(train.get_column(label_column).unique().to_list())
    counts = (
        train.group_by(condition_columns + [label_column])
        .agg(pl.len().alias("n"))
        .pivot(on=label_column, index=condition_columns, values="n")
        .fill_null(0)
    )
    for cls in classes:
        if cls not in counts.columns:
            counts = counts.with_columns(pl.lit(0).alias(cls))
    total = sum(pl.col(cls) for cls in classes) + smoothing * len(classes)
    counts = counts.with_columns(
        [((pl.col(cls) + smoothing) / total).alias(f"p_{cls}") for cls in classes]
    ).select(condition_columns + [f"p_{cls}" for cls in classes])

    joined = eval_frame.join(counts, on=condition_columns, how="left").with_columns(
        [pl.col(f"p_{cls}").fill_null(1.0 / len(classes)) for cls in classes]
    )
    probabilities = joined.select([f"p_{cls}" for cls in classes]).to_numpy()
    labels = joined.get_column(label_column).to_list()
    return float(log_loss(labels, probabilities, labels=classes))


def save_model(model, output_dir: Path, name: str, metrics: dict) -> Path:
    """Persist model, feature schema, and metrics under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{name}.cbm"
    model.save_model(str(model_path))
    (output_dir / f"{name}_features.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "categorical_features": CATEGORICAL_FEATURES,
                "classes": [str(c) for c in model.classes_],
            },
            indent=2,
        )
    )
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return model_path
