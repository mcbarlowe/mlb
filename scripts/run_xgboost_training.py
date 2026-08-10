from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import polars as pl

from src.ml.catboost_model import PitchXGBoostModel
from src.ml.features import PITCH_TYPE_CODES, PitchFeatureEngine
from src.ml.mlflow_utils import (
    build_metric_dict,
    build_param_dict,
    configure_mlflow,
    log_path_if_exists,
)
from src.ml.season_splits import default_data_source_train_seasons

DEFAULT_VAL_SEASON = "2024"
DEFAULT_TEST_SEASON = "2025"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost pitch prediction baseline.")
    parser.add_argument("--data-path", type=str, default="postgres")
    parser.add_argument("--train-seasons", nargs="+", type=str, default=None)
    parser.add_argument("--val-season", type=str, default=DEFAULT_VAL_SEASON)
    parser.add_argument("--test-season", type=str, default=DEFAULT_TEST_SEASON)
    parser.add_argument("--exclude-2020", action="store_true", default=True)
    parser.add_argument("--include-2020", action="store_true")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="models/xgboost")
    parser.add_argument("--mlflow-experiment", type=str, default="mlb-model-training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(data_path: str, train_seasons: list[str], val_season: str, test_season: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, PitchFeatureEngine]:
    feature_engine = PitchFeatureEngine(data_path)

    print("Loading training data...")
    train_dfs = []
    for season in train_seasons:
        df = feature_engine.load_data(seasons=[season])
        train_dfs.append(df)
        print(f"  {season}: {len(df):,} pitches")
    train_df = pl.concat(train_dfs, how="diagonal")
    print(f"Total training: {len(train_df):,} pitches")

    print(f"\nLoading validation data: {val_season}")
    val_df = feature_engine.load_data(seasons=[val_season])
    print(f"  {val_season}: {len(val_df):,} pitches")

    print(f"\nLoading test data: {test_season}")
    test_df = feature_engine.load_data(seasons=[test_season])
    print(f"  {test_season}: {len(test_df):,} pitches")

    return train_df, val_df, test_df, feature_engine

def log_artifacts_if_available(results: dict, output_dir: Path, model_dir: Path) -> None:
    try:
        mlflow.log_dict(results, "results.json")
        log_path_if_exists(model_dir, "artifacts")
        log_path_if_exists(output_dir / "results.json", "artifacts")
    except (OSError, mlflow.exceptions.MlflowException) as exc:
        message = f"{type(exc).__name__}: {exc}"
        mlflow.set_tag("artifact_logging_failed", "true")
        mlflow.set_tag("artifact_logging_error", message[:500])
        print(
            "WARNING: MLflow artifact logging failed; "
            f"local artifacts remain at {output_dir}: {message}"
        )


def main() -> None:
    args = parse_args()
    if args.include_2020:
        args.exclude_2020 = False
    if args.quick:
        args.train_seasons = ["2023"]
        args.iterations = 50
        args.early_stopping = 5

    train_seasons = args.train_seasons or default_data_source_train_seasons(
        args.data_path,
        val_season=args.val_season,
        test_season=args.test_season,
        exclude_2020=args.exclude_2020,
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri, require_tracking_uri=True)

    print("=" * 70)
    print("XGBOOST PITCH PREDICTION BASELINE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Tracking URI: {tracking_uri}")
    print("Data Split:")
    print(f"  Train: {train_seasons}")
    print(f"  Validation: {args.val_season}")
    print(f"  Test: {args.test_season}")

    train_df, val_df, test_df, feature_engine = load_data(
        args.data_path,
        train_seasons,
        args.val_season,
        args.test_season,
    )

    print("\nFitting feature engine...")
    all_df = pl.concat([train_df, val_df, test_df], how="diagonal")
    feature_engine.fit(all_df)
    print(f"Pitchers: {feature_engine.n_pitchers:,}")
    print(f"Batters: {feature_engine.n_batters:,}")

    model = PitchXGBoostModel(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        early_stopping_rounds=args.early_stopping,
        random_seed=args.seed,
        verbose=50 if not args.quick else 10,
    )

    X_train, y_type_train, y_px_train, y_pz_train, cat_features = model.prepare_data(train_df, feature_engine)
    X_val, y_type_val, y_px_val, y_pz_val, _ = model.prepare_data(val_df, feature_engine)
    X_test, y_type_test, y_px_test, y_pz_test, _ = model.prepare_data(test_df, feature_engine)

    print(f"Train: {len(X_train):,} samples, {len(model.feature_columns)} features")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")

    with mlflow.start_run(run_name=f"xgboost-{timestamp}"):
        mlflow.set_tags({
            "model_family": "xgboost_pitch_baseline",
            "tracking_uri": tracking_uri,
            "data_path": args.data_path,
            "train_seasons": ",".join(train_seasons),
            "val_season": args.val_season,
            "test_season": args.test_season,
            "quick": str(args.quick),
        })
        mlflow.log_params(build_param_dict(vars(args), prefix="arg"))

        train_results = model.train(
            X_train,
            y_type_train,
            y_px_train,
            y_pz_train,
            X_val,
            y_type_val,
            y_px_val,
            y_pz_val,
            cat_features=cat_features,
            n_classes=len(PITCH_TYPE_CODES),
        )
        test_results = model.evaluate(
            X_test,
            y_type_test,
            y_px_test,
            y_pz_test,
            cat_features=cat_features,
        )
        importance = model.get_feature_importance(top_n=20)

        results = {
            "timestamp": timestamp,
            "train_seasons": train_seasons,
            "val_season": args.val_season,
            "test_season": args.test_season,
            "xgboost": {
                "accuracy": test_results["accuracy"],
                "top3_accuracy": test_results["top3_accuracy"],
                "f1_macro": test_results["f1_macro"],
                "f1_weighted": test_results["f1_weighted"],
                "mae_px": test_results["mae_px"],
                "mae_pz": test_results["mae_pz"],
                "rmse_px": test_results["rmse_px"],
                "rmse_pz": test_results["rmse_pz"],
                "euclidean_error": test_results["euclidean_error"],
                "feature_columns": model.feature_columns,
                "categorical_features": model.categorical_features,
                "feature_importance": importance,
                "best_iteration": train_results,
            },
        }

        model_dir = output_dir / "xgboost"
        model.save(model_dir)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

        mlflow.log_metrics(build_metric_dict(results["xgboost"], prefix="test"))
        log_artifacts_if_available(results, output_dir, model_dir)

    print("\nXGBoost Test Results:")
    print(f"  Accuracy:        {test_results['accuracy']:.1%}")
    print(f"  Top-3 Accuracy:  {test_results['top3_accuracy']:.1%}")
    print(f"  Macro F1:        {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1:     {test_results['f1_weighted']:.4f}")
    print(f"  MAE px:          {test_results['mae_px']:.4f} ft")
    print(f"  MAE pz:          {test_results['mae_pz']:.4f} ft")
    print(f"  Euclidean Error: {test_results['euclidean_error']:.4f} ft")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == '__main__':
    main()
