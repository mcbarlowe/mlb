"""
Training script for CatBoost pitch prediction model.

This provides a comparison baseline against the LSTM model.
CatBoost often excels on tabular data with mixed feature types.

Usage:
    # Run full training
    uv run python scripts/run_catboost_training.py

    # Quick test
    uv run python scripts/run_catboost_training.py --quick

    # Custom settings
    uv run python scripts/run_catboost_training.py --iterations 2000 --depth 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl

from mlb.ml.catboost_model import PitchCatBoostModel
from mlb.ml.evaluate import plot_confusion_matrix, plot_location_predictions
from mlb.ml.features import PITCH_TYPE_CODES, PitchFeatureEngine
from mlb.ml.season_splits import default_data_source_train_seasons


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_data(
    data_path: str,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
) -> tuple:
    """
    Load and prepare data for CatBoost training.

    Returns:
        Tuple of DataFrames and feature engine.
    """
    feature_engine = PitchFeatureEngine(data_path)

    # Load training data
    print(f"Loading training data: {train_seasons}")
    train_dfs = []
    for season in train_seasons:
        df = feature_engine.load_data(seasons=[season])
        train_dfs.append(df)
        print(f"  {season}: {len(df):,} pitches")

    train_df = pl.concat(train_dfs, how="diagonal")
    print(f"Total training: {len(train_df):,} pitches")

    # Load validation data
    print(f"\nLoading validation data: {val_season}")
    val_df = feature_engine.load_data(seasons=[val_season])
    print(f"Validation: {len(val_df):,} pitches")

    # Load test data
    print(f"\nLoading test data: {test_season}")
    test_df = feature_engine.load_data(seasons=[test_season])
    print(f"Test: {len(test_df):,} pitches")

    # Fit feature engine on all data for consistent mappings
    print("\nFitting feature engine...")
    all_df = pl.concat([train_df, val_df, test_df], how="diagonal")
    feature_engine.fit(all_df)

    print(f"Pitchers: {feature_engine.n_pitchers:,}")
    print(f"Batters: {feature_engine.n_batters:,}")

    return train_df, val_df, test_df, feature_engine


def run_training(args) -> dict:
    """Run the CatBoost training pipeline."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"catboost_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CATBOOST PITCH PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    set_seed(args.seed)

    # Model configuration
    config = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    print("Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    train_seasons = args.train_seasons or default_data_source_train_seasons(
        args.data_path,
        val_season=args.val_season,
        test_season=args.test_season,
        exclude_2020=args.exclude_2020,
    )

    print("Data Split:")
    print(f"  Train: {train_seasons}")
    print(f"  Validation: {args.val_season}")
    print(f"  Test: {args.test_season}")
    print()

    results = {
        "config": config,
        "timestamp": timestamp,
        "args": vars(args),
        "train_seasons": train_seasons,
    }

    # =========================================================================
    # PHASE 1: Load and Prepare Data
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: LOADING DATA")
    print("=" * 70)

    train_df, val_df, test_df, feature_engine = load_and_prepare_data(
        args.data_path,
        train_seasons,
        args.val_season,
        args.test_season,
    )

    # Create CatBoost model
    model = PitchCatBoostModel(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        task_type=args.task_type,
        random_seed=args.seed,
        verbose=args.verbose,
    )

    # Prepare data
    print("\n" + "=" * 70)
    print("PHASE 2: PREPARING FEATURES")
    print("=" * 70)

    print("\nPreparing training data...")
    X_train, y_type_train, y_px_train, y_pz_train, cat_features = model.prepare_data(
        train_df, feature_engine
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Features: {len(model.feature_columns)}")
    print(f"Categorical features: {model.categorical_features}")

    print("\nPreparing validation data...")
    X_val, y_type_val, y_px_val, y_pz_val, _ = model.prepare_data(
        val_df, feature_engine
    )
    print(f"Validation samples: {len(X_val):,}")

    print("\nPreparing test data...")
    X_test, y_type_test, y_px_test, y_pz_test, _ = model.prepare_data(
        test_df, feature_engine
    )
    print(f"Test samples: {len(X_test):,}")

    results["data"] = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "n_features": len(model.feature_columns),
    }

    # =========================================================================
    # PHASE 3: Train Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING MODELS")
    print("=" * 70)

    train_results = model.train(
        X_train, y_type_train, y_px_train, y_pz_train,
        X_val, y_type_val, y_px_val, y_pz_val,
        cat_features=cat_features,
        n_classes=len(PITCH_TYPE_CODES),
    )

    results["training"] = train_results

    # =========================================================================
    # PHASE 4: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: TEST SET EVALUATION")
    print("=" * 70)

    test_results = model.evaluate(
        X_test, y_type_test, y_px_test, y_pz_test,
        cat_features=cat_features,
    )

    print("\nClassification Metrics (Pitch Type):")
    print(f"  Accuracy:      {test_results['accuracy']:.4f}")
    print(f"  Top-3 Acc:     {test_results['top3_accuracy']:.4f}")
    print(f"  Macro F1:      {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1:   {test_results['f1_weighted']:.4f}")

    print("\nLocation Metrics:")
    print(f"  MAE px:        {test_results['mae_px']:.4f} ft")
    print(f"  MAE pz:        {test_results['mae_pz']:.4f} ft")
    print(f"  RMSE px:       {test_results['rmse_px']:.4f} ft")
    print(f"  RMSE pz:       {test_results['rmse_pz']:.4f} ft")
    print(f"  Euclidean:     {test_results['euclidean_error']:.4f} ft")

    results["test_results"] = {
        "accuracy": test_results["accuracy"],
        "top3_accuracy": test_results["top3_accuracy"],
        "f1_macro": test_results["f1_macro"],
        "f1_weighted": test_results["f1_weighted"],
        "mae_px": test_results["mae_px"],
        "mae_pz": test_results["mae_pz"],
        "rmse_px": test_results["rmse_px"],
        "rmse_pz": test_results["rmse_pz"],
        "euclidean_error": test_results["euclidean_error"],
    }

    # =========================================================================
    # PHASE 5: Feature Importance
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: FEATURE IMPORTANCE")
    print("=" * 70)

    importance = model.get_feature_importance(top_n=15)

    print("\nTop Features for Pitch Type:")
    for feat, imp in importance["pitch_type"][:10]:
        print(f"  {feat}: {imp:.2f}")

    print("\nTop Features for Location (px):")
    for feat, imp in importance["px"][:10]:
        print(f"  {feat}: {imp:.2f}")

    print("\nTop Features for Location (pz):")
    for feat, imp in importance["pz"][:10]:
        print(f"  {feat}: {imp:.2f}")

    results["feature_importance"] = {
        "pitch_type": [(f, float(i)) for f, i in importance["pitch_type"]],
        "px": [(f, float(i)) for f, i in importance["px"]],
        "pz": [(f, float(i)) for f, i in importance["pz"]],
    }

    # =========================================================================
    # PHASE 6: Save Outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING OUTPUTS")
    print("=" * 70)

    # Save models
    model.save(output_dir / "models")
    print(f"Models saved: {output_dir / 'models'}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Generate plots
    if args.plots:
        print("\nGenerating plots...")

        plot_confusion_matrix(
            y_type_test,
            test_results["type_preds"],
            save_path=str(plots_dir / "confusion_matrix.png"),
        )
        print(f"  {plots_dir / 'confusion_matrix.png'}")

        plot_location_predictions(
            np.column_stack([test_results["px_preds"], test_results["pz_preds"]]),
            np.column_stack([y_px_test, y_pz_test]),
            save_path=str(plots_dir / "location_predictions.png"),
        )
        print(f"  {plots_dir / 'location_predictions.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Output: {output_dir}")
    print()
    print("Key Metrics:")
    print(f"  Pitch Type Accuracy: {test_results['accuracy']:.1%}")
    print(f"  Top-3 Accuracy:      {test_results['top3_accuracy']:.1%}")
    print(f"  Location MAE:        {(test_results['mae_px'] + test_results['mae_pz']) / 2:.3f} ft")
    print(f"  Euclidean Error:     {test_results['euclidean_error']:.3f} ft")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost pitch prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="postgres",
        help="Training data source: 'postgres' or a parquet path",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=str,
        default=None,
        help="Optional explicit training seasons; default uses all available pre-validation seasons except 2020",
    )
    parser.add_argument("--val-season", type=str, default="2024")
    parser.add_argument("--test-season", type=str, default="2025")
    parser.add_argument("--exclude-2020", action="store_true", default=True)
    parser.add_argument("--include-2020", action="store_true")

    # Model
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--task-type", type=str, default="CPU", choices=["CPU", "GPU"])

    # Output
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", type=int, default=100)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick test (100 iterations)")

    args = parser.parse_args()

    if args.include_2020:
        args.exclude_2020 = False
    if args.no_plots:
        args.plots = False
    if args.quick:
        args.iterations = 100
        args.early_stopping_rounds = 10
        args.verbose = 10

    run_training(args)


if __name__ == "__main__":
    main()
