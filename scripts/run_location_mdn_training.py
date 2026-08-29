"""
Training script for standalone MDN location prediction.

This trains a feedforward MDN specifically for pitch location density estimation,
separate from pitch type prediction. The output is a bivariate Gaussian mixture
that can be used to generate kernel density estimates.

Usage:
    uv run python scripts/run_location_mdn_training.py
    uv run python scripts/run_location_mdn_training.py --quick
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlb.ml.features import PitchFeatureEngine
from mlb.ml.mdn_location_model import (
    BivariateMDN,
    MDNLocationTrainer,
    plot_multiple_densities,
)
from mlb.ml.season_splits import default_data_source_train_seasons


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_location_features(
    df: pl.DataFrame,
    feature_engine: PitchFeatureEngine,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare features specifically for location prediction.

    Uses all engineered features plus pitch_type_code as a feature
    (since we're predicting location, knowing the pitch type helps).
    """

    # Transform data
    df = feature_engine.transform(df)

    # Filter nulls
    df = df.filter(
        pl.col("px").is_not_null()
        & pl.col("pz").is_not_null()
        & pl.col("pitch_type_idx").is_not_null()
    )

    # Feature columns for location prediction
    # Include pitch_type_idx since we know what pitch is being thrown
    feature_cols = [
        # Count features
        "balls", "strikes", "two_strike_count", "hitters_count",
        "first_pitch", "pitcher_ahead",
        # Game state
        "inning", "outs", "runners_bitmap", "score_diff",
        # IDs (as numeric for NN)
        "pitcher_idx", "batter_idx",
        # Handedness
        "throw_side_enc", "bat_side_enc", "platoon_same_side",
        # Pitcher tendencies
        "pitcher_ff_pct", "pitcher_repertoire",
        # Batter zone
        "batter_zone_height", "batter_zone_mid",
        # Current pitch type (this is known when predicting location)
        "pitch_type_idx",
        # Previous pitch features
        "prev_pitch_type_idx", "prev_px", "prev_pz", "prev_speed",
        "prev_is_strike", "velocity_delta",
        # Swing/result
        "prev_swing", "prev_result_type",
        # Cumulative
        "n_fastballs_in_ab", "n_breaking_in_ab",
        # Sequence
        "same_pitch_streak", "pitch_number",
    ]

    # Verify columns exist
    available = [c for c in feature_cols if c in df.columns]

    X = df.select(available).to_numpy().astype(np.float32)
    y = df.select(["px", "pz"]).to_numpy().astype(np.float32)

    # Handle any remaining NaNs
    X = np.nan_to_num(X, nan=0.0)

    return X, y, available


def run_training(args):
    """Run the MDN location training pipeline."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"mdn_location_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MDN LOCATION DENSITY MODEL TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    config = {
        "hidden_dims": args.hidden_dims,
        "n_components": args.n_components,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "train_seasons": args.train_seasons,
        "val_season": args.val_season,
        "test_season": args.test_season,
    }
    print("\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    feature_engine = PitchFeatureEngine(args.data_path)

    # Load training data
    train_seasons = args.train_seasons
    print(f"\nTraining seasons: {train_seasons}")
    train_dfs = []
    for season in train_seasons:
        df = feature_engine.load_data(seasons=[season])
        train_dfs.append(df)
        print(f"  {season}: {len(df):,} pitches")
    train_df = pl.concat(train_dfs, how="diagonal")

    # Validation
    print(f"\nValidation: {args.val_season}")
    val_df = feature_engine.load_data(seasons=[args.val_season])
    print(f"  {args.val_season}: {len(val_df):,} pitches")

    # Test
    print(f"\nTest: {args.test_season}")
    test_df = feature_engine.load_data(seasons=[args.test_season])
    print(f"  {args.test_season}: {len(test_df):,} pitches")

    # Fit feature engine
    print("\nFitting feature engine...")
    all_df = pl.concat([train_df, val_df, test_df], how="diagonal")
    feature_engine.fit(all_df)

    # Prepare data
    print("\nPreparing features...")
    X_train, y_train, feature_cols = prepare_location_features(train_df, feature_engine)
    X_val, y_val, _ = prepare_location_features(val_df, feature_engine)
    X_test, y_test, _ = prepare_location_features(test_df, feature_engine)

    print(f"Train: {len(X_train):,} samples, {len(feature_cols)} features")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # =========================================================================
    # Create and Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING MDN MODEL")
    print("=" * 70)

    model = BivariateMDN(
        n_features=len(feature_cols),
        hidden_dims=args.hidden_dims,
        n_components=args.n_components,
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    trainer = MDNLocationTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=device,
    )

    train_results = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.n_epochs,
        early_stopping_patience=args.patience,
    )

    print(f"\nBest validation NLL: {train_results['best_val_nll']:.4f}")

    # =========================================================================
    # Evaluate
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    test_metrics = trainer.validate(test_loader)

    print("\nTest Metrics:")
    print(f"  NLL:        {test_metrics['nll']:.4f}")
    print(f"  MAE px:     {test_metrics['mae_px']:.4f} ft")
    print(f"  MAE pz:     {test_metrics['mae_pz']:.4f} ft")
    print(f"  Euclidean:  {test_metrics['euclidean']:.4f} ft")

    # =========================================================================
    # Generate Example Density Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING DENSITY PLOTS")
    print("=" * 70)

    # Get some example predictions
    model.eval()
    n_examples = 6

    # Sample different scenarios
    test_X = torch.tensor(X_test[:100], dtype=torch.float32)
    test_y = torch.tensor(y_test[:100], dtype=torch.float32)

    # Select diverse examples
    example_indices = np.linspace(0, 99, n_examples, dtype=int)

    features_list = [test_X[i:i+1] for i in example_indices]
    targets_list = [test_y[i] for i in example_indices]
    titles = [f"Pitch {i+1}" for i in range(n_examples)]

    plot_multiple_densities(
        model,
        features_list,
        targets_list,
        titles,
        n_samples=500,
        save_path=str(plots_dir / "density_examples.png"),
    )
    print(f"Saved: {plots_dir / 'density_examples.png'}")

    # =========================================================================
    # Save Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    model_path = output_dir / "mdn_location_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
    }, model_path)
    print(f"Model saved: {model_path}")

    # Save results
    results = {
        "config": config,
        "timestamp": timestamp,
        "test_metrics": test_metrics,
        "feature_columns": feature_cols,
        "train_history": train_results["history"],
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")
    print("\nKey Metrics:")
    print(f"  Location NLL:    {test_metrics['nll']:.4f}")
    print(f"  MAE (avg):       {(test_metrics['mae_px'] + test_metrics['mae_pz'])/2:.4f} ft")
    print(f"  Euclidean Error: {test_metrics['euclidean']:.4f} ft")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train MDN for pitch location density estimation",
    )

    parser.add_argument(
        "--data-path",
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
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--n-components", type=int, default=5,
                        help="Number of Gaussian mixture components")
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)

    # Output
    parser.add_argument("--output-dir", default="models")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    if args.include_2020:
        args.exclude_2020 = False
    if args.quick:
        args.n_epochs = 10
        args.patience = 3
    if args.train_seasons is None:
        args.train_seasons = default_data_source_train_seasons(
            args.data_path,
            val_season=args.val_season,
            test_season=args.test_season,
            exclude_2020=args.exclude_2020,
        )

    run_training(args)


if __name__ == "__main__":
    main()
