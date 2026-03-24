#!/usr/bin/env python
"""
Combined training script for pitch prediction models.

Trains both:
1. CatBoost model for pitch type classification
2. MDN model for pitch location density estimation

Usage:
    uv run python scripts/run_combined_training.py
    uv run python scripts/run_combined_training.py --quick

Output Structure:
    models/combined_YYYYMMDD_HHMMSS/
    ├── catboost/
    │   ├── type_model.cbm      # Pitch type classifier
    │   ├── px_model.cbm        # Horizontal location regressor
    │   ├── pz_model.cbm        # Vertical location regressor
    │   └── feature_info.json   # Feature column info
    ├── mdn_location_model.pt   # MDN location density model
    ├── feature_engine.json     # Pitcher/batter ID mappings
    ├── results.json            # Evaluation metrics
    └── plots/
        └── mdn_density_examples.png

Inference Example:
    # Load CatBoost for pitch type prediction
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model("models/combined_xxx/catboost/type_model.cbm")
    pitch_probs = model.predict_proba(features)

    # Load MDN for location density
    import torch
    from src.ml.mdn_location_model import BivariateMDN, get_point_estimate
    checkpoint = torch.load("models/combined_xxx/mdn_location_model.pt")
    mdn = BivariateMDN(**checkpoint["config"])
    mdn.load_state_dict(checkpoint["model_state_dict"])

    # Get point estimate
    point = get_point_estimate(mdn, features)  # [px, pz]

    # Or get full density
    from src.ml.mdn_location_model import get_location_density
    px_grid, pz_grid, density = get_location_density(mdn, features)
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

from src.ml.features import PitchFeatureEngine, PITCH_TYPE_CODES
from src.ml.catboost_model import PitchCatBoostModel
from src.ml.mdn_location_model import (
    BivariateMDN,
    MDNLocationTrainer,
    plot_multiple_densities,
    get_point_estimate,
    predict_location_batch,
)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(data_path: Path, train_seasons: list[str], val_season: str, test_season: str):
    """Load data for all seasons."""
    print("Loading training data...")
    train_dfs = []
    for season in train_seasons:
        path = data_path / season
        if path.exists():
            df = pl.scan_parquet(str(path / "*.parquet")).collect()
            train_dfs.append(df)
            print(f"  {season}: {len(df):,} pitches")
    train_df = pl.concat(train_dfs, how="diagonal")
    print(f"Total training: {len(train_df):,} pitches")

    print(f"\nLoading validation data: {val_season}")
    val_df = pl.scan_parquet(str(data_path / val_season / "*.parquet")).collect()
    print(f"  {val_season}: {len(val_df):,} pitches")

    print(f"\nLoading test data: {test_season}")
    test_df = pl.scan_parquet(str(data_path / test_season / "*.parquet")).collect()
    print(f"  {test_season}: {len(test_df):,} pitches")

    return train_df, val_df, test_df


def prepare_mdn_features(
    df: pl.DataFrame,
    feature_engine: PitchFeatureEngine,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features for MDN location prediction."""
    from src.ml.features import PITCH_TYPE_TO_IDX

    df = feature_engine.transform(df)

    df = df.filter(
        pl.col("px").is_not_null()
        & pl.col("pz").is_not_null()
        & pl.col("pitch_type_idx").is_not_null()
    )

    # Get feature columns from the feature engine and add pitch_type_idx for location prediction
    feature_cols = feature_engine.get_feature_columns() + ["pitch_type_idx"]

    # Remove any duplicates while preserving order
    seen = set()
    feature_cols = [x for x in feature_cols if not (x in seen or seen.add(x))]

    available = [c for c in feature_cols if c in df.columns]

    X = df.select(available).to_numpy().astype(np.float32)
    y = df.select(["px", "pz"]).to_numpy().astype(np.float32)

    X = np.nan_to_num(X, nan=0.0)

    return X, y, available


def train_catboost(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_engine: PitchFeatureEngine,
    output_dir: Path,
    args,
) -> dict:
    """Train CatBoost pitch type model."""
    print("\n" + "=" * 70)
    print("TRAINING CATBOOST PITCH TYPE MODEL")
    print("=" * 70)

    model = PitchCatBoostModel(
        iterations=args.catboost_iterations,
        learning_rate=args.catboost_lr,
        depth=args.catboost_depth,
        l2_leaf_reg=3.0,
        early_stopping_rounds=args.early_stopping,
        task_type="CPU",
        random_seed=args.seed,
        verbose=50 if not args.quick else 10,
    )

    print("\nPreparing CatBoost data...")
    X_train, y_type_train, y_px_train, y_pz_train, cat_features = model.prepare_data(
        train_df, feature_engine
    )
    X_val, y_type_val, y_px_val, y_pz_val, _ = model.prepare_data(
        val_df, feature_engine
    )
    X_test, y_type_test, y_px_test, y_pz_test, _ = model.prepare_data(
        test_df, feature_engine
    )

    print(f"Train: {len(X_train):,} samples, {len(model.feature_columns)} features")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")

    train_results = model.train(
        X_train, y_type_train, y_px_train, y_pz_train,
        X_val, y_type_val, y_px_val, y_pz_val,
        cat_features=cat_features,
        n_classes=len(PITCH_TYPE_CODES),
    )

    test_results = model.evaluate(
        X_test, y_type_test, y_px_test, y_pz_test,
        cat_features=cat_features,
    )

    print("\nCatBoost Test Results:")
    print(f"  Pitch Type Accuracy: {test_results['accuracy']:.1%}")
    print(f"  Top-3 Accuracy:      {test_results['top3_accuracy']:.1%}")
    print(f"  Macro F1:            {test_results['f1_macro']:.4f}")
    print(f"  Location MAE px:     {test_results['mae_px']:.4f} ft")
    print(f"  Location MAE pz:     {test_results['mae_pz']:.4f} ft")
    print(f"  Euclidean Error:     {test_results['euclidean_error']:.4f} ft")

    # Save model
    catboost_dir = output_dir / "catboost"
    model.save(catboost_dir)
    print(f"\nCatBoost model saved: {catboost_dir}")

    return {
        "accuracy": test_results["accuracy"],
        "top3_accuracy": test_results["top3_accuracy"],
        "f1_macro": test_results["f1_macro"],
        "f1_weighted": test_results["f1_weighted"],
        "mae_px": test_results["mae_px"],
        "mae_pz": test_results["mae_pz"],
        "euclidean_error": test_results["euclidean_error"],
        "feature_columns": model.feature_columns,
        "categorical_features": model.categorical_features,
    }


def train_mdn(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_engine: PitchFeatureEngine,
    output_dir: Path,
    args,
) -> dict:
    """Train MDN location density model."""
    print("\n" + "=" * 70)
    print("TRAINING MDN LOCATION DENSITY MODEL")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nPreparing MDN data...")
    X_train, y_train, feature_cols = prepare_mdn_features(train_df, feature_engine)
    X_val, y_val, _ = prepare_mdn_features(val_df, feature_engine)
    X_test, y_test, _ = prepare_mdn_features(test_df, feature_engine)

    print(f"Train: {len(X_train):,} samples, {len(feature_cols)} features")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")

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

    train_loader = DataLoader(train_dataset, batch_size=args.mdn_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.mdn_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.mdn_batch_size)

    model = BivariateMDN(
        n_features=len(feature_cols),
        hidden_dims=args.mdn_hidden_dims,
        n_components=args.mdn_components,
        dropout=0.2,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDN parameters: {n_params:,}")

    trainer = MDNLocationTrainer(
        model=model,
        learning_rate=args.mdn_lr,
        device=device,
    )

    train_results = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.mdn_epochs,
        early_stopping_patience=args.early_stopping,
    )

    print(f"\nBest validation NLL: {train_results['best_val_nll']:.4f}")

    test_metrics = trainer.validate(test_loader)

    print("\nMDN Test Results:")
    print(f"  NLL:           {test_metrics['nll']:.4f}")
    print(f"  MAE px:        {test_metrics['mae_px']:.4f} ft")
    print(f"  MAE pz:        {test_metrics['mae_pz']:.4f} ft")
    print(f"  Euclidean:     {test_metrics['euclidean']:.4f} ft")

    # Generate example plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    model.eval()
    test_X = torch.tensor(X_test[:100], dtype=torch.float32)
    test_y = torch.tensor(y_test[:100], dtype=torch.float32)
    example_indices = np.linspace(0, 99, 6, dtype=int)
    features_list = [test_X[i:i+1] for i in example_indices]
    targets_list = [test_y[i] for i in example_indices]
    titles = [f"Pitch {i+1}" for i in range(6)]

    plot_multiple_densities(
        model,
        features_list,
        targets_list,
        titles,
        n_samples=500,
        save_path=str(plots_dir / "mdn_density_examples.png"),
    )
    print(f"\nDensity plots saved: {plots_dir / 'mdn_density_examples.png'}")

    # Save model
    mdn_path = output_dir / "mdn_location_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dims": args.mdn_hidden_dims,
            "n_components": args.mdn_components,
            "dropout": 0.2,
            "n_features": len(feature_cols),
        },
        "feature_columns": feature_cols,
    }, mdn_path)
    print(f"MDN model saved: {mdn_path}")

    return {
        "nll": test_metrics["nll"],
        "mae_px": test_metrics["mae_px"],
        "mae_pz": test_metrics["mae_pz"],
        "euclidean": test_metrics["euclidean"],
        "feature_columns": feature_cols,
        "n_components": args.mdn_components,
    }


def run_combined_training(args):
    """Run training for both models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"combined_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMBINED PITCH PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print()

    set_seed(args.seed)

    # Define seasons
    train_seasons = ["2018", "2019", "2021", "2022", "2023"]
    val_season = "2024"
    test_season = "2025"

    print("Data Split:")
    print(f"  Train: {train_seasons}")
    print(f"  Validation: {val_season}")
    print(f"  Test: {test_season}")

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data_path = Path(args.data_path)
    train_df, val_df, test_df = load_data(data_path, train_seasons, val_season, test_season)

    # Fit feature engine
    print("\nFitting feature engine...")
    all_df = pl.concat([train_df, val_df, test_df], how="diagonal")
    feature_engine = PitchFeatureEngine(data_path)
    feature_engine.fit(all_df)
    print(f"Pitchers: {feature_engine.n_pitchers:,}")
    print(f"Batters: {feature_engine.n_batters:,}")

    # Train CatBoost
    catboost_results = train_catboost(
        train_df, val_df, test_df, feature_engine, output_dir, args
    )

    # Train MDN
    mdn_results = train_mdn(
        train_df, val_df, test_df, feature_engine, output_dir, args
    )

    # Save feature engine
    feature_engine_path = output_dir / "feature_engine.json"
    feature_engine_data = {
        "pitcher_to_idx": feature_engine.pitcher_to_idx,
        "batter_to_idx": feature_engine.batter_to_idx,
        "n_pitchers": feature_engine.n_pitchers,
        "n_batters": feature_engine.n_batters,
        "pitcher_ff_pct": {str(k): v for k, v in feature_engine.pitcher_ff_pct.items()},
        "pitcher_repertoire_size": {str(k): v for k, v in feature_engine.pitcher_repertoire_size.items()},
    }
    with open(feature_engine_path, "w") as f:
        json.dump(feature_engine_data, f, indent=2)
    print(f"\nFeature engine saved: {feature_engine_path}")

    # Combined results summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 70)

    results = {
        "timestamp": timestamp,
        "train_seasons": train_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "catboost": catboost_results,
        "mdn": mdn_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n+----------------------+----------------------+")
    print("|        METRIC        |        VALUE         |")
    print("+----------------------+----------------------+")
    print("|  CATBOOST (Pitch Type)                      |")
    print("+----------------------+----------------------+")
    print(f"|  Accuracy            | {catboost_results['accuracy']:>18.1%}   |")
    print(f"|  Top-3 Accuracy      | {catboost_results['top3_accuracy']:>18.1%}   |")
    print(f"|  Macro F1            | {catboost_results['f1_macro']:>20.4f} |")
    print("+----------------------+----------------------+")
    print("|  MDN (Location Density)                     |")
    print("+----------------------+----------------------+")
    print(f"|  NLL                 | {mdn_results['nll']:>20.4f} |")
    print(f"|  MAE px              | {mdn_results['mae_px']:>17.4f} ft |")
    print(f"|  MAE pz              | {mdn_results['mae_pz']:>17.4f} ft |")
    print(f"|  Euclidean           | {mdn_results['euclidean']:>17.4f} ft |")
    print("+----------------------+----------------------+")

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Results JSON: {results_path}")

    # Print information about handling new/unseen pitchers
    print("\n" + "=" * 70)
    print("HANDLING NEW/UNSEEN PITCHERS")
    print("=" * 70)
    print("""
The models handle new pitchers differently:

CatBoost Model:
  - Uses pitcher_id as a categorical feature handled natively by CatBoost
  - New pitchers get assigned to an "unknown" category automatically
  - CatBoost's optimal handling of unseen categories means it will fall back
    to learning from other features (count, handedness, game situation, etc.)
  - Performance degrades gracefully - the model still makes reasonable
    predictions based on situational context

MDN Location Model:
  - Uses pitcher_idx as an embedding lookup (similar to word embeddings)
  - Unknown pitchers get mapped to the last index (n_pitchers - 1)
  - The embedding table has n_pitchers entries (known + 1 for unknown)
  - The model learns a general "average pitcher" representation for unknown
  - Other features (pitch type, count, handedness) still inform the prediction

Best Practices for New Pitchers:
  1. Periodically retrain models to incorporate new pitchers
  2. Use pitcher tendency features (pitcher_ff_pct, repertoire) which can be
     computed from even a few games of data
  3. Monitor prediction confidence for unknown pitchers
  4. Consider using handedness and repertoire-based clustering for similar
     pitcher lookup

Feature Engine Pitcher Handling:
  - The PitchFeatureEngine maps pitcher IDs to indices
  - Known pitchers get indices 0 to N-1, unknown get index N
  - pitcher_to_idx mapping is saved in feature_engine.json
  - To add a new pitcher: update the mapping or retrain
""")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost (pitch type) + MDN (location density) models",
    )

    # Data
    parser.add_argument("--data-path", default="data/processed/livefeeds")

    # CatBoost settings
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-lr", type=float, default=0.05)
    parser.add_argument("--catboost-depth", type=int, default=8)

    # MDN settings
    parser.add_argument("--mdn-hidden-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--mdn-batch-size", type=int, default=512)
    parser.add_argument("--mdn-epochs", type=int, default=100)
    parser.add_argument("--mdn-lr", type=float, default=1e-3)

    # Common settings
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick test run")

    args = parser.parse_args()

    if args.quick:
        args.catboost_iterations = 100
        args.mdn_epochs = 10
        args.early_stopping = 5

    run_combined_training(args)


if __name__ == "__main__":
    main()
