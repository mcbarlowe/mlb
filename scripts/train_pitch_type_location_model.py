"""
Training script for Pitch-Type-Conditioned Location Model.

This trains an MDN where each pitch type has its own location distribution,
treating pitch type as a random effect. This is more granular than the
HierarchicalMDN which groups by pitch family.

Usage:
    uv run python scripts/train_pitch_type_location_model.py
    uv run python scripts/train_pitch_type_location_model.py --quick
    uv run python scripts/train_pitch_type_location_model.py --compare-baseline
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.ml.evaluate import compute_mdn_coverage, compute_mdn_nll
from src.ml.features import IDX_TO_PITCH_TYPE, PITCH_TYPE_CODES, PitchFeatureEngine
from src.ml.mdn_location_model import BivariateMDN, MDNLocationTrainer
from src.ml.pitch_type_location_model import (
    PitchTypeConditionedMDN,
    PitchTypeLocationBatchIterableDataset,
    PitchTypeLocationDataset,
    PitchTypeLocationTrainer,
    compare_to_baseline,
)
from src.ml.season_splits import default_data_source_train_seasons


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature columns for location prediction (excluding pitch_type_idx)."""
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
        # Previous pitch features (NOT pitch_type_idx - that's the conditioning variable)
        "prev_pitch_type_idx", "prev_px", "prev_pz", "prev_speed",
        "prev_is_strike", "velocity_delta",
        # Swing/result
        "prev_swing", "prev_result_type",
        # Cumulative
        "n_fastballs_in_ab", "n_breaking_in_ab",
        # Sequence
        "same_pitch_streak", "pitch_number",
        # Weather (if available)
        "temp_normalized", "wind_speed",
        # Situation
        "runners_in_scoring_position", "leverage_approx",
        # Pitcher fatigue
        "pitcher_pitch_count",
        # Interaction features
        "prev_fb_x_rhb", "prev_off_x_rhb", "prev_brk_x_rhb",
        "ahead_x_rhb", "two_strike_x_rhb", "hitters_x_rhb", "platoon_x_breaking",
    ]

    # Movement profile features ride along when the feature engine attached
    # them (opt-in via --movement-profiles-dir); the presence filter below
    # keeps this a no-op otherwise.
    from src.ml.movement_profiles import movement_profile_columns

    feature_cols.extend(movement_profile_columns())

    # Filter to columns that exist
    available = [c for c in feature_cols if c in df.columns]
    return available


def evaluate_per_pitch_type(
    model: PitchTypeConditionedMDN,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model with detailed per-pitch-type breakdown."""
    model.eval()

    all_params = {"pi": [], "mu": [], "sigma": [], "rho": []}
    all_targets = []
    all_pitch_types = []
    all_preds = []

    with torch.no_grad():
        for features, pitch_type_idx, location in test_loader:
            features = features.to(device)
            pitch_type_idx = pitch_type_idx.to(device)
            location = location.to(device)

            params = model(features, pitch_type_idx)
            pred = model.get_expected_value(params)

            # Collect results
            all_params["pi"].append(params["pi"].cpu().numpy())
            all_params["mu"].append(params["mu"].cpu().numpy())
            all_params["sigma"].append(params["sigma"].cpu().numpy())
            all_params["rho"].append(params["rho"].cpu().numpy())
            all_targets.append(location.cpu().numpy())
            all_pitch_types.append(pitch_type_idx.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    # Concatenate
    mdn_params = {
        "pi": np.concatenate(all_params["pi"]),
        "mu": np.concatenate(all_params["mu"]),
        "sigma": np.concatenate(all_params["sigma"]),
        "rho": np.concatenate(all_params["rho"]),
    }
    targets = np.concatenate(all_targets)
    pitch_types = np.concatenate(all_pitch_types)
    preds = np.concatenate(all_preds)

    # Overall metrics
    nll_values = compute_mdn_nll(mdn_params, targets)
    coverage_90 = compute_mdn_coverage(mdn_params, targets, 0.90)
    coverage_95 = compute_mdn_coverage(mdn_params, targets, 0.95)

    errors = preds - targets
    mae_px = np.abs(errors[:, 0]).mean()
    mae_pz = np.abs(errors[:, 1]).mean()
    euclidean = np.sqrt((errors ** 2).sum(axis=1)).mean()

    metrics = {
        "overall": {
            "nll": float(nll_values.mean()),
            "mae_px": float(mae_px),
            "mae_pz": float(mae_pz),
            "euclidean": float(euclidean),
            "coverage_90": float(coverage_90),
            "coverage_95": float(coverage_95),
            "n_samples": len(targets),
        },
        "per_pitch_type": {},
    }

    # Per-pitch-type breakdown
    for pt_idx in range(len(PITCH_TYPE_CODES)):
        mask = pitch_types == pt_idx
        if mask.sum() == 0:
            continue

        pt_code = IDX_TO_PITCH_TYPE.get(pt_idx, f"UNK{pt_idx}")

        # Filter for this pitch type
        pt_mdn_params = {
            "pi": mdn_params["pi"][mask],
            "mu": mdn_params["mu"][mask],
            "sigma": mdn_params["sigma"][mask],
            "rho": mdn_params["rho"][mask],
        }
        pt_targets = targets[mask]
        pt_preds = preds[mask]

        # Compute metrics
        pt_nll = compute_mdn_nll(pt_mdn_params, pt_targets)
        pt_coverage_90 = compute_mdn_coverage(pt_mdn_params, pt_targets, 0.90)
        pt_coverage_95 = compute_mdn_coverage(pt_mdn_params, pt_targets, 0.95)

        pt_errors = pt_preds - pt_targets
        pt_mae_px = np.abs(pt_errors[:, 0]).mean()
        pt_mae_pz = np.abs(pt_errors[:, 1]).mean()
        pt_euclidean = np.sqrt((pt_errors ** 2).sum(axis=1)).mean()

        metrics["per_pitch_type"][pt_code] = {
            "nll": float(pt_nll.mean()),
            "mae_px": float(pt_mae_px),
            "mae_pz": float(pt_mae_pz),
            "euclidean": float(pt_euclidean),
            "coverage_90": float(pt_coverage_90),
            "coverage_95": float(pt_coverage_95),
            "count": int(mask.sum()),
        }

    return metrics


def train_baseline_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    hidden_dims: list[int],
    n_components: int,
    dropout: float,
    n_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> BivariateMDN:
    """Train a baseline BivariateMDN without pitch type conditioning."""
    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODEL (No Pitch Type Conditioning)")
    print("=" * 70)

    baseline_model = BivariateMDN(
        n_features=n_features,
        hidden_dims=hidden_dims,
        n_components=n_components,
        dropout=dropout,
    )

    baseline_trainer = MDNLocationTrainer(
        model=baseline_model,
        learning_rate=learning_rate,
        device=str(device),
    )

    # Adapt train/val loaders for baseline (no pitch type)
    def baseline_collate(batch):
        features = torch.stack([b[0] for b in batch])
        locations = torch.stack([b[2] for b in batch])
        return features, locations

    # Create new loaders
    baseline_train = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        collate_fn=baseline_collate,
    )
    baseline_val = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        collate_fn=baseline_collate,
    )

    baseline_trainer.train(
        baseline_train,
        baseline_val,
        n_epochs=n_epochs,
        early_stopping_patience=10,
    )

    return baseline_model


def run_training(args):
    """Run the pitch-type-conditioned location model training pipeline."""

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"pitch_type_location_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PITCH-TYPE-CONDITIONED LOCATION MODEL TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

    set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
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
        "movement_profiles_dir": getattr(args, "movement_profiles_dir", None),
    }
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data_path = Path(args.data_path)
    feature_engine = PitchFeatureEngine(
        data_path,
        movement_profiles_dir=getattr(args, "movement_profiles_dir", None),
    )

    def load_season_frame(season: str) -> pl.DataFrame:
        df = feature_engine.load_data(seasons=[season])
        print(f"  Loaded {len(df):,} pitches for {season}")
        return df

    # Fit feature engine
    print(f"\nTraining seasons: {args.train_seasons}")
    print("\nFitting feature engine...")
    if args.low_memory:
        feature_engine.fit_frames(
            load_season_frame(season) for season in args.train_seasons
        )
    else:
        train_df = feature_engine.load_data(seasons=args.train_seasons)
        print(f"  Loaded {len(train_df):,} pitches")
        feature_engine.fit(train_df)
    print(f"  Pitchers: {len(feature_engine.pitcher_to_idx):,}")
    print(f"  Batters: {len(feature_engine.batter_to_idx):,}")

    if args.low_memory:
        feature_cols = feature_engine.get_feature_columns()
        print(f"\nUsing {len(feature_cols)} features")

        print("\n" + "=" * 70)
        print("CREATING STREAMING DATASETS")
        print("=" * 70)
        train_dataset = PitchTypeLocationBatchIterableDataset(
            seasons=args.train_seasons,
            load_season=load_season_frame,
            transform_season=feature_engine.transform,
            feature_columns=feature_cols,
            batch_size=args.batch_size,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
            shuffle=True,
            seed=args.seed,
        )
        val_dataset = PitchTypeLocationBatchIterableDataset(
            seasons=[args.val_season],
            load_season=load_season_frame,
            transform_season=feature_engine.transform,
            feature_columns=feature_cols,
            batch_size=args.batch_size,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
            shuffle=False,
            seed=args.seed,
        )
        test_dataset = PitchTypeLocationBatchIterableDataset(
            seasons=[args.test_season],
            load_season=load_season_frame,
            transform_season=feature_engine.transform,
            feature_columns=feature_cols,
            batch_size=args.batch_size,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
            shuffle=False,
            seed=args.seed,
        )

        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=0)
        n_features = train_dataset.n_features
    else:
        print("\nTransforming training data...")
        train_df = feature_engine.transform(train_df)

        print(f"\nValidation season: {args.val_season}")
        val_df = load_season_frame(args.val_season)
        val_df = feature_engine.transform(val_df)

        print(f"\nTest season: {args.test_season}")
        test_df = load_season_frame(args.test_season)
        test_df = feature_engine.transform(test_df)

        feature_cols = get_feature_columns(train_df)
        print(f"\nUsing {len(feature_cols)} features")

        print("\n" + "=" * 70)
        print("CREATING DATASETS")
        print("=" * 70)

        print("\nTraining set:")
        train_dataset = PitchTypeLocationDataset(
            train_df,
            feature_columns=feature_cols,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
        )

        print("\nValidation set:")
        val_dataset = PitchTypeLocationDataset(
            val_df,
            feature_columns=feature_cols,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
        )

        print("\nTest set:")
        test_dataset = PitchTypeLocationDataset(
            test_df,
            feature_columns=feature_cols,
            pitch_type_column="pitch_type_idx",
            location_columns=["px", "pz"],
            exclude_from_features=["pitch_type_idx"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        n_features = train_dataset.n_features

    print(f"\nInput features: {n_features}")

    # =========================================================================
    # Create and Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING PITCH-TYPE-CONDITIONED MODEL")
    print("=" * 70)

    model = PitchTypeConditionedMDN(
        n_features=n_features,
        n_pitch_types=len(PITCH_TYPE_CODES),
        hidden_dims=args.hidden_dims,
        n_components=args.n_components,
        dropout=args.dropout,
    )

    print("\nModel architecture:")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Components per pitch type: {args.n_components}")
    print(f"  Total pitch type heads: {len(PITCH_TYPE_CODES)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    trainer = PitchTypeLocationTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=str(device),
    )

    print("\nTraining...")
    train_results = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.n_epochs,
        early_stopping_patience=10,
        verbose=True,
    )

    print(f"\nBest validation NLL: {train_results['best_val_nll']:.4f}")

    # =========================================================================
    # Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)

    test_metrics = evaluate_per_pitch_type(model, test_loader, device)

    print("\nOverall Test Metrics:")
    for k, v in test_metrics["overall"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-Pitch-Type Metrics:")
    print(f"{'Pitch':<8} {'NLL':>8} {'MAE_px':>8} {'MAE_pz':>8} {'Eucl':>8} {'Cov90':>8} {'Count':>10}")
    print("-" * 70)
    for pt_code, pt_metrics in sorted(test_metrics["per_pitch_type"].items()):
        print(
            f"{pt_code:<8} "
            f"{pt_metrics['nll']:>8.3f} "
            f"{pt_metrics['mae_px']:>8.3f} "
            f"{pt_metrics['mae_pz']:>8.3f} "
            f"{pt_metrics['euclidean']:>8.3f} "
            f"{pt_metrics['coverage_90']:>8.1%} "
            f"{pt_metrics['count']:>10,}"
        )

    # =========================================================================
    # Compare to Baseline (Optional)
    # =========================================================================
    if args.compare_baseline:
        baseline_model = train_baseline_model(
            train_loader,
            val_loader,
            n_features=n_features,
            hidden_dims=args.hidden_dims,
            n_components=args.n_components,
            dropout=args.dropout,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            device=device,
        )

        print("\n" + "=" * 70)
        print("COMPARING TO BASELINE")
        print("=" * 70)

        comparison = compare_to_baseline(model, baseline_model, test_loader, str(device))

        print(f"\nConditioned Model NLL: {comparison['conditioned_nll']:.4f}")
        print(f"Baseline Model NLL:    {comparison['baseline_nll']:.4f}")
        print(f"NLL Improvement:       {comparison['nll_improvement']:.4f}")
        print(f"NLL Improvement %:     {comparison['nll_improvement_pct']:.2f}%")

        test_metrics["baseline_comparison"] = comparison

        # Save baseline model
        baseline_path = output_dir / "baseline_model.pt"
        torch.save(baseline_model.state_dict(), baseline_path)
        print(f"\nSaved baseline model to: {baseline_path}")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save model
    model_path = output_dir / "pitch_type_location_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Save config
    config_path = output_dir / "config.json"
    config["feature_columns"] = feature_cols
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    history = {k: [float(v) for v in vals] for k, vals in train_results["history"].items()}
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    # Save feature engine
    fe_path = output_dir / "feature_engine.pt"
    torch.save({
        "pitcher_to_idx": feature_engine.pitcher_to_idx,
        "batter_to_idx": feature_engine.batter_to_idx,
        "pitcher_ff_pct": feature_engine.pitcher_ff_pct,
        "pitcher_repertoire_size": feature_engine.pitcher_repertoire_size,
    }, fe_path)
    print(f"Saved feature engine to: {fe_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train pitch-type-conditioned location model"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="postgres",
        help="Training data source: 'postgres' or a parquet path",
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit training seasons; default uses all available pre-validation seasons except 2020",
    )
    parser.add_argument(
        "--val-season",
        type=str,
        default="2024",
        help="Validation season",
    )
    parser.add_argument(
        "--test-season",
        type=str,
        default="2025",
        help="Test season",
    )

    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Stream season-sized batches instead of materializing the full training window in memory",
    )
    # Model arguments
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of mixture components per pitch type",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--movement-profiles-dir",
        type=str,
        default=None,
        help="Enable pitcher movement profile features from this store directory",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory",
    )

    # Comparison arguments
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Train and compare to baseline model (no pitch type conditioning)",
    )

    # Quick mode for testing
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer epochs and smaller data",
    )

    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.n_epochs = 3
        args.train_seasons = ["2023"]
        args.hidden_dims = [128, 64]
        print("Quick mode enabled: reduced epochs, data, and model size")
    if args.train_seasons is None:
        args.train_seasons = default_data_source_train_seasons(
            args.data_path,
            val_season=args.val_season,
            test_season=args.test_season,
            exclude_2020=True,
        )

    run_training(args)


if __name__ == "__main__":
    main()
