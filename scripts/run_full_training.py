"""
Training script for pitch prediction model.

By default, this uses every available season before the validation year from
the selected data source, excluding 2020 unless ``--include-2020`` is passed.
Validation defaults to 2024 and the held-out test season defaults to 2025.

Usage:
    # Run full training
    uv run python scripts/run_full_training.py

    # Quick test
    uv run python scripts/run_full_training.py --quick

    # Custom settings
    uv run python scripts/run_full_training.py --n-epochs 100 --hidden-dim 256
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.ml.cross_validation import TimeSeriesCrossValidator
from src.ml.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_location_predictions,
    plot_mdn_predictions,
    print_classification_report,
)
from src.ml.features import PITCH_TYPE_CODES, compute_class_weights
from src.ml.model import create_model
from src.ml.train import PitchPredictionTrainer


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def run_training(args) -> dict:
    """Run the training pipeline."""

    # Create output directory
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("PITCH PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device(args.device)}")
    print()

    set_seed(args.seed)

    # Model configuration
    config = {
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "n_location_components": args.n_components,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "type_weight": args.type_weight,
        "location_weight": args.location_weight,
    }
    if args.model_type in ["lstm_attention", "enhanced_attention"]:
        config["n_attention_heads"] = args.n_attention_heads
        config["n_attention_layers"] = args.n_attention_layers

    print("Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    # Setup data loader helper
    exclude_seasons = ["2020"] if args.exclude_2020 else None
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=exclude_seasons,
        train_seasons=args.train_seasons,
        val_seasons=[args.val_season],
        test_season=args.test_season,
    )

    print("Data Split:")
    train_seasons = args.train_seasons or [s for s in cv.train_seasons if s != args.val_season]
    print(f"  Train: {train_seasons}")
    print(f"  Validation: {args.val_season}")
    print(f"  Test: {args.test_season} (held out)")
    if exclude_seasons:
        print(f"  Excluded: {exclude_seasons}")
    print()

    results = {
        "config": config,
        "timestamp": timestamp,
        "args": vars(args),
    }

    # =========================================================================
    # PHASE 1: Load and Prepare Data
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: PREPARING DATA")
    print("=" * 70)
    print()

    train_loader, val_loader, n_train, n_val = cv.get_final_train_val_loaders(
        val_season=args.val_season
    )
    assert cv.feature_engine is not None
    print(f"\nTraining: {n_train:,} at-bats")
    print(f"Validation: {n_val:,} at-bats")

    # Validate feature dimensions
    feature_cols = cv.feature_engine.get_feature_columns()
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Check actual tensor shape from dataloader
    sample_batch = next(iter(train_loader))
    actual_features = sample_batch["features"].shape[-1]
    print(f"Actual tensor features: {actual_features}")

    if actual_features != len(feature_cols):
        raise ValueError(
            f"Feature dimension mismatch! Expected {len(feature_cols)} features "
            f"but got {actual_features}. This may be due to cached data or code changes."
        )
    print()

    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights from training data...")
        # Get training seasons (excluding validation)
        train_seasons = [s for s in cv.train_seasons if s != args.val_season]
        # Access cached season data
        import polars as pl
        train_dfs = [cv._season_data[s] for s in train_seasons if s in cv._season_data]
        if train_dfs:
            combined_train = pl.concat(train_dfs, how="diagonal")
            # Transform to get pitch_type_idx
            transformed = cv.feature_engine.transform(combined_train)
            # Compute class weights
            class_weights = compute_class_weights(
                transformed,
                pitch_type_col="pitch_type_idx",
                n_classes=len(PITCH_TYPE_CODES),
                smoothing=args.class_weight_smoothing,
            )
            print(f"Class weights (smoothing={args.class_weight_smoothing}):")
            for i, code in enumerate(PITCH_TYPE_CODES):
                print(f"  {code}: {class_weights[i]:.3f}")
            print()

    # =========================================================================
    # PHASE 2: Train Model
    # =========================================================================
    print("=" * 70)
    print("PHASE 2: TRAINING MODEL")
    print("=" * 70)
    print()

    # Create model with dynamic feature indices
    feature_indices = cv.feature_engine.get_feature_indices()
    model_kwargs = {
        "n_pitch_types": cv.n_pitch_types,
        "n_pitchers": cv.n_pitchers,
        "n_batters": cv.n_batters,
        "n_features": cv.n_features,
        "model_type": args.model_type,
        "feature_indices": feature_indices,
        "hidden_dim": config["hidden_dim"],
        "n_layers": config["n_layers"],
        "dropout": config["dropout"],
        "embedding_dim": config["embedding_dim"],
        "n_location_components": config["n_location_components"],
    }

    # Add attention-specific args if using attention model
    if args.model_type in ["lstm_attention", "enhanced_attention"]:
        model_kwargs["n_attention_heads"] = args.n_attention_heads
        model_kwargs["n_attention_layers"] = args.n_attention_layers

    model = create_model(**model_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Train
    trainer = PitchPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=config["learning_rate"],
        type_weight=config["type_weight"],
        location_weight=config["location_weight"],
        checkpoint_dir=checkpoint_dir,
        class_weights=class_weights,
    )

    train_results = trainer.train(
        n_epochs=args.n_epochs,
        early_stopping_patience=args.patience,
        show_batch_progress=args.show_batch_progress,
    )

    results["training"] = {
        "best_val_loss": train_results["best_val_loss"],
        "total_epochs": train_results["total_epochs"],
        "n_parameters": n_params,
    }

    print()
    print(f"Training complete: {train_results['total_epochs']} epochs")
    print(f"Best validation loss: {train_results['best_val_loss']:.4f}")
    print()

    # Load best model
    best_checkpoint = checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # =========================================================================
    # PHASE 3: Evaluate on Test Set
    # =========================================================================
    print("=" * 70)
    print("PHASE 3: TEST SET EVALUATION")
    print("=" * 70)
    print()

    test_loader, n_test = cv.get_test_loader()
    print(f"Test set: {n_test:,} at-bats from season {cv.test_season}")
    print()

    test_results = evaluate_model(model, test_loader, device=args.device)

    print("Classification Metrics (Pitch Type):")
    print(f"  Accuracy:      {test_results['accuracy']:.4f}")
    print(f"  Top-3 Acc:     {test_results['top3_accuracy']:.4f}")
    print(f"  Macro F1:      {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1:   {test_results['f1_weighted']:.4f}")
    print()

    print("Location Metrics (MDN):")
    print(f"  NLL:           {test_results['nll']:.4f}")
    print(f"  MAE px:        {test_results['mae_px']:.4f} ft")
    print(f"  MAE pz:        {test_results['mae_pz']:.4f} ft")
    print(f"  Euclidean:     {test_results['euclidean_error']:.4f} ft")
    print()

    print("Calibration:")
    print(f"  Coverage @90%: {test_results['coverage_90']:.1%}")
    print(f"  Coverage @95%: {test_results['coverage_95']:.1%}")
    print()

    results["test_results"] = {
        "accuracy": test_results["accuracy"],
        "top3_accuracy": test_results["top3_accuracy"],
        "f1_macro": test_results["f1_macro"],
        "f1_weighted": test_results["f1_weighted"],
        "nll": test_results["nll"],
        "mae_px": test_results["mae_px"],
        "mae_pz": test_results["mae_pz"],
        "rmse_px": test_results["rmse_px"],
        "rmse_pz": test_results["rmse_pz"],
        "euclidean_error": test_results["euclidean_error"],
        "coverage_90": test_results["coverage_90"],
        "coverage_95": test_results["coverage_95"],
    }

    # =========================================================================
    # PHASE 4: Save Outputs
    # =========================================================================
    print("=" * 70)
    print("PHASE 4: SAVING OUTPUTS")
    print("=" * 70)
    print()

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved: {final_model_path}")

    # Save feature engine (needed for inference)
    feature_engine_path = output_dir / "feature_engine.json"
    assert cv.feature_engine is not None
    cv.feature_engine.save(feature_engine_path)
    print(f"Feature engine saved: {feature_engine_path}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    report_path = None
    # Generate plots
    if args.plots:
        print("\nGenerating plots...")

        plot_confusion_matrix(
            test_results["type_targets"],
            test_results["type_preds"],
            save_path=str(plots_dir / "confusion_matrix.png"),
        )
        print(f"  {plots_dir / 'confusion_matrix.png'}")

        plot_location_predictions(
            test_results["loc_preds"],
            test_results["loc_targets"],
            save_path=str(plots_dir / "location_predictions.png"),
        )
        print(f"  {plots_dir / 'location_predictions.png'}")

        plot_mdn_predictions(
            test_results["mdn_params"],
            test_results["loc_targets"],
            n_samples=30,
            save_path=str(plots_dir / "mdn_predictions.png"),
        )
        print(f"  {plots_dir / 'mdn_predictions.png'}")

        # Classification report
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            print_classification_report(
                test_results["type_targets"],
                test_results["type_preds"],
            )
            sys.stdout = old_stdout
        print(f"  {report_path}")

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
    print(f"  Location NLL:        {test_results['nll']:.3f}")
    print(f"  Coverage @95%:       {test_results['coverage_95']:.1%}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train pitch prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data-path", type=str, default="postgres",
                        help="Training data source: 'postgres' or a parquet path")
    parser.add_argument("--train-seasons", nargs="+", type=str, default=None,
                        help="Optional explicit training seasons; default uses all available pre-validation seasons")
    parser.add_argument("--val-season", type=str, default="2024")
    parser.add_argument("--test-season", type=str, default="2025")
    parser.add_argument("--exclude-2020", action="store_true", default=True)
    parser.add_argument("--include-2020", action="store_true")

    # Model
    parser.add_argument("--model-type", type=str, default="lstm",
                        choices=["lstm", "lstm_attention", "enhanced", "enhanced_attention", "simple"],
                        help="Model architecture type: lstm, lstm_attention, enhanced (pitch-conditioned location), enhanced_attention, or simple")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--n-attention-heads", type=int, default=4,
                        help="Number of attention heads (for lstm_attention)")
    parser.add_argument("--n-attention-layers", type=int, default=1,
                        help="Number of stacked attention layers (for lstm_attention)")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--type-weight", type=float, default=1.0)
    parser.add_argument("--location-weight", type=float, default=0.5)
    parser.add_argument("--use-class-weights", action="store_true", default=True,
                        help="Use inverse frequency class weighting (default: True)")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weighting")
    parser.add_argument("--class-weight-smoothing", type=float, default=0.5,
                        help="Smoothing factor for class weights (0=uniform, 1=full inverse freq)")

    # Output
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-batch-progress", action="store_true", default=False)

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick test (10 epochs)")

    args = parser.parse_args()

    if args.include_2020:
        args.exclude_2020 = False
    if args.no_plots:
        args.plots = False
    if args.no_class_weights:
        args.use_class_weights = False
    if args.quick:
        args.n_epochs = 10
        args.patience = 3

    run_training(args)


if __name__ == "__main__":
    main()
