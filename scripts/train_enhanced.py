"""
Train the enhanced pitch prediction model with all improvements.

This script trains the PitchPredictorEnhanced model which includes:
1. Pitch-type-conditioned location prediction
2. Hierarchical MDN with pitch-family-specific components
3. Pitch family x bat_side interaction features

Usage:
    uv run python scripts/train_enhanced.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json

import numpy as np
import polars as pl
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


def get_device() -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    print("=" * 70)
    print("ENHANCED MODEL TRAINING")
    print("=" * 70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("models/enhanced") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Output: {output_dir}")

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Configuration
    config = {
        "model_type": "enhanced_attention",
        "hidden_dim": 128,
        "n_layers": 2,
        "dropout": 0.3,
        "n_location_components": 8,  # 2 per family x 4 families
        "embedding_dim": 32,
        "n_attention_heads": 4,
        "n_attention_layers": 2,
        "learning_rate": 1e-3,
        "type_weight": 1.0,
        "location_weight": 0.5,
        "batch_size": 64,
        "n_epochs": 50,
        "patience": 7,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Setup data - keep this script on 2023-only training to stay within memory,
    # but source the data from PostgreSQL so the validation/test seasons stay aligned
    # with the broader full-history training pipeline.
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Use only 2023 for training (full-history settings exceed this script's memory budget)
    cv = TimeSeriesCrossValidator(
        data_path="postgres",
        batch_size=config["batch_size"],
        exclude_seasons=["2020", "2018", "2019", "2021", "2022"],  # Only use 2023
        val_seasons=["2024"],
        test_season="2025",
    )

    train_loader, val_loader, n_train, n_val = cv.get_final_train_val_loaders(
        val_season="2024",
    )

    assert cv.feature_engine is not None
    print(f"\nTrain: {n_train:,} at-bats (2023)")
    print(f"Val: {n_val:,} at-bats (2024)")

    # Compute class weights
    print("\nComputing class weights...")
    train_dfs = [cv._season_data[s] for s in ["2023"] if s in cv._season_data]
    if train_dfs:
        combined_train = pl.concat(train_dfs, how="diagonal")
        transformed = cv.feature_engine.transform(combined_train)
        class_weights = compute_class_weights(
            transformed,
            pitch_type_col="pitch_type_idx",
            n_classes=len(PITCH_TYPE_CODES),
            smoothing=0.5,
        )
        print("Class weights:")
        for i, code in enumerate(PITCH_TYPE_CODES):
            print(f"  {code}: {class_weights[i]:.3f}")
    else:
        class_weights = None

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    feature_indices = cv.feature_engine.get_feature_indices()

    # Create the model - careful not to pass model_type twice
    model = create_model(
        n_pitch_types=cv.n_pitch_types,
        n_pitchers=cv.n_pitchers,
        n_batters=cv.n_batters,
        n_features=cv.n_features,
        model_type=config["model_type"],
        feature_indices=feature_indices,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        embedding_dim=config["embedding_dim"],
        n_location_components=config["n_location_components"],
        n_attention_heads=config["n_attention_heads"],
        n_attention_layers=config["n_attention_layers"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type']}")
    print(f"Parameters: {n_params:,}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    trainer = PitchPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=config["learning_rate"],
        type_weight=config["type_weight"],
        location_weight=config["location_weight"],
        checkpoint_dir=checkpoint_dir,
        class_weights=class_weights,
    )

    train_results = trainer.train(
        n_epochs=config["n_epochs"],
        early_stopping_patience=config["patience"],
        show_batch_progress=False,
    )

    print(f"\nTraining complete: {train_results['total_epochs']} epochs")
    print(f"Best validation loss: {train_results['best_val_loss']:.4f}")

    # Load best model
    best_checkpoint = checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    test_loader, n_test = cv.get_test_loader()
    print(f"Test set: {n_test:,} at-bats (2025)")

    test_results = evaluate_model(model, test_loader, device=str(device))

    print("\nPitch Type Metrics:")
    print(f"  Accuracy:      {test_results['accuracy']:.4f}")
    print(f"  Top-3 Acc:     {test_results['top3_accuracy']:.4f}")
    print(f"  Macro F1:      {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1:   {test_results['f1_weighted']:.4f}")

    print("\nLocation Metrics:")
    print(f"  NLL:           {test_results['nll']:.4f}")
    print(f"  MAE px:        {test_results['mae_px']:.4f} ft")
    print(f"  MAE pz:        {test_results['mae_pz']:.4f} ft")
    print(f"  Euclidean:     {test_results['euclidean_error']:.4f} ft")

    print("\nCalibration:")
    print(f"  Coverage @90%: {test_results['coverage_90']:.1%}")
    print(f"  Coverage @95%: {test_results['coverage_95']:.1%}")

    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Save model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model: {final_model_path}")

    # Save feature engine
    feature_engine_path = output_dir / "feature_engine.json"
    cv.feature_engine.save(feature_engine_path)
    print(f"Feature engine: {feature_engine_path}")

    # Save results
    results = {
        "config": config,
        "timestamp": timestamp,
        "training": {
            "best_val_loss": train_results["best_val_loss"],
            "total_epochs": train_results["total_epochs"],
            "n_parameters": n_params,
        },
        "test_results": {
            "accuracy": test_results["accuracy"],
            "top3_accuracy": test_results["top3_accuracy"],
            "f1_macro": test_results["f1_macro"],
            "f1_weighted": test_results["f1_weighted"],
            "nll": test_results["nll"],
            "mae_px": test_results["mae_px"],
            "mae_pz": test_results["mae_pz"],
            "euclidean_error": test_results["euclidean_error"],
            "coverage_90": test_results["coverage_90"],
            "coverage_95": test_results["coverage_95"],
        },
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrix(
        test_results["type_targets"],
        test_results["type_preds"],
        save_path=str(plots_dir / "confusion_matrix.png"),
    )
    plot_location_predictions(
        test_results["loc_preds"],
        test_results["loc_targets"],
        save_path=str(plots_dir / "location_predictions.png"),
    )
    plot_mdn_predictions(
        test_results["mdn_params"],
        test_results["loc_targets"],
        n_samples=30,
        save_path=str(plots_dir / "mdn_predictions.png"),
    )

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
    print(f"Classification report: {report_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")
    print("\nKey Results:")
    print(f"  Pitch Type Accuracy: {test_results['accuracy']:.1%}")
    print(f"  Top-3 Accuracy:      {test_results['top3_accuracy']:.1%}")
    print(f"  Location NLL:        {test_results['nll']:.3f}")


if __name__ == "__main__":
    main()
