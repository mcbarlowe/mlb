#!/usr/bin/env python
"""
Main training script for pitch prediction model.

Usage:
    # Run cross-validation with default settings
    uv run python scripts/train_model.py --mode cv

    # Run hyperparameter search
    uv run python scripts/train_model.py --mode search --n-configs 20

    # Train final model
    uv run python scripts/train_model.py --mode final

    # Evaluate on test set
    uv run python scripts/train_model.py --mode test --checkpoint models/checkpoints/best_model.pt
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.ml.model import create_model
from src.ml.train import PitchPredictionTrainer, PitchPredictionLoss
from src.ml.evaluate import evaluate_model, plot_mdn_predictions, plot_confusion_matrix
from src.ml.cross_validation import (
    TimeSeriesCrossValidator,
    CVResults,
    run_cross_validation,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
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


def create_model_from_config(config: dict, cv: TimeSeriesCrossValidator):
    """Create a model from a configuration dict."""
    return create_model(
        n_pitch_types=cv.n_pitch_types,
        n_pitchers=cv.n_pitchers,
        n_batters=cv.n_batters,
        n_features=cv.n_features,
        model_type=config.get("model_type", "lstm"),
        hidden_dim=config.get("hidden_dim", 128),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        embedding_dim=config.get("embedding_dim", 32),
        n_location_components=config.get("n_location_components", 3),
    )


def train_fold(
    model,
    train_loader,
    val_loader,
    n_epochs: int = 30,
    early_stopping_patience: int = 5,
    device: str = "auto",
    learning_rate: float = 1e-3,
    type_weight: float = 1.0,
    location_weight: float = 0.5,
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """Train a model on a single fold."""
    trainer = PitchPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        type_weight=type_weight,
        location_weight=location_weight,
        checkpoint_dir=checkpoint_dir,
    )

    results = trainer.train(
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
    )

    return results


def run_cv_mode(args) -> None:
    """Run cross-validation."""
    print("=" * 60)
    print("Running Cross-Validation")
    print("=" * 60)

    set_seed(args.seed)

    # Setup CV
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=["2020"] if args.exclude_2020 else None,
    )

    # Configuration
    config = {
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "n_location_components": args.n_components,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "type_weight": args.type_weight,
        "location_weight": args.location_weight,
    }

    print(f"\nConfiguration: {config}")

    # Run CV
    results = run_cross_validation(
        model_fn=lambda: create_model_from_config(config, cv),
        cv=cv,
        train_fn=lambda model, train_loader, val_loader, **kwargs: train_fold(
            model,
            train_loader,
            val_loader,
            learning_rate=config["learning_rate"],
            type_weight=config["type_weight"],
            location_weight=config["location_weight"],
            **kwargs,
        ),
        eval_fn=evaluate_model,
        config=config,
        n_epochs=args.n_epochs,
        early_stopping_patience=args.patience,
        device=args.device,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"cv_results_{timestamp}.json"
    results.to_json(results_path)
    print(f"\nResults saved to: {results_path}")


def run_search_mode(args) -> None:
    """Run hyperparameter search."""
    print("=" * 60)
    print("Running Hyperparameter Search")
    print("=" * 60)

    set_seed(args.seed)

    # Setup CV
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=["2020"] if args.exclude_2020 else None,
    )

    # Hyperparameter search space
    search_space = {
        "hidden_dim": [64, 128, 256],
        "n_layers": [1, 2, 3],
        "dropout": [0.1, 0.3, 0.5],
        "n_location_components": [1, 3, 5],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "type_weight": [0.5, 1.0, 2.0],
        "location_weight": [0.25, 0.5, 1.0],
        "embedding_dim": [16, 32, 64],
    }

    # Generate random configurations
    configs = []
    for _ in range(args.n_configs):
        config = {key: random.choice(values) for key, values in search_space.items()}
        configs.append(config)

    print(f"\nSearching {len(configs)} configurations")

    # Initialize feature engine by getting first fold
    _ = list(cv.get_folds())  # This fits the feature engine

    all_results = []
    best_score = float("inf")
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*60}")
        print(f"Config: {config}")

        try:
            # Re-create CV iterator for each config
            cv._season_data = {}  # Clear cached data

            results = run_cross_validation(
                model_fn=lambda c=config: create_model_from_config(c, cv),
                cv=cv,
                train_fn=lambda model, train_loader, val_loader, c=config, **kwargs: train_fold(
                    model,
                    train_loader,
                    val_loader,
                    learning_rate=c["learning_rate"],
                    type_weight=c["type_weight"],
                    location_weight=c["location_weight"],
                    **kwargs,
                ),
                eval_fn=evaluate_model,
                config=config,
                n_epochs=args.n_epochs,
                early_stopping_patience=args.patience,
                device=args.device,
            )

            summary = results.summary()
            mean_nll = summary.get("nll", float("inf"))

            all_results.append({
                "config": config,
                "summary": summary,
                "fold_results": results.fold_results,
            })

            if mean_nll < best_score:
                best_score = mean_nll
                best_config = config
                print(f"\n*** New best config! NLL: {mean_nll:.4f}")

        except Exception as e:
            print(f"Error with config {config}: {e}")
            continue

    # Save all results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_results = {
        "search_space": search_space,
        "n_configs": len(configs),
        "best_config": best_config,
        "best_score": best_score,
        "all_results": all_results,
    }

    results_path = output_dir / f"hp_search_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(search_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Hyperparameter Search Complete")
    print(f"{'='*60}")
    print(f"Best NLL: {best_score:.4f}")
    print(f"Best Config: {best_config}")
    print(f"Results saved to: {results_path}")


def run_final_mode(args) -> None:
    """Train final model on all training data."""
    print("=" * 60)
    print("Training Final Model")
    print("=" * 60)

    set_seed(args.seed)

    # Setup CV for data loading
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=["2020"] if args.exclude_2020 else None,
    )

    # Get final train/val loaders
    train_loader, val_loader, n_train, n_val = cv.get_final_train_val_loaders(
        val_season="2024"
    )
    print(f"Training on {n_train:,} at-bats, validating on {n_val:,} at-bats")

    # Configuration (use best from search or defaults)
    config = {
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "n_location_components": args.n_components,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "type_weight": args.type_weight,
        "location_weight": args.location_weight,
    }

    print(f"\nConfiguration: {config}")

    # Create model
    model = create_model_from_config(config, cv)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Setup checkpoint directory
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    )

    results = trainer.train(
        n_epochs=args.n_epochs,
        early_stopping_patience=args.patience,
    )

    # Save training info
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_path = output_dir / f"final_model_info_{timestamp}.json"

    with open(info_path, "w") as f:
        json.dump({
            "config": config,
            "n_parameters": n_params,
            "best_val_loss": results["best_val_loss"],
            "total_epochs": results["total_epochs"],
            "checkpoint_path": str(checkpoint_dir / "best_model.pt"),
        }, f, indent=2)

    print(f"\nFinal model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Training info saved to: {info_path}")


def run_test_mode(args) -> None:
    """Evaluate model on held-out test set."""
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    set_seed(args.seed)

    # Setup CV for data loading
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=["2020"] if args.exclude_2020 else None,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Need to get model dimensions from CV
    # Initialize feature engine by iterating folds
    _ = list(cv.get_folds())

    # Configuration (from checkpoint or defaults)
    config = {
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "n_location_components": args.n_components,
        "embedding_dim": args.embedding_dim,
    }

    # Create model and load weights
    model = create_model_from_config(config, cv)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded from checkpoint")

    # Get test loader
    test_loader, n_test = cv.get_test_loader()
    print(f"Test set: {n_test:,} at-bats from season {cv.test_season}")

    # Evaluate
    device = get_device(args.device)
    results = evaluate_model(model, test_loader, device=args.device)

    print(f"\n{'='*60}")
    print("Test Set Results")
    print(f"{'='*60}")

    # Classification metrics
    print("\nPitch Type Classification:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    print(f"  Macro F1: {results['f1_macro']:.4f}")
    print(f"  Weighted F1: {results['f1_weighted']:.4f}")

    # Location metrics
    print("\nLocation Prediction (MDN):")
    print(f"  NLL: {results['nll']:.4f}")
    print(f"  MAE px: {results['mae_px']:.4f} ft")
    print(f"  MAE pz: {results['mae_pz']:.4f} ft")
    print(f"  RMSE px: {results['rmse_px']:.4f} ft")
    print(f"  RMSE pz: {results['rmse_pz']:.4f} ft")
    print(f"  Euclidean Error: {results['euclidean_error']:.4f} ft")

    # Calibration
    print("\nCalibration (Coverage):")
    print(f"  Coverage @90%: {results['coverage_90']:.2%}")
    print(f"  Coverage @95%: {results['coverage_95']:.2%}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"test_results_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in results.items()
        if k not in ["mdn_params"]  # Skip large arrays
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Generate plots
    if args.plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            results["type_targets"],
            results["type_preds"],
            save_path=str(plots_dir / f"confusion_matrix_{timestamp}.png"),
        )

        # MDN predictions
        plot_mdn_predictions(
            results["mdn_params"],
            results["loc_targets"],
            n_samples=50,
            save_path=str(plots_dir / f"mdn_predictions_{timestamp}.png"),
        )

        print(f"Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train pitch prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cv", "search", "final", "test"],
        default="cv",
        help="Training mode: cv (cross-validation), search (hyperparameter search), "
        "final (train final model), test (evaluate on test set)",
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/livefeeds",
        help="Path to processed parquet files",
    )
    parser.add_argument(
        "--exclude-2020",
        action="store_true",
        help="Exclude 2020 COVID-shortened season",
    )

    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--n-components", type=int, default=3, help="MDN mixture components")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=30, help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--type-weight", type=float, default=1.0, help="Pitch type loss weight")
    parser.add_argument("--location-weight", type=float, default=0.5, help="Location loss weight")

    # Search arguments
    parser.add_argument("--n-configs", type=int, default=20, help="Number of HP configs to try")

    # Test arguments
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for results",
    )

    # Misc arguments
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == "cv":
        run_cv_mode(args)
    elif args.mode == "search":
        run_search_mode(args)
    elif args.mode == "final":
        run_final_mode(args)
    elif args.mode == "test":
        if not args.checkpoint:
            parser.error("--checkpoint is required for test mode")
        run_test_mode(args)


if __name__ == "__main__":
    main()
