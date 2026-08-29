"""
Fine-tune only the location head on a pre-trained pitch prediction model.

This script:
1. Loads an existing trained LSTM model
2. Replaces the location head with the new HierarchicalMDN
3. Freezes all parameters except the location head
4. Trains only the location prediction

Usage:
    uv run python scripts/finetune_location.py --checkpoint models/attention_full/run_20260119_124719

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
import torch
import torch.nn.functional as F
from torch import nn

from mlb.ml.cross_validation import TimeSeriesCrossValidator
from mlb.ml.evaluate import (
    evaluate_model,
    plot_location_predictions,
    plot_mdn_predictions,
)
from mlb.ml.features import PitchFeatureEngine
from mlb.ml.model import (
    HierarchicalMDN,
    PitchPredictor,
    PitchPredictorWithAttention,
)
from mlb.ml.train import PitchPredictionTrainer


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


def upgrade_model_location_head(
    model: nn.Module,
    n_pitch_types: int,
    embedding_dim: int = 32,
    n_components_per_family: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Replace a model's location head with the new HierarchicalMDN.

    Args:
        model: Pre-trained model (PitchPredictor or PitchPredictorWithAttention)
        n_pitch_types: Number of pitch type classes
        embedding_dim: Embedding dimension for pitch type conditioning
        n_components_per_family: MDN components per pitch family
        dropout: Dropout rate

    Returns:
        Model with upgraded location head
    """
    hidden_dim = model.hidden_dim

    # Create new hierarchical location head
    new_location_head = HierarchicalMDN(
        hidden_dim=hidden_dim,
        n_pitch_types=n_pitch_types,
        embedding_dim=embedding_dim,
        n_components_per_family=n_components_per_family,
        dropout=dropout,
    )

    # Store original location head for reference
    model._original_location_head = model.location_head

    # Replace with new head
    model.location_head = new_location_head

    # Update n_location_components to match new head
    model.n_location_components = new_location_head.total_components

    # Override the forward method to use pitch-type conditioning
    def new_forward(features, lengths, mask, return_attention=False):
        """Modified forward that conditions location on pitch type."""
        _, seq_len, _ = features.shape

        # Extract embedding indices
        pitcher_idx = features[:, :, model.pitcher_idx_pos].long().clamp(
            0, model.pitcher_embedding.num_embeddings - 1
        )
        batter_idx = features[:, :, model.batter_idx_pos].long().clamp(
            0, model.batter_embedding.num_embeddings - 1
        )
        prev_pitch_idx = features[:, :, model.prev_pitch_idx_pos].long().clamp(
            -1, model.n_pitch_types - 1
        )
        prev_pitch_idx = torch.where(
            prev_pitch_idx < 0,
            torch.tensor(model.n_pitch_types, device=prev_pitch_idx.device),
            prev_pitch_idx,
        )

        # Get embeddings
        pitcher_emb = model.pitcher_embedding(pitcher_idx)
        batter_emb = model.batter_embedding(batter_idx)
        prev_pitch_emb = model.prev_pitch_embedding(prev_pitch_idx)

        # Extract continuous features
        continuous_features = features[:, :, model.continuous_indices]

        # Concatenate
        x = torch.cat([
            pitcher_emb,
            batter_emb,
            prev_pitch_emb,
            continuous_features,
        ], dim=-1)

        # LSTM
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = model.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        # Attention (if model has it)
        attn_weights = None
        if hasattr(model, 'attention_layers'):
            for attn_layer in model.attention_layers:
                lstm_out, attn_weights = attn_layer(
                    lstm_out, mask, return_weights=return_attention
                )

        lstm_out = model.dropout(lstm_out)

        # Predict pitch type
        pitch_type_logits = model.pitch_type_head(lstm_out)
        pitch_type_probs = F.softmax(pitch_type_logits, dim=-1)

        # Predict location conditioned on pitch type (NEW!)
        mdn_params = model.location_head(lstm_out, pitch_type_probs)

        if return_attention:
            return pitch_type_logits, mdn_params, attn_weights
        return pitch_type_logits, mdn_params

    # Bind the new forward method
    import types
    model.forward = types.MethodType(lambda self, *args, **kwargs: new_forward(*args, **kwargs), model)
    # Actually, simpler approach - just replace forward directly
    model.forward = new_forward

    return model


def freeze_except_location_head(model: nn.Module) -> None:
    """Freeze all parameters except the location head."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze location head
    for param in model.location_head.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def run_finetuning(args) -> dict:
    """Run the location head fine-tuning pipeline."""

    # Find checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        # Look for best_model.pt or final_model.pt
        if (checkpoint_path / "checkpoints" / "best_model.pt").exists():
            model_path = checkpoint_path / "checkpoints" / "best_model.pt"
        elif (checkpoint_path / "final_model.pt").exists():
            model_path = checkpoint_path / "final_model.pt"
        else:
            raise FileNotFoundError(f"No model found in {checkpoint_path}")
        feature_engine_path = checkpoint_path / "feature_engine.json"
    else:
        model_path = checkpoint_path
        feature_engine_path = checkpoint_path.parent / "feature_engine.json"

    print("=" * 70)
    print("LOCATION HEAD FINE-TUNING")
    print("=" * 70)
    print(f"Loading model from: {model_path}")
    print(f"Feature engine from: {feature_engine_path}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"finetune_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device(args.device)}")
    print()

    set_seed(args.seed)

    # Load feature engine
    feature_engine = PitchFeatureEngine.load(feature_engine_path)
    print(f"Loaded feature engine with {feature_engine.n_pitchers} pitchers, {feature_engine.n_batters} batters")

    # Setup data
    cv = TimeSeriesCrossValidator(
        data_path=args.data_path,
        batch_size=args.batch_size,
        exclude_seasons=["2020"] if args.exclude_2020 else None,
        val_seasons=["2024"],
        test_season="2025",
    )
    # Use the loaded feature engine
    cv.feature_engine = feature_engine

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, n_train, n_val = cv.get_final_train_val_loaders(val_season="2024")
    print(f"Training: {n_train:,} at-bats")
    print(f"Validation: {n_val:,} at-bats")

    # Load pre-trained model
    print("\nLoading pre-trained model...")
    checkpoint = torch.load(model_path, map_location="cpu")

    # Determine model type from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Check if it has attention layers
    has_attention = any("attention_layers" in k for k in state_dict)
    print(f"Model has attention: {has_attention}")

    # Get dimensions from state dict
    hidden_dim = state_dict["lstm.weight_hh_l0"].shape[1]
    n_pitch_types = state_dict["pitch_type_head.3.weight"].shape[0]
    embedding_dim = state_dict["pitcher_embedding.weight"].shape[1]
    n_pitchers = state_dict["pitcher_embedding.weight"].shape[0]
    n_batters = state_dict["batter_embedding.weight"].shape[0]

    print(f"Hidden dim: {hidden_dim}")
    print(f"Pitch types: {n_pitch_types}")
    print(f"Embedding dim: {embedding_dim}")

    # Get feature indices
    feature_indices = feature_engine.get_feature_indices()
    n_continuous = len(feature_indices["continuous_indices"])

    # Recreate model architecture
    if has_attention:
        # Count attention layers
        n_attn_layers = len([k for k in state_dict if "attention_layers" in k and "attention.q_proj.weight" in k])

        model = PitchPredictorWithAttention(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_continuous,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=args.dropout,
            n_location_components=3,  # Will be replaced
            n_attention_heads=4,
            n_attention_layers=n_attn_layers,
            feature_indices=feature_indices,
        )
    else:
        model = PitchPredictor(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_continuous,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=args.dropout,
            n_location_components=3,  # Will be replaced
            feature_indices=feature_indices,
        )

    # Load pre-trained weights
    model.load_state_dict(state_dict)
    print("Loaded pre-trained weights")

    # Upgrade location head
    print("\nUpgrading location head to HierarchicalMDN...")
    model = upgrade_model_location_head(
        model,
        n_pitch_types=n_pitch_types,
        embedding_dim=embedding_dim,
        n_components_per_family=args.n_components_per_family,
        dropout=args.dropout,
    )
    print(f"New location head has {model.n_location_components} mixture components")

    # Freeze everything except location head
    print("\nFreezing pre-trained layers...")
    freeze_except_location_head(model)

    # Training config
    config = {
        "base_checkpoint": str(model_path),
        "n_components_per_family": args.n_components_per_family,
        "total_components": model.n_location_components,
        "learning_rate": args.learning_rate,
        "location_weight": args.location_weight,
    }

    print("\nFine-tuning Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create trainer (only trains unfrozen params)
    trainer = PitchPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        type_weight=0.0,  # Don't train pitch type - it's frozen anyway
        location_weight=args.location_weight,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    print("=" * 70)
    print("TRAINING LOCATION HEAD")
    print("=" * 70)

    train_results = trainer.train(
        n_epochs=args.n_epochs,
        early_stopping_patience=args.patience,
        show_batch_progress=args.show_batch_progress,
    )

    print(f"\nTraining complete: {train_results['total_epochs']} epochs")
    print(f"Best validation loss: {train_results['best_val_loss']:.4f}")

    # Load best model
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    test_loader, n_test = cv.get_test_loader()
    print(f"Test set: {n_test:,} at-bats")

    test_results = evaluate_model(model, test_loader, device=args.device)

    print("\nClassification Metrics (Pitch Type - should be similar to original):")
    print(f"  Accuracy:      {test_results['accuracy']:.4f}")
    print(f"  Top-3 Acc:     {test_results['top3_accuracy']:.4f}")

    print("\nLocation Metrics (IMPROVED):")
    print(f"  NLL:           {test_results['nll']:.4f}")
    print(f"  MAE px:        {test_results['mae_px']:.4f} ft")
    print(f"  MAE pz:        {test_results['mae_pz']:.4f} ft")
    print(f"  Euclidean:     {test_results['euclidean_error']:.4f} ft")
    print(f"  Coverage @90%: {test_results['coverage_90']:.1%}")
    print(f"  Coverage @95%: {test_results['coverage_95']:.1%}")

    # Save
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    # Save model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved: {final_model_path}")

    # Save results
    results = {
        "config": config,
        "timestamp": timestamp,
        "training": {
            "best_val_loss": train_results["best_val_loss"],
            "total_epochs": train_results["total_epochs"],
        },
        "test_results": {
            "accuracy": test_results["accuracy"],
            "top3_accuracy": test_results["top3_accuracy"],
            "nll": test_results["nll"],
            "mae_px": test_results["mae_px"],
            "mae_pz": test_results["mae_pz"],
            "euclidean_error": test_results["euclidean_error"],
            "coverage_90": test_results["coverage_90"],
            "coverage_95": test_results["coverage_95"],
        }
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Plots
    if args.plots:
        print("\nGenerating plots...")

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

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune location head on pre-trained pitch prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained model checkpoint or directory")

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default="postgres",
        help="Training data source: 'postgres' or a parquet path",
    )
    parser.add_argument("--exclude-2020", action="store_true", default=True)

    # Model
    parser.add_argument("--n-components-per-family", type=int, default=2,
                        help="MDN components per pitch family (total = 4 * this)")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--location-weight", type=float, default=1.0)

    # Output
    parser.add_argument("--output-dir", type=str, default="models/location_finetuned")
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-batch-progress", action="store_true", default=False)

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.no_plots:
        args.plots = False

    run_finetuning(args)


if __name__ == "__main__":
    main()
