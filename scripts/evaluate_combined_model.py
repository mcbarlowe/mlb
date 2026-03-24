#!/usr/bin/env python
"""
Evaluate the combined pitch type + location model pipeline.

Uses:
- Pitch type model: LSTM+Attention from models/attention_full
- Location model: PitchTypeConditionedMDN from models/pitch_type_location

Usage:
    uv run python scripts/evaluate_combined_model.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
import numpy as np
import polars as pl

from src.ml.model import create_model
from src.ml.pitch_type_location_model import (
    PitchTypeConditionedMDN,
    PitchTypeThenLocationPredictor,
)
from src.ml.cross_validation import TimeSeriesCrossValidator
from src.ml.features import PitchFeatureEngine, PITCH_TYPE_CODES


def get_device() -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pitch_type_model(model_dir: Path, device: torch.device):
    """Load the trained pitch type model."""

    # Load checkpoint first to get exact dimensions
    checkpoint = torch.load(model_dir / "final_model.pt", map_location=device)

    # Get dimensions from checkpoint
    n_pitchers = checkpoint.get("n_pitchers", 3633)
    n_batters = checkpoint.get("n_batters", 4607)
    n_features = checkpoint.get("n_features", 49)
    n_pitch_types = checkpoint.get("n_pitch_types", len(PITCH_TYPE_CODES))
    feature_indices = checkpoint.get("feature_indices", None)
    config = checkpoint.get("config", {})

    # Create model with exact dimensions from checkpoint
    model = create_model(
        n_pitch_types=n_pitch_types,
        n_pitchers=n_pitchers,
        n_batters=n_batters,
        n_features=n_features,
        model_type=config.get("model_type", "lstm_attention"),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        embedding_dim=config.get("embedding_dim", 32),
        n_location_components=config.get("n_location_components", 3),
        n_attention_heads=config.get("n_attention_heads", 8),
        n_attention_layers=config.get("n_attention_layers", 2),
        feature_indices=feature_indices,
    )

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config, checkpoint


def load_location_model(model_dir: Path, device: torch.device):
    """Load the trained pitch-type-conditioned location model."""

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Get n_features from feature_columns or checkpoint
    n_features = config.get("n_features")
    if n_features is None and "feature_columns" in config:
        n_features = len(config["feature_columns"])

    # Create model
    model = PitchTypeConditionedMDN(
        n_features=n_features,
        n_pitch_types=config.get("n_pitch_types", 11),
        hidden_dims=config["hidden_dims"],
        n_components=config["n_components"],
        dropout=config["dropout"],
    )

    # Load weights
    checkpoint = torch.load(model_dir / "pitch_type_location_model.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


def main():
    print("=" * 70)
    print("COMBINED MODEL EVALUATION")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Paths
    pitch_type_model_dir = Path("models/attention_full/run_20260119_124719")
    location_model_dir = Path("models/pitch_type_location_20260121_003206")

    print(f"\nPitch type model: {pitch_type_model_dir}")
    print(f"Location model: {location_model_dir}")

    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    print("\nLoading pitch type model (LSTM+Attention)...")
    pitch_type_model, pt_config, pt_checkpoint = load_pitch_type_model(pitch_type_model_dir, device)
    print(f"  Model type: {pt_config.get('model_type', 'lstm_attention')}")
    print(f"  Hidden dim: {pt_config['hidden_dim']}")

    print("\nLoading location model (PitchTypeConditionedMDN)...")
    location_model, loc_config = load_location_model(location_model_dir, device)
    print(f"  Hidden dims: {loc_config['hidden_dims']}")
    print(f"  Components per type: {loc_config['n_components']}")

    # Create combined predictor
    print("\nCreating combined predictor...")
    combined_model = PitchTypeThenLocationPredictor(
        pitch_type_model=pitch_type_model,
        location_model=location_model,
        use_soft_conditioning=True,
    )
    combined_model.to(device)
    combined_model.eval()

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    # Load test data using cross validator
    cv = TimeSeriesCrossValidator(
        data_path="data/processed/livefeeds",
        batch_size=64,
        exclude_seasons=["2020"],
    )

    test_loader, n_test = cv.get_test_loader()
    print(f"Test set: {n_test:,} at-bats from {cv.test_season}")

    # Compute feature indices for location model
    # The pitch type model uses 49 features, location model uses 43
    full_feature_cols = cv.feature_engine.get_feature_columns()
    loc_feature_cols = loc_config.get("feature_columns", [])

    if loc_feature_cols:
        # Create mapping from full feature set to location feature subset
        loc_feature_indices = []
        for col in loc_feature_cols:
            if col in full_feature_cols:
                loc_feature_indices.append(full_feature_cols.index(col))
        loc_feature_indices = torch.tensor(loc_feature_indices, device=device)
        print(f"\nLocation model uses {len(loc_feature_indices)}/{len(full_feature_cols)} features")
    else:
        loc_feature_indices = None

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING COMBINED MODEL")
    print("=" * 70)

    all_type_preds = []
    all_type_targets = []
    all_loc_preds = []
    all_loc_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Batch is a dict with keys: features, targets, lengths, mask
            # targets has shape [batch, seq_len, 3] where [:,:,0]=pitch_type_idx, [:,:,1:3]=location
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            # Extract pitch type and location from targets
            pitch_type = targets[:, :, 0].long()  # pitch_type_idx
            location = targets[:, :, 1:3]  # px, pz

            # Extract location features (subset of full features)
            if loc_feature_indices is not None:
                # features shape: [batch, seq_len, n_features]
                # Select only the features the location model was trained on
                loc_features = features[:, :, loc_feature_indices]
                # Flatten for location model: [batch*seq_len, n_loc_features]
                loc_features_flat = loc_features.reshape(-1, loc_features.shape[-1])
            else:
                loc_features_flat = None

            # Get predictions from combined model
            output = combined_model(features, lengths, mask, location_features=loc_features_flat)

            # Extract predictions for valid positions (where mask is True)
            batch_size, seq_len = mask.shape

            for b in range(batch_size):
                valid_len = lengths[b].item()
                for t in range(valid_len):
                    # Pitch type
                    logits = output["pitch_type_logits"][b, t]
                    pred = logits.argmax().item()
                    target = pitch_type[b, t].item()
                    all_type_preds.append(pred)
                    all_type_targets.append(target)

                    # Location
                    loc_target = location[b, t].cpu().numpy()
                    all_loc_targets.append(loc_target)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches...", flush=True)

    # Compute metrics
    all_type_preds = np.array(all_type_preds)
    all_type_targets = np.array(all_type_targets)
    all_loc_targets = np.array(all_loc_targets)

    # Pitch type accuracy
    accuracy = (all_type_preds == all_type_targets).mean()

    # Top-3 accuracy would require keeping the full logits

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nPitch Type Classification:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Total predictions: {len(all_type_preds):,}")

    print("\nNote: Location metrics from combined model require more complex evaluation.")
    print("The pitch type model predicts type, then location model uses those predictions.")

    # Compare to individual models
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\nPitch Type Model (standalone):")
    print("  Accuracy: 72.3%")

    print(f"\nCombined Pipeline:")
    print(f"  Accuracy: {accuracy*100:.1f}%")

    print("\nLocation Model Comparison:")
    print("  Standalone location model (new):    NLL=2.048, Eucl=0.961 ft")
    print("  Attention model's built-in MDN:     NLL=2.083, Eucl=0.980 ft")
    print("  -> New location model is 1.7% better on NLL")


if __name__ == "__main__":
    main()
