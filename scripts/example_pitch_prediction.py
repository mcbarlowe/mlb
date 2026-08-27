#!/usr/bin/env python
"""
Example script demonstrating pitch prediction inference.

Shows how to:
1. Load trained models
2. Make predictions for individual pitches
3. Get pitch type probabilities, location point estimates, and density
4. Visualize predictions

Usage:
    uv run python scripts/example_pitch_prediction.py --model-dir models/combined_XXXXXX
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch

from src.ml.features import PitchFeatureEngine
from src.ml.pitch_predictor import (
    GameContext,
    PitchPredictor,
    create_pitch_card_from_row,
)


def main():
    parser = argparse.ArgumentParser(description="Example pitch prediction")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--data-path", default="data/processed/livefeeds")
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("PITCH PREDICTION EXAMPLE")
    print("=" * 70)

    # Load the predictor
    print(f"\nLoading models from: {args.model_dir}")
    predictor = PitchPredictor.load(args.model_dir)
    print("Models loaded successfully!")

    # Load some sample data for demonstration
    print("\nLoading sample data from 2025 season...")
    data_path = Path(args.data_path) / "2025"
    df = pl.scan_parquet(str(data_path / "*.parquet")).head(1000).collect()
    print(f"Loaded {len(df):,} pitches")

    # Prepare features using feature engine
    print("\nPreparing features...")
    feature_engine = PitchFeatureEngine(Path(args.data_path))

    # Load feature engine mappings from saved model
    import json
    with open(Path(args.model_dir) / "feature_engine.json") as f:
        fe_data = json.load(f)
        feature_engine.pitcher_to_idx = fe_data["pitcher_to_idx"]
        feature_engine.batter_to_idx = fe_data["batter_to_idx"]
        feature_engine.pitcher_ff_pct = {int(k): v for k, v in fe_data.get("pitcher_ff_pct", {}).items()}
        feature_engine.pitcher_repertoire = {int(k): v for k, v in fe_data.get("pitcher_repertoire", {}).items()}
        feature_engine._fitted = True

    # Transform data
    df = feature_engine.transform(df)

    # Filter to valid pitches
    df = df.filter(
        pl.col("pitch_type_idx").is_not_null()
        & pl.col("px").is_not_null()
        & pl.col("pz").is_not_null()
    )

    # Prepare CatBoost features
    catboost_cols = predictor.feature_columns

    # Add prev_pitch_type for CatBoost
    df = df.with_columns([
        pl.col("pitch_type_code")
        .shift(1)
        .over(["game_pk", "at_bat_index"])
        .fill_null("NONE")
        .alias("prev_pitch_type"),
    ])

    # Check which columns are available
    available_catboost = [c for c in catboost_cols if c in df.columns]
    print(f"CatBoost features: {len(available_catboost)}/{len(catboost_cols)}")

    # Prepare MDN features
    mdn_cols = predictor.mdn_feature_columns
    available_mdn = [c for c in mdn_cols if c in df.columns]
    print(f"MDN features: {len(available_mdn)}/{len(mdn_cols)}")

    # Make predictions for sample pitches
    print(f"\n{'=' * 70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 70}")

    # Select random examples
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), min(args.n_examples, len(df)), replace=False)

    for i, idx in enumerate(sample_indices):
        row = df[idx]

        # Get actual values
        actual_type = row["pitch_type_code"][0] if "pitch_type_code" in row.columns else "UNK"
        actual_px = row["px"][0] if "px" in row.columns else None
        actual_pz = row["pz"][0] if "pz" in row.columns else None

        # Prepare features
        catboost_features = row.select(available_catboost).to_pandas()
        mdn_features = torch.tensor(
            row.select(available_mdn).to_numpy().astype(np.float32),
            dtype=torch.float32,
        )
        mdn_features = torch.nan_to_num(mdn_features, nan=0.0)

        # Get prediction
        prediction = predictor.predict(catboost_features, mdn_features)

        # Get strike zone probability
        strike_prob = predictor.get_strike_zone_probability(prediction)

        print(f"\n--- Pitch {i+1} ---")
        print(f"Actual: {actual_type} at ({actual_px:.2f}, {actual_pz:.2f})")
        print("\nPitch Type Prediction:")
        for pitch_type, prob in prediction.top_3_types:
            marker = " <--" if pitch_type == actual_type else ""
            print(f"  {pitch_type}: {prob:.1%}{marker}")

        print("\nLocation Prediction:")
        print(f"  Expected: ({prediction.location_point[0]:.2f}, {prediction.location_point[1]:.2f})")
        print(f"  Mode:     ({prediction.location_mode[0]:.2f}, {prediction.location_mode[1]:.2f})")
        print(f"  Actual:   ({actual_px:.2f}, {actual_pz:.2f})")

        error = np.sqrt(
            (prediction.location_point[0] - actual_px)**2 +
            (prediction.location_point[1] - actual_pz)**2
        )
        print(f"  Error:    {error:.2f} ft")
        print(f"\nStrike Zone Probability: {strike_prob:.1%}")

        # Save visualizations for first example
        if i == 0:
            # Simple visualization
            output_path = Path(args.model_dir) / "example_prediction.png"
            predictor.plot_prediction(
                prediction,
                title=f"Prediction vs Actual: {actual_type}",
                actual_location=(actual_px, actual_pz),
                save_path=str(output_path),
            )
            print(f"\nSimple visualization saved: {output_path}")

            # Full pitch card with game context
            card_path = Path(args.model_dir) / "example_pitch_card.png"
            create_pitch_card_from_row(
                predictor,
                row,
                catboost_features,
                mdn_features,
                save_path=str(card_path),
            )
            print(f"Pitch card saved: {card_path}")

    # Batch prediction example
    print(f"\n{'=' * 70}")
    print("BATCH PREDICTION EXAMPLE")
    print(f"{'=' * 70}")

    batch_size = min(100, len(df))
    batch_df = df.head(batch_size)

    catboost_batch = batch_df.select(available_catboost).to_pandas()
    mdn_batch = torch.tensor(
        batch_df.select(available_mdn).to_numpy().astype(np.float32),
        dtype=torch.float32,
    )
    mdn_batch = torch.nan_to_num(mdn_batch, nan=0.0)

    batch_results = predictor.predict_batch(catboost_batch, mdn_batch)

    # Calculate accuracy
    actual_types = batch_df["pitch_type_idx"].to_numpy()
    correct = (batch_results["predicted_type_indices"] == actual_types).sum()
    accuracy = correct / batch_size

    # Calculate location error
    actual_locations = batch_df.select(["px", "pz"]).to_numpy()
    errors = np.sqrt(
        (batch_results["location_points"][:, 0] - actual_locations[:, 0])**2 +
        (batch_results["location_points"][:, 1] - actual_locations[:, 1])**2
    )

    print(f"\nBatch of {batch_size} pitches:")
    print(f"  Pitch Type Accuracy: {accuracy:.1%}")
    print(f"  Location MAE: {errors.mean():.3f} ft")
    print(f"  Location Median Error: {np.median(errors):.3f} ft")

    # =========================================================================
    # Manual Pitch Card Example
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("MANUAL PITCH CARD EXAMPLE")
    print(f"{'=' * 70}")

    # Demonstrate creating a pitch card with manually specified context
    sample_row = df[0]
    sample_catboost = sample_row.select(available_catboost).to_pandas()
    sample_mdn = torch.tensor(
        sample_row.select(available_mdn).to_numpy().astype(np.float32),
        dtype=torch.float32,
    )
    sample_mdn = torch.nan_to_num(sample_mdn, nan=0.0)

    # Create custom game context
    context = GameContext(
        pitcher_name="Gerrit Cole",
        batter_name="Shohei Ohtani",
        pitcher_hand="R",
        batter_hand="L",
        home_team="NYY",
        away_team="LAD",
        inning=7,
        inning_half="Top",
        balls=1,
        strikes=2,
        outs=2,
        date="2025-06-15",
        runners_on="1st & 2nd",
        score_home=3,
        score_away=4,
        pitch_number=5,
    )

    # Make prediction
    prediction = predictor.predict(sample_catboost, sample_mdn)

    # Create pitch card
    manual_card_path = Path(args.model_dir) / "manual_pitch_card.png"
    predictor.create_pitch_card(
        prediction=prediction,
        context=context,
        actual_pitch_type="SL",  # Example: actual pitch was a slider
        actual_location=(0.3, 2.1),  # Example actual location
        save_path=str(manual_card_path),
    )
    print(f"Manual pitch card saved: {manual_card_path}")
    print("\nThis demonstrates creating a pitch card with custom game context.")

    print(f"\n{'=' * 70}")
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
