#!/usr/bin/env python
"""
Generate sample pitch cards using the LSTM+Attention model.

This script loads pitch data, creates sequences for at-bats, and generates
pitch cards showing predictions from the LSTM model.

Usage:
    uv run python scripts/generate_lstm_pitch_cards.py
    uv run python scripts/generate_lstm_pitch_cards.py --n-cards 10
    uv run python scripts/generate_lstm_pitch_cards.py --output-dir output/cards
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from src.ml.pitch_predictor import GameContext, PitchPredictor


def load_sample_at_bats(
    data_path: Path,
    season: str = "2025",
    n_at_bats: int = 10,
    min_pitches: int = 3,
    seed: int = 42,
) -> list[pl.DataFrame]:
    """
    Load sample at-bats from parquet files.

    Args:
        data_path: Path to processed livefeeds directory
        season: Season to load data from
        n_at_bats: Number of at-bats to sample
        min_pitches: Minimum pitches per at-bat
        seed: Random seed

    Returns:
        List of DataFrames, each containing one at-bat
    """
    print(f"Loading data from {season}...")
    df = pl.scan_parquet(str(data_path / season / "*.parquet")).collect()

    print(f"Loaded {len(df):,} pitches")

    # Sort by game and at-bat
    df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

    # Group by at-bat and filter for minimum pitches
    at_bat_groups = df.group_by(["game_pk", "at_bat_index"]).agg(
        pl.count().alias("n_pitches"),
        pl.first("pitcher_name").alias("pitcher_name"),
        pl.first("batter_name").alias("batter_name"),
    ).filter(
        pl.col("n_pitches") >= min_pitches
    )

    print(f"Found {len(at_bat_groups):,} at-bats with >= {min_pitches} pitches")

    # Sample at-bats
    np.random.seed(seed)
    sampled = at_bat_groups.sample(n=min(n_at_bats, len(at_bat_groups)), seed=seed)

    # Get full at-bat data for each sampled at-bat
    at_bats = []
    for row in sampled.iter_rows(named=True):
        at_bat_df = df.filter(
            (pl.col("game_pk") == row["game_pk"]) &
            (pl.col("at_bat_index") == row["at_bat_index"])
        ).sort("pitch_number")
        at_bats.append(at_bat_df)

    return at_bats


def prepare_lstm_features(
    at_bat_df: pl.DataFrame,
    feature_engine,
    pitch_index: int = -1,
) -> tuple[torch.Tensor, pl.DataFrame]:
    """
    Prepare features for LSTM prediction.

    Args:
        at_bat_df: DataFrame containing the at-bat
        feature_engine: Fitted PitchFeatureEngine
        pitch_index: Index of the pitch to predict (-1 for last)

    Returns:
        Tuple of (features tensor, single row DataFrame for the target pitch)
    """
    # Transform the at-bat data
    transformed = feature_engine.transform(at_bat_df)

    # Get feature columns
    feature_cols = feature_engine.get_feature_columns()

    # Extract features as tensor (use all pitches up to and including target)
    if pitch_index < 0:
        pitch_index = len(transformed) + pitch_index

    # Use sequence up to and including target pitch
    sequence_df = transformed.slice(0, pitch_index + 1)
    features = sequence_df.select(feature_cols).to_numpy()
    features = torch.tensor(features, dtype=torch.float32)

    # Get the target pitch row for context
    target_row = at_bat_df.slice(pitch_index, 1)

    return features, target_row


def get_context_from_row(row: pl.DataFrame) -> GameContext:
    """Extract GameContext from a pitch row."""
    def get_val(col, default=None):
        if col in row.columns:
            val = row[col][0]
            return val if val is not None else default
        return default

    from datetime import datetime

    def format_date(date_val):
        if date_val is None:
            return None
        date_str = str(date_val)
        try:
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str)
            else:
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
            return dt.strftime('%B %d, %Y')
        except Exception:
            return date_str[:10] if len(date_str) >= 10 else date_str

    # Get runner states
    runner_on_1b = bool(get_val("is_runner_on_first", False))
    runner_on_2b = bool(get_val("is_runner_on_second", False))
    runner_on_3b = bool(get_val("is_runner_on_third", False))

    return GameContext(
        pitcher_name=get_val("pitcher_name", "Unknown"),
        batter_name=get_val("batter_name", "Unknown"),
        pitcher_hand=get_val("throw_side", "R"),
        batter_hand=get_val("bat_side", "R"),
        home_team=get_val("home_team_name", "HOME"),
        away_team=get_val("away_team_name", "AWAY"),
        inning=get_val("inning", 1),
        inning_half="Top" if get_val("half_inning", "top") == "top" else "Bot",
        balls=get_val("balls", 0),
        strikes=get_val("strikes", 0),
        outs=get_val("outs", 0),
        date=format_date(get_val("game_date")),
        runner_on_1b=runner_on_1b,
        runner_on_2b=runner_on_2b,
        runner_on_3b=runner_on_3b,
        score_home=get_val("home_score"),
        score_away=get_val("away_score"),
        pitch_number=get_val("pitch_number"),
        pitcher_id=get_val("pitcher_id"),
        batter_id=get_val("batter_id"),
        pitch_result=get_val("pitch_call_description"),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate pitch cards using LSTM model")
    parser.add_argument("--model-dir", default="models/attention_full/run_20260119_124719",
                        help="Path to LSTM model directory")
    parser.add_argument("--data-path", default="data/processed/livefeeds",
                        help="Path to processed pitch data")
    parser.add_argument("--season", default="2025", help="Season to sample from")
    parser.add_argument("--n-cards", type=int, default=5, help="Number of pitch cards to generate")
    parser.add_argument("--output-dir", default="output/lstm_pitch_cards",
                        help="Output directory for pitch cards")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LSTM PITCH CARD GENERATOR")
    print("=" * 70)

    # Load the LSTM model
    print(f"\nLoading LSTM model from {args.model_dir}...")
    predictor = PitchPredictor.load_lstm(args.model_dir, device=args.device)
    print(f"Model loaded successfully (type: {predictor.model_type})")

    if predictor.feature_engine is None:
        print("ERROR: Feature engine not loaded. Cannot transform data.")
        return

    # Load sample at-bats
    at_bats = load_sample_at_bats(
        data_path=Path(args.data_path),
        season=args.season,
        n_at_bats=args.n_cards,
        seed=args.seed,
    )

    print(f"\nGenerating {len(at_bats)} pitch cards...")

    correct_predictions = 0
    total_predictions = 0

    for i, at_bat_df in enumerate(tqdm(at_bats, desc="Generating cards")):
        # Use the last pitch in the at-bat for prediction
        try:
            # Prepare features
            features, target_row = prepare_lstm_features(
                at_bat_df,
                predictor.feature_engine,
                pitch_index=-1,
            )

            # Make prediction
            prediction = predictor.predict(lstm_features=features)

            # Get context
            context = get_context_from_row(target_row)

            # Get actual values
            actual_type = target_row["pitch_type_code"][0]
            actual_px = target_row["px"][0]
            actual_pz = target_row["pz"][0]
            actual_location = (actual_px, actual_pz) if actual_px is not None and actual_pz is not None else None

            # Track accuracy
            total_predictions += 1
            if prediction.predicted_type == actual_type:
                correct_predictions += 1

            # Generate pitch card
            filename = f"pitch_card_{i+1:02d}_{context.pitcher_name.replace(' ', '_')}_{context.batter_name.replace(' ', '_')}.png"
            save_path = output_dir / filename

            fig = predictor.create_pitch_card(
                prediction=prediction,
                context=context,
                actual_pitch_type=actual_type,
                actual_location=actual_location,
                save_path=str(save_path),
            )

            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)

        except Exception as e:
            print(f"\nError processing at-bat {i+1}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {total_predictions} pitch cards in {output_dir}")
    print(f"Pitch type accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")
    print()

    # List generated files
    print("Generated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
