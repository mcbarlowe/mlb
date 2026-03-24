#!/usr/bin/env python
"""
Train per-pitcher pitch type prediction models.

This script trains individual CatBoost models for high-volume pitchers
and evaluates them against the global model baseline.

Usage:
    PYTHONPATH=. uv run python scripts/train_per_pitcher_models.py
    PYTHONPATH=. uv run python scripts/train_per_pitcher_models.py --min-pitches 5000 --max-pitchers 100
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from src.ml.features import PitchFeatureEngine
from src.ml.per_pitcher_trainer import PerPitcherTrainer


def load_season_data(season: str) -> pl.DataFrame:
    """Load all parquet files for a season."""
    season_dir = Path(f"data/processed/livefeeds/{season}")
    files = list(season_dir.glob("*.parquet"))
    dfs = [pl.read_parquet(f) for f in tqdm(files, desc=f"Loading {season}", leave=False)]
    return pl.concat(dfs) if dfs else pl.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Train per-pitcher models")
    parser.add_argument(
        "--min-pitches",
        type=int,
        default=3000,
        help="Minimum pitches for pitcher-specific model (default: 3000)",
    )
    parser.add_argument(
        "--max-pitchers",
        type=int,
        default=None,
        help="Maximum number of pitcher models to train (for testing)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Max CatBoost iterations per model (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/per_pitcher_TIMESTAMP)",
    )
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"models/per_pitcher_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-PITCHER PITCH PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Min pitches for individual model: {args.min_pitches}")
    print(f"Max iterations per model: {args.iterations}")
    if args.max_pitchers:
        print(f"Max pitchers to train: {args.max_pitchers}")
    print()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_seasons = ["2018", "2019", "2021", "2022", "2023"]
    val_season = "2024"
    test_season = "2025"

    print("Loading training data...")
    train_dfs = []
    for season in train_seasons:
        df = load_season_data(season)
        print(f"  {season}: {len(df):,} pitches")
        train_dfs.append(df)
    train_df = pl.concat(train_dfs)
    print(f"Total training: {len(train_df):,} pitches")

    print(f"\nLoading validation data: {val_season}")
    val_df = load_season_data(val_season)
    print(f"  {val_season}: {len(val_df):,} pitches")

    print(f"\nLoading test data: {test_season}")
    test_df = load_season_data(test_season)
    print(f"  {test_season}: {len(test_df):,} pitches")

    # Fit feature engine on training data
    print("\nFitting feature engine...")
    feature_engine = PitchFeatureEngine()
    feature_engine.fit(train_df)
    print(f"Pitchers: {len(feature_engine.pitcher_to_idx):,}")
    print(f"Batters: {len(feature_engine.batter_to_idx):,}")

    # Get pitcher counts
    pitcher_counts = pl.read_parquet("data/pitcher_counts.parquet")
    eligible = pitcher_counts.filter(pl.col("len") >= args.min_pitches)
    print(f"\nPitchers with >={args.min_pitches} pitches: {len(eligible)}")

    # Initialize trainer
    print("\n" + "=" * 70)
    print("TRAINING PER-PITCHER MODELS")
    print("=" * 70)

    trainer = PerPitcherTrainer(
        min_pitches=args.min_pitches,
        iterations=args.iterations,
        learning_rate=0.1,
        depth=6,
        early_stopping_rounds=20,
        verbose=0,
    )

    # Train models
    train_results = trainer.train(
        train_df=train_df,
        val_df=val_df,
        feature_engine=feature_engine,
        pitcher_counts=pitcher_counts,
        max_pitchers=args.max_pitchers,
    )

    print("\n" + "-" * 50)
    print("Training Results:")
    print(f"  Pitcher models trained: {train_results['n_pitcher_models']}")
    if train_results.get("mean_pitcher_accuracy"):
        print(f"  Mean pitcher accuracy (val): {train_results['mean_pitcher_accuracy']:.1%}")
        print(f"  Median pitcher accuracy (val): {train_results['median_pitcher_accuracy']:.1%}")
        print(f"  Min pitcher accuracy (val): {train_results['min_pitcher_accuracy']:.1%}")
        print(f"  Max pitcher accuracy (val): {train_results['max_pitcher_accuracy']:.1%}")
    print(f"  Global model accuracy (val): {train_results['global_accuracy']:.1%}")

    # Evaluate on test data
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST DATA")
    print("=" * 70)

    test_results = trainer.evaluate(test_df, feature_engine)

    print(f"\nTest Results:")
    print(f"  Overall Accuracy:      {test_results['overall_accuracy']:.1%}")
    print(f"  Overall Top-3 Accuracy: {test_results['overall_top3_accuracy']:.1%}")
    print(f"  Overall F1 (macro):    {test_results['overall_f1_macro']:.4f}")
    print(f"  Overall F1 (weighted): {test_results['overall_f1_weighted']:.4f}")
    print()
    print(f"  Samples using pitcher models: {test_results['n_pitcher_model_samples']:,}")
    print(f"  Samples using global model:   {test_results['n_global_model_samples']:,}")

    if test_results.get("pitcher_model_accuracy"):
        print(f"\n  Pitcher model accuracy: {test_results['pitcher_model_accuracy']:.1%}")
    if test_results.get("global_model_accuracy"):
        print(f"  Global model accuracy:  {test_results['global_model_accuracy']:.1%}")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    trainer.save(output_dir)
    feature_engine.save(output_dir / "feature_engine.json")

    # Save results
    results = {
        "timestamp": timestamp,
        "config": {
            "min_pitches": args.min_pitches,
            "max_pitchers": args.max_pitchers,
            "iterations": args.iterations,
            "train_seasons": train_seasons,
            "val_season": val_season,
            "test_season": test_season,
        },
        "training": {
            "n_pitcher_models": train_results["n_pitcher_models"],
            "mean_pitcher_accuracy": train_results.get("mean_pitcher_accuracy"),
            "median_pitcher_accuracy": train_results.get("median_pitcher_accuracy"),
            "global_accuracy": train_results["global_accuracy"],
        },
        "test": test_results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'results.json'}")

    # Print comparison to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print("""
Baseline (single global model):
  - Accuracy: 66.4%
  - Top-3: 91.9%

Per-pitcher models:
  - Overall Accuracy: {:.1%}
  - Top-3 Accuracy: {:.1%}
  - Pitcher-specific accuracy: {:.1%}

Improvement: {:+.1%} accuracy
""".format(
        test_results["overall_accuracy"],
        test_results["overall_top3_accuracy"],
        test_results.get("pitcher_model_accuracy", 0),
        test_results["overall_accuracy"] - 0.664,
    ))

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
