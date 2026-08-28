"""Analyze held-out model probability calibration across seasons."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim.team_strength import (
    FEATURE_NAMES,
    load_completed_games,
    train_strength_model,
)


def heldout_home_probabilities(
    season: int, train_seasons: list[int]
) -> pd.DataFrame:
    """Fit on prior seasons and return held-out home-win probabilities."""
    games = load_completed_games(
        start_season=min(train_seasons),
        end_season=season,
    )
    fitted = train_strength_model(
        games,
        prediction_season=season,
        train_seasons=train_seasons,
    )
    heldout = fitted.feature_frame[fitted.feature_frame["season"] == season].copy()
    if heldout.empty:
        raise ValueError(f"No held-out games found for {season}")
    heldout["model_prob_home"] = fitted.estimator.predict_proba(
        heldout[list(FEATURE_NAMES)]
    )[:, 1]
    return heldout[["game_pk", "model_prob_home", "home_won"]]


def analyze_calibration(
    season: int, train_seasons: list[int]
) -> tuple[pd.DataFrame, float]:
    """Check whether held-out model probabilities are well calibrated."""
    df = heldout_home_probabilities(season, train_seasons)
    
    print(f"\n{'='*60}")
    print(f"Season {season} (trained on {min(train_seasons)}-{max(train_seasons)})")
    print('='*60)
    
    # Overall stats
    print(f"\nGames analyzed: {len(df)}")
    print(f"Home team won: {df['home_won'].mean():.1%}")
    print(f"Model predicted home win prob: {df['model_prob_home'].mean():.1%}")
    
    # Probability distribution
    p = df['model_prob_home']
    print("\nProbability distribution:")
    print(f"  Min:  {p.min():.3f}")
    print(f"  25%:  {p.quantile(0.25):.3f}")
    print(f"  50%:  {p.quantile(0.50):.3f}")
    print(f"  75%:  {p.quantile(0.75):.3f}")
    print(f"  Max:  {p.max():.3f}")
    print(f"  Std:  {p.std():.3f}")
    
    # Extreme predictions
    extreme_low = (p < 0.30).sum()
    extreme_high = (p > 0.70).sum()
    print("\nExtreme predictions:")
    print(f"  < 30%: {extreme_low:4d} ({extreme_low/len(df)*100:5.1f}%)")
    print(f"  > 70%: {extreme_high:4d} ({extreme_high/len(df)*100:5.1f}%)")
    
    # Calibration analysis: bin predictions and check actual outcomes
    bins = [0, 0.4, 0.45, 0.50, 0.55, 0.60, 1.0]
    df['prob_bin'] = pd.cut(df['model_prob_home'], bins=bins)
    
    print("\nCalibration by probability bin:")
    print(f"{'Bin':>15} | {'Count':>6} | {'Predicted':>9} | {'Actual':>8} | {'Error':>7}")
    print('-' * 60)
    
    total_error = 0
    for bin_label in df['prob_bin'].cat.categories:
        bin_df = df[df['prob_bin'] == bin_label]
        if len(bin_df) == 0:
            continue
        
        predicted = bin_df['model_prob_home'].mean()
        actual = bin_df['home_won'].mean()
        error = predicted - actual
        total_error += abs(error) * len(bin_df)
        
        print(f"{bin_label!s:>15} | {len(bin_df):6d} | {predicted:8.1%} | {actual:7.1%} | {error:+6.1%}")
    
    mean_abs_error = total_error / len(df)
    print(f"\nMean absolute calibration error: {mean_abs_error:.3%}")
    
    # Brier score (lower is better)
    brier = ((df['model_prob_home'] - df['home_won']) ** 2).mean()
    print(f"Brier score: {brier:.4f}")
    
    return df, mean_abs_error


def main():
    """Analyze calibration for all seasons."""
    
    seasons = [
        (2021, list(range(2015, 2021))),
        (2022, list(range(2015, 2022))),
        (2024, list(range(2015, 2024))),
        (2025, list(range(2015, 2025))),
    ]
    
    results = {}
    for season, train_years in seasons:
        df, mae = analyze_calibration(season, train_years)
        results[season] = {'df': df, 'mae': mae}
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"\n{'Season':>6} | {'Training Data':>15} | {'Calib Error':>12} | {'Status':>20}")
    print('-' * 60)
    
    for season, train_years in seasons:
        mae = results[season]['mae']
        train_range = f"{min(train_years)}-{max(train_years)}"
        
        if mae < 0.02:
            status = "✅ Well-calibrated"
        elif mae < 0.04:
            status = "⚠️  Slightly off"
        else:
            status = "❌ Miscalibrated"
        
        print(f"{season:6d} | {train_range:>15} | {mae:11.2%} | {status:>20}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print('='*60)
    print()
    print("Compare reliability, Brier score, and probability distributions across")
    print("seasons. If predictive scores deteriorate while calibration remains stable,")
    print("inspect feature drift, training-window changes, and input-data integrity.")


if __name__ == "__main__":
    main()
