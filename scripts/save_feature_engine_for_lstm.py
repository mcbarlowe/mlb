#!/usr/bin/env python
"""
Utility script to save feature_engine.json for an existing LSTM model.

This is needed because earlier training runs didn't save the feature engine.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.cross_validation import TimeSeriesCrossValidator

# Model directory to save feature_engine to
model_dir = Path("models/attention_full/run_20260119_124719")

print("Loading data and fitting feature engine...")
cv = TimeSeriesCrossValidator(
    data_path=Path("data/processed/livefeeds"),
    test_season="2025",
    val_seasons=["2024"],
    exclude_seasons=["2020"],
)

# Just need to prepare the data to fit the feature engine
cv.prepare_data(quick=False)

# Save the feature engine
feature_engine_path = model_dir / "feature_engine.json"
cv.feature_engine.save(feature_engine_path)
print(f"Feature engine saved to: {feature_engine_path}")
print(f"  - {cv.n_pitchers:,} pitchers")
print(f"  - {cv.n_batters:,} batters")
print(f"  - {cv.n_features} features")
