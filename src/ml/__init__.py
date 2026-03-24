"""
MLB Pitch Prediction ML Module.

This module provides tools for predicting pitcher behavior:
- Pitch type classification (FF, CH, SL, CU, etc.)
- Pitch location prediction via Mixture Density Network (MDN)
- Pitch-type-conditioned location models
"""

from src.ml.features import PitchFeatureEngine, compute_class_weights
from src.ml.dataset import PitchSequenceDataset, create_dataloaders
from src.ml.model import PitchPredictor, create_model
from src.ml.train import train_model, PitchPredictionTrainer, MDNLoss, PitchPredictionLoss
from src.ml.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_location_predictions,
    sample_from_mdn,
    plot_mdn_predictions,
    compute_mdn_nll,
    compute_mdn_coverage,
    get_mdn_mean_prediction,
    # Per-pitch-type evaluation
    evaluate_pitch_type_location_model,
    compute_per_pitch_type_metrics,
    print_per_pitch_type_metrics,
    plot_location_by_pitch_type,
)
from src.ml.cross_validation import (
    TimeSeriesCrossValidator,
    CVFold,
    CVResults,
    run_cross_validation,
)
from src.ml.pitch_type_location_model import (
    PitchTypeConditionedMDN,
    PitchTypeThenLocationPredictor,
    PitchTypeLocationDataset,
    PitchTypeLocationTrainer,
    create_pitch_type_location_dataloaders,
    compare_to_baseline,
)

__all__ = [
    # Features
    "PitchFeatureEngine",
    "compute_class_weights",
    # Dataset
    "PitchSequenceDataset",
    "create_dataloaders",
    # Model
    "PitchPredictor",
    "create_model",
    # Training
    "train_model",
    "PitchPredictionTrainer",
    "MDNLoss",
    "PitchPredictionLoss",
    # Evaluation
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_location_predictions",
    # MDN utilities
    "sample_from_mdn",
    "plot_mdn_predictions",
    "compute_mdn_nll",
    "compute_mdn_coverage",
    "get_mdn_mean_prediction",
    # Per-pitch-type evaluation
    "evaluate_pitch_type_location_model",
    "compute_per_pitch_type_metrics",
    "print_per_pitch_type_metrics",
    "plot_location_by_pitch_type",
    # Cross-validation
    "TimeSeriesCrossValidator",
    "CVFold",
    "CVResults",
    "run_cross_validation",
    # Pitch-type-conditioned location model
    "PitchTypeConditionedMDN",
    "PitchTypeThenLocationPredictor",
    "PitchTypeLocationDataset",
    "PitchTypeLocationTrainer",
    "create_pitch_type_location_dataloaders",
    "compare_to_baseline",
]
