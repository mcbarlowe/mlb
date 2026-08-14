"""
MLB Pitch Prediction ML Module.

This module provides tools for predicting pitcher behavior:
- Pitch type classification (FF, CH, SL, CU, etc.)
- Pitch location prediction via Mixture Density Network (MDN)
- Pitch-type-conditioned location models
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ml.cross_validation import CVFold as CVFold
    from src.ml.cross_validation import CVResults as CVResults
    from src.ml.cross_validation import (
        TimeSeriesCrossValidator as TimeSeriesCrossValidator,
    )
    from src.ml.cross_validation import run_cross_validation as run_cross_validation
    from src.ml.dataset import PitchSequenceDataset as PitchSequenceDataset
    from src.ml.dataset import create_dataloaders as create_dataloaders
    from src.ml.evaluate import compute_mdn_coverage as compute_mdn_coverage
    from src.ml.evaluate import compute_mdn_nll as compute_mdn_nll
    from src.ml.evaluate import (
        compute_per_pitch_type_metrics as compute_per_pitch_type_metrics,
    )
    from src.ml.evaluate import evaluate_model as evaluate_model
    from src.ml.evaluate import (
        evaluate_pitch_type_location_model as evaluate_pitch_type_location_model,
    )
    from src.ml.evaluate import get_mdn_mean_prediction as get_mdn_mean_prediction
    from src.ml.evaluate import plot_confusion_matrix as plot_confusion_matrix
    from src.ml.evaluate import (
        plot_location_by_pitch_type as plot_location_by_pitch_type,
    )
    from src.ml.evaluate import plot_location_predictions as plot_location_predictions
    from src.ml.evaluate import plot_mdn_predictions as plot_mdn_predictions
    from src.ml.evaluate import (
        print_per_pitch_type_metrics as print_per_pitch_type_metrics,
    )
    from src.ml.evaluate import sample_from_mdn as sample_from_mdn
    from src.ml.features import PitchFeatureEngine as PitchFeatureEngine
    from src.ml.features import compute_class_weights as compute_class_weights
    from src.ml.model import PitchPredictor as PitchPredictor
    from src.ml.model import create_model as create_model
    from src.ml.pitch_type_location_model import (
        PitchTypeConditionedMDN as PitchTypeConditionedMDN,
    )
    from src.ml.pitch_type_location_model import (
        PitchTypeLocationDataset as PitchTypeLocationDataset,
    )
    from src.ml.pitch_type_location_model import (
        PitchTypeLocationTrainer as PitchTypeLocationTrainer,
    )
    from src.ml.pitch_type_location_model import (
        PitchTypeThenLocationPredictor as PitchTypeThenLocationPredictor,
    )
    from src.ml.pitch_type_location_model import (
        compare_to_baseline as compare_to_baseline,
    )
    from src.ml.pitch_type_location_model import (
        create_pitch_type_location_dataloaders as create_pitch_type_location_dataloaders,
    )
    from src.ml.train import MDNLoss as MDNLoss
    from src.ml.train import PitchPredictionLoss as PitchPredictionLoss
    from src.ml.train import PitchPredictionTrainer as PitchPredictionTrainer
    from src.ml.train import train_model as train_model

__all__ = [
    "CVFold",
    "CVResults",
    "MDNLoss",
    "PitchFeatureEngine",
    "PitchPredictionLoss",
    "PitchPredictionTrainer",
    "PitchPredictor",
    "PitchSequenceDataset",
    "PitchTypeConditionedMDN",
    "PitchTypeLocationDataset",
    "PitchTypeLocationTrainer",
    "PitchTypeThenLocationPredictor",
    "TimeSeriesCrossValidator",
    "compare_to_baseline",
    "compute_class_weights",
    "compute_mdn_coverage",
    "compute_mdn_nll",
    "compute_per_pitch_type_metrics",
    "create_dataloaders",
    "create_model",
    "create_pitch_type_location_dataloaders",
    "evaluate_model",
    "evaluate_pitch_type_location_model",
    "get_mdn_mean_prediction",
    "plot_confusion_matrix",
    "plot_location_by_pitch_type",
    "plot_location_predictions",
    "plot_mdn_predictions",
    "print_per_pitch_type_metrics",
    "run_cross_validation",
    "sample_from_mdn",
    "train_model",
]

_EXPORT_MODULES = {
    "CVFold": "src.ml.cross_validation",
    "CVResults": "src.ml.cross_validation",
    "MDNLoss": "src.ml.train",
    "PitchFeatureEngine": "src.ml.features",
    "PitchPredictionLoss": "src.ml.train",
    "PitchPredictionTrainer": "src.ml.train",
    "PitchPredictor": "src.ml.model",
    "PitchSequenceDataset": "src.ml.dataset",
    "PitchTypeConditionedMDN": "src.ml.pitch_type_location_model",
    "PitchTypeLocationDataset": "src.ml.pitch_type_location_model",
    "PitchTypeLocationTrainer": "src.ml.pitch_type_location_model",
    "PitchTypeThenLocationPredictor": "src.ml.pitch_type_location_model",
    "TimeSeriesCrossValidator": "src.ml.cross_validation",
    "compare_to_baseline": "src.ml.pitch_type_location_model",
    "compute_class_weights": "src.ml.features",
    "compute_mdn_coverage": "src.ml.evaluate",
    "compute_mdn_nll": "src.ml.evaluate",
    "compute_per_pitch_type_metrics": "src.ml.evaluate",
    "create_dataloaders": "src.ml.dataset",
    "create_model": "src.ml.model",
    "create_pitch_type_location_dataloaders": "src.ml.pitch_type_location_model",
    "evaluate_model": "src.ml.evaluate",
    "evaluate_pitch_type_location_model": "src.ml.evaluate",
    "get_mdn_mean_prediction": "src.ml.evaluate",
    "plot_confusion_matrix": "src.ml.evaluate",
    "plot_location_by_pitch_type": "src.ml.evaluate",
    "plot_location_predictions": "src.ml.evaluate",
    "plot_mdn_predictions": "src.ml.evaluate",
    "print_per_pitch_type_metrics": "src.ml.evaluate",
    "run_cross_validation": "src.ml.cross_validation",
    "sample_from_mdn": "src.ml.evaluate",
    "train_model": "src.ml.train",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(module_name, fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
