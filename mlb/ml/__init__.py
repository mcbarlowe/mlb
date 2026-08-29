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
    from mlb.ml.cross_validation import CVFold as CVFold
    from mlb.ml.cross_validation import CVResults as CVResults
    from mlb.ml.cross_validation import (
        TimeSeriesCrossValidator as TimeSeriesCrossValidator,
    )
    from mlb.ml.cross_validation import run_cross_validation as run_cross_validation
    from mlb.ml.dataset import PitchSequenceDataset as PitchSequenceDataset
    from mlb.ml.dataset import create_dataloaders as create_dataloaders
    from mlb.ml.evaluate import compute_mdn_coverage as compute_mdn_coverage
    from mlb.ml.evaluate import compute_mdn_nll as compute_mdn_nll
    from mlb.ml.evaluate import (
        compute_per_pitch_type_metrics as compute_per_pitch_type_metrics,
    )
    from mlb.ml.evaluate import evaluate_model as evaluate_model
    from mlb.ml.evaluate import (
        evaluate_pitch_type_location_model as evaluate_pitch_type_location_model,
    )
    from mlb.ml.evaluate import get_mdn_mean_prediction as get_mdn_mean_prediction
    from mlb.ml.evaluate import plot_confusion_matrix as plot_confusion_matrix
    from mlb.ml.evaluate import (
        plot_location_by_pitch_type as plot_location_by_pitch_type,
    )
    from mlb.ml.evaluate import plot_location_predictions as plot_location_predictions
    from mlb.ml.evaluate import plot_mdn_predictions as plot_mdn_predictions
    from mlb.ml.evaluate import (
        print_per_pitch_type_metrics as print_per_pitch_type_metrics,
    )
    from mlb.ml.evaluate import sample_from_mdn as sample_from_mdn
    from mlb.ml.features import PitchFeatureEngine as PitchFeatureEngine
    from mlb.ml.features import compute_class_weights as compute_class_weights
    from mlb.ml.model import PitchPredictor as PitchPredictor
    from mlb.ml.model import create_model as create_model
    from mlb.ml.pitch_type_location_model import (
        PitchTypeConditionedMDN as PitchTypeConditionedMDN,
    )
    from mlb.ml.pitch_type_location_model import (
        PitchTypeLocationDataset as PitchTypeLocationDataset,
    )
    from mlb.ml.pitch_type_location_model import (
        PitchTypeLocationTrainer as PitchTypeLocationTrainer,
    )
    from mlb.ml.pitch_type_location_model import (
        PitchTypeThenLocationPredictor as PitchTypeThenLocationPredictor,
    )
    from mlb.ml.pitch_type_location_model import (
        compare_to_baseline as compare_to_baseline,
    )
    from mlb.ml.pitch_type_location_model import (
        create_pitch_type_location_dataloaders as create_pitch_type_location_dataloaders,
    )
    from mlb.ml.train import MDNLoss as MDNLoss
    from mlb.ml.train import PitchPredictionLoss as PitchPredictionLoss
    from mlb.ml.train import PitchPredictionTrainer as PitchPredictionTrainer
    from mlb.ml.train import train_model as train_model

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
    "CVFold": "mlb.ml.cross_validation",
    "CVResults": "mlb.ml.cross_validation",
    "MDNLoss": "mlb.ml.train",
    "PitchFeatureEngine": "mlb.ml.features",
    "PitchPredictionLoss": "mlb.ml.train",
    "PitchPredictionTrainer": "mlb.ml.train",
    "PitchPredictor": "mlb.ml.model",
    "PitchSequenceDataset": "mlb.ml.dataset",
    "PitchTypeConditionedMDN": "mlb.ml.pitch_type_location_model",
    "PitchTypeLocationDataset": "mlb.ml.pitch_type_location_model",
    "PitchTypeLocationTrainer": "mlb.ml.pitch_type_location_model",
    "PitchTypeThenLocationPredictor": "mlb.ml.pitch_type_location_model",
    "TimeSeriesCrossValidator": "mlb.ml.cross_validation",
    "compare_to_baseline": "mlb.ml.pitch_type_location_model",
    "compute_class_weights": "mlb.ml.features",
    "compute_mdn_coverage": "mlb.ml.evaluate",
    "compute_mdn_nll": "mlb.ml.evaluate",
    "compute_per_pitch_type_metrics": "mlb.ml.evaluate",
    "create_dataloaders": "mlb.ml.dataset",
    "create_model": "mlb.ml.model",
    "create_pitch_type_location_dataloaders": "mlb.ml.pitch_type_location_model",
    "evaluate_model": "mlb.ml.evaluate",
    "evaluate_pitch_type_location_model": "mlb.ml.evaluate",
    "get_mdn_mean_prediction": "mlb.ml.evaluate",
    "plot_confusion_matrix": "mlb.ml.evaluate",
    "plot_location_by_pitch_type": "mlb.ml.evaluate",
    "plot_location_predictions": "mlb.ml.evaluate",
    "plot_mdn_predictions": "mlb.ml.evaluate",
    "print_per_pitch_type_metrics": "mlb.ml.evaluate",
    "run_cross_validation": "mlb.ml.cross_validation",
    "sample_from_mdn": "mlb.ml.evaluate",
    "train_model": "mlb.ml.train",
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
