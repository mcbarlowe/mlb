"""
Time-series cross-validation for pitch prediction models.

Implements season-based expanding window cross-validation that respects
the temporal nature of baseball data.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.ml.dataset import PitchSequenceDataset, collate_pitch_sequences
from src.ml.features import PitchFeatureEngine


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""

    fold_idx: int
    train_seasons: list[str]
    val_season: str
    train_loader: DataLoader
    val_loader: DataLoader
    n_train_samples: int
    n_val_samples: int


@dataclass
class CVResults:
    """Results from cross-validation."""

    fold_results: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def add_fold_result(self, fold_idx: int, metrics: dict) -> None:
        """Add results from a single fold."""
        self.fold_results.append({"fold": fold_idx, **metrics})

    def get_mean_metrics(self) -> dict:
        """Get mean metrics across all folds."""
        if not self.fold_results:
            return {}

        keys = [k for k in self.fold_results[0].keys() if k != "fold"]
        return {
            key: np.mean([r[key] for r in self.fold_results if key in r])
            for key in keys
        }

    def get_std_metrics(self) -> dict:
        """Get standard deviation of metrics across folds."""
        if not self.fold_results:
            return {}

        keys = [k for k in self.fold_results[0].keys() if k != "fold"]
        return {
            f"{key}_std": np.std([r[key] for r in self.fold_results if key in r])
            for key in keys
        }

    def summary(self) -> dict:
        """Get summary with mean and std for all metrics."""
        return {**self.get_mean_metrics(), **self.get_std_metrics()}

    def to_json(self, path: Path) -> None:
        """Save results to JSON file."""
        data = {
            "config": self.config,
            "fold_results": self.fold_results,
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class TimeSeriesCrossValidator:
    """
    Season-based expanding window cross-validator.

    Creates folds where training data is from earlier seasons and
    validation is from a later season, respecting temporal ordering.
    """

    def __init__(
        self,
        data_path: str = "data/processed/livefeeds",
        train_seasons: Optional[list[str]] = None,
        val_seasons: Optional[list[str]] = None,
        test_season: Optional[str] = None,
        batch_size: int = 64,
        max_seq_len: int = 20,
        exclude_seasons: Optional[list[str]] = None,
    ):
        """
        Initialize the cross-validator.

        Args:
            data_path: Path to processed parquet files.
            train_seasons: Seasons available for training (default: 2018-2023).
            val_seasons: Seasons to use for validation folds (default: 2022-2024).
            test_season: Season held out for final testing (default: 2025).
            batch_size: Batch size for data loaders.
            max_seq_len: Maximum sequence length for at-bats.
            exclude_seasons: Seasons to exclude (e.g., ["2020"] for COVID year).
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Default seasons
        all_seasons = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

        # Remove excluded seasons
        if exclude_seasons:
            all_seasons = [s for s in all_seasons if s not in exclude_seasons]

        self.test_season = test_season or "2025"
        self.train_seasons = train_seasons or [
            s for s in all_seasons if s < self.test_season
        ]
        self.val_seasons = val_seasons or ["2022", "2023", "2024"]

        # Feature engine (fitted once on all available training data)
        self.feature_engine: Optional[PitchFeatureEngine] = None
        self._season_data: dict[str, pl.DataFrame] = {}

    def _load_season(self, season: str) -> pl.DataFrame:
        """Load and cache data for a single season."""
        if season not in self._season_data:
            pattern = self.data_path / season / "*.parquet"
            if not pattern.parent.exists():
                raise ValueError(f"Season {season} not found at {pattern.parent}")

            df = pl.scan_parquet(str(pattern)).collect()
            self._season_data[season] = df
            print(f"Loaded season {season}: {len(df):,} pitches")

        return self._season_data[season]

    def _fit_feature_engine(self, seasons: list[str]) -> None:
        """Fit the feature engine on specified seasons."""
        # Load all seasons for fitting
        dfs = [self._load_season(s) for s in seasons]
        # Use how="diagonal" to handle schema differences across seasons
        combined_df = pl.concat(dfs, how="diagonal")

        # Fit feature engine
        self.feature_engine = PitchFeatureEngine(self.data_path)
        self.feature_engine.fit(combined_df)
        print(
            f"Feature engine fitted on {len(seasons)} seasons: "
            f"{self.feature_engine.n_pitchers:,} pitchers, "
            f"{self.feature_engine.n_batters:,} batters"
        )

    def _create_dataloader(
        self, seasons: list[str], shuffle: bool = True
    ) -> tuple[DataLoader, int]:
        """Create a DataLoader for specified seasons."""
        print(f"  Loading {len(seasons)} season(s): {seasons}")
        dfs = [self._load_season(s) for s in seasons]
        # Use how="diagonal" to handle schema differences across seasons
        combined_df = pl.concat(dfs, how="diagonal")
        print(f"  Total pitches: {len(combined_df):,}")

        # Transform features
        print("  Transforming features...")
        transformed_df = self.feature_engine.transform(combined_df)

        # Filter nulls
        transformed_df = transformed_df.filter(
            pl.col("pitch_type_idx").is_not_null()
            & pl.col("px").is_not_null()
            & pl.col("pz").is_not_null()
        )
        print(f"  Valid pitches after filtering: {len(transformed_df):,}")

        # Create dataset (this groups pitches into at-bat sequences)
        print("  Creating at-bat sequences...")
        dataset = PitchSequenceDataset(
            transformed_df,
            self.feature_engine.get_feature_columns(),
            self.feature_engine.get_target_columns(),
            self.max_seq_len,
        )
        print(f"  Created {len(dataset):,} at-bat sequences")

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_pitch_sequences,
            num_workers=0,
        )

        return loader, len(dataset)

    def get_folds(self) -> Iterator[CVFold]:
        """
        Generate cross-validation folds.

        Yields CVFold objects with train/val loaders for each fold.
        Uses expanding window: each fold adds more training data.
        """
        # Fit feature engine on all training seasons
        if self.feature_engine is None:
            self._fit_feature_engine(self.train_seasons)

        for fold_idx, val_season in enumerate(self.val_seasons):
            # Training seasons are all seasons before validation season
            train_seasons = [s for s in self.train_seasons if s < val_season]

            if not train_seasons:
                print(f"Skipping fold {fold_idx}: no training seasons before {val_season}")
                continue

            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}: Train {train_seasons} → Val [{val_season}]")
            print(f"{'='*60}")

            # Create data loaders
            print("\nPreparing training data:")
            train_loader, n_train = self._create_dataloader(train_seasons, shuffle=True)
            print("\nPreparing validation data:")
            val_loader, n_val = self._create_dataloader([val_season], shuffle=False)

            print(f"\nData ready: {n_train:,} train / {n_val:,} val at-bats")
            print("Starting training...\n")

            yield CVFold(
                fold_idx=fold_idx,
                train_seasons=train_seasons,
                val_season=val_season,
                train_loader=train_loader,
                val_loader=val_loader,
                n_train_samples=n_train,
                n_val_samples=n_val,
            )

    def get_test_loader(self) -> tuple[DataLoader, int]:
        """Get the held-out test set loader."""
        if self.feature_engine is None:
            self._fit_feature_engine(self.train_seasons)

        return self._create_dataloader([self.test_season], shuffle=False)

    def get_final_train_val_loaders(
        self, val_season: str = "2024"
    ) -> tuple[DataLoader, DataLoader, int, int]:
        """
        Get loaders for final model training.

        Uses all data except test season for training, with specified
        season for validation/early stopping.

        Args:
            val_season: Season to use for validation during final training.

        Returns:
            Tuple of (train_loader, val_loader, n_train, n_val).
        """
        if self.feature_engine is None:
            self._fit_feature_engine(self.train_seasons)

        train_seasons = [s for s in self.train_seasons if s != val_season]
        train_loader, n_train = self._create_dataloader(train_seasons, shuffle=True)
        val_loader, n_val = self._create_dataloader([val_season], shuffle=False)

        return train_loader, val_loader, n_train, n_val

    @property
    def n_pitchers(self) -> int:
        """Number of unique pitchers in the dataset."""
        if self.feature_engine is None:
            raise RuntimeError("Feature engine not fitted. Call get_folds() first.")
        return self.feature_engine.n_pitchers

    @property
    def n_batters(self) -> int:
        """Number of unique batters in the dataset."""
        if self.feature_engine is None:
            raise RuntimeError("Feature engine not fitted. Call get_folds() first.")
        return self.feature_engine.n_batters

    @property
    def n_pitch_types(self) -> int:
        """Number of pitch type classes."""
        if self.feature_engine is None:
            raise RuntimeError("Feature engine not fitted. Call get_folds() first.")
        return self.feature_engine.n_pitch_types

    @property
    def n_features(self) -> int:
        """Number of input features."""
        if self.feature_engine is None:
            raise RuntimeError("Feature engine not fitted. Call get_folds() first.")
        return len(self.feature_engine.get_feature_columns())


def run_cross_validation(
    model_fn,
    cv: TimeSeriesCrossValidator,
    train_fn,
    eval_fn,
    config: dict,
    n_epochs: int = 30,
    early_stopping_patience: int = 5,
    device: str = "auto",
) -> CVResults:
    """
    Run full cross-validation with a model.

    Args:
        model_fn: Function that creates a new model instance.
        cv: TimeSeriesCrossValidator instance.
        train_fn: Function to train model (model, train_loader, val_loader, ...) -> results.
        eval_fn: Function to evaluate model (model, data_loader) -> metrics dict.
        config: Model/training configuration dict.
        n_epochs: Maximum epochs per fold.
        early_stopping_patience: Early stopping patience.
        device: Device to train on.

    Returns:
        CVResults with metrics from all folds.
    """
    results = CVResults(config=config)

    for fold in cv.get_folds():
        print(f"\n{'='*60}")
        print(f"Training Fold {fold.fold_idx + 1}")
        print(f"Train: {fold.train_seasons} ({fold.n_train_samples:,} at-bats)")
        print(f"Val: {fold.val_season} ({fold.n_val_samples:,} at-bats)")
        print(f"{'='*60}")

        # Create fresh model
        model = model_fn()

        # Train
        train_results = train_fn(
            model=model,
            train_loader=fold.train_loader,
            val_loader=fold.val_loader,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            device=device,
        )

        # Evaluate
        metrics = eval_fn(model, fold.val_loader, device=device)

        # Add training info
        metrics["train_epochs"] = train_results.get("total_epochs", n_epochs)
        metrics["best_val_loss"] = train_results.get("best_val_loss", float("inf"))

        results.add_fold_result(fold.fold_idx, metrics)

        print(f"\nFold {fold.fold_idx + 1} Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    summary = results.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return results
