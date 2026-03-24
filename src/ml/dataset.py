"""
PyTorch Dataset for pitch sequence prediction.

This module handles data loading and batching for training pitch prediction models.
"""

from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class PitchSequenceDataset(Dataset):
    """
    PyTorch Dataset for pitch sequences grouped by at-bat.

    Each sample represents one at-bat with a variable-length sequence of pitches.
    The model predicts the pitch type and location for each pitch in the sequence.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
        max_seq_len: int = 20,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with pitch data (already feature-engineered).
            feature_columns: List of feature column names.
            target_columns: List of target column names.
            max_seq_len: Maximum sequence length (pitches per at-bat).
        """
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.max_seq_len = max_seq_len

        # Group pitches by at-bat
        self.at_bats = self._group_by_at_bat(df)

    def _group_by_at_bat(self, df: pl.DataFrame) -> list[dict]:
        """
        Group pitches into at-bat sequences.

        Returns list of dicts with 'features' and 'targets' tensors.
        """
        at_bats = []

        # Sort by game and at-bat, then pitch number
        print("    Sorting pitches...")
        df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

        # Group by game and at-bat
        print("    Grouping by at-bat...")
        print(f"    Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        grouped = df.group_by(["game_pk", "at_bat_index"], maintain_order=True)
        groups_list = list(grouped)

        for _, group_df in tqdm(groups_list, desc="    Creating sequences", leave=False):
            # Get features and targets as numpy arrays
            features = group_df.select(self.feature_columns).to_numpy()
            targets = group_df.select(self.target_columns).to_numpy()

            # Skip empty at-bats or at-bats with missing data
            if len(features) == 0 or np.isnan(features).any():
                continue

            # Truncate if too long
            if len(features) > self.max_seq_len:
                features = features[: self.max_seq_len]
                targets = targets[: self.max_seq_len]

            at_bats.append({
                "features": torch.tensor(features, dtype=torch.float32),
                "targets": torch.tensor(targets, dtype=torch.float32),
                "length": len(features),
            })

        return at_bats

    def __len__(self) -> int:
        return len(self.at_bats)

    def __getitem__(self, idx: int) -> dict:
        return self.at_bats[idx]


def collate_pitch_sequences(batch: list[dict]) -> dict:
    """
    Collate function for batching variable-length pitch sequences.

    Pads sequences to the max length in the batch.

    Args:
        batch: List of samples from PitchSequenceDataset.

    Returns:
        Dict with padded tensors and lengths.
    """
    features = [item["features"] for item in batch]
    targets = [item["targets"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])

    # Pad sequences (batch_first=True)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1.0)

    # Create attention mask (1 for real tokens, 0 for padding)
    max_len = features_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "features": features_padded,
        "targets": targets_padded,
        "lengths": lengths,
        "mask": mask,
    }


def create_dataloaders(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    batch_size: int = 64,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    max_seq_len: int = 20,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with temporal split.

    Args:
        df: DataFrame with pitch data.
        feature_columns: List of feature column names.
        target_columns: List of target column names.
        batch_size: Batch size for training.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        max_seq_len: Maximum sequence length.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Sort by game date for temporal split
    if "game_date" in df.columns:
        df = df.sort("game_date")
    elif "game_pk" in df.columns:
        # game_pk is roughly chronological within a season
        df = df.sort("game_pk")

    n_rows = len(df)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))

    train_df = df.head(train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n_rows - val_end)

    print(f"Dataset split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Create datasets
    train_dataset = PitchSequenceDataset(
        train_df, feature_columns, target_columns, max_seq_len
    )
    val_dataset = PitchSequenceDataset(
        val_df, feature_columns, target_columns, max_seq_len
    )
    test_dataset = PitchSequenceDataset(
        test_df, feature_columns, target_columns, max_seq_len
    )

    print(f"At-bats: train={len(train_dataset):,}, val={len(val_dataset):,}, test={len(test_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pitch_sequences,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pitch_sequences,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pitch_sequences,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


class PitchDataModule:
    """
    Convenience class to manage data loading pipeline.

    Combines feature engineering and dataset creation.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        seasons: Optional[list[str]] = None,
        batch_size: int = 64,
        max_seq_len: int = 20,
        sample_frac: Optional[float] = None,
    ):
        """
        Initialize the data module.

        Args:
            data_path: Path to parquet data files.
            seasons: List of seasons to load.
            batch_size: Batch size for training.
            max_seq_len: Maximum sequence length.
            sample_frac: Optional fraction to sample for faster iteration.
        """
        from pathlib import Path
        from src.ml.features import PitchFeatureEngine

        self.data_path = Path(data_path) if data_path else None
        self.seasons = seasons
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.sample_frac = sample_frac

        self.feature_engine = PitchFeatureEngine(self.data_path)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def setup(self) -> None:
        """Load and prepare data."""
        # Load raw data
        df = self.feature_engine.load_data(
            seasons=self.seasons,
            sample_frac=self.sample_frac,
        )

        # Feature engineering
        df = self.feature_engine.fit_transform(df)

        # Filter out rows with null targets
        df = df.filter(
            pl.col("pitch_type_idx").is_not_null()
            & pl.col("px").is_not_null()
            & pl.col("pz").is_not_null()
        )

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            df=df,
            feature_columns=self.feature_engine.get_feature_columns(),
            target_columns=self.feature_engine.get_target_columns(),
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
        )

    @property
    def n_pitchers(self) -> int:
        return self.feature_engine.n_pitchers

    @property
    def n_batters(self) -> int:
        return self.feature_engine.n_batters

    @property
    def n_pitch_types(self) -> int:
        return self.feature_engine.n_pitch_types

    @property
    def n_features(self) -> int:
        return len(self.feature_engine.get_feature_columns())
