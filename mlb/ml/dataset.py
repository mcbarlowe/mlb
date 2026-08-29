"""
PyTorch Dataset for pitch sequence prediction.

This module handles data loading and batching for training pitch prediction models.
"""

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm


def _iter_pitch_sequence_arrays(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    max_seq_len: int,
) -> Iterator[dict[str, np.ndarray | int]]:
    """Yield per-at-bat feature and target arrays from a transformed frame."""
    print("    Sorting pitches...")
    df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

    print("    Grouping by at-bat...")
    print(f"    Feature columns ({len(feature_columns)}): {feature_columns}")
    grouped = df.group_by(["game_pk", "at_bat_index"], maintain_order=True)

    for _, group_df in tqdm(grouped, desc="    Creating sequences", leave=False):
        features = np.array(
            group_df.select(feature_columns).cast(pl.Float32).to_numpy(),
            dtype=np.float32,
            copy=True,
        )
        targets = np.array(
            group_df.select(target_columns).cast(pl.Float32).to_numpy(),
            dtype=np.float32,
            copy=True,
        )

        if len(features) == 0 or np.isnan(features).any():
            continue

        if len(features) > max_seq_len:
            features = features[: max_seq_len]
            targets = targets[: max_seq_len]

        yield {
            "features": features,
            "targets": targets,
            "length": len(features),
        }


@dataclass(frozen=True)
class PlayerDropoutSpec:
    """Randomly disguise players as out-of-vocabulary during training.

    Mirrors inference for unseen players (see PitchFeatureEngine.transform):
    the embedding index moves to the unknown slot and the pitcher tendency
    features take the transform defaults. Training with a small rate gives
    the unknown embedding gradient signal and calibrated behavior.
    """

    rate: float
    pitcher_idx_pos: int
    batter_idx_pos: int
    pitcher_ff_pct_pos: int
    pitcher_repertoire_pos: int
    pitcher_unknown_idx: float
    batter_unknown_idx: float
    ff_pct_default: float = 0.5
    repertoire_default: float = 4.0 / 7.0

    @classmethod
    def build(
        cls,
        *,
        rate: float,
        feature_columns: list[str],
        n_known_pitchers: int,
        n_known_batters: int,
    ) -> "PlayerDropoutSpec":
        positions = {column: i for i, column in enumerate(feature_columns)}
        return cls(
            rate=rate,
            pitcher_idx_pos=positions["pitcher_idx"],
            batter_idx_pos=positions["batter_idx"],
            pitcher_ff_pct_pos=positions["pitcher_ff_pct"],
            pitcher_repertoire_pos=positions["pitcher_repertoire"],
            pitcher_unknown_idx=float(n_known_pitchers),
            batter_unknown_idx=float(n_known_batters),
        )


def _apply_player_dropout(features, spec: PlayerDropoutSpec, drop_pitcher: bool, drop_batter: bool) -> None:
    """Overwrite identity features in place on a [seq, features] array/tensor."""
    if drop_pitcher:
        features[:, spec.pitcher_idx_pos] = spec.pitcher_unknown_idx
        features[:, spec.pitcher_ff_pct_pos] = spec.ff_pct_default
        features[:, spec.pitcher_repertoire_pos] = spec.repertoire_default
    if drop_batter:
        features[:, spec.batter_idx_pos] = spec.batter_unknown_idx


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
        player_dropout: PlayerDropoutSpec | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with pitch data (already feature-engineered).
            feature_columns: List of feature column names.
            target_columns: List of target column names.
            max_seq_len: Maximum sequence length (pitches per at-bat).
            player_dropout: Optional training-time player identity dropout.
        """
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.max_seq_len = max_seq_len
        self.player_dropout = player_dropout

        # Group pitches by at-bat
        self.at_bats = self._group_by_at_bat(df)

    def _group_by_at_bat(self, df: pl.DataFrame) -> list[dict]:
        """
        Group pitches into at-bat sequences.

        Returns list of dicts with 'features' and 'targets' tensors.
        """
        at_bats = []
        for sample in _iter_pitch_sequence_arrays(
            df,
            self.feature_columns,
            self.target_columns,
            self.max_seq_len,
        ):
            at_bats.append(
                {
                    "features": torch.tensor(sample["features"], dtype=torch.float32),
                    "targets": torch.tensor(sample["targets"], dtype=torch.float32),
                    "length": int(sample["length"]),
                }
            )

        return at_bats

    def __len__(self) -> int:
        return len(self.at_bats)

    def __getitem__(self, idx: int) -> dict:
        sample = self.at_bats[idx]
        spec = self.player_dropout
        if spec is None or spec.rate <= 0:
            return sample
        drop_pitcher = bool(torch.rand(()) < spec.rate)
        drop_batter = bool(torch.rand(()) < spec.rate)
        if not (drop_pitcher or drop_batter):
            return sample
        features = sample["features"].clone()
        _apply_player_dropout(features, spec, drop_pitcher, drop_batter)
        return {
            "features": features,
            "targets": sample["targets"],
            "length": sample["length"],
        }


class PitchSequenceIterableDataset(IterableDataset):
    """Low-memory iterable sequence dataset that streams one season at a time."""

    def __init__(
        self,
        seasons: list[str],
        load_season: Callable[[str], pl.DataFrame],
        transform_season: Callable[[pl.DataFrame], pl.DataFrame],
        feature_columns: list[str],
        target_columns: list[str],
        max_seq_len: int = 20,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer_size: int = 1024,
        player_dropout: PlayerDropoutSpec | None = None,
    ):
        self.seasons = list(seasons)
        self.load_season = load_season
        self.transform_season = transform_season
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.player_dropout = player_dropout
        self._iteration = 0

    def __iter__(self) -> Iterator[dict[str, np.ndarray | int]]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        seasons = list(self.seasons)
        if self.shuffle:
            rng.shuffle(seasons)

        shuffle_buffer: list[dict[str, np.ndarray | int]] = []
        for season in seasons:
            season_df = self.transform_season(self.load_season(season))
            if season_df.is_empty():
                continue

            for sample in _iter_pitch_sequence_arrays(
                season_df,
                self.feature_columns,
                self.target_columns,
                self.max_seq_len,
            ):
                spec = self.player_dropout
                if spec is not None and spec.rate > 0:
                    drop_pitcher = rng.random() < spec.rate
                    drop_batter = rng.random() < spec.rate
                    if drop_pitcher or drop_batter:
                        _apply_player_dropout(
                            sample["features"], spec, drop_pitcher, drop_batter
                        )
                if self.shuffle and self.shuffle_buffer_size > 1:
                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.shuffle_buffer_size:
                        yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))
                else:
                    yield sample

        while shuffle_buffer:
            yield shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))


def collate_pitch_sequences(batch: list[dict]) -> dict:
    """
    Collate function for batching variable-length pitch sequences.

    Pads sequences to the max length in the batch.

    Args:
        batch: List of samples from PitchSequenceDataset.

    Returns:
        Dict with padded tensors and lengths.
    """
    features = [
        item["features"]
        if torch.is_tensor(item["features"])
        else torch.tensor(item["features"], dtype=torch.float32)
        for item in batch
    ]
    targets = [
        item["targets"]
        if torch.is_tensor(item["targets"])
        else torch.tensor(item["targets"], dtype=torch.float32)
        for item in batch
    ]
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)

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
        data_path: str | None = None,
        seasons: list[str] | None = None,
        batch_size: int = 64,
        max_seq_len: int = 20,
        sample_frac: float | None = None,
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

        from mlb.ml.features import PitchFeatureEngine

        self.data_path = Path(data_path) if data_path else None
        self.seasons = seasons
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.sample_frac = sample_frac

        self.feature_engine = PitchFeatureEngine(self.data_path)
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

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
