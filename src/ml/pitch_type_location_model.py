"""
Pitch-Type-Conditioned Location Model.

This module provides an MDN where pitch type is treated as a random effect,
with each of the 11 pitch types having its own location distribution parameters.
This is more granular than the HierarchicalMDN which groups by pitch family.

Classes:
    PitchTypeConditionedMDN: MDN with separate output head per pitch type
    PitchTypeThenLocationPredictor: Combined predictor using pitch type model -> location model
    PitchTypeLocationDataset: Dataset returning (features, pitch_type_idx, location) tuples
    PitchTypeLocationTrainer: Trainer with per-pitch-type metrics
"""

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from src.ml.features import PITCH_TYPE_CODES, IDX_TO_PITCH_TYPE


class PitchTypeConditionedMDN(nn.Module):
    """
    MDN with separate output head per pitch type (random effect).

    Each of the 11 pitch types gets its own complete MDN head, providing
    completely separate location distributions:
    - Fastballs (FF, SI, FC): Tend to be up in zone
    - Breaking (SL, CU, KC, ST): Low and glove-side
    - Offspeed (CH, FS): Low and arm-side
    - Other (KN, OTHER): Variable

    This is more granular than HierarchicalMDN which groups by pitch family.
    """

    def __init__(
        self,
        n_features: int,
        n_pitch_types: int = 11,
        hidden_dims: list[int] | None = None,
        n_components: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize the pitch-type-conditioned MDN.

        Args:
            n_features: Number of input features (excluding pitch_type_idx).
            n_pitch_types: Number of pitch type classes (default 11).
            hidden_dims: List of hidden layer dimensions for shared backbone.
            n_components: Number of mixture components per pitch type.
            dropout: Dropout rate.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_features = n_features
        self.n_pitch_types = n_pitch_types
        self.n_components = n_components

        # Build shared MLP backbone
        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.backbone_out_dim = in_dim

        # Create separate output heads for each pitch type
        # Each head outputs: K weights + 2K means + 2K sigmas + K rhos = 6K params
        self.pitch_type_heads = nn.ModuleList([
            nn.Linear(in_dim, 6 * n_components)
            for _ in range(n_pitch_types)
        ])

    def _parse_mdn_params(self, raw_output: torch.Tensor) -> dict:
        """
        Parse raw output from a head into MDN parameters.

        Args:
            raw_output: Raw output [batch, 6*K]

        Returns:
            Dictionary with pi, mu, sigma, rho parameters.
        """
        K = self.n_components
        batch = raw_output.shape[0]

        pi_logits = raw_output[:, :K]
        mu = raw_output[:, K:3*K].reshape(batch, K, 2)
        log_sigma = raw_output[:, 3*K:5*K].reshape(batch, K, 2)
        rho_raw = raw_output[:, 5*K:6*K]

        # Apply activations
        pi = F.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=0.01, max=5.0)
        rho = torch.tanh(rho_raw) * 0.99  # Avoid exactly +/-1

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward(self, x: torch.Tensor, pitch_type_idx: torch.Tensor) -> dict:
        """
        Forward pass with ground-truth pitch type.

        Args:
            x: Input features [batch, n_features]
            pitch_type_idx: Pitch type indices [batch] (integer tensor)

        Returns:
            Dictionary with MDN parameters for the specified pitch types:
            - pi: [batch, K] mixture weights
            - mu: [batch, K, 2] means
            - sigma: [batch, K, 2] standard deviations
            - rho: [batch, K] correlations
        """
        batch_size = x.shape[0]
        device = x.device

        # Shared backbone
        hidden = self.backbone(x)

        # Initialize output tensors
        K = self.n_components
        pi = torch.zeros(batch_size, K, device=device)
        mu = torch.zeros(batch_size, K, 2, device=device)
        sigma = torch.zeros(batch_size, K, 2, device=device)
        rho = torch.zeros(batch_size, K, device=device)

        # Process each pitch type present in the batch
        for pt_idx in range(self.n_pitch_types):
            mask = pitch_type_idx == pt_idx
            if not mask.any():
                continue

            # Get predictions from the appropriate head
            raw_output = self.pitch_type_heads[pt_idx](hidden[mask])
            params = self._parse_mdn_params(raw_output)

            # Fill in the results
            pi[mask] = params["pi"]
            mu[mask] = params["mu"]
            sigma[mask] = params["sigma"]
            rho[mask] = params["rho"]

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward_soft(self, x: torch.Tensor, pitch_type_probs: torch.Tensor) -> dict:
        """
        Forward pass with pitch type probabilities (for inference with predicted types).

        Computes weighted combination of all pitch-type-specific predictions.

        Args:
            x: Input features [batch, n_features]
            pitch_type_probs: Pitch type probabilities [batch, n_pitch_types]

        Returns:
            Dictionary with combined MDN parameters (weighted by pitch type probs).
        """
        batch_size = x.shape[0]
        device = x.device
        K = self.n_components

        # Shared backbone
        hidden = self.backbone(x)

        # Collect parameters from all pitch type heads
        all_pi = []
        all_mu = []
        all_sigma = []
        all_rho = []

        for pt_idx in range(self.n_pitch_types):
            raw_output = self.pitch_type_heads[pt_idx](hidden)
            params = self._parse_mdn_params(raw_output)
            all_pi.append(params["pi"])  # [batch, K]
            all_mu.append(params["mu"])  # [batch, K, 2]
            all_sigma.append(params["sigma"])  # [batch, K, 2]
            all_rho.append(params["rho"])  # [batch, K]

        # Stack: [n_pitch_types, batch, K, ...]
        all_pi = torch.stack(all_pi, dim=0)  # [n_pitch_types, batch, K]
        all_mu = torch.stack(all_mu, dim=0)  # [n_pitch_types, batch, K, 2]
        all_sigma = torch.stack(all_sigma, dim=0)  # [n_pitch_types, batch, K, 2]
        all_rho = torch.stack(all_rho, dim=0)  # [n_pitch_types, batch, K]

        # Weight by pitch type probabilities
        # pitch_type_probs: [batch, n_pitch_types] -> [n_pitch_types, batch, 1]
        weights = pitch_type_probs.T.unsqueeze(-1)  # [n_pitch_types, batch, 1]

        # Weighted combination
        # For pi: renormalize after weighting
        weighted_pi = (all_pi * weights).sum(dim=0)  # [batch, K]
        weighted_pi = weighted_pi / weighted_pi.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # For mu: weighted average
        weights_expanded = weights.unsqueeze(-1)  # [n_pitch_types, batch, 1, 1]
        weighted_mu = (all_mu * weights_expanded).sum(dim=0)  # [batch, K, 2]

        # For sigma: weighted average (could also use geometric mean)
        weighted_sigma = (all_sigma * weights_expanded).sum(dim=0)  # [batch, K, 2]

        # For rho: weighted average
        weighted_rho = (all_rho * weights).sum(dim=0)  # [batch, K]

        return {
            "pi": weighted_pi,
            "mu": weighted_mu,
            "sigma": weighted_sigma,
            "rho": weighted_rho,
        }

    def forward_all_types(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning parameters for all pitch types.

        Useful for analysis and visualization.

        Args:
            x: Input features [batch, n_features]

        Returns:
            Dictionary with MDN parameters for each pitch type:
            - pi: [batch, n_pitch_types, K]
            - mu: [batch, n_pitch_types, K, 2]
            - sigma: [batch, n_pitch_types, K, 2]
            - rho: [batch, n_pitch_types, K]
        """
        batch_size = x.shape[0]
        K = self.n_components

        # Shared backbone
        hidden = self.backbone(x)

        # Collect parameters from all pitch type heads
        all_pi = []
        all_mu = []
        all_sigma = []
        all_rho = []

        for pt_idx in range(self.n_pitch_types):
            raw_output = self.pitch_type_heads[pt_idx](hidden)
            params = self._parse_mdn_params(raw_output)
            all_pi.append(params["pi"])
            all_mu.append(params["mu"])
            all_sigma.append(params["sigma"])
            all_rho.append(params["rho"])

        return {
            "pi": torch.stack(all_pi, dim=1),  # [batch, n_pitch_types, K]
            "mu": torch.stack(all_mu, dim=1),  # [batch, n_pitch_types, K, 2]
            "sigma": torch.stack(all_sigma, dim=1),  # [batch, n_pitch_types, K, 2]
            "rho": torch.stack(all_rho, dim=1),  # [batch, n_pitch_types, K]
        }

    def log_prob(self, params: dict, target: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of target under the mixture.

        Args:
            params: MDN parameters from forward()
            target: Target locations [batch, 2]

        Returns:
            Log probability [batch]
        """
        pi = params["pi"]  # [batch, K]
        mu = params["mu"]  # [batch, K, 2]
        sigma = params["sigma"]  # [batch, K, 2]
        rho = params["rho"]  # [batch, K]

        # Expand target for broadcasting
        target = target.unsqueeze(1)  # [batch, 1, 2]

        # Compute bivariate Gaussian log probability for each component
        dx = target[..., 0] - mu[..., 0]  # [batch, K]
        dy = target[..., 1] - mu[..., 1]  # [batch, K]
        sx = sigma[..., 0]
        sy = sigma[..., 1]

        # Bivariate Gaussian log probability
        one_minus_rho_sq = (1 - rho ** 2).clamp(min=1e-6)
        z = (dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * (dx / sx) * (dy / sy)
        log_exp = -z / (2 * one_minus_rho_sq)
        log_norm = (
            -math.log(2 * math.pi)
            - torch.log(sx)
            - torch.log(sy)
            - 0.5 * torch.log(one_minus_rho_sq)
        )

        # Log probability for each component
        log_component = log_norm + log_exp  # [batch, K]

        # Log-sum-exp for mixture
        log_prob = torch.logsumexp(torch.log(pi + 1e-10) + log_component, dim=-1)

        return log_prob

    def sample(self, params: dict, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the mixture distribution.

        Args:
            params: MDN parameters from forward()
            n_samples: Number of samples per input

        Returns:
            Samples [batch, n_samples, 2]
        """
        pi = params["pi"]
        mu = params["mu"]
        sigma = params["sigma"]
        rho = params["rho"]

        batch_size = pi.shape[0]
        device = pi.device

        samples = []
        for _ in range(n_samples):
            # Sample component indices
            component_idx = torch.multinomial(pi, 1).squeeze(-1)

            # Get parameters for selected components
            batch_idx = torch.arange(batch_size, device=device)
            mu_selected = mu[batch_idx, component_idx]
            sigma_selected = sigma[batch_idx, component_idx]
            rho_selected = rho[batch_idx, component_idx]

            # Sample from bivariate Gaussian using Cholesky decomposition
            z = torch.randn(batch_size, 2, device=device)

            # Cholesky factor for correlation
            L = torch.zeros(batch_size, 2, 2, device=device)
            L[:, 0, 0] = 1.0
            L[:, 1, 0] = rho_selected
            L[:, 1, 1] = torch.sqrt((1 - rho_selected ** 2).clamp(min=1e-6))

            # Transform standard normal to correlated
            correlated = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)

            # Scale and shift
            sample = mu_selected + sigma_selected * correlated
            samples.append(sample)

        return torch.stack(samples, dim=1)

    def get_expected_value(self, params: dict) -> torch.Tensor:
        """
        Get the expected value (weighted mean) from the mixture.

        Args:
            params: MDN parameters

        Returns:
            Expected locations [batch, 2]
        """
        pi = params["pi"]  # [batch, K]
        mu = params["mu"]  # [batch, K, 2]

        # Weighted sum of means
        return torch.sum(pi.unsqueeze(-1) * mu, dim=1)

    def get_mode(self, params: dict) -> torch.Tensor:
        """
        Get the mode (mean of highest-weight component) from the mixture.

        Args:
            params: MDN parameters

        Returns:
            Mode locations [batch, 2]
        """
        pi = params["pi"]
        mu = params["mu"]

        max_idx = torch.argmax(pi, dim=-1)
        batch_idx = torch.arange(pi.shape[0], device=pi.device)

        return mu[batch_idx, max_idx]


class PitchTypeThenLocationPredictor(nn.Module):
    """
    Combined predictor: pitch type model -> location model.

    Uses an existing pitch type prediction model (e.g., LSTM) to predict
    pitch type, then uses those predictions to condition the location model.
    """

    def __init__(
        self,
        pitch_type_model: nn.Module,
        location_model: PitchTypeConditionedMDN,
        use_soft_conditioning: bool = True,
    ):
        """
        Initialize the combined predictor.

        Args:
            pitch_type_model: Existing pitch type model (e.g., LSTM).
                Should return (pitch_type_logits, mdn_params) or just pitch_type_logits.
            location_model: PitchTypeConditionedMDN for location prediction.
            use_soft_conditioning: If True, use pitch type probabilities (soft).
                If False, use argmax predictions (hard).
        """
        super().__init__()
        self.pitch_type_model = pitch_type_model
        self.location_model = location_model
        self.use_soft_conditioning = use_soft_conditioning

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        location_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Predict pitch type, then use it for location.

        Args:
            features: Input features for pitch type model [batch, seq_len, n_features].
            lengths: Sequence lengths [batch].
            mask: Attention mask [batch, seq_len].
            location_features: Optional separate features for location model [batch*seq, n_loc_features].
                If None, will be extracted from features.

        Returns:
            Dictionary with:
            - pitch_type_logits: [batch, seq_len, n_pitch_types]
            - pitch_type_probs: [batch, seq_len, n_pitch_types]
            - mdn_params: dict with pi, mu, sigma, rho (flattened to [batch*seq_len, ...])
        """
        # Get pitch type predictions
        pitch_type_output = self.pitch_type_model(features, lengths, mask)
        if isinstance(pitch_type_output, tuple):
            pitch_type_logits = pitch_type_output[0]
        else:
            pitch_type_logits = pitch_type_output

        pitch_type_probs = F.softmax(pitch_type_logits, dim=-1)

        # Flatten for location model
        batch_size, seq_len, _ = pitch_type_logits.shape
        flat_probs = pitch_type_probs.reshape(-1, pitch_type_probs.shape[-1])

        # Get location features (flattened)
        if location_features is None:
            # Use same features, flattened
            location_features = features.reshape(-1, features.shape[-1])

        # Get location predictions
        if self.use_soft_conditioning:
            mdn_params = self.location_model.forward_soft(location_features, flat_probs)
        else:
            # Hard conditioning: use argmax pitch type
            pitch_type_idx = torch.argmax(flat_probs, dim=-1)
            mdn_params = self.location_model.forward(location_features, pitch_type_idx)

        return {
            "pitch_type_logits": pitch_type_logits,
            "pitch_type_probs": pitch_type_probs,
            "mdn_params": mdn_params,
        }

    def predict_location_given_type(
        self,
        features: torch.Tensor,
        pitch_type_idx: torch.Tensor,
    ) -> dict:
        """
        Predict location for a specific pitch type (analysis mode).

        Args:
            features: Input features [batch, n_features].
            pitch_type_idx: Pitch type indices [batch].

        Returns:
            Dictionary with MDN parameters.
        """
        return self.location_model.forward(features, pitch_type_idx)


class PitchTypeLocationDataset(Dataset):
    """
    Dataset returning (features, pitch_type_idx, location) tuples.

    This is a flattened dataset (not sequence-based) for training
    the pitch-type-conditioned location model directly.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
        pitch_type_column: str = "pitch_type_idx",
        location_columns: list[str] | None = None,
        exclude_from_features: list[str] | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame with pitch data.
            feature_columns: List of feature column names to use.
            pitch_type_column: Column name for pitch type index.
            location_columns: Column names for location (default: ["px", "pz"]).
            exclude_from_features: Columns to exclude from features (conditioning variables).
        """
        if location_columns is None:
            location_columns = ["px", "pz"]

        if exclude_from_features is None:
            exclude_from_features = [pitch_type_column]

        # Filter out excluded columns
        self.feature_columns = [
            c for c in feature_columns
            if c not in exclude_from_features
        ]

        # Filter out invalid rows (NaN locations, invalid pitch types)
        valid_mask = (
            df[location_columns[0]].is_not_null()
            & df[location_columns[1]].is_not_null()
            & df[pitch_type_column].is_not_null()
            & (df[pitch_type_column] >= 0)
        )
        df = df.filter(valid_mask)

        self.features = torch.tensor(
            np.array(
                df.select(self.feature_columns).cast(pl.Float32).to_numpy(),
                dtype=np.float32,
                copy=True,
            ),
            dtype=torch.float32,
        )
        self.pitch_type_idx = torch.tensor(
            np.array(df[pitch_type_column].cast(pl.Int64).to_numpy(), dtype=np.int64, copy=True),
            dtype=torch.long,
        )
        self.location = torch.tensor(
            np.array(
                df.select(location_columns).cast(pl.Float32).to_numpy(),
                dtype=np.float32,
                copy=True,
            ),
            dtype=torch.float32,
        )

        # Handle NaN in features by replacing with 0
        self.features = torch.nan_to_num(self.features, nan=0.0)

        print(f"    PitchTypeLocationDataset: {len(self.features):,} samples")
        print(f"    Features: {len(self.feature_columns)} columns")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.pitch_type_idx[idx], self.location[idx]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

class PitchTypeLocationBatchIterableDataset(IterableDataset):
    """Low-memory iterable dataset that streams pitch-location batches by season."""

    def __init__(
        self,
        seasons: list[str],
        load_season: Callable[[str], pl.DataFrame],
        transform_season: Callable[[pl.DataFrame], pl.DataFrame],
        feature_columns: list[str],
        batch_size: int,
        pitch_type_column: str = "pitch_type_idx",
        location_columns: list[str] | None = None,
        exclude_from_features: list[str] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        if location_columns is None:
            location_columns = ["px", "pz"]
        if exclude_from_features is None:
            exclude_from_features = [pitch_type_column]

        self.seasons = list(seasons)
        self.load_season = load_season
        self.transform_season = transform_season
        self.feature_columns = [
            column for column in feature_columns if column not in exclude_from_features
        ]
        self.batch_size = batch_size
        self.pitch_type_column = pitch_type_column
        self.location_columns = location_columns
        self.shuffle = shuffle
        self.seed = seed
        self._iteration = 0

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1

        season_order = list(self.seasons)
        if self.shuffle and season_order:
            season_indices = rng.permutation(len(season_order))
            season_order = [season_order[index] for index in season_indices]

        for season in season_order:
            season_df = self.transform_season(self.load_season(season))
            valid_mask = (
                season_df[self.location_columns[0]].is_not_null()
                & season_df[self.location_columns[1]].is_not_null()
                & season_df[self.pitch_type_column].is_not_null()
                & (season_df[self.pitch_type_column] >= 0)
            )
            season_df = season_df.filter(valid_mask)
            if season_df.is_empty():
                continue

            print(f"    Streaming season {season}: {len(season_df):,} samples")
            features = np.array(
                season_df.select(self.feature_columns).cast(pl.Float32).to_numpy(),
                dtype=np.float32,
                copy=True,
            )
            features = np.nan_to_num(features, nan=0.0)
            pitch_type_idx = np.array(
                season_df[self.pitch_type_column].cast(pl.Int64).to_numpy(),
                dtype=np.int64,
                copy=True,
            )
            location = np.array(
                season_df.select(self.location_columns).cast(pl.Float32).to_numpy(),
                dtype=np.float32,
                copy=True,
            )

            if self.shuffle:
                row_indices = rng.permutation(len(features))
                features = features[row_indices]
                pitch_type_idx = pitch_type_idx[row_indices]
                location = location[row_indices]

            for start in range(0, len(features), self.batch_size):
                end = start + self.batch_size
                yield (
                    torch.tensor(features[start:end], dtype=torch.float32),
                    torch.tensor(pitch_type_idx[start:end], dtype=torch.long),
                    torch.tensor(location[start:end], dtype=torch.float32),
                )


def create_pitch_type_location_dataloaders(
    df: pl.DataFrame,
    feature_columns: list[str],
    pitch_type_column: str = "pitch_type_idx",
    location_columns: list[str] | None = None,
    exclude_from_features: list[str] | None = None,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for pitch type location model.

    Args:
        df: DataFrame with pitch data.
        feature_columns: List of feature column names.
        pitch_type_column: Column name for pitch type index.
        location_columns: Column names for location.
        exclude_from_features: Columns to exclude from features.
        batch_size: Batch size for training.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        num_workers: Number of data loading workers.
        seed: Random seed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if location_columns is None:
        location_columns = ["px", "pz"]

    # Sort by game for temporal split
    if "game_pk" in df.columns:
        df = df.sort("game_pk")

    n_rows = len(df)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))

    train_df = df.head(train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n_rows - val_end)

    print(f"Dataset split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Create datasets
    train_dataset = PitchTypeLocationDataset(
        train_df, feature_columns, pitch_type_column,
        location_columns, exclude_from_features
    )
    val_dataset = PitchTypeLocationDataset(
        val_df, feature_columns, pitch_type_column,
        location_columns, exclude_from_features
    )
    test_dataset = PitchTypeLocationDataset(
        test_df, feature_columns, pitch_type_column,
        location_columns, exclude_from_features
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


class PitchTypeLocationTrainer:
    """
    Trainer for PitchTypeConditionedMDN with per-pitch-type metrics.
    """

    def __init__(
        self,
        model: PitchTypeConditionedMDN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
    ):
        """
        Initialize the trainer.

        Args:
            model: PitchTypeConditionedMDN model.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            device: Device to use ("auto", "cuda", "mps", "cpu").
        """
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for features, pitch_type_idx, location in pbar:
            features = features.to(self.device)
            pitch_type_idx = pitch_type_idx.to(self.device)
            location = location.to(self.device)

            self.optimizer.zero_grad()

            params = self.model(features, pitch_type_idx)
            log_prob = self.model.log_prob(params, location)
            loss = -log_prob.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate the model with per-pitch-type metrics."""
        self.model.eval()
        total_nll = 0.0
        all_preds = []
        all_targets = []
        all_pitch_types = []
        n_samples = 0

        with torch.no_grad():
            for features, pitch_type_idx, location in dataloader:
                features = features.to(self.device)
                pitch_type_idx = pitch_type_idx.to(self.device)
                location = location.to(self.device)

                params = self.model(features, pitch_type_idx)
                log_prob = self.model.log_prob(params, location)
                total_nll -= log_prob.sum().item()

                # Get point predictions for MAE
                pred = self.model.get_expected_value(params)
                all_preds.append(pred.cpu())
                all_targets.append(location.cpu())
                all_pitch_types.append(pitch_type_idx.cpu())
                n_samples += len(features)

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        pitch_types = torch.cat(all_pitch_types, dim=0).numpy()

        # Overall metrics
        mae_px = np.abs(preds[:, 0] - targets[:, 0]).mean()
        mae_pz = np.abs(preds[:, 1] - targets[:, 1]).mean()
        euclidean = np.sqrt(
            (preds[:, 0] - targets[:, 0])**2 +
            (preds[:, 1] - targets[:, 1])**2
        ).mean()

        metrics = {
            "nll": total_nll / n_samples,
            "mae_px": mae_px,
            "mae_pz": mae_pz,
            "euclidean": euclidean,
        }

        # Per-pitch-type metrics
        per_type_metrics = {}
        for pt_idx in range(self.model.n_pitch_types):
            mask = pitch_types == pt_idx
            if mask.sum() == 0:
                continue

            pt_code = IDX_TO_PITCH_TYPE.get(pt_idx, f"UNK{pt_idx}")
            pt_preds = preds[mask]
            pt_targets = targets[mask]

            pt_mae_px = np.abs(pt_preds[:, 0] - pt_targets[:, 0]).mean()
            pt_mae_pz = np.abs(pt_preds[:, 1] - pt_targets[:, 1]).mean()
            pt_euclidean = np.sqrt(
                (pt_preds[:, 0] - pt_targets[:, 0])**2 +
                (pt_preds[:, 1] - pt_targets[:, 1])**2
            ).mean()

            per_type_metrics[pt_code] = {
                "count": int(mask.sum()),
                "mae_px": pt_mae_px,
                "mae_pz": pt_mae_pz,
                "euclidean": pt_euclidean,
            }

        metrics["per_pitch_type"] = per_type_metrics

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            n_epochs: Maximum number of epochs.
            early_stopping_patience: Patience for early stopping.
            verbose: Whether to print progress.

        Returns:
            Dictionary with training history and best metrics.
        """
        best_val_nll = float("inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_nll": [],
            "val_mae_px": [],
            "val_mae_pz": [],
            "val_euclidean": [],
        }
        best_state = None

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, epoch)

            if verbose:
                print(f"Epoch {epoch+1}: Validating...", flush=True)

            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["nll"])

            history["train_loss"].append(train_loss)
            history["val_nll"].append(val_metrics["nll"])
            history["val_mae_px"].append(val_metrics["mae_px"])
            history["val_mae_pz"].append(val_metrics["mae_pz"])
            history["val_euclidean"].append(val_metrics["euclidean"])

            if val_metrics["nll"] < best_val_nll:
                best_val_nll = val_metrics["nll"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_per_type = val_metrics.get("per_pitch_type", {})
            else:
                patience_counter += 1

            if verbose:
                print(
                    f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                    f"val_nll={val_metrics['nll']:.4f}, "
                    f"val_mae={val_metrics['euclidean']:.4f}",
                    flush=True
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "history": history,
            "best_val_nll": best_val_nll,
            "best_per_pitch_type": best_per_type,
        }


def compare_to_baseline(
    conditioned_model: PitchTypeConditionedMDN,
    baseline_model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device = "auto",
) -> dict:
    """
    Compare pitch-type-conditioned model to a baseline (no pitch type info).

    Args:
        conditioned_model: Trained PitchTypeConditionedMDN.
        baseline_model: Trained BivariateMDN (no pitch type conditioning).
        test_loader: Test DataLoader.
        device: Device to use.

    Returns:
        Dictionary with comparison metrics.
    """
    if isinstance(device, torch.device):
        resolved_device = device
    elif device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    conditioned_model = conditioned_model.to(resolved_device)
    baseline_model = baseline_model.to(resolved_device)
    conditioned_model.eval()
    baseline_model.eval()

    cond_nll = 0.0
    base_nll = 0.0
    n_samples = 0

    with torch.no_grad():
        for features, pitch_type_idx, location in test_loader:
            features = features.to(resolved_device)
            pitch_type_idx = pitch_type_idx.to(resolved_device)
            location = location.to(resolved_device)

            # Conditioned model
            cond_params = conditioned_model(features, pitch_type_idx)
            cond_log_prob = conditioned_model.log_prob(cond_params, location)
            cond_nll -= cond_log_prob.sum().item()

            # Baseline model (no pitch type)
            base_params = baseline_model(features)
            base_log_prob = baseline_model.log_prob(base_params, location)
            base_nll -= base_log_prob.sum().item()

            n_samples += len(features)

    return {
        "conditioned_nll": cond_nll / n_samples,
        "baseline_nll": base_nll / n_samples,
        "nll_improvement": (base_nll - cond_nll) / n_samples,
        "nll_improvement_pct": 100 * (base_nll - cond_nll) / base_nll,
    }
