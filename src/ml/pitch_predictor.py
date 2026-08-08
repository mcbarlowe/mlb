"""
Unified pitch predictor combining pitch type and location models.

This module provides a single interface for predicting:
1. Pitch type probabilities (from CatBoost)
2. Location point estimate (from MDN)
3. Location probability density / KDE (from MDN)

Example:
    predictor = PitchPredictor.load("models/combined_20240115_120000")

    # Single pitch prediction
    result = predictor.predict(features)
    print(f"Most likely pitch: {result['predicted_type']}")
    print(f"Location estimate: ({result['location_point'][0]:.2f}, {result['location_point'][1]:.2f})")

    # Plot the location density
    predictor.plot_prediction(features, save_path="prediction.png")
"""

import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.ml.mdn_location_model import (
    BivariateMDN,
    get_location_density,
    get_point_estimate,
    get_mixture_parameters,
)
from src.ml.features import PITCH_TYPE_CODES, IDX_TO_PITCH_TYPE, PitchFeatureEngine
from src.ml.model import create_model
from datetime import datetime

# Full pitch type names for display
PITCH_TYPE_FULL_NAMES = {
    "FF": "Four-Seam Fastball",
    "FA": "Fastball",
    "FT": "Two-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "CH": "Changeup",
    "SL": "Slider",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "ST": "Sweeper",
    "SV": "Slurve",
    "FS": "Splitter",
    "FO": "Forkball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "SC": "Screwball",
    "IN": "Intentional Ball",
    "PO": "Pitchout",
    "AB": "Automatic Ball",
    "UN": "Unknown",
    "OTHER": "Other",
}

# Cache for headshots to avoid repeated downloads
_headshot_cache: dict[int, Optional[np.ndarray]] = {}


def fetch_mlb_headshot(player_id: int, size: int = 100) -> Optional[np.ndarray]:
    """
    Fetch a player's headshot from MLB's CDN.

    Args:
        player_id: MLB player ID
        size: Desired width in pixels (height will be proportional)

    Returns:
        numpy array of the image, or None if fetch failed
    """
    if player_id is None:
        return None

    # Check cache first
    if player_id in _headshot_cache:
        return _headshot_cache[player_id]

    url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_{size},q_auto:best/v1/people/{player_id}/headshot/67/current"

    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=5) as response:
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            _headshot_cache[player_id] = img_array
            return img_array
    except (URLError, HTTPError, Exception):
        # Cache the failure to avoid repeated attempts
        _headshot_cache[player_id] = None
        return None


@dataclass
class GameContext:
    """Container for game context information."""
    pitcher_name: str
    batter_name: str
    pitcher_hand: str  # "L" or "R"
    batter_hand: str  # "L" or "R"
    home_team: str
    away_team: str
    inning: int
    inning_half: str  # "Top" or "Bot"
    balls: int
    strikes: int
    outs: int
    date: Optional[str] = None  # "YYYY-MM-DD" or display format
    runners_on: Optional[str] = None  # e.g., "1st & 3rd", "Bases Empty" (legacy)
    runner_on_1b: bool = False  # Runner on first base
    runner_on_2b: bool = False  # Runner on second base
    runner_on_3b: bool = False  # Runner on third base
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    pitch_number: Optional[int] = None  # Pitch number in at-bat
    pitcher_id: Optional[int] = None  # MLB player ID for headshot
    batter_id: Optional[int] = None  # MLB player ID for headshot
    pitch_result: Optional[str] = None  # e.g., "Called Strike", "Ball", "Swinging Strike", "In play, no out"

    @property
    def count_str(self) -> str:
        return f"{self.balls}-{self.strikes}"

    @property
    def matchup_str(self) -> str:
        return f"{self.pitcher_hand}HP vs {self.batter_hand}HB"

    @property
    def game_str(self) -> str:
        if self.score_home is not None and self.score_away is not None:
            return f"{self.away_team} {self.score_away} @ {self.home_team} {self.score_home}"
        return f"{self.away_team} @ {self.home_team}"


@dataclass
class PitchPrediction:
    """Container for a single pitch prediction."""

    # Pitch type
    type_probabilities: np.ndarray  # [n_classes] probability for each pitch type
    predicted_type_idx: int  # Index of most likely pitch type
    predicted_type: str  # Code of most likely pitch type (e.g., "FF")
    top_3_types: list[tuple[str, float]]  # Top 3 pitch types with probabilities

    # Location point estimate
    location_point: np.ndarray  # [px, pz] expected location
    location_mode: np.ndarray  # [px, pz] mode (highest density point)

    # Location density grid
    px_grid: np.ndarray  # Horizontal axis values
    pz_grid: np.ndarray  # Vertical axis values
    location_density: np.ndarray  # [pz_size, px_size] probability density

    # MDN mixture parameters (for advanced use)
    mixture_weights: np.ndarray  # [K] component weights
    mixture_means: np.ndarray  # [K, 2] component means
    mixture_stds: np.ndarray  # [K, 2] component standard deviations

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type_probabilities": {
                PITCH_TYPE_CODES[i]: float(p)
                for i, p in enumerate(self.type_probabilities)
            },
            "predicted_type": self.predicted_type,
            "top_3_types": [
                {"type": t, "probability": float(p)}
                for t, p in self.top_3_types
            ],
            "location_point": {
                "px": float(self.location_point[0]),
                "pz": float(self.location_point[1]),
            },
            "location_mode": {
                "px": float(self.location_mode[0]),
                "pz": float(self.location_mode[1]),
            },
        }


class PitchPredictor:
    """
    Unified predictor for pitch type and location.

    Supports multiple model types:
    - CatBoost + MDN (separate models for type and location)
    - LSTM + Attention (single model for both)

    The predictor can use either:
    - Raw features (requires feature preparation)
    - Pre-prepared features matching the training format
    """

    def __init__(
        self,
        type_model: Optional[CatBoostClassifier] = None,
        mdn_model: Optional[BivariateMDN] = None,
        lstm_model: Optional[torch.nn.Module] = None,
        feature_columns: list[str] = None,
        categorical_features: list[str] = None,
        mdn_feature_columns: list[str] = None,
        feature_engine: Optional[PitchFeatureEngine] = None,
        pitcher_to_idx: Optional[dict] = None,
        batter_to_idx: Optional[dict] = None,
        device: str = "cpu",
        model_type: str = "catboost",  # "catboost" or "lstm"
    ):
        """
        Initialize the predictor.

        Args:
            type_model: Trained CatBoost pitch type classifier (for catboost mode)
            mdn_model: Trained MDN location model (for catboost mode)
            lstm_model: Trained LSTM+Attention model (for lstm mode)
            feature_columns: Feature columns for CatBoost
            categorical_features: Categorical feature names for CatBoost
            mdn_feature_columns: Feature columns for MDN
            feature_engine: PitchFeatureEngine for data transformation
            pitcher_to_idx: Pitcher ID to index mapping
            batter_to_idx: Batter ID to index mapping
            device: Device for inference
            model_type: "catboost" or "lstm"
        """
        self.model_type = model_type
        self.device = device

        if model_type == "catboost":
            self.type_model = type_model
            self.mdn_model = mdn_model.to(device) if mdn_model else None
            if self.mdn_model:
                self.mdn_model.eval()
            self.lstm_model = None
        else:  # lstm
            self.lstm_model = lstm_model.to(device) if lstm_model else None
            if self.lstm_model:
                self.lstm_model.eval()
            self.type_model = None
            self.mdn_model = None

        self.feature_columns = feature_columns or []
        self.categorical_features = categorical_features or []
        self.mdn_feature_columns = mdn_feature_columns or []
        self.feature_engine = feature_engine

        self.pitcher_to_idx = pitcher_to_idx or {}
        self.batter_to_idx = batter_to_idx or {}

        # Get categorical feature indices (for CatBoost)
        self.cat_indices = [
            i for i, col in enumerate(self.feature_columns)
            if col in self.categorical_features
        ]

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: str = "cpu") -> "PitchPredictor":
        """
        Load a trained CatBoost + MDN predictor from disk.

        Args:
            model_dir: Path to the combined model directory
            device: Device for MDN inference ("cpu", "cuda", or "mps")

        Returns:
            Loaded PitchPredictor instance
        """
        model_dir = Path(model_dir)

        # Load CatBoost pitch type model
        type_model = CatBoostClassifier()
        type_model.load_model(str(model_dir / "catboost" / "type_model.cbm"))

        # Load CatBoost feature info
        with open(model_dir / "catboost" / "feature_info.json") as f:
            catboost_info = json.load(f)

        # Load MDN model
        checkpoint = torch.load(
            model_dir / "mdn_location_model.pt",
            map_location=device,
            weights_only=False,
        )

        mdn_model = BivariateMDN(
            n_features=checkpoint["config"]["n_features"],
            hidden_dims=checkpoint["config"]["hidden_dims"],
            n_components=checkpoint["config"]["n_components"],
            dropout=checkpoint["config"]["dropout"],
        )
        mdn_model.load_state_dict(checkpoint["model_state_dict"])

        # Load feature engine data
        with open(model_dir / "feature_engine.json") as f:
            feature_engine_data = json.load(f)

        return cls(
            type_model=type_model,
            mdn_model=mdn_model,
            feature_columns=catboost_info["feature_columns"],
            categorical_features=catboost_info["categorical_features"],
            mdn_feature_columns=checkpoint["feature_columns"],
            pitcher_to_idx=feature_engine_data.get("pitcher_to_idx", {}),
            batter_to_idx=feature_engine_data.get("batter_to_idx", {}),
            device=device,
            model_type="catboost",
        )

    @classmethod
    def load_lstm(cls, model_dir: Union[str, Path], device: str = "cpu") -> "PitchPredictor":
        """
        Load a trained LSTM (or LSTM+Attention) predictor from disk.

        Args:
            model_dir: Path to the LSTM model directory (e.g., models/attention_full/run_xxx)
            device: Device for inference ("cpu", "cuda", or "mps")

        Returns:
            Loaded PitchPredictor instance configured for LSTM inference
        """
        model_dir = Path(model_dir)

        # Load checkpoint first to get model metadata
        checkpoint = torch.load(
            model_dir / "final_model.pt",
            map_location=device,
            weights_only=False,
        )

        # Extract metadata from checkpoint (for models saved with full metadata)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config = checkpoint["config"]
            n_pitchers = checkpoint["n_pitchers"]
            n_batters = checkpoint["n_batters"]
            n_features = checkpoint["n_features"]
            feature_indices = checkpoint.get("feature_indices")
            model_state_dict = checkpoint["model_state_dict"]
        else:
            # Fallback to results.json for older checkpoints
            with open(model_dir / "results.json") as f:
                results = json.load(f)
            config = results["config"]
            model_state_dict = checkpoint
            n_pitchers = None
            n_batters = None
            n_features = None
            feature_indices = None

        model_type = config.get("model_type", "lstm")

        # Load the feature engine for transform functionality
        feature_engine_path = model_dir / "feature_engine.json"
        feature_engine = None
        if feature_engine_path.exists():
            feature_engine = PitchFeatureEngine.load(feature_engine_path)
        else:
            # Try parent directory (for backward compatibility)
            parent_engine_path = model_dir.parent / "feature_engine.json"
            if parent_engine_path.exists():
                feature_engine = PitchFeatureEngine.load(parent_engine_path)

        # Use checkpoint metadata if available, otherwise fall back to feature_engine
        if n_pitchers is None:
            if feature_engine is None:
                raise FileNotFoundError(
                    f"Could not find feature_engine.json in {model_dir} or {model_dir.parent}"
                )
            n_pitchers = feature_engine.n_pitchers
            n_batters = feature_engine.n_batters
            n_features = len(feature_engine.get_feature_columns())
            feature_indices = feature_engine.get_feature_indices()
        elif feature_engine is not None:
            # Use feature_indices from feature_engine if not in checkpoint
            if feature_indices is None:
                feature_indices = feature_engine.get_feature_indices()

        # Create the model architecture with exact dimensions from training
        lstm_model = create_model(
            n_pitch_types=len(PITCH_TYPE_CODES),
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_features=n_features,
            model_type=model_type,
            feature_indices=feature_indices,
            embedding_dim=config.get("embedding_dim", 32),
            hidden_dim=config.get("hidden_dim", 128),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.3),
            n_location_components=config.get("n_location_components", 3),
            n_attention_heads=config.get("n_attention_heads", 4),
            n_attention_layers=config.get("n_attention_layers", 1),
        )

        # Load model weights
        lstm_model.load_state_dict(model_state_dict)
        lstm_model.to(device)
        lstm_model.eval()

        return cls(
            lstm_model=lstm_model,
            feature_engine=feature_engine,
            feature_columns=feature_engine.get_feature_columns() if feature_engine else [],
            pitcher_to_idx=feature_engine.pitcher_to_idx if feature_engine else {},
            batter_to_idx=feature_engine.batter_to_idx if feature_engine else {},
            device=device,
            model_type="lstm",
        )

    def predict(
        self,
        catboost_features=None,
        mdn_features: torch.Tensor = None,
        lstm_features: torch.Tensor = None,
        grid_size: int = 100,
        n_density_samples: int = 1000,
    ) -> PitchPrediction:
        """
        Make a full prediction for a single pitch.

        For CatBoost mode:
            catboost_features: Features for CatBoost (DataFrame or array-like)
            mdn_features: Features for MDN [n_features] or [1, n_features]

        For LSTM mode:
            lstm_features: Features for LSTM [seq_len, n_features] or [1, seq_len, n_features]
                          The prediction is made for the last position in the sequence.

        Args:
            catboost_features: Features for CatBoost (DataFrame or array-like)
            mdn_features: Features for MDN [n_features] or [1, n_features]
            lstm_features: Features for LSTM [seq_len, n_features]
            grid_size: Resolution for density grid
            n_density_samples: Samples for KDE estimation

        Returns:
            PitchPrediction with all prediction outputs
        """
        if self.model_type == "lstm":
            return self._predict_lstm(lstm_features, grid_size, n_density_samples)
        else:
            return self._predict_catboost(catboost_features, mdn_features, grid_size, n_density_samples)

    def _predict_catboost(
        self,
        catboost_features,
        mdn_features: torch.Tensor,
        grid_size: int = 100,
        n_density_samples: int = 1000,
    ) -> PitchPrediction:
        """Make prediction using CatBoost + MDN models."""
        # Ensure mdn_features is properly shaped
        if isinstance(mdn_features, np.ndarray):
            mdn_features = torch.tensor(mdn_features, dtype=torch.float32)
        if mdn_features.dim() == 1:
            mdn_features = mdn_features.unsqueeze(0)
        mdn_features = mdn_features.to(self.device)

        # Get pitch type probabilities from CatBoost
        pool = Pool(catboost_features, cat_features=self.cat_indices)
        type_probs = self.type_model.predict_proba(pool)

        # Handle batch dimension
        if type_probs.ndim == 2:
            type_probs = type_probs[0]

        predicted_idx = int(np.argmax(type_probs))
        predicted_type = IDX_TO_PITCH_TYPE.get(predicted_idx, "UNK")

        # Get top 3 pitch types
        top_3_indices = np.argsort(type_probs)[::-1][:3]
        top_3_types = [
            (IDX_TO_PITCH_TYPE.get(i, "UNK"), type_probs[i])
            for i in top_3_indices
        ]

        # Get MDN predictions
        with torch.no_grad():
            params = self.mdn_model(mdn_features)

            # Point estimates
            location_point = self.mdn_model.get_expected_value(params)[0].cpu().numpy()
            location_mode = self.mdn_model.get_mode(params)[0].cpu().numpy()

            # Mixture parameters
            mixture_weights = params["pi"][0].cpu().numpy()
            mixture_means = params["mu"][0].cpu().numpy()
            mixture_stds = params["sigma"][0].cpu().numpy()

        # Get location density grid
        px_grid, pz_grid, density = get_location_density(
            self.mdn_model,
            mdn_features,
            px_range=(-2.5, 2.5),
            pz_range=(0.5, 4.5),
            grid_size=grid_size,
            n_samples=n_density_samples,
        )

        return PitchPrediction(
            type_probabilities=type_probs,
            predicted_type_idx=predicted_idx,
            predicted_type=predicted_type,
            top_3_types=top_3_types,
            location_point=location_point,
            location_mode=location_mode,
            px_grid=px_grid,
            pz_grid=pz_grid,
            location_density=density,
            mixture_weights=mixture_weights,
            mixture_means=mixture_means,
            mixture_stds=mixture_stds,
        )

    def _predict_lstm(
        self,
        features: torch.Tensor,
        grid_size: int = 100,
        n_density_samples: int = 1000,
    ) -> PitchPrediction:
        """
        Make prediction using LSTM model.

        Args:
            features: Sequence features [seq_len, n_features] or [1, seq_len, n_features]
            grid_size: Resolution for density grid
            n_density_samples: Samples for location density estimation

        Returns:
            PitchPrediction for the last position in the sequence
        """
        # Ensure features is properly shaped [batch, seq_len, n_features]
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension
        features = features.to(self.device)

        batch_size, seq_len, _ = features.shape
        lengths = torch.tensor([seq_len], dtype=torch.long)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # Get model output (handles both lstm and lstm_attention)
            output = self.lstm_model(features, lengths, mask)
            if len(output) == 2:
                pitch_type_logits, mdn_params = output
            else:
                pitch_type_logits, mdn_params, _ = output  # Attention weights

            # Get predictions for last position in sequence
            type_logits_last = pitch_type_logits[0, -1, :]  # [n_pitch_types]
            type_probs = torch.softmax(type_logits_last, dim=-1).cpu().numpy()

            predicted_idx = int(np.argmax(type_probs))
            predicted_type = IDX_TO_PITCH_TYPE.get(predicted_idx, "UNK")

            # Get top 3 pitch types
            top_3_indices = np.argsort(type_probs)[::-1][:3]
            top_3_types = [
                (IDX_TO_PITCH_TYPE.get(i, "UNK"), type_probs[i])
                for i in top_3_indices
            ]

            # Get MDN parameters for last position
            pi = mdn_params["pi"][0, -1, :]  # [K]
            mu = mdn_params["mu"][0, -1, :, :]  # [K, 2]
            sigma = mdn_params["sigma"][0, -1, :, :]  # [K, 2]
            rho = mdn_params["rho"][0, -1, :]  # [K]

            mixture_weights = pi.cpu().numpy()
            mixture_means = mu.cpu().numpy()
            mixture_stds = sigma.cpu().numpy()

            # Compute expected value (weighted mean of components)
            location_point = (pi.unsqueeze(-1) * mu).sum(dim=0).cpu().numpy()

        # Generate location density grid using sampling
        px_grid, pz_grid, density = self._compute_lstm_location_density(
            mdn_params, grid_size, n_density_samples
        )

        # Compute mode as the peak of the KDE (true mode of mixture)
        max_idx = np.unravel_index(np.argmax(density), density.shape)
        location_mode = np.array([px_grid[max_idx[1]], pz_grid[max_idx[0]]])

        return PitchPrediction(
            type_probabilities=type_probs,
            predicted_type_idx=predicted_idx,
            predicted_type=predicted_type,
            top_3_types=top_3_types,
            location_point=location_point,
            location_mode=location_mode,
            px_grid=px_grid,
            pz_grid=pz_grid,
            location_density=density,
            mixture_weights=mixture_weights,
            mixture_means=mixture_means,
            mixture_stds=mixture_stds,
        )

    def _compute_lstm_location_density(
        self,
        mdn_params: dict,
        grid_size: int = 100,
        n_samples: int = 1000,
        px_range: tuple = (-2.5, 2.5),
        pz_range: tuple = (0.5, 4.5),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute location density grid from LSTM MDN parameters.

        Uses kernel density estimation on samples from the mixture distribution.
        """
        from scipy import stats

        # Get parameters for last position
        pi = mdn_params["pi"][0, -1, :].cpu().numpy()  # [K]
        mu = mdn_params["mu"][0, -1, :, :].cpu().numpy()  # [K, 2]
        sigma = mdn_params["sigma"][0, -1, :, :].cpu().numpy()  # [K, 2]
        rho = mdn_params["rho"][0, -1, :].cpu().numpy()  # [K]

        K = len(pi)

        # Sample from the mixture
        samples = []
        for _ in range(n_samples):
            # Select component based on mixture weights
            k = np.random.choice(K, p=pi)

            # Sample from bivariate normal with correlation
            mean = mu[k]
            std = sigma[k]
            correlation = rho[k]

            # Create covariance matrix
            cov = np.array([
                [std[0]**2, correlation * std[0] * std[1]],
                [correlation * std[0] * std[1], std[1]**2]
            ])

            sample = np.random.multivariate_normal(mean, cov)
            samples.append(sample)

        samples = np.array(samples)

        # Create grid
        px_grid = np.linspace(px_range[0], px_range[1], grid_size)
        pz_grid = np.linspace(pz_range[0], pz_range[1], grid_size)

        # Use KDE to estimate density
        try:
            kernel = stats.gaussian_kde(samples.T)
            PX, PZ = np.meshgrid(px_grid, pz_grid)
            positions = np.vstack([PX.ravel(), PZ.ravel()])
            density = kernel(positions).reshape(grid_size, grid_size)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback if KDE fails (e.g., singular covariance)
            density = np.zeros((grid_size, grid_size))

        return px_grid, pz_grid, density

    def predict_batch(
        self,
        catboost_features=None,
        mdn_features: torch.Tensor = None,
        lstm_features: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        """
        Make predictions for a batch of pitches (without density grids).

        For efficiency, this returns only point estimates and probabilities,
        not the full density grids.

        For CatBoost mode:
            catboost_features: Features for CatBoost [batch, n_features]
            mdn_features: Features for MDN [batch, n_features]

        For LSTM mode:
            lstm_features: Features for LSTM [batch, seq_len, n_features]
            lengths: Sequence lengths [batch]

        Returns:
            Dictionary with batch predictions
        """
        if self.model_type == "lstm":
            return self._predict_batch_lstm(lstm_features, lengths)
        else:
            return self._predict_batch_catboost(catboost_features, mdn_features)

    def _predict_batch_catboost(
        self,
        catboost_features,
        mdn_features: torch.Tensor,
    ) -> dict:
        """Make batch predictions using CatBoost + MDN."""
        if isinstance(mdn_features, np.ndarray):
            mdn_features = torch.tensor(mdn_features, dtype=torch.float32)
        mdn_features = mdn_features.to(self.device)

        # CatBoost predictions
        pool = Pool(catboost_features, cat_features=self.cat_indices)
        type_probs = self.type_model.predict_proba(pool)
        predicted_types = np.argmax(type_probs, axis=1)

        # MDN predictions
        with torch.no_grad():
            params = self.mdn_model(mdn_features)
            location_points = self.mdn_model.get_expected_value(params).cpu().numpy()
            location_modes = self.mdn_model.get_mode(params).cpu().numpy()

        return {
            "type_probabilities": type_probs,
            "predicted_type_indices": predicted_types,
            "predicted_types": [IDX_TO_PITCH_TYPE.get(i, "UNK") for i in predicted_types],
            "location_points": location_points,
            "location_modes": location_modes,
        }

    def _predict_batch_lstm(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> dict:
        """
        Make batch predictions using LSTM model.

        Args:
            features: Sequence features [batch, seq_len, n_features]
            lengths: Sequence lengths [batch]. If None, assumes all sequences are same length.

        Returns:
            Dictionary with predictions for each position in each sequence
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)

        batch_size, seq_len, _ = features.shape

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        mask = torch.arange(seq_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)

        with torch.no_grad():
            output = self.lstm_model(features, lengths.cpu(), mask)
            if len(output) == 2:
                pitch_type_logits, mdn_params = output
            else:
                pitch_type_logits, mdn_params, _ = output

            # Convert logits to probabilities
            type_probs = torch.softmax(pitch_type_logits, dim=-1).cpu().numpy()
            predicted_types = np.argmax(type_probs, axis=-1)

            # Get location predictions (expected value from mixture)
            pi = mdn_params["pi"]  # [batch, seq, K]
            mu = mdn_params["mu"]  # [batch, seq, K, 2]

            # Weighted sum of means
            location_points = (pi.unsqueeze(-1) * mu).sum(dim=2).cpu().numpy()

            # Mode: mean of highest-weight component
            max_weight_idx = torch.argmax(pi, dim=-1)  # [batch, seq]
            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len)
            seq_idx = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            location_modes = mu[batch_idx, seq_idx, max_weight_idx].cpu().numpy()

        return {
            "type_probabilities": type_probs,
            "predicted_type_indices": predicted_types,
            "predicted_types": [[IDX_TO_PITCH_TYPE.get(i, "UNK") for i in seq] for seq in predicted_types],
            "location_points": location_points,
            "location_modes": location_modes,
            "mask": mask.cpu().numpy(),
        }

    def plot_prediction(
        self,
        prediction: PitchPrediction,
        title: Optional[str] = None,
        actual_location: Optional[tuple[float, float]] = None,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot a complete prediction visualization.

        Creates a two-panel figure:
        - Left: Pitch type probabilities (bar chart)
        - Right: Location density with strike zone

        Args:
            prediction: PitchPrediction from predict()
            title: Optional overall title
            actual_location: Optional (px, pz) of actual pitch for comparison
            save_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left panel: Pitch type probabilities
        pitch_types = PITCH_TYPE_CODES
        probs = prediction.type_probabilities

        # Sort by probability for better visualization
        sorted_idx = np.argsort(probs)[::-1]
        sorted_types = [pitch_types[i] for i in sorted_idx]
        sorted_probs = probs[sorted_idx]

        # Only show types with >1% probability
        mask = sorted_probs > 0.01
        bars = ax1.barh(
            range(sum(mask)),
            sorted_probs[mask],
            color='steelblue',
            edgecolor='navy',
        )
        ax1.set_yticks(range(sum(mask)))
        ax1.set_yticklabels([sorted_types[i] for i in range(len(sorted_types)) if mask[i]])
        ax1.set_xlabel('Probability')
        ax1.set_title('Pitch Type Probabilities')
        ax1.set_xlim(0, 1)
        ax1.invert_yaxis()

        # Add probability labels
        for i, (bar, prob) in enumerate(zip(bars, sorted_probs[mask])):
            ax1.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}',
                va='center',
                fontsize=9,
            )

        # Right panel: Location density
        PX, PZ = np.meshgrid(prediction.px_grid, prediction.pz_grid)

        # Plot density contours
        contour = ax2.contourf(
            PX, PZ, prediction.location_density,
            levels=20,
            cmap='YlOrRd',
            alpha=0.8,
        )
        ax2.contour(
            PX, PZ, prediction.location_density,
            levels=10,
            colors='darkred',
            alpha=0.3,
            linewidths=0.5,
        )

        # Draw strike zone
        strike_zone = plt.Rectangle(
            (-0.83, 1.5), 1.66, 2.0,
            fill=False, edgecolor='black', linewidth=2,
        )
        ax2.add_patch(strike_zone)

        # Plot point estimate
        ax2.scatter(
            prediction.location_point[0],
            prediction.location_point[1],
            c='green', s=100, marker='*',
            label=f'Expected ({prediction.location_point[0]:.2f}, {prediction.location_point[1]:.2f})',
            zorder=10, edgecolors='darkgreen', linewidths=1,
        )

        # Plot mixture component means (significant ones only)
        for i, (w, mean) in enumerate(zip(prediction.mixture_weights, prediction.mixture_means)):
            if w > 0.1:
                ax2.scatter(
                    mean[0], mean[1],
                    c='blue', s=50*w*10, marker='o',
                    alpha=0.6, edgecolors='navy',
                )

        # Plot actual location if provided
        if actual_location is not None:
            ax2.scatter(
                actual_location[0], actual_location[1],
                c='red', s=150, marker='x',
                label=f'Actual ({actual_location[0]:.2f}, {actual_location[1]:.2f})',
                zorder=11, linewidths=3,
            )

        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(0.5, 4.5)
        ax2.set_xlabel('Horizontal Position (ft)')
        ax2.set_ylabel('Vertical Position (ft)')
        ax2.set_title('Location Probability Density')
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right')

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax2, shrink=0.8)
        cbar.set_label('Probability Density')

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_strike_zone_probability(
        self,
        prediction: PitchPrediction,
        zone_left: float = -0.83,
        zone_right: float = 0.83,
        zone_bottom: float = 1.5,
        zone_top: float = 3.5,
    ) -> float:
        """
        Calculate probability of pitch being in the strike zone.

        Args:
            prediction: PitchPrediction from predict()
            zone_left: Left edge of strike zone (ft)
            zone_right: Right edge of strike zone (ft)
            zone_bottom: Bottom of strike zone (ft)
            zone_top: Top of strike zone (ft)

        Returns:
            Probability (0-1) of pitch being in strike zone
        """
        # Create mask for strike zone
        px_mask = (prediction.px_grid >= zone_left) & (prediction.px_grid <= zone_right)
        pz_mask = (prediction.pz_grid >= zone_bottom) & (prediction.pz_grid <= zone_top)

        # Get density within zone
        zone_density = prediction.location_density[np.ix_(pz_mask, px_mask)]

        # Approximate integral (sum * cell area)
        dx = prediction.px_grid[1] - prediction.px_grid[0]
        dz = prediction.pz_grid[1] - prediction.pz_grid[0]

        zone_prob = zone_density.sum() * dx * dz

        # Normalize by total density
        total_prob = prediction.location_density.sum() * dx * dz

        return zone_prob / total_prob if total_prob > 0 else 0.0

    def _draw_baseball_diamond(
        self,
        ax: plt.Axes,
        runner_on_1b: bool = False,
        runner_on_2b: bool = False,
        runner_on_3b: bool = False,
        outs: int = 0,
    ) -> None:
        """
        Draw a baseball diamond with runners highlighted.

        Args:
            ax: Matplotlib axes to draw on
            runner_on_1b: Whether there's a runner on first
            runner_on_2b: Whether there's a runner on second
            runner_on_3b: Whether there's a runner on third
            outs: Number of outs (0-2)
        """
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.68, 1.68)
        ax.set_aspect('equal')
        ax.axis('off')

        # Base positions (diamond shape)
        home = (0, 0)
        first = (1, 0.7)
        second = (0, 1.4)
        third = (-1, 0.7)

        # Draw the diamond outline
        diamond_coords = [home, first, second, third, home]
        xs = [p[0] for p in diamond_coords]
        ys = [p[1] for p in diamond_coords]
        ax.plot(xs, ys, 'k-', linewidth=2)

        # Base size
        base_size = 0.18

        # Draw home plate (pentagon)
        home_plate = plt.Polygon(
            [(-0.1, 0), (0.1, 0), (0.1, 0.1), (0, 0.18), (-0.1, 0.1)],
            fill=True, facecolor='white', edgecolor='black', linewidth=2,
        )
        ax.add_patch(home_plate)

        # Draw bases (squares rotated 45 degrees)
        def draw_base(center, is_occupied):
            x, y = center
            color = '#f1c40f' if is_occupied else 'white'  # Yellow if occupied
            edge_color = '#d68910' if is_occupied else 'black'
            base = plt.Polygon(
                [(x, y - base_size), (x + base_size, y),
                 (x, y + base_size), (x - base_size, y)],
                fill=True, facecolor=color, edgecolor=edge_color, linewidth=2,
            )
            ax.add_patch(base)

        draw_base(first, runner_on_1b)
        draw_base(second, runner_on_2b)
        draw_base(third, runner_on_3b)

        # Draw outs indicator
        out_y = -0.35
        for i in range(3):
            out_x = -0.3 + i * 0.3
            is_out = i < outs
            circle = plt.Circle(
                (out_x, out_y), 0.08,
                fill=True,
                facecolor='#e74c3c' if is_out else 'white',
                edgecolor='#c0392b' if is_out else '#666666',
                linewidth=1.5,
            )
            ax.add_patch(circle)

        # Label for outs
        ax.text(0, out_y - 0.22, 'OUTS', ha='center', va='center',
                fontsize=8, color='#666666', fontweight='bold')

    def create_pitch_card(
        self,
        prediction: PitchPrediction,
        context: GameContext,
        actual_pitch_type: Optional[str] = None,
        actual_location: Optional[tuple[float, float]] = None,
        save_path: Optional[str] = None,
        figsize: tuple[float, float] = (10.5, 8.0),
    ) -> plt.Figure:
        """
        Create a compact pitch prediction card with full game context.

        Layout (two tight columns, no dead rows):
        - Navy header band: teams, score, inning, count, pitch number
        - Matchup row with headshots
        - Left column: pitch type probabilities, bases/outs diamond below
        - Right column: strike zone density, in-zone probability below
        - Optional result band when the actual pitch is provided

        Args:
            prediction: PitchPrediction from predict()
            context: GameContext with game information
            actual_pitch_type: Actual pitch thrown (e.g., "FF")
            actual_location: Actual (px, pz) location
            save_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        header_bg = '#14213d'
        ink = '#1f2937'
        muted = '#6b7280'
        track_color = '#e8ecf2'
        bar_color = '#94a9c9'
        top_color = '#fca311'
        actual_color = '#27ae60'

        fig = plt.figure(figsize=figsize, facecolor='white')

        has_result = bool(actual_pitch_type or actual_location)
        height_ratios = [0.62, 0.72, 1.55, 1.35, 0.30]
        n_rows = 5
        if has_result:
            height_ratios.append(0.34)
            n_rows = 6

        gs = fig.add_gridspec(
            n_rows, 2,
            height_ratios=height_ratios,
            width_ratios=[1.0, 1.12],
            hspace=0.30,
            wspace=0.10,
            left=0.05, right=0.95, top=0.975, bottom=0.03,
        )

        # =====================================================================
        # Header band: scoreboard
        # =====================================================================
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        ax_header.add_patch(plt.Rectangle(
            (0, 0), 1, 1, transform=ax_header.transAxes,
            facecolor=header_bg, edgecolor='none',
        ))

        home_team = context.home_team or "HOME"
        away_team = context.away_team or "AWAY"
        if context.score_home is not None and context.score_away is not None:
            score_text = f"{away_team} {context.score_away}  @  {home_team} {context.score_home}"
        else:
            score_text = f"{away_team}  @  {home_team}"

        ax_header.text(
            0.025, 0.62, score_text,
            ha='left', va='center', fontsize=17, fontweight='bold',
            color='white', transform=ax_header.transAxes,
        )
        if context.date:
            ax_header.text(
                0.025, 0.20, context.date,
                ha='left', va='center', fontsize=9, color='#9fb3c8',
                transform=ax_header.transAxes,
            )

        situation = f"{context.inning_half} {context.inning}  •  Count {context.count_str}"
        ax_header.text(
            0.975, 0.62, situation,
            ha='right', va='center', fontsize=13, fontweight='bold',
            color='white', transform=ax_header.transAxes,
        )
        if context.pitch_number:
            ax_header.text(
                0.975, 0.20, f"Pitch #{context.pitch_number} of at-bat",
                ha='right', va='center', fontsize=9, color='#9fb3c8',
                transform=ax_header.transAxes,
            )

        # =====================================================================
        # Matchup row with headshots
        # =====================================================================
        ax_matchup = fig.add_subplot(gs[1, :])
        ax_matchup.axis('off')
        ax_matchup.set_xlim(0, 1)
        ax_matchup.set_ylim(0, 1)

        pitcher_headshot = fetch_mlb_headshot(context.pitcher_id, size=80)
        batter_headshot = fetch_mlb_headshot(context.batter_id, size=80)

        pitcher_text_x = 0.135
        if pitcher_headshot is not None:
            im_pitcher = OffsetImage(pitcher_headshot, zoom=0.52)
            ax_matchup.add_artist(
                AnnotationBbox(im_pitcher, (0.06, 0.5), frameon=False)
            )
        ax_matchup.text(
            pitcher_text_x, 0.62, context.pitcher_name,
            ha='left', va='center', fontsize=12, fontweight='bold',
            color=ink, transform=ax_matchup.transAxes,
        )
        ax_matchup.text(
            pitcher_text_x, 0.30, f"{context.pitcher_hand}HP  •  pitching",
            ha='left', va='center', fontsize=9, color=muted,
            transform=ax_matchup.transAxes,
        )

        ax_matchup.text(
            0.5, 0.5, "vs",
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#b0b7c3', transform=ax_matchup.transAxes,
        )

        batter_text_x = 0.865
        if batter_headshot is not None:
            im_batter = OffsetImage(batter_headshot, zoom=0.52)
            ax_matchup.add_artist(
                AnnotationBbox(im_batter, (0.94, 0.5), frameon=False)
            )
        ax_matchup.text(
            batter_text_x, 0.62, context.batter_name,
            ha='right', va='center', fontsize=12, fontweight='bold',
            color=ink, transform=ax_matchup.transAxes,
        )
        ax_matchup.text(
            batter_text_x, 0.30, f"{context.batter_hand}HB  •  batting",
            ha='right', va='center', fontsize=9, color=muted,
            transform=ax_matchup.transAxes,
        )

        # =====================================================================
        # Left column, top: pitch type probabilities
        # =====================================================================
        ax_probs = fig.add_subplot(gs[2, 0])
        ax_probs.set_xlim(0, 1)
        ax_probs.set_ylim(0, 1)
        ax_probs.axis('off')
        ax_probs.text(
            0.0, 1.04, 'N E X T   P I T C H', ha='left', va='bottom',
            fontsize=9, fontweight='bold', color=muted,
            transform=ax_probs.transAxes,
        )

        probs = prediction.type_probabilities
        sorted_idx = np.argsort(probs)[::-1]
        sorted_codes = [PITCH_TYPE_CODES[i] for i in sorted_idx]
        sorted_probs = probs[sorted_idx]

        mask = sorted_probs > 0.01
        n_show = min(int(np.sum(mask)), 5)

        row_top = 0.92
        row_height = 0.21
        bar_y_offset = 0.055

        for i in range(n_show):
            y_pos = row_top - i * row_height
            pitch_code = sorted_codes[i]
            pitch_name = PITCH_TYPE_FULL_NAMES.get(pitch_code, pitch_code)
            prob = float(sorted_probs[i])

            is_actual = bool(actual_pitch_type and pitch_code == actual_pitch_type)
            if is_actual:
                name_color, fill_color, weight = actual_color, actual_color, 'bold'
            elif i == 0:
                name_color, fill_color, weight = ink, top_color, 'bold'
            else:
                name_color, fill_color, weight = muted, bar_color, 'normal'

            ax_probs.text(
                0.0, y_pos, pitch_name, ha='left', va='bottom',
                fontsize=10.5, color=name_color, fontweight=weight,
                transform=ax_probs.transAxes,
            )
            ax_probs.text(
                1.0, y_pos, f"{prob:.0%}", ha='right', va='bottom',
                fontsize=10.5, color=name_color, fontweight=weight,
                transform=ax_probs.transAxes,
            )
            bar_y = y_pos - bar_y_offset
            ax_probs.plot(
                [0.0, 1.0], [bar_y, bar_y], color=track_color,
                linewidth=5, solid_capstyle='round',
                transform=ax_probs.transAxes, zorder=1,
            )
            ax_probs.plot(
                [0.0, max(prob, 0.012)], [bar_y, bar_y], color=fill_color,
                linewidth=5, solid_capstyle='round',
                transform=ax_probs.transAxes, zorder=2,
            )

        if context.pitch_result:
            ax_probs.text(
                0.0, -0.08, f"Result: {context.pitch_result}",
                fontsize=9, va='top', color='#c0392b', fontweight='bold',
                transform=ax_probs.transAxes,
            )

        # =====================================================================
        # Left column, bottom: bases and outs
        # =====================================================================
        ax_diamond = fig.add_subplot(gs[3, 0])
        ax_diamond.text(
            0.0, 1.02, 'S I T U A T I O N', ha='left', va='bottom',
            fontsize=9, fontweight='bold', color=muted,
            transform=ax_diamond.transAxes,
        )
        self._draw_baseball_diamond(
            ax_diamond,
            runner_on_1b=context.runner_on_1b,
            runner_on_2b=context.runner_on_2b,
            runner_on_3b=context.runner_on_3b,
            outs=context.outs,
        )

        # =====================================================================
        # Right column: strike zone density (spans probability + diamond rows)
        # =====================================================================
        ax_zone = fig.add_subplot(gs[2:4, 1])
        ax_zone.text(
            0.0, 1.015, "P R E D I C T E D   L O C A T I O N   (P I T C H E R ' S   V I E W)",
            ha='left', va='bottom', fontsize=9, fontweight='bold', color=muted,
            transform=ax_zone.transAxes,
        )

        PX, PZ = np.meshgrid(-prediction.px_grid, prediction.pz_grid)
        ax_zone.contourf(
            PX, PZ, prediction.location_density,
            levels=20, cmap='YlOrRd', alpha=0.85,
        )
        ax_zone.contour(
            PX, PZ, prediction.location_density,
            levels=8, colors='darkred', alpha=0.25, linewidths=0.5,
        )

        zone_width = 17 / 12
        strike_zone = plt.Rectangle(
            (-zone_width / 2, 1.5), zone_width, 2.0,
            fill=False, edgecolor='black', linewidth=2.4,
        )
        ax_zone.add_patch(strike_zone)

        plate_y = 0.82
        plate = plt.Polygon(
            [[-0.708, plate_y + 0.3], [0.708, plate_y + 0.3],
             [0.708, plate_y + 0.2], [0, plate_y],
             [-0.708, plate_y + 0.2]],
            fill=True, facecolor='white', edgecolor='black', linewidth=1.6,
        )
        ax_zone.add_patch(plate)

        ax_zone.scatter(
            -prediction.location_point[0],
            prediction.location_point[1],
            c='#2e86de', s=90, marker='D', label='Expected',
            zorder=9, edgecolors='#1b4f8f', linewidths=1.4,
        )
        if actual_location is not None:
            ax_zone.scatter(
                -actual_location[0], actual_location[1],
                c='#e74c3c', s=180, marker='X', label='Actual',
                zorder=11, edgecolors='#c0392b', linewidths=2,
            )

        ax_zone.set_xlim(-1.9, 1.9)
        ax_zone.set_ylim(0.7, 4.3)
        ax_zone.set_aspect('equal')
        ax_zone.set_xticks([])
        ax_zone.set_yticks([])
        for spine in ax_zone.spines.values():
            spine.set_color('#d5dbe4')
        ax_zone.legend(
            loc='upper right', fontsize=8.5, frameon=False,
            handletextpad=0.5, borderaxespad=0.2,
        )
        ax_zone.text(-1.78, 0.82, '← LHB', fontsize=8, color=muted)
        ax_zone.text(1.32, 0.82, 'RHB →', fontsize=8, color=muted)

        # =====================================================================
        # Under the zone: in-zone probability
        # =====================================================================
        ax_zone_prob = fig.add_subplot(gs[4, 1])
        ax_zone_prob.axis('off')
        strike_prob = self.get_strike_zone_probability(prediction)
        ax_zone_prob.text(
            0.5, 0.5, f"In-zone probability: {strike_prob:.0%}",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=ink, transform=ax_zone_prob.transAxes,
        )

        # =====================================================================
        # Optional result band (only when the actual pitch is known)
        # =====================================================================
        if has_result:
            ax_footer = fig.add_subplot(gs[5, :])
            ax_footer.axis('off')

            result_parts = []
            if actual_pitch_type:
                actual_full_name = PITCH_TYPE_FULL_NAMES.get(actual_pitch_type, actual_pitch_type)
                result_parts.append(f"Actual: {actual_full_name}")
            if actual_location:
                result_parts.append(
                    f"Location: ({actual_location[0]:.2f}, {actual_location[1]:.2f})"
                )
                exp_error = np.sqrt(
                    (prediction.location_point[0] - actual_location[0])**2 +
                    (prediction.location_point[1] - actual_location[1])**2
                )
                result_parts.append(f"Exp Err: {exp_error:.2f} ft")
            if actual_pitch_type:
                if actual_pitch_type == prediction.predicted_type:
                    result_parts.append("✓ Type Correct")
                else:
                    actual_prob = (
                        probs[PITCH_TYPE_CODES.index(actual_pitch_type)]
                        if actual_pitch_type in PITCH_TYPE_CODES
                        else 0
                    )
                    result_parts.append(
                        f"Predicted: {prediction.predicted_type} (actual had {actual_prob:.0%})"
                    )

            footer_color = (
                actual_color
                if actual_pitch_type == prediction.predicted_type
                else '#e74c3c'
            )
            ax_footer.text(
                0.5, 0.5, "  •  ".join(result_parts),
                ha='center', va='center', fontsize=10.5,
                color=footer_color, fontweight='bold',
                transform=ax_footer.transAxes,
            )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        return fig


def create_pitch_card_from_row(
    predictor: PitchPredictor,
    row: pl.DataFrame,
    catboost_features=None,
    mdn_features: torch.Tensor = None,
    lstm_features: torch.Tensor = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Convenience function to create a pitch card directly from a data row.

    Args:
        predictor: Loaded PitchPredictor
        row: Single row DataFrame with pitch data
        catboost_features: Prepared CatBoost features (for CatBoost mode)
        mdn_features: Prepared MDN features (for CatBoost mode)
        lstm_features: Prepared LSTM features (for LSTM mode)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Extract context from row
    def get_val(col, default=None):
        if col in row.columns:
            val = row[col][0]
            return val if val is not None else default
        return default

    # Get runner states - check both naming conventions
    runner_on_1b = bool(get_val("is_runner_on_first", False) or get_val("runner_on_1b", False))
    runner_on_2b = bool(get_val("is_runner_on_second", False) or get_val("runner_on_2b", False))
    runner_on_3b = bool(get_val("is_runner_on_third", False) or get_val("runner_on_3b", False))

    # Get team names - check both naming conventions
    home_team = get_val("home_team_name") or get_val("home_team", "HOME")
    away_team = get_val("away_team_name") or get_val("away_team", "AWAY")

    # Format game date nicely
    def format_game_date(date_val):
        if date_val is None:
            return None
        date_str = str(date_val)
        try:
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
            return dt.strftime('%B %d, %Y')
        except:
            return date_str[:10] if len(date_str) >= 10 else date_str

    game_date = format_game_date(get_val("game_date"))

    context = GameContext(
        pitcher_name=get_val("pitcher_name", "Unknown"),
        batter_name=get_val("batter_name", "Unknown"),
        pitcher_hand=get_val("throw_side", "R"),
        batter_hand=get_val("bat_side", "R"),
        home_team=home_team,
        away_team=away_team,
        inning=get_val("inning", 1),
        inning_half="Top" if get_val("half_inning", "top") == "top" else "Bot",
        balls=get_val("balls", 0),
        strikes=get_val("strikes", 0),
        outs=get_val("outs", 0),
        date=game_date,
        runner_on_1b=runner_on_1b,
        runner_on_2b=runner_on_2b,
        runner_on_3b=runner_on_3b,
        score_home=get_val("home_score"),
        score_away=get_val("away_score"),
        pitch_number=get_val("pitch_number"),
        pitcher_id=get_val("pitcher_id"),
        batter_id=get_val("batter_id"),
        pitch_result=get_val("pitch_call_description"),
    )

    # Make prediction based on model type
    if predictor.model_type == "lstm":
        prediction = predictor.predict(lstm_features=lstm_features)
    else:
        prediction = predictor.predict(catboost_features=catboost_features, mdn_features=mdn_features)

    # Get actual values
    actual_type = get_val("pitch_type_code")
    actual_px = get_val("px")
    actual_pz = get_val("pz")
    actual_location = (actual_px, actual_pz) if actual_px is not None and actual_pz is not None else None

    return predictor.create_pitch_card(
        prediction=prediction,
        context=context,
        actual_pitch_type=actual_type,
        actual_location=actual_location,
        save_path=save_path,
    )


def create_pitch_card_from_at_bat(
    predictor: PitchPredictor,
    at_bat_df: pl.DataFrame,
    pitch_index: int = -1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a pitch card for a specific pitch in an at-bat using LSTM model.

    This function prepares the sequence features and creates a pitch card
    for the specified pitch in the at-bat sequence.

    Args:
        predictor: Loaded PitchPredictor (must be LSTM type)
        at_bat_df: DataFrame containing all pitches in the at-bat (sorted by pitch_number)
        pitch_index: Index of the pitch to predict (-1 for last pitch)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if predictor.model_type != "lstm":
        raise ValueError("create_pitch_card_from_at_bat requires an LSTM model")

    if predictor.feature_engine is None:
        raise ValueError("LSTM predictor must have a feature_engine")

    # Transform the at-bat data using the feature engine
    at_bat_transformed = predictor.feature_engine.transform(at_bat_df)

    # Get feature columns
    feature_cols = predictor.feature_engine.get_feature_columns()

    # Extract features as tensor
    features = at_bat_transformed.select(feature_cols).to_numpy()
    features = torch.tensor(features, dtype=torch.float32)

    # Get the row for context (the pitch we're predicting)
    if pitch_index < 0:
        pitch_index = len(at_bat_df) + pitch_index
    row = at_bat_df.slice(pitch_index, 1)

    # For LSTM, we use all pitches up to and including the target pitch
    sequence_features = features[:pitch_index + 1]

    return create_pitch_card_from_row(
        predictor=predictor,
        row=row,
        lstm_features=sequence_features,
        save_path=save_path,
    )
