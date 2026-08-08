"""Next-pitch prediction service for live games.

Loads the trained pitch-type model (LSTM+Attention) once, optionally
refines the location density with the standalone pitch-type-conditioned
MDN, and turns a :class:`LiveSnapshot` into a :class:`PitchPrediction`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.live.game_state import LiveSnapshot
from src.ml.pitch_predictor import PitchPrediction, PitchPredictor
from src.ml.pitch_type_location_model import PitchTypeConditionedMDN

PX_RANGE = (-2.5, 2.5)
PZ_RANGE = (0.5, 4.5)


def mixture_density_grid(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    grid_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a bivariate Gaussian mixture density on a fixed grid.

    Closed-form evaluation (no sampling) so live predictions are
    deterministic and fast.
    """
    px_grid = np.linspace(PX_RANGE[0], PX_RANGE[1], grid_size)
    pz_grid = np.linspace(PZ_RANGE[0], PZ_RANGE[1], grid_size)
    xx, zz = np.meshgrid(px_grid, pz_grid)

    density = np.zeros_like(xx)
    for k in range(len(pi)):
        sx = max(float(sigma[k, 0]), 1e-3)
        sz = max(float(sigma[k, 1]), 1e-3)
        r = float(np.clip(rho[k], -0.99, 0.99))
        dx = (xx - float(mu[k, 0])) / sx
        dz = (zz - float(mu[k, 1])) / sz
        one_minus_r2 = 1.0 - r * r
        exponent = -(dx * dx - 2.0 * r * dx * dz + dz * dz) / (2.0 * one_minus_r2)
        norm = 1.0 / (2.0 * math.pi * sx * sz * math.sqrt(one_minus_r2))
        density += float(pi[k]) * norm * np.exp(exponent)

    return px_grid, pz_grid, density


class LiveNextPitchPredictor:
    """Model bundle that predicts the next pitch from a live snapshot."""

    def __init__(
        self,
        pitch_type_model_dir: str | Path,
        location_model_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.pitch_predictor = PitchPredictor.load_lstm(
            pitch_type_model_dir, device=device
        )
        if self.pitch_predictor.feature_engine is None:
            raise FileNotFoundError(
                f"feature_engine.json not found under {pitch_type_model_dir}"
            )
        self.feature_columns = (
            self.pitch_predictor.feature_engine.get_feature_columns()
        )

        self.location_model: PitchTypeConditionedMDN | None = None
        self.location_feature_columns: list[str] = []
        if location_model_dir is not None:
            self.location_model, self.location_feature_columns = (
                self._load_location_model(Path(location_model_dir))
            )

    def _load_location_model(
        self, model_dir: Path
    ) -> tuple[PitchTypeConditionedMDN, list[str]]:
        with open(model_dir / "config.json") as f:
            config = json.load(f)

        feature_columns = [
            column
            for column in config.get("feature_columns", [])
            if column != "pitch_type_idx"
        ]
        model = PitchTypeConditionedMDN(
            n_features=len(feature_columns),
            hidden_dims=config.get("hidden_dims", [256, 128]),
            n_components=config.get("n_components", 3),
            dropout=config.get("dropout", 0.2),
        )
        state_dict = torch.load(
            model_dir / "pitch_type_location_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model, feature_columns

    def _at_bat_features(self, snapshot: LiveSnapshot) -> pl.DataFrame:
        engine = self.pitch_predictor.feature_engine
        assert engine is not None
        transformed = engine.transform(snapshot.frame)
        at_bat = transformed.filter(
            pl.col("at_bat_index") == snapshot.at_bat_index
        ).sort("pitch_number")
        if at_bat.is_empty():
            raise ValueError(
                f"No rows for at-bat {snapshot.at_bat_index} in game {snapshot.game_pk}"
            )
        return at_bat

    def predict(self, snapshot: LiveSnapshot) -> PitchPrediction:
        at_bat = self._at_bat_features(snapshot)

        features = torch.tensor(
            np.nan_to_num(
                at_bat.select(self.feature_columns)
                .cast(pl.Float32, strict=False)
                .to_numpy(),
                nan=0.0,
            ),
            dtype=torch.float32,
        )
        prediction = self.pitch_predictor.predict(lstm_features=features)

        if self.location_model is not None and self.location_feature_columns:
            prediction = self._refine_location(at_bat, prediction)
        return prediction

    def _refine_location(
        self, at_bat: pl.DataFrame, prediction: PitchPrediction
    ) -> PitchPrediction:
        assert self.location_model is not None
        available = [
            column
            for column in self.location_feature_columns
            if column in at_bat.columns
        ]
        if len(available) != len(self.location_feature_columns):
            return prediction

        loc_features = torch.tensor(
            np.nan_to_num(
                at_bat.select(self.location_feature_columns)
                .cast(pl.Float32, strict=False)
                .to_numpy()[-1:],
                nan=0.0,
            ),
            dtype=torch.float32,
        ).to(self.device)
        type_probs = torch.tensor(
            prediction.type_probabilities, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            params = self.location_model.forward_soft(loc_features, type_probs)

        pi = params["pi"][0].cpu().numpy()
        mu = params["mu"][0].cpu().numpy()
        sigma = params["sigma"][0].cpu().numpy()
        rho = params["rho"][0].cpu().numpy()

        px_grid, pz_grid, density = mixture_density_grid(pi, mu, sigma, rho)
        peak = np.unravel_index(np.argmax(density), density.shape)

        prediction.location_point = (pi[:, None] * mu).sum(axis=0)
        prediction.location_mode = np.array(
            [px_grid[peak[1]], pz_grid[peak[0]]]
        )
        prediction.px_grid = px_grid
        prediction.pz_grid = pz_grid
        prediction.location_density = density
        prediction.mixture_weights = pi
        prediction.mixture_means = mu
        prediction.mixture_stds = sigma
        return prediction
