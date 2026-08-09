"""Live inference for the pitch outcome models.

Marginalizes Stage A (pitch result) and Stage B (in-play event) over the
upstream pitch prediction: a distribution over pitch types and a sample of
plausible locations. The upstream models are opaque here — this module
consumes only pitch-type code strings, location samples, and game state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.outcome.dataset import (
    CATEGORICAL_FEATURES,
    DEFAULT_SZ_BOTTOM,
    DEFAULT_SZ_TOP,
    FEATURE_COLUMNS,
    ZONE_HALF_WIDTH_FT,
)
from src.outcome.labels import canonicalize_pitch_type
from src.outcome.profiles import load_profile_stores

RESULT_CLASS_ORDER = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
]
EVENT_CLASS_ORDER = [
    "out",
    "single",
    "double",
    "triple",
    "home_run",
    "reached_on_error",
]


@dataclass(frozen=True)
class OutcomeGameState:
    """Everything the outcome models need about the moment before a pitch."""

    balls: int
    strikes: int
    outs: int
    runner_on_first: bool
    runner_on_second: bool
    runner_on_third: bool
    inning: int
    is_top_half: bool
    score_diff: int  # batting team minus fielding team
    season: int
    times_through_order: int
    pitcher_id: int | None
    batter_id: int | None
    throw_side: str
    bat_side: str
    sz_top: float = DEFAULT_SZ_TOP
    sz_bottom: float = DEFAULT_SZ_BOTTOM


def sample_locations_from_grid(
    px_grid: np.ndarray,
    pz_grid: np.ndarray,
    density: np.ndarray,
    n_samples: int = 40,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """Draw plate locations from a gridded density (jittered cell centers)."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(density, dtype=np.float64).ravel()
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        weights = np.ones_like(weights)
        total = weights.sum()
    indices = rng.choice(weights.size, size=n_samples, p=weights / total)
    rows, cols = np.unravel_index(indices, density.shape)
    dx = float(px_grid[1] - px_grid[0]) if len(px_grid) > 1 else 0.05
    dz = float(pz_grid[1] - pz_grid[0]) if len(pz_grid) > 1 else 0.05
    jitter_x = rng.uniform(-dx / 2, dx / 2, size=n_samples)
    jitter_z = rng.uniform(-dz / 2, dz / 2, size=n_samples)
    return [
        (float(px_grid[c] + jx), float(pz_grid[r] + jz))
        for r, c, jx, jz in zip(rows, cols, jitter_x, jitter_z)
    ]


def build_feature_frame(
    state: OutcomeGameState,
    type_probabilities: dict[str, float],
    locations: list[tuple[float, float]],
    pitcher_profiles: pl.DataFrame,
    batter_priors: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray]:
    """One feature row per (pitch type × location), plus row weights.

    Weights are P(type) / n_locations, so weighted sums marginalize the
    outcome probabilities over the predicted pitch distribution.
    """
    canonical: dict[str, float] = {}
    for code, prob in type_probabilities.items():
        mapped = canonicalize_pitch_type(code)
        if mapped is None or prob <= 0:
            continue
        canonical[mapped] = canonical.get(mapped, 0.0) + float(prob)
    if not canonical:
        raise ValueError("No usable pitch types in the predicted distribution")

    zone_center = (state.sz_top + state.sz_bottom) / 2.0
    zone_half_height = max((state.sz_top - state.sz_bottom) / 2.0, 1e-3)

    rows: list[dict] = []
    weights: list[float] = []
    n_locations = max(len(locations), 1)
    for pitch_type, type_prob in canonical.items():
        for px, pz in locations:
            rows.append(
                {
                    "pitch_type": pitch_type,
                    "throw_side": state.throw_side,
                    "bat_side": state.bat_side,
                    "pitcher_id": state.pitcher_id,
                    "batter_id": state.batter_id,
                    "pitcher_id_cat": str(state.pitcher_id) if state.pitcher_id else "unknown",
                    "batter_id_cat": str(state.batter_id) if state.batter_id else "unknown",
                    "balls_before": state.balls,
                    "strikes_before": state.strikes,
                    "outs": state.outs,
                    "runner_on_first": int(state.runner_on_first),
                    "runner_on_second": int(state.runner_on_second),
                    "runner_on_third": int(state.runner_on_third),
                    "inning": state.inning,
                    "is_top_half": int(state.is_top_half),
                    "score_diff": state.score_diff,
                    "season": state.season,
                    "times_through_order": state.times_through_order,
                    "px": px,
                    "pz": pz,
                    "abs_px": abs(px),
                    "zone_norm_height": (pz - state.sz_bottom)
                    / (state.sz_top - state.sz_bottom),
                    "zone_dist_center": float(
                        np.sqrt(
                            (px / ZONE_HALF_WIDTH_FT) ** 2
                            + ((pz - zone_center) / zone_half_height) ** 2
                        )
                    ),
                    "in_zone": int(
                        abs(px) <= ZONE_HALF_WIDTH_FT
                        and state.sz_bottom <= pz <= state.sz_top
                    ),
                }
            )
            weights.append(type_prob / n_locations)

    frame = pl.DataFrame(rows).join(
        pitcher_profiles,
        on=["pitcher_id", "pitch_type"],
        how="left",
    ).join(
        batter_priors,
        on="batter_id",
        how="left",
    )
    return frame.select(FEATURE_COLUMNS), np.asarray(weights)


class PitchOutcomePredictor:
    """Loads Stage A/B models + profile stores and predicts outcome odds."""

    def __init__(self, run_dir: str | Path, profiles_dir: str | Path | None = None):
        from catboost import CatBoostClassifier

        run_dir = Path(run_dir)
        self.stage_a = CatBoostClassifier()
        self.stage_a.load_model(str(run_dir / "stage_a.cbm"))
        self.stage_b = CatBoostClassifier()
        self.stage_b.load_model(str(run_dir / "stage_b.cbm"))
        self._check_features(run_dir)

        profiles_dir = Path(profiles_dir) if profiles_dir else run_dir.parent
        self.pitcher_profiles, self.batter_priors = load_profile_stores(profiles_dir)

    @staticmethod
    def _check_features(run_dir: Path) -> None:
        for stage in ("stage_a", "stage_b"):
            meta = json.loads((run_dir / f"{stage}_features.json").read_text())
            if meta["feature_columns"] != FEATURE_COLUMNS:
                raise ValueError(
                    f"{stage} was trained with different features than this "
                    "code builds; retrain or check out the matching revision"
                )

    def _predict_proba(self, model, features: pl.DataFrame) -> np.ndarray:
        pandas_frame = features.to_pandas()
        for column in CATEGORICAL_FEATURES:
            pandas_frame[column] = pandas_frame[column].fillna("unknown").astype(str)
        numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
        pandas_frame[numeric] = pandas_frame[numeric].astype("float64")
        return model.predict_proba(pandas_frame)

    def predict(
        self,
        state: OutcomeGameState,
        type_probabilities: dict[str, float],
        locations: list[tuple[float, float]],
    ) -> dict:
        """Marginal outcome odds for the upcoming pitch.

        Returns ``result`` (P over Stage A classes, sums to 1) and
        ``event_given_in_play`` (P over Stage B classes conditional on
        the ball being put in play).
        """
        features, weights = build_feature_frame(
            state,
            type_probabilities,
            locations,
            self.pitcher_profiles,
            self.batter_priors,
        )
        weights = weights / weights.sum()

        result_probs = self._predict_proba(self.stage_a, features)
        result_classes = [str(c) for c in np.asarray(self.stage_a.classes_)]
        result = {
            cls: float((weights * result_probs[:, i]).sum())
            for i, cls in enumerate(result_classes)
        }

        in_play_index = result_classes.index("in_play")
        in_play_row_weights = weights * result_probs[:, in_play_index]
        event_probs = self._predict_proba(self.stage_b, features)
        event_classes = [str(c) for c in np.asarray(self.stage_b.classes_)]
        denominator = in_play_row_weights.sum()
        if denominator > 0:
            event_given_in_play = {
                cls: float((in_play_row_weights * event_probs[:, i]).sum() / denominator)
                for i, cls in enumerate(event_classes)
            }
        else:
            event_given_in_play = {cls: 0.0 for cls in event_classes}

        return {
            "result": {cls: result.get(cls, 0.0) for cls in RESULT_CLASS_ORDER},
            "event_given_in_play": {
                cls: event_given_in_play.get(cls, 0.0) for cls in EVENT_CLASS_ORDER
            },
        }
