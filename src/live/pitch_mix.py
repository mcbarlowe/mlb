"""Empirical pitch-mix backoff for pitchers unknown to the trained model.

Pitchers outside the model's training vocabulary hit an untrained embedding
row, producing arbitrary yet overconfident pitch-type distributions. For those
pitchers the live predictor blends the model output with the pitcher's recent
empirical pitch mix; the empirical weight grows with sample size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ml.features import PITCH_TYPE_CODES, PITCH_TYPE_TO_IDX

if TYPE_CHECKING:
    from collections.abc import Mapping

# Blend weight for the empirical mix is n / (n + EMPIRICAL_MIX_PSEUDOCOUNT).
EMPIRICAL_MIX_PSEUDOCOUNT = 50.0
# Add-alpha smoothing across the pitch-type codes inside the empirical mix.
EMPIRICAL_MIX_SMOOTHING = 1.0


def pitch_mix_counts_from_postgres(pitcher_id: int) -> dict[str, int]:
    """Count the pitcher's pitches by type over the current and prior season."""
    from src.database import PostgresHandler

    query = f"""
        SELECT pitch_type_code, COUNT(*) AS n
        FROM mlb.pitches
        WHERE pitcher_id = {int(pitcher_id)}
          AND season >= EXTRACT(YEAR FROM CURRENT_DATE) - 1
          AND pitch_type_code IS NOT NULL
        GROUP BY pitch_type_code
    """
    with PostgresHandler() as db:
        frame = db.query(query)
    return {
        str(code): int(count)
        for code, count in zip(frame["pitch_type_code"], frame["n"])
    }


def counts_to_vector(counts: Mapping[str, int]) -> np.ndarray:
    """Map raw pitch-type counts onto the model's code ordering.

    Codes outside the canonical list fold into ``OTHER``.
    """
    vector = np.zeros(len(PITCH_TYPE_CODES), dtype=np.float64)
    other_idx = PITCH_TYPE_TO_IDX["OTHER"]
    for code, count in counts.items():
        if count and count > 0:
            vector[PITCH_TYPE_TO_IDX.get(code, other_idx)] += int(count)
    return vector


def blend_with_empirical_mix(
    model_probs: np.ndarray,
    counts_vector: np.ndarray,
    *,
    pseudocount: float = EMPIRICAL_MIX_PSEUDOCOUNT,
    smoothing: float = EMPIRICAL_MIX_SMOOTHING,
) -> np.ndarray:
    """Blend model probabilities with a smoothed empirical mix.

    With no observed pitches the model distribution is returned unchanged;
    as observations grow the blend converges to the empirical mix.
    """
    model = np.asarray(model_probs, dtype=np.float64)
    model = model / max(float(model.sum()), 1e-9)
    n = float(counts_vector.sum())
    if n <= 0:
        return model
    empirical = (counts_vector + smoothing) / (n + smoothing * len(counts_vector))
    weight = n / (n + pseudocount)
    blended = weight * empirical + (1.0 - weight) * model
    return blended / blended.sum()
