"""Empirical pitch-mix backoff for pitchers unknown to the trained model.

Pitchers outside the model's training vocabulary hit an untrained embedding
row, producing arbitrary yet overconfident pitch-type distributions. For those
pitchers the live predictor blends the model output with the pitcher's recent
empirical pitch mix; the empirical weight grows with sample size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mlb.ml.features import PITCH_TYPE_CODES, PITCH_TYPE_TO_IDX

if TYPE_CHECKING:
    from collections.abc import Mapping

# Add-alpha smoothing across the pitch-type codes inside the empirical mix.
EMPIRICAL_MIX_SMOOTHING = 1.0
# Log-space prior-correction exponent and confidence temperature. Chosen on
# the 2025 unknown-pitcher segment (123,830 pitches, leak-free as-of mixes):
# vs the raw model this lifts top-1 0.568->0.642, top-3 0.852->0.902, and
# halves log-loss 2.213->1.058, dominating linear blends on all metrics.
EMPIRICAL_MIX_GAMMA = 1.0
EMPIRICAL_MIX_TEMPERATURE = 2.0

# League pitch mix, canonical codes, mlb.pitches seasons 2021-2025.
_LEAGUE_COUNTS = {
    "FF": 1_287_629,
    "SI": 604_086,
    "FC": 298_758,
    "CH": 418_301,
    "SL": 607_339,
    "CU": 264_927,
    "KC": 77_325,
    "ST": 219_052,
    "FS": 91_757,
    "KN": 1_591,
    "OTHER": 140_929,
}
LEAGUE_PITCH_MIX = np.array(
    [_LEAGUE_COUNTS[code] for code in PITCH_TYPE_CODES], dtype=np.float64
)
LEAGUE_PITCH_MIX /= LEAGUE_PITCH_MIX.sum()


def pitch_mix_counts_from_postgres(pitcher_id: int) -> dict[str, int]:
    """Count the pitcher's pitches by type over the current and prior season."""
    from mlb.database import PostgresHandler

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
    gamma: float = EMPIRICAL_MIX_GAMMA,
    temperature: float = EMPIRICAL_MIX_TEMPERATURE,
    smoothing: float = EMPIRICAL_MIX_SMOOTHING,
) -> np.ndarray:
    """Correct model probabilities with the pitcher's empirical mix.

    Log-space prior correction: ``p ~ model * (mix / league)**gamma``,
    flattened by ``temperature``. This preserves the model's
    count-conditional ordering (unlike a linear blend, whose argmax
    collapses to the pitcher's modal pitch) while injecting the
    pitcher-specific arsenal and de-hallucinating confidence. With no
    observed pitches the model distribution is returned unchanged.
    """
    model = np.asarray(model_probs, dtype=np.float64)
    model = model / max(float(model.sum()), 1e-9)
    n = float(counts_vector.sum())
    if n <= 0:
        return model
    empirical = (counts_vector + smoothing) / (n + smoothing * len(counts_vector))
    blended = model * (empirical / LEAGUE_PITCH_MIX) ** gamma
    blended = blended ** (1.0 / temperature)
    return blended / blended.sum()
