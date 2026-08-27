"""American/decimal/implied-probability conversions and de-vigging.

All functions are pure and side-effect free so the conversions and the vig
removal can be unit-tested against hand-computed values. "American" odds use
the sportsbook convention: +150 pays 150 on 100 staked; -150 risks 150 to win
100. "Decimal" odds are the total return per unit staked (stake included), so
+150 == 2.5 and -150 == 1.6667.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "american_to_decimal",
    "american_to_prob",
    "decimal_to_american",
    "decimal_to_prob",
    "devig_proportional",
    "devig_proportional_many",
    "no_vig_outright",
    "no_vig_two_way",
    "prob_to_american",
    "prob_to_decimal",
    "two_way_overround",
]


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (total return per unit staked)."""
    if american == 0:
        raise ValueError("American odds cannot be zero")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds (>1) to American odds."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def decimal_to_prob(decimal: float) -> float:
    """Implied probability (with vig) from decimal odds."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    return 1.0 / decimal


def prob_to_decimal(prob: float) -> float:
    """Fair decimal odds for a probability in (0, 1)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return 1.0 / prob


def american_to_prob(american: float) -> float:
    """Implied probability (with vig) from American odds."""
    return decimal_to_prob(american_to_decimal(american))


def prob_to_american(prob: float) -> float:
    """Fair American odds for a probability in (0, 1)."""
    return decimal_to_american(prob_to_decimal(prob))


def two_way_overround(prob_a: float, prob_b: float) -> float:
    """Book overround (vig) for a two-way market: sum of implied probs minus 1."""
    return prob_a + prob_b - 1.0


def devig_proportional(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalize two implied probabilities so they sum to one (multiplicative).

    The standard, assumption-light de-vig: scale both sides by the same factor.
    It preserves the odds ratio between the two outcomes.
    """
    total = prob_a + prob_b
    if total <= 0.0:
        raise ValueError("Implied probabilities must be positive")
    return prob_a / total, prob_b / total


def devig_proportional_many(
    probabilities: Sequence[float], *, target_total: float = 1.0
) -> tuple[float, ...]:
    """Normalize multi-runner implied probabilities to ``target_total``.

    This is the outright/futures analogue of two-way proportional de-vigging:
    every runner is scaled by the same factor, preserving relative prices while
    removing the book overround. ``target_total`` is normally 1 for a single
    winner outright, but can be higher for stage markets with multiple winners.
    """
    if len(probabilities) < 2:
        raise ValueError("At least two runners are required")
    if target_total <= 0.0:
        raise ValueError("Target total must be positive")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("Implied probabilities must be positive")
    if any(prob <= 0.0 or not math.isfinite(prob) for prob in probabilities):
        raise ValueError("Implied probabilities must be finite and positive")
    scale = target_total / total
    return tuple(prob * scale for prob in probabilities)


def no_vig_outright(
    american_odds: Sequence[float], *, target_total: float = 1.0
) -> tuple[float, ...]:
    """Fair probabilities for a multi-runner American-odds futures market."""
    return devig_proportional_many(
        tuple(american_to_prob(american) for american in american_odds),
        target_total=target_total,
    )


def _shin_z(prob_a: float, prob_b: float) -> float:
    """Solve Shin's insider-trading fraction z for a two-way market.

    Shin (1993) models the observed prices as containing a fraction ``z`` of
    insider money; removing it shrinks favorites less than proportional
    de-vigging and corrects some favorite-longshot bias. Closed form for the
    two-outcome case.
    """
    booksum = prob_a + prob_b
    # Normalized implied probs (sum to 1) feed the closed-form solution.
    qa, qb = prob_a / booksum, prob_b / booksum
    inner = booksum * booksum - 4.0 * booksum * (booksum - 1.0) * (qa * qa + qb * qb)
    # Guard tiny negatives from floating point when the book is near-fair.
    inner = max(inner, 0.0)
    denom = booksum - (qa * qa + qb * qb) * booksum
    if denom == 0.0:
        return 0.0
    z = (booksum - math.sqrt(inner)) / denom
    return min(max(z, 0.0), 0.999)


def _shin_prob(q_norm: float, z: float) -> float:
    return (math.sqrt(z * z + 4.0 * (1.0 - z) * q_norm * q_norm) - z) / (2.0 * (1.0 - z))


def no_vig_two_way(
    american_a: float, american_b: float, method: str = "proportional"
) -> tuple[float, float]:
    """Fair (no-vig) probabilities for the two sides of a market.

    ``method`` is ``"proportional"`` (normalize implied probs) or ``"shin"``
    (Shin's insider-trading correction). Both return probabilities summing to 1.
    """
    prob_a = american_to_prob(american_a)
    prob_b = american_to_prob(american_b)
    if method == "proportional":
        return devig_proportional(prob_a, prob_b)
    if method == "shin":
        booksum = prob_a + prob_b
        qa, qb = prob_a / booksum, prob_b / booksum
        z = _shin_z(prob_a, prob_b)
        fair_a = _shin_prob(qa, z)
        fair_b = _shin_prob(qb, z)
        total = fair_a + fair_b
        return fair_a / total, fair_b / total
    raise ValueError(f"Unknown de-vig method {method!r}")
