"""Count/at-bat state machine for plate-appearance simulation.

Maps Stage A per-pitch results onto count transitions and terminal
plate-appearance outcomes. Pure functions, no model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Terminal plate-appearance outcomes. The first three come from the count
# machine itself; the rest are Stage B in-play events.
PA_OUTCOMES = [
    "walk",
    "strikeout",
    "hit_by_pitch",
    "out",
    "single",
    "double",
    "triple",
    "home_run",
    "reached_on_error",
]


@dataclass(frozen=True)
class CountTransition:
    """Result of applying one pitch result to a count."""

    balls: int
    strikes: int
    terminal: str | None  # None = at-bat continues; else a PA_OUTCOMES entry
    in_play: bool = False  # True when the PA resolves via Stage B


def apply_pitch_result(balls: int, strikes: int, result: str) -> CountTransition:
    """Apply a Stage A pitch result to the current count.

    Rules:
    - ball: 4th ball is a walk
    - called/swinging strike: 3rd strike is a strikeout
    - foul: adds a strike but never the third (2 strikes stays 2)
    - hit_by_pitch: immediate terminal
    - in_play: terminal; the event comes from Stage B
    """
    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
        raise ValueError(f"Invalid count {balls}-{strikes}")

    if result == "ball":
        if balls == 3:
            return CountTransition(balls, strikes, "walk")
        return CountTransition(balls + 1, strikes, None)
    if result in ("called_strike", "swinging_strike"):
        if strikes == 2:
            return CountTransition(balls, strikes, "strikeout")
        return CountTransition(balls, strikes + 1, None)
    if result == "foul":
        return CountTransition(balls, min(strikes + 1, 2), None)
    if result == "hit_by_pitch":
        return CountTransition(balls, strikes, "hit_by_pitch")
    if result == "in_play":
        return CountTransition(balls, strikes, "in_play", in_play=True)
    raise ValueError(f"Unknown pitch result {result!r}")
