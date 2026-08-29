"""Label mappings for the pitch outcome models.

This package is deliberately independent of the pitch prediction models in
``src/ml``: the only shared vocabulary is pitch-type code strings and plate
coordinates, so the upstream type/location models can change freely.
"""

from __future__ import annotations

# Stage A: per-pitch result classes.
PITCH_RESULT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
]

# Mapping from GUMBO ``pitch_call_description`` to a Stage A class.
PITCH_CALL_TO_RESULT: dict[str, str] = {
    "Ball": "ball",
    "Ball In Dirt": "ball",
    "Automatic Ball": "ball",
    "Automatic Ball - Intentional": "ball",
    "Automatic Ball - Pitcher Pitch Timer Violation": "ball",
    "Called Strike": "called_strike",
    "Automatic Strike": "called_strike",
    "Automatic Strike - Batter Pitch Timer Violation": "called_strike",
    "Automatic Strike - Batter Timeout Violation": "called_strike",
    "Swinging Strike": "swinging_strike",
    "Swinging Strike (Blocked)": "swinging_strike",
    "Missed Bunt": "swinging_strike",
    "Foul Tip": "swinging_strike",
    "Foul": "foul",
    "Foul Bunt": "foul",
    "Foul Pitchout": "foul",
    "In play, out(s)": "in_play",
    "In play, no out": "in_play",
    "In play, run(s)": "in_play",
    "Hit By Pitch": "hit_by_pitch",
}

# Non-pitch events and rows we exclude from training entirely. Intentional
# balls are a managerial decision handled by the simulator, not a model output.
EXCLUDED_PITCH_CALLS = {
    "Pickoff Attempt 1B",
    "Pickoff Attempt 2B",
    "Pickoff Attempt 3B",
    "Pickoff Error 1B",
    "Pickoff Error 2B",
    "Pickoff Error 3B",
    "Pitcher Step Off",
    "Pitchout",
    "Intent Ball",
    "None",
}

# Stage B: terminal event classes for balls in play.
IN_PLAY_EVENT_CLASSES = [
    "out",
    "single",
    "double",
    "triple",
    "home_run",
    "reached_on_error",
]

EVENT_TYPE_TO_CLASS: dict[str, str] = {
    "field_out": "out",
    "force_out": "out",
    "grounded_into_double_play": "out",
    "double_play": "out",
    "triple_play": "out",
    "fielders_choice": "out",
    "fielders_choice_out": "out",
    "sac_fly": "out",
    "sac_fly_double_play": "out",
    "sac_bunt": "out",
    "sac_bunt_double_play": "out",
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "field_error": "reached_on_error",
    "catcher_interf": "reached_on_error",
}

# Canonical pitch-type vocabulary. Codes outside this set map to OTHER; rows
# with no code at all are dropped (they cannot be conditioned on).
CANONICAL_PITCH_TYPES = [
    "FF",
    "SI",
    "SL",
    "CH",
    "CU",
    "FC",
    "ST",
    "KC",
    "FS",
    "KN",
    "OTHER",
]

_MISSING_TYPE_CODES = {"", "None", "UN", "IN", "PO", "AB"}


def map_pitch_call(description: str | None) -> str | None:
    """Stage A label for a pitch call, or ``None`` when the row is excluded."""
    if description is None or description in EXCLUDED_PITCH_CALLS:
        return None
    return PITCH_CALL_TO_RESULT.get(description)


def map_event_type(event_type: str | None) -> str | None:
    """Stage B label for an in-play terminal event, or ``None`` to drop."""
    if event_type is None:
        return None
    return EVENT_TYPE_TO_CLASS.get(event_type)


def canonicalize_pitch_type(code: str | None) -> str | None:
    """Canonical pitch-type code, ``OTHER`` for rare types, ``None`` to drop."""
    if code is None or code in _MISSING_TYPE_CODES:
        return None
    if code in CANONICAL_PITCH_TYPES:
        return code
    return "OTHER"
