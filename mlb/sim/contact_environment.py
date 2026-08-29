"""Per-game contact environment -> PA-outcome multipliers.

The sim otherwise runs every game at a league-neutral run environment, which
misses the biggest totals driver (ballpark) plus weather. This maps a game's
venue and weather into multipliers on the hit/HR PA-outcome classes, applied on
top of the league PA-outcome calibration (see ``MatchupProviderFactory``).

Park factors are leak-free (each season's factors are computed from strictly
prior seasons; see ``scripts/compute_park_factors.py``). Umpire strike-zone
effects (strikeout/walk) are supported by the same multiplier mechanism but need
home-plate-umpire data not yet ingested, so they stay neutral for now.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PARK_FACTORS_PATH = Path("models/sim/park_factors.json")

# PA-outcome classes a ballpark reshapes.
PARK_CLASSES = ("home_run", "single", "double", "triple")

# Weather HR sensitivity (sabermetric consensus magnitudes [INFERENCE]).
_TEMP_BASELINE_F = 70.0
_TEMP_HR_PER_DEG = 0.012  # ~1.2% more HR per degF above baseline
_WIND_HR_PER_MPH = 0.010  # ~1% per mph blowing out to the field (in = negative)
_HR_FACTOR_CLAMP = (0.80, 1.25)
_ROOF_TERMS = ("dome", "roof closed", "closed roof")


@dataclass(frozen=True)
class GameWeather:
    temp_f: float | None = None
    wind_mph: float | None = None
    wind_dir: str | None = None  # "out" | "in" | "cross" | None
    condition: str | None = None
    indoor: bool = False


def _num(value: object) -> float | None:
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def parse_weather(
    temp: object, wind: object, condition: object
) -> GameWeather:
    """Parse the raw ``mlb.games`` weather strings into a typed record.

    temp e.g. ``"71"``; wind e.g. ``"8 mph, Out To RF"``; condition e.g.
    ``"Overcast"`` / ``"Dome"`` / ``"Roof Closed"``.
    """
    cond = str(condition).strip().lower() if condition else None
    indoor = bool(cond and any(term in cond for term in _ROOF_TERMS))
    wind_s = str(wind).lower() if wind else ""
    wind_dir: str | None = None
    if "out" in wind_s:
        wind_dir = "out"
    elif "in" in wind_s:
        wind_dir = "in"
    elif wind_s:
        wind_dir = "cross"
    return GameWeather(
        temp_f=_num(temp),
        wind_mph=_num(wind_s),
        wind_dir=wind_dir,
        condition=cond,
        indoor=indoor,
    )


def weather_hr_factor(weather: GameWeather | None) -> float:
    """Home-run multiplier from temperature and wind (1.0 = neutral)."""
    if weather is None or weather.indoor:
        return 1.0
    factor = 1.0
    if weather.temp_f is not None:
        factor *= 1.0 + _TEMP_HR_PER_DEG * (weather.temp_f - _TEMP_BASELINE_F)
    if weather.wind_mph is not None and weather.wind_dir in ("out", "in"):
        sign = 1.0 if weather.wind_dir == "out" else -1.0
        factor *= 1.0 + sign * _WIND_HR_PER_MPH * weather.wind_mph
    lo, hi = _HR_FACTOR_CLAMP
    return max(lo, min(hi, factor))


class ContactEnvironment:
    """Maps ``venue_id`` + weather to PA-outcome multipliers for one season."""

    def __init__(self, park_factors: dict[str, dict[str, float]]):
        self._park = park_factors

    @classmethod
    def load(
        cls, season: int, path: Path = DEFAULT_PARK_FACTORS_PATH
    ) -> ContactEnvironment | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        factors = data.get("factors", {}).get(str(season))
        if not factors:
            return None
        return cls(factors)

    def multipliers(
        self, venue_id: int | None, weather: GameWeather | None
    ) -> dict[str, float]:
        """Combined park x weather multipliers; only non-neutral classes."""
        mult: dict[str, float] = {}
        park = self._park.get(str(venue_id)) if venue_id is not None else None
        if park:
            for cls_ in PARK_CLASSES:
                if cls_ in park and float(park[cls_]) != 1.0:
                    mult[cls_] = float(park[cls_])
        wx = weather_hr_factor(weather)
        if wx != 1.0:
            mult["home_run"] = mult.get("home_run", 1.0) * wx
        return mult
