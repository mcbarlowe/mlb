"""Count-conditioned pitch mix and location profiles for simulation.

The pitch-type LSTM needs real per-pitch sequence context, which simulated
plate appearances do not have. For bulk game simulation we instead condition
the outcome models on empirical per-pitcher inputs at each count:

- P(pitch type | pitcher, count) with league shrinkage
- plate locations sampled from that pitcher's actual pitches at that count,
  topped up from a league pool when the pitcher's sample is thin

Built from PostgreSQL pitch rows by ``scripts/export_pitch_mix.py``.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl

from src.outcome.labels import CANONICAL_PITCH_TYPES

MIX_TABLE_PATH = Path("models/sim/pitch_mix.parquet")
LOCATION_TABLE_PATH = Path("models/sim/pitch_locations.parquet")

_NULL_TYPE_CODES = ["", "None", "UN", "IN", "PO", "AB"]

COUNTS: list[tuple[int, int]] = [(b, s) for b in range(4) for s in range(3)]


def _canonical_type_expr() -> pl.Expr:
    code = pl.col("pitch_type_code")
    return (
        pl.when(code.is_null() | code.is_in(_NULL_TYPE_CODES))
        .then(pl.lit(None, dtype=pl.String))
        .when(code.is_in(CANONICAL_PITCH_TYPES))
        .then(code)
        .otherwise(pl.lit("OTHER"))
    )


def build_pitch_mix_tables(
    raw: pl.DataFrame,
    max_locations_per_count: int = 80,
    seed: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate raw pitch rows into mix counts and location samples.

    ``raw`` needs: ``pitcher_id``, ``game_pk``, ``at_bat_index``,
    ``pitch_number``, ``count_after_pitch`` (post-pitch), ``pitch_type_code``,
    ``px``, ``pz``.
    """
    ab_key = ["game_pk", "at_bat_index"]
    counts = pl.col("count_after_pitch").str.split_exact("-", 1)
    frame = (
        raw.sort(ab_key + ["pitch_number"])
        .with_columns(
            counts.struct.field("field_0").cast(pl.Int64, strict=False).alias("balls_after"),
            counts.struct.field("field_1").cast(pl.Int64, strict=False).alias("strikes_after"),
        )
        .with_columns(
            pl.col("balls_after").shift(1).over(ab_key).fill_null(0).alias("balls"),
            pl.col("strikes_after").shift(1).over(ab_key).fill_null(0).alias("strikes"),
            _canonical_type_expr().alias("pitch_type"),
        )
        .filter(
            pl.col("pitch_type").is_not_null()
            & pl.col("balls").is_between(0, 3)
            & pl.col("strikes").is_between(0, 2)
            & pl.col("px").is_finite()
            & pl.col("pz").is_finite()
        )
    )

    mix = (
        frame.group_by(["pitcher_id", "balls", "strikes", "pitch_type"])
        .agg(pl.len().alias("n"))
        .sort(["pitcher_id", "balls", "strikes", "pitch_type"])
    )

    location_group = ["pitcher_id", "balls", "strikes", "pitch_type"]
    locations = (
        frame.filter(
            pl.int_range(pl.len()).shuffle(seed=seed).over(location_group)
            < max_locations_per_count
        )
        .select(location_group + ["px", "pz"])
        .sort(location_group)
    )
    return mix, locations


class PitchMixProfiles:
    """Blended per-pitcher, per-count type distributions and location pools."""

    def __init__(
        self,
        mix: pl.DataFrame,
        locations: pl.DataFrame,
        shrinkage: float = 60.0,
        league_pool_size: int = 400,
        seed: int | None = None,
    ):
        self._shrinkage = shrinkage
        rng = random.Random(seed)

        self._pitcher_mix: dict[tuple[int, int, int], dict[str, int]] = {}
        for key, group in mix.group_by(["pitcher_id", "balls", "strikes"]):
            pid, balls, strikes = (int(str(part)) for part in key)
            self._pitcher_mix[(pid, balls, strikes)] = dict(
                zip(group["pitch_type"].to_list(), group["n"].to_list())
            )

        league = mix.group_by(["balls", "strikes", "pitch_type"]).agg(pl.col("n").sum())
        self._league_mix: dict[tuple[int, int], dict[str, float]] = {}
        for key, group in league.group_by(["balls", "strikes"]):
            balls, strikes = (int(str(part)) for part in key)
            total = float(group["n"].sum())
            self._league_mix[(balls, strikes)] = {
                t: n / total for t, n in zip(group["pitch_type"].to_list(), group["n"].to_list())
            }

        self._pitcher_locations: dict[
            tuple[int, int, int, str], list[tuple[float, float]]
        ] = {}
        for key, group in locations.group_by(
            ["pitcher_id", "balls", "strikes", "pitch_type"]
        ):
            pid, balls, strikes = (int(str(part)) for part in key[:3])
            pitch_type = str(key[3])
            self._pitcher_locations[(pid, balls, strikes, pitch_type)] = list(
                zip(group["px"].to_list(), group["pz"].to_list())
            )

        self._league_locations: dict[tuple[int, int, str], list[tuple[float, float]]] = {}
        for key, group in locations.group_by(["balls", "strikes", "pitch_type"]):
            balls, strikes = int(str(key[0])), int(str(key[1]))
            pitch_type = str(key[2])
            pool = list(zip(group["px"].to_list(), group["pz"].to_list()))
            if len(pool) > league_pool_size:
                pool = rng.sample(pool, league_pool_size)
            self._league_locations[(balls, strikes, pitch_type)] = pool

        self._league_any_type: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for key, group in locations.group_by(["balls", "strikes"]):
            balls, strikes = (int(str(part)) for part in key)
            pool = list(zip(group["px"].to_list(), group["pz"].to_list()))
            if len(pool) > league_pool_size:
                pool = rng.sample(pool, league_pool_size)
            self._league_any_type[(balls, strikes)] = pool

    @classmethod
    def load(
        cls,
        mix_path: Path = MIX_TABLE_PATH,
        location_path: Path = LOCATION_TABLE_PATH,
        **kwargs,
    ) -> PitchMixProfiles:
        return cls(pl.read_parquet(mix_path), pl.read_parquet(location_path), **kwargs)

    def type_distribution(self, pitcher_id: int, balls: int, strikes: int) -> dict[str, float]:
        """League-shrunk pitch type distribution at a count."""
        league = self._league_mix.get((balls, strikes))
        if league is None:
            raise KeyError(f"No league mix for count {balls}-{strikes}")
        own = self._pitcher_mix.get((pitcher_id, balls, strikes))
        if not own:
            return dict(league)
        total = sum(own.values())
        k = self._shrinkage
        blended = {
            t: (own.get(t, 0) + k * league.get(t, 0.0)) / (total + k)
            for t in set(own) | set(league)
        }
        norm = sum(blended.values())
        return {t: p / norm for t, p in blended.items()}

    def sample_locations(
        self,
        pitcher_id: int,
        balls: int,
        strikes: int,
        pitch_type: str,
        n: int,
        rng: random.Random,
    ) -> list[tuple[float, float]]:
        """``n`` locations for one pitch type at one count.

        Preference order: the pitcher's own pitches of that type at that
        count, league pitches of that type at that count, then any league
        pitch at that count.
        """
        own = self._pitcher_locations.get((pitcher_id, balls, strikes, pitch_type), [])
        picked = list(own) if len(own) <= n else rng.sample(own, n)
        for pool in (
            self._league_locations.get((balls, strikes, pitch_type), []),
            self._league_any_type.get((balls, strikes), []),
        ):
            while len(picked) < n and pool:
                picked.append(rng.choice(pool))
        if not picked:
            raise KeyError(
                f"No locations available for count {balls}-{strikes} {pitch_type}"
            )
        return picked[:n]

    def inputs_by_count(
        self,
        pitcher_id: int,
        n_locations: int = 12,
        rng: random.Random | None = None,
    ) -> dict[
        tuple[int, int],
        tuple[dict[str, float], dict[str, list[tuple[float, float]]]],
    ]:
        """Provider inputs for every count: type dist + per-type locations."""
        rng = rng or random.Random(0)
        inputs = {}
        for balls, strikes in COUNTS:
            types = self.type_distribution(pitcher_id, balls, strikes)
            locations_by_type = {
                pitch_type: self.sample_locations(
                    pitcher_id, balls, strikes, pitch_type, n_locations, rng
                )
                for pitch_type in types
            }
            inputs[(balls, strikes)] = (types, locations_by_type)
        return inputs
