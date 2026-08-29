"""Count- and stretch-conditioned pitch mix and location profiles.

The pitch-type LSTM needs real per-pitch sequence context, which simulated
plate appearances do not have. For bulk game simulation we instead condition
the outcome models on empirical per-pitcher inputs at each count:

- P(pitch type | pitcher, count, stretch) with league shrinkage
- plate locations sampled from that pitcher's actual pitches at that count,
  topped up from league pools when the pitcher's sample is thin

``stretch`` distinguishes pitching from the windup (bases empty) vs the
stretch (any runner on) — pitch selection differs measurably, and the
repaired base/out state makes the split trustworthy. Fallback chains walk
stretch-specific pools first, then stretch-blind, then league.

Built from PostgreSQL pitch rows by ``scripts/export_pitch_mix.py``.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl

from mlb.outcome.labels import CANONICAL_PITCH_TYPES

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
    ``pitch_number``, ``count_after_pitch`` (post-pitch),
    ``pitch_type_code``, ``px``, ``pz``, and the ``is_runner_on_*`` flags
    (true at-bat start state).
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
            (
                pl.col("is_runner_on_first")
                | pl.col("is_runner_on_second")
                | pl.col("is_runner_on_third")
            )
            .fill_null(False)
            .alias("stretch"),
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
        frame.group_by(["pitcher_id", "balls", "strikes", "stretch", "pitch_type"])
        .agg(pl.len().alias("n"))
        .sort(["pitcher_id", "balls", "strikes", "stretch", "pitch_type"])
    )

    location_group = ["pitcher_id", "balls", "strikes", "stretch", "pitch_type"]
    locations = (
        frame.filter(
            pl.int_range(pl.len()).shuffle(seed=seed).over(location_group)
            < max_locations_per_count
        )
        .select(location_group + ["px", "pz"])
        # Sorted on the coordinates as well as the group keys. Sorting by group key alone leaves
        # within-group row order unpinned, and these rows become the location pools that are
        # sampled from, so an unpinned order yields different draws for the same seed.
        .sort(location_group + ["px", "pz"])
    )
    return mix, locations


class PitchMixProfiles:
    """Blended per-pitcher type distributions and location pools."""

    def __init__(
        self,
        mix: pl.DataFrame,
        locations: pl.DataFrame,
        shrinkage: float = 60.0,
        league_pool_size: int = 400,
        seed: int | None = None,
    ):
        self._shrinkage = shrinkage

        if "stretch" not in mix.columns:
            raise ValueError(
                "pitch mix tables lack the stretch column; re-run "
                "scripts/export_pitch_mix.py"
            )

        # (pid, balls, strikes, stretch) -> type -> n
        self._pitcher_mix: dict[tuple[int, int, int, bool], dict[str, int]] = {}
        for key, group in mix.group_by(["pitcher_id", "balls", "strikes", "stretch"]):
            pid, balls, strikes = (int(str(part)) for part in key[:3])
            stretch = bool(key[3])
            self._pitcher_mix[(pid, balls, strikes, stretch)] = dict(
                zip(group["pitch_type"].to_list(), group["n"].to_list())
            )

        league = mix.group_by(["balls", "strikes", "stretch", "pitch_type"]).agg(
            pl.col("n").sum()
        )
        self._league_mix: dict[tuple[int, int, bool], dict[str, float]] = {}
        for key, group in league.group_by(["balls", "strikes", "stretch"]):
            balls, strikes = int(str(key[0])), int(str(key[1]))
            stretch = bool(key[2])
            total = float(group["n"].sum())
            # Sorted by pitch type so the dict's insertion order does not depend on the
            # group_by output order, which polars does not guarantee. Downstream code sums these
            # values, and an order-dependent sum differs by 1 ULP between processes.
            self._league_mix[(balls, strikes, stretch)] = {
                t: n / total
                for t, n in sorted(
                    zip(group["pitch_type"].to_list(), group["n"].to_list()),
                    key=lambda pair: pair[0],
                )
            }

        self._pitcher_locations: dict[
            tuple[int, int, int, bool, str], list[tuple[float, float]]
        ] = {}
        for key, group in locations.group_by(
            ["pitcher_id", "balls", "strikes", "stretch", "pitch_type"]
        ):
            pid, balls, strikes = (int(str(part)) for part in key[:3])
            stretch = bool(key[3])
            pitch_type = str(key[4])
            self._pitcher_locations[(pid, balls, strikes, stretch, pitch_type)] = list(
                zip(group["px"].to_list(), group["pz"].to_list())
            )

        self._league_locations: dict[
            tuple[int, int, bool, str], list[tuple[float, float]]
        ] = {}
        # Subsampling draws from a stream seeded by the group key rather than from the shared
        # ``rng``. A shared stream is consumed in polars' group_by iteration order, which is not
        # guaranteed, so the pools differed between processes and rare pitch types drew different
        # locations for the same seed.
        for key, group in locations.group_by(["balls", "strikes", "stretch", "pitch_type"]):
            balls, strikes = int(str(key[0])), int(str(key[1]))
            stretch = bool(key[2])
            pitch_type = str(key[3])
            pool = list(zip(group["px"].to_list(), group["pz"].to_list()))
            if len(pool) > league_pool_size:
                pool = random.Random(
                    f"{seed}|league|{balls}|{strikes}|{stretch}|{pitch_type}"
                ).sample(pool, league_pool_size)
            self._league_locations[(balls, strikes, stretch, pitch_type)] = pool

        self._league_any_type: dict[tuple[int, int, bool], list[tuple[float, float]]] = {}
        for key, group in locations.group_by(["balls", "strikes", "stretch"]):
            balls, strikes = int(str(key[0])), int(str(key[1]))
            stretch = bool(key[2])
            pool = list(zip(group["px"].to_list(), group["pz"].to_list()))
            if len(pool) > league_pool_size:
                pool = random.Random(
                    f"{seed}|any|{balls}|{strikes}|{stretch}"
                ).sample(pool, league_pool_size)
            self._league_any_type[(balls, strikes, stretch)] = pool

    @classmethod
    def load(
        cls,
        mix_path: Path = MIX_TABLE_PATH,
        location_path: Path = LOCATION_TABLE_PATH,
        **kwargs,
    ) -> PitchMixProfiles:
        return cls(pl.read_parquet(mix_path), pl.read_parquet(location_path), **kwargs)

    def type_distribution(
        self, pitcher_id: int, balls: int, strikes: int, stretch: bool = False
    ) -> dict[str, float]:
        """League-shrunk pitch type distribution at a count/stretch state."""
        league = self._league_mix.get((balls, strikes, stretch)) or self._league_mix.get(
            (balls, strikes, not stretch)
        )
        if league is None:
            raise KeyError(f"No league mix for count {balls}-{strikes}")
        own = self._pitcher_mix.get((pitcher_id, balls, strikes, stretch))
        if not own:
            # Stretch-specific sample missing: fall back to the pitcher's
            # stretch-blind usage at this count before going pure league.
            other = self._pitcher_mix.get((pitcher_id, balls, strikes, not stretch))
            own = other or {}
        if not own:
            return dict(league)
        # Keys are sorted before the dict is built and before any float is summed. Iterating a
        # set of strings takes an order that varies with hash randomisation, and polars group_by
        # output order is not guaranteed either, so accumulating floats in that order produced
        # 1-ULP differences between processes. Those differences flip comparisons in the weighted
        # location draw, which cascaded into entirely different simulated games: two runs with an
        # identical seed diverged. Sorting makes the arithmetic order-independent.
        total = sum(own[t] for t in sorted(own))
        k = self._shrinkage
        blended = {
            t: (own.get(t, 0) + k * league.get(t, 0.0)) / (total + k)
            for t in sorted(set(own) | set(league))
        }
        norm = sum(blended[t] for t in sorted(blended))
        return {t: blended[t] / norm for t in sorted(blended)}

    def sample_locations(
        self,
        pitcher_id: int,
        balls: int,
        strikes: int,
        pitch_type: str,
        n: int,
        rng: random.Random,
        stretch: bool = False,
    ) -> list[tuple[float, float]]:
        """``n`` locations for one pitch type at one count/stretch state."""
        own = self._pitcher_locations.get(
            (pitcher_id, balls, strikes, stretch, pitch_type), []
        )
        if len(own) < n:
            own = own + self._pitcher_locations.get(
                (pitcher_id, balls, strikes, not stretch, pitch_type), []
            )
        picked = list(own) if len(own) <= n else rng.sample(own, n)
        for pool in (
            self._league_locations.get((balls, strikes, stretch, pitch_type), []),
            self._league_locations.get((balls, strikes, not stretch, pitch_type), []),
            self._league_any_type.get((balls, strikes, stretch), []),
            self._league_any_type.get((balls, strikes, not stretch), []),
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
        stretch: bool = False,
    ) -> dict[
        tuple[int, int],
        tuple[dict[str, float], dict[str, list[tuple[float, float]]]],
    ]:
        """Provider inputs for every count, for one pitcher/stretch state."""
        rng = rng or random.Random(0)
        inputs = {}
        for balls, strikes in COUNTS:
            types = self.type_distribution(pitcher_id, balls, strikes, stretch)
            locations_by_type = {
                pitch_type: self.sample_locations(
                    pitcher_id, balls, strikes, pitch_type, n_locations, rng, stretch
                )
                for pitch_type in types
            }
            inputs[(balls, strikes)] = (types, locations_by_type)
        return inputs


class PitchModelCountProfiles:
    """Pitch-model type distributions with empirical location backoff.

    This is an opt-in simulation provider. It keeps the fast existing
    precomputed outcome path by asking the LSTM for one neutral, first-pitch
    sequence per count, then reuses ``PitchMixProfiles`` for location samples.
    """

    def __init__(
        self,
        predictor,
        fallback: PitchMixProfiles,
        season: int,
        *,
        model_weight: float = 1.0,
    ):
        if not (0.0 <= model_weight <= 1.0):
            raise ValueError("model_weight must be between 0 and 1")
        if predictor.feature_engine is None:
            raise ValueError("pitch predictor has no feature engine")
        if predictor.lstm_model is None:
            raise ValueError("pitch predictor must be an LSTM pitch-type model")
        self._predictor = predictor
        self._fallback = fallback
        self._season = season
        self._model_weight = model_weight
        self._feature_columns = predictor.feature_engine.get_feature_columns()
        self._type_cache: dict[
            tuple[int, int, str, str, bool, bool, int],
            dict[tuple[int, int], dict[str, float]],
        ] = {}

    @classmethod
    def load(
        cls,
        model_dir: str,
        fallback: PitchMixProfiles,
        season: int,
        *,
        device: str = "cpu",
        model_weight: float = 1.0,
    ) -> PitchModelCountProfiles:
        from mlb.ml.pitch_predictor import PitchPredictor
        from mlb.ml.run_dirs import resolve_pitch_type_run_dir

        return cls(
            PitchPredictor.load_lstm(resolve_pitch_type_run_dir(model_dir), device=device),
            fallback,
            season,
            model_weight=model_weight,
        )

    def inputs_for_matchup(
        self,
        *,
        pitcher_id: int,
        throw_side: str,
        batter_id: int,
        bat_side: str,
        is_top_half: bool,
        times_through: int,
        n_locations: int = 12,
        rng: random.Random | None = None,
        stretch: bool = False,
    ) -> dict[
        tuple[int, int],
        tuple[dict[str, float], dict[str, list[tuple[float, float]]]],
    ]:
        rng = rng or random.Random(0)
        key = (
            pitcher_id,
            batter_id,
            throw_side,
            bat_side,
            is_top_half,
            stretch,
            times_through,
        )
        types_by_count = self._type_cache.get(key)
        if types_by_count is None:
            types_by_count = self._count_type_distributions(
                pitcher_id=pitcher_id,
                throw_side=throw_side,
                batter_id=batter_id,
                bat_side=bat_side,
                is_top_half=is_top_half,
                times_through=times_through,
                stretch=stretch,
            )
            self._type_cache[key] = types_by_count
        inputs = {}
        for balls, strikes in COUNTS:
            types = types_by_count[(balls, strikes)]
            locations_by_type = {
                pitch_type: self._fallback.sample_locations(
                    pitcher_id, balls, strikes, pitch_type, n_locations, rng, stretch
                )
                for pitch_type in types
            }
            inputs[(balls, strikes)] = (types, locations_by_type)
        return inputs

    def _count_type_distributions(
        self,
        *,
        pitcher_id: int,
        throw_side: str,
        batter_id: int,
        bat_side: str,
        is_top_half: bool,
        times_through: int,
        stretch: bool,
    ) -> dict[tuple[int, int], dict[str, float]]:
        import numpy as np
        import torch

        from mlb.ml.features import PITCH_TYPE_CODES

        rows = []
        half_inning = "top" if is_top_half else "bottom"
        for at_bat_index, (balls, strikes) in enumerate(COUNTS):
            rows.append(
                {
                    "game_pk": -1,
                    "season": self._season,
                    "game_date": f"{self._season}-07-01",
                    "day_night": "night",
                    "weather_temp": 70.0,
                    "weather_wind": "0 mph, Calm",
                    "at_bat_index": at_bat_index,
                    "half_inning": half_inning,
                    "inning": 5,
                    "batter_id": batter_id,
                    "bat_side": bat_side,
                    "pitcher_id": pitcher_id,
                    "throw_side": throw_side,
                    "is_runner_on_first": stretch,
                    "is_runner_on_second": False,
                    "is_runner_on_third": False,
                    "away_score": 0,
                    "home_score": 0,
                    "description": "",
                    "pitch_number": 1,
                    "count_after_pitch": f"{balls}-{strikes}",
                    "outs": 1,
                    "pitch_type_code": "OTHER",
                    "px": 0.0,
                    "pz": 2.5,
                    "pitch_start_speed": 90.0,
                    "pitch_strike_zone_top": 3.5,
                    "pitch_strike_zone_bottom": 1.5,
                    "is_strike": False,
                    "is_ball": False,
                    "is_in_play": False,
                    "times_through_order": times_through,
                }
            )

        engine = self._predictor.feature_engine
        assert engine is not None
        transformed = engine.transform(pl.DataFrame(rows))
        features = transformed.select(self._feature_columns).cast(pl.Float32).to_numpy()
        tensor = torch.tensor(
            np.nan_to_num(features, nan=0.0), dtype=torch.float32
        ).unsqueeze(1)
        lengths = torch.ones(len(COUNTS), dtype=torch.long)
        predicted = self._predictor.predict_batch(
            lstm_features=tensor,
            lengths=lengths,
        )
        probabilities = predicted["type_probabilities"][:, 0, :]
        return {
            count: self._distribution_from_probs(
                pitcher_id,
                count[0],
                count[1],
                stretch,
                {
                    PITCH_TYPE_CODES[i]: float(prob)
                    for i, prob in enumerate(row)
                    if prob >= 1e-4
                },
            )
            for count, row in zip(COUNTS, probabilities)
        }

    def _distribution_from_probs(
        self,
        pitcher_id: int,
        balls: int,
        strikes: int,
        stretch: bool,
        model: dict[str, float],
    ) -> dict[str, float]:
        if self._model_weight < 1.0:
            fallback = self._fallback.type_distribution(
                pitcher_id, balls, strikes, stretch
            )
            # Sorted for the same reason as type_distribution: set-of-strings iteration order
            # varies between processes and feeds a float sum.
            model = {
                pitch_type: self._model_weight * model.get(pitch_type, 0.0)
                + (1.0 - self._model_weight) * fallback.get(pitch_type, 0.0)
                for pitch_type in sorted(set(model) | set(fallback))
            }
        norm = sum(model[t] for t in sorted(model))
        if norm <= 0.0:
            return self._fallback.type_distribution(pitcher_id, balls, strikes, stretch)
        return {t: model[t] / norm for t in sorted(model)}
