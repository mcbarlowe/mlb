"""Empirical base-out transition tables (RE288-style).

Built from our own at-bat data: for each (PA outcome, runners bitmap, outs)
we record the empirical distribution over (runners after, outs after, runs
scored). Mid-at-bat events (steals, wild pitches) are absorbed into the
transitions on average, which matches the v1 simulation scope.

Runner bitmap encoding: bit 0 = first, bit 1 = second, bit 2 = third
(0 = bases empty, 7 = bases loaded).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlb.outcome.labels import EVENT_TYPE_TO_CLASS

# At-bat level event mapping for the transition tables: the count-machine
# terminals plus the Stage B in-play classes.
AB_EVENT_TO_PA_OUTCOME: dict[str, str] = {
    "walk": "walk",
    "intent_walk": "walk",
    "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
    "strikeout_triple_play": "strikeout",
    "hit_by_pitch": "hit_by_pitch",
    **EVENT_TYPE_TO_CLASS,
}

DEFAULT_TABLE_PATH = Path("models/sim/base_out_tables.parquet")


def runners_bitmap(on_first: bool, on_second: bool, on_third: bool) -> int:
    return int(on_first) | (int(on_second) << 1) | (int(on_third) << 2)


def build_transition_frame(at_bats: pl.DataFrame) -> pl.DataFrame:
    """At-bat transition counts from reconstructed at-bat state rows.

    Expects one row per at-bat with true start-of-AB state (see
    ``src.data.base_state.compute_at_bat_states``): ``game_pk``,
    ``at_bat_index``, ``inning``, ``half_inning``, ``event_type``,
    ``outs_before``, ``runners_before`` (bitmap), ``outs_after``
    (authoritative post-play outs), ``away_score``/``home_score``
    (post-play).
    """
    at_bats = at_bats.sort(["game_pk", "at_bat_index"])

    is_top = pl.col("half_inning").str.to_lowercase() == "top"
    at_bats = at_bats.with_columns(
        pl.when(is_top)
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .alias("bat_score_post"),
    )

    half_key = ["game_pk", "half_inning"]
    inning_key = ["game_pk", "inning", "half_inning"]
    at_bats = at_bats.with_columns(
        (
            pl.col("bat_score_post")
            - pl.col("bat_score_post").shift(1).over(half_key).fill_null(0)
        ).alias("runs"),
        pl.col("runners_before").shift(-1).over(inning_key).alias("runners_after"),
    )

    at_bats = at_bats.with_columns(
        pl.col("event_type")
        .replace_strict(AB_EVENT_TO_PA_OUTCOME, default=None)
        .alias("pa_outcome"),
    )

    cleaned = at_bats.filter(
        pl.col("pa_outcome").is_not_null()
        # Keep rows whose post state is knowable: either the inning ended
        # (post-play outs reached 3) or a following at-bat exists. Drops
        # walk-off finals, which have no successor state.
        & (pl.col("runners_after").is_not_null() | (pl.col("outs_after") == 3))
        & (pl.col("runs") >= 0)
        & (pl.col("runs") <= 4)
        & (pl.col("outs_after") >= pl.col("outs_before"))
    ).with_columns(pl.col("runners_after").fill_null(0))

    return (
        cleaned.group_by(
            [
                "pa_outcome",
                "runners_before",
                "outs_before",
                "runners_after",
                "outs_after",
                "runs",
            ]
        )
        .agg(pl.len().alias("n"))
        .sort(
            ["pa_outcome", "runners_before", "outs_before", "n"],
            descending=[False, False, False, True],
        )
    )


# Deterministic fallbacks for (outcome, runners, outs) states never observed.
# Conservative single-advance semantics; runs only on forced pushes and homers.
def _fallback_transition(outcome: str, runners: int, outs: int) -> tuple[int, int, int]:
    first, second, third = bool(runners & 1), bool(runners & 2), bool(runners & 4)
    if outcome in ("walk", "hit_by_pitch"):
        run = int(first and second and third)
        new_first = True
        new_second = first or second
        new_third = (first and second) or third
        return runners_bitmap(new_first, new_second, new_third), outs, run
    if outcome == "strikeout":
        return runners, min(outs + 1, 3), 0
    if outcome == "out":
        return runners, min(outs + 1, 3), 0
    if outcome in ("single", "reached_on_error"):
        runs = int(third)
        return runners_bitmap(True, first, second), outs, runs
    if outcome == "double":
        runs = int(second) + int(third)
        return runners_bitmap(False, True, first), outs, runs
    if outcome == "triple":
        runs = int(first) + int(second) + int(third)
        return runners_bitmap(False, False, True), outs, runs
    if outcome == "home_run":
        runs = 1 + int(first) + int(second) + int(third)
        return 0, outs, runs
    raise ValueError(f"Unknown PA outcome {outcome!r}")


@dataclass(frozen=True)
class BaseOutTransition:
    runners_after: int
    outs_after: int
    runs: int


class BaseOutEngine:
    """Samples base-out transitions from empirical tables with fallbacks."""

    def __init__(self, table: pl.DataFrame, seed: int | None = None):
        self._rng = random.Random(seed)
        self._table: dict[tuple[str, int, int], tuple[list[tuple[int, int, int]], list[int]]] = {}
        for key, group in table.group_by(
            ["pa_outcome", "runners_before", "outs_before"]
        ):
            outcome, runners, outs = key
            outcomes = list(
                zip(
                    group["runners_after"].to_list(),
                    group["outs_after"].to_list(),
                    group["runs"].to_list(),
                )
            )
            weights = group["n"].to_list()
            self._table[(str(outcome), int(str(runners)), int(str(outs)))] = (
                outcomes,
                weights,
            )

    @classmethod
    def load(cls, path: Path = DEFAULT_TABLE_PATH, seed: int | None = None) -> BaseOutEngine:
        return cls(pl.read_parquet(path), seed=seed)

    def sample(self, outcome: str, runners: int, outs: int) -> BaseOutTransition:
        entry = self._table.get((outcome, runners, outs))
        if entry is None:
            runners_after, outs_after, runs = _fallback_transition(outcome, runners, outs)
            return BaseOutTransition(runners_after, outs_after, runs)
        outcomes, weights = entry
        runners_after, outs_after, runs = self._rng.choices(outcomes, weights=weights)[0]
        return BaseOutTransition(int(runners_after), int(outs_after), int(runs))
