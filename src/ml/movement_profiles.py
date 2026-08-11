"""Leak-free pitcher movement profiles from measured pitch characteristics.

Builds, for every (pitcher, game) appearance, the trailing distribution of
what the pitcher has actually thrown BEFORE that game: per-pitch-type usage
share plus mean velocity, horizontal/vertical movement, and spin rate over a
trailing window of appearances. Profiles are as-of game start, so attaching
them to pitches can never leak same-game or future information.

The canonical pitch-type axis matches the prediction models
(``PITCH_TYPE_CODES``); non-canonical codes fold into ``OTHER``.
"""

from __future__ import annotations

import polars as pl

from src.ml.features import PITCH_TYPE_CODES

# spin_rate was backfilled on 2026-08-11 (scripts/fix_pitches_spin.py)
# after the extraction-level fix; it is a first-class profile stat again.
MOVEMENT_STATS = ("velo", "pfx_x", "pfx_z", "spin_rate")
DEFAULT_WINDOW_GAMES = 40

PER_GAME_REQUIRED_COLUMNS = (
    "pitcher_id",
    "game_pk",
    "game_date",
    "pitch_type_code",
    "n",
    *MOVEMENT_STATS,
)


def canonical_pitch_code_expr(column: str = "pitch_type_code") -> pl.Expr:
    """Fold non-canonical pitch codes into OTHER."""
    return (
        pl.when(pl.col(column).is_in(list(PITCH_TYPE_CODES)))
        .then(pl.col(column))
        .otherwise(pl.lit("OTHER"))
        .alias(column)
    )


def _dense_appearance_grid(per_game: pl.DataFrame) -> pl.DataFrame:
    """Cross every (pitcher, game) appearance with the pitcher's type inventory.

    A type the pitcher did not throw today still needs a profile row for
    today's game, so trailing usage decays instead of disappearing.
    """
    appearances = per_game.select(
        "pitcher_id", "game_pk", "game_date"
    ).unique()
    inventory = per_game.select("pitcher_id", "pitch_type_code").unique()
    grid = appearances.join(inventory, on="pitcher_id", how="inner")
    return grid.join(
        per_game,
        on=["pitcher_id", "game_pk", "game_date", "pitch_type_code"],
        how="left",
    ).with_columns(pl.col("n").fill_null(0))


def compute_trailing_profiles(
    per_game: pl.DataFrame,
    window_games: int = DEFAULT_WINDOW_GAMES,
) -> pl.DataFrame:
    """Compute as-of-game trailing profiles from per-game pitch-type stats.

    Args:
        per_game: One row per (pitcher_id, game_pk, game_date,
            pitch_type_code) with pitch count ``n`` and per-game means for
            each stat in ``MOVEMENT_STATS``. Stats may be null when
            unmeasured.
        window_games: Trailing window measured in the pitcher's appearances.

    Returns:
        Long frame keyed by (pitcher_id, game_pk, pitch_type_code) with
        ``trailing_n``, ``usage``, and trailing weighted means for each
        stat, all computed from strictly earlier games. Appearances with no
        prior history (a pitcher's first game) have ``trailing_n == 0`` and
        null stats.
    """
    missing = [c for c in PER_GAME_REQUIRED_COLUMNS if c not in per_game.columns]
    if missing:
        raise ValueError(f"per_game frame is missing columns: {missing}")

    dense = (
        _dense_appearance_grid(per_game)
        .with_columns(
            [
                # DB nulls arrive as NaN through pandas; non-finite values
                # must read as "unmeasured" or they poison rolling sums.
                pl.when(pl.col(stat).is_finite())
                .then(pl.col(stat))
                .otherwise(None)
                .alias(stat)
                for stat in MOVEMENT_STATS
            ]
        )
        .sort(["pitcher_id", "pitch_type_code", "game_date", "game_pk"])
    )

    group = ["pitcher_id", "pitch_type_code"]

    def trailing_sum(expr: pl.Expr) -> pl.Expr:
        return (
            expr.fill_null(0.0)
            .shift(1)
            .rolling_sum(window_size=window_games, min_samples=1)
            .over(group)
        )

    stat_exprs: list[pl.Expr] = [trailing_sum(pl.col("n")).alias("trailing_n")]
    for stat in MOVEMENT_STATS:
        weight = (
            pl.when(pl.col(stat).is_not_null())
            .then(pl.col("n"))
            .otherwise(0)
        )
        weight_sum = trailing_sum(weight)
        stat_exprs.append(
            pl.when(weight_sum > 0)
            .then(trailing_sum(pl.col(stat) * pl.col("n")) / weight_sum)
            .otherwise(None)
            .alias(stat)
        )

    trailing = dense.with_columns(stat_exprs).with_columns(
        pl.col("trailing_n").fill_null(0).cast(pl.Int64)
    )

    totals = trailing.group_by(["pitcher_id", "game_pk"]).agg(
        pl.col("trailing_n").sum().alias("trailing_n_total")
    )
    trailing = trailing.join(totals, on=["pitcher_id", "game_pk"], how="left")
    return trailing.with_columns(
        pl.when(pl.col("trailing_n_total") > 0)
        .then(pl.col("trailing_n") / pl.col("trailing_n_total"))
        .otherwise(None)
        .alias("usage")
    ).select(
        "pitcher_id",
        "game_pk",
        "game_date",
        "pitch_type_code",
        "trailing_n",
        "trailing_n_total",
        "usage",
        *MOVEMENT_STATS,
    )


def profile_column_name(stat: str, pitch_code: str) -> str:
    return f"profile_{stat}_{pitch_code.lower()}"


def pivot_profiles_wide(trailing: pl.DataFrame) -> pl.DataFrame:
    """Pivot long trailing profiles into one row per (pitcher, game)."""
    wide = trailing.pivot(
        on="pitch_type_code",
        index=["pitcher_id", "game_pk", "game_date", "trailing_n_total"],
        values=["usage", *MOVEMENT_STATS],
        aggregate_function="first",
    )
    renames = {}
    for column in wide.columns:
        for stat in ("usage", *MOVEMENT_STATS):
            prefix = f"{stat}_"
            if column.startswith(prefix) and column[len(prefix):] in PITCH_TYPE_CODES:
                renames[column] = profile_column_name(stat, column[len(prefix):])
    return wide.rename(renames)


MOVEMENT_PROFILE_STAT_SCALES = {
    "usage": 1.0,
    "velo": 1.0 / 100.0,
    "pfx_x": 1.0 / 12.0,
    "pfx_z": 1.0 / 12.0,
    "spin_rate": 1.0 / 3000.0,
}


def movement_profile_columns() -> list[str]:
    """Feature columns contributed by movement profiles, in stable order."""
    return [
        profile_column_name(stat, code)
        for stat in ("usage", *MOVEMENT_STATS)
        for code in PITCH_TYPE_CODES
    ]


def attach_movement_profiles(
    frame: pl.DataFrame,
    wide_profiles: pl.DataFrame,
    league_defaults: dict[str, float],
) -> pl.DataFrame:
    """Attach normalized profile features to a pitch frame.

    Historical games hit their exact (pitcher_id, game_pk) appearance row;
    null stats there mean "no prior history" and take league defaults —
    NEVER the pitcher's latest profile, which would leak the future into
    training data. Games absent from the store (live games) fall back to
    the pitcher's latest stored profile, which is past by construction,
    then to league defaults for true debuts.
    """
    columns = movement_profile_columns()
    missing = [c for c in columns if c not in wide_profiles.columns]
    if missing:
        wide_profiles = wide_profiles.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
        )
    marker = "__profile_row_exists"
    exact = wide_profiles.select(
        "pitcher_id",
        "game_pk",
        pl.col("trailing_n_total").alias(marker),
        *columns,
    )
    result = frame.join(exact, on=["pitcher_id", "game_pk"], how="left")

    latest = (
        wide_profiles.sort(["game_date", "game_pk"])
        .group_by("pitcher_id", maintain_order=True)
        .last()
        .select("pitcher_id", *columns)
        .rename({c: f"{c}__latest" for c in columns})
    )
    result = result.join(latest, on="pitcher_id", how="left")

    def _scale_for(column: str) -> float:
        for stat, scale in MOVEMENT_PROFILE_STAT_SCALES.items():
            if column.startswith(f"profile_{stat}_"):
                return scale
        raise ValueError(f"No scale for profile column {column}")

    fills = []
    for column in columns:
        default = league_defaults.get(column, 0.0)
        row_exists = pl.col(marker).is_not_null()
        value = (
            pl.when(row_exists)
            .then(pl.col(column).fill_null(default))
            .otherwise(pl.col(f"{column}__latest").fill_null(default))
        )
        fills.append((value * _scale_for(column)).alias(column))
    result = result.with_columns(fills)
    return result.drop([marker, *[f"{c}__latest" for c in columns]])


def league_default_profiles(trailing: pl.DataFrame) -> dict[str, float]:
    """League-average fallback values for pitchers with no history."""
    defaults: dict[str, float] = {}
    aggregated = (
        trailing.filter(pl.col("trailing_n") > 0)
        .group_by("pitch_type_code")
        .agg(
            pl.col("usage").mean(),
            *[pl.col(stat).mean() for stat in MOVEMENT_STATS],
        )
    )
    for row in aggregated.iter_rows(named=True):
        code = row["pitch_type_code"]
        for stat in ("usage", *MOVEMENT_STATS):
            value = row[stat]
            if value is not None:
                defaults[profile_column_name(stat, code)] = float(value)
    return defaults
