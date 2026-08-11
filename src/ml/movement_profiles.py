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

    dense = _dense_appearance_grid(per_game).sort(
        ["pitcher_id", "pitch_type_code", "game_date", "game_pk"]
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
        stat_exprs.append(
            (
                trailing_sum(pl.col(stat) * pl.col("n"))
                / trailing_sum(weight)
            ).alias(stat)
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
        index=["pitcher_id", "game_pk", "trailing_n_total"],
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
