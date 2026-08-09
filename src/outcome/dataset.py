"""Training datasets for the pitch outcome models.

Builds leak-free Stage A (per-pitch result) and Stage B (in-play event)
frames directly from the PostgreSQL pitches table. Every feature here is
computable at simulation time from (a) game state, (b) a pitch described by
type + location, and (c) rolling profile lookups — never from measurements
of the pitch being predicted.

Leakage rules verified against the ETL (see docs/pitch_outcome_model_plan.md):
- ``outs`` and runner flags are at-bat-start state (safe as-is)
- ``count_after_pitch`` is post-pitch (shifted within the at-bat)
- ``away_score``/``home_score`` are post-play (shifted across at-bats)
- rolling profiles/priors use expanding means that exclude the current row
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from src.outcome.labels import (
    CANONICAL_PITCH_TYPES,
    map_event_type,
    map_pitch_call,
)

# Strike zone geometry (feet). Half-width is the plate half-width.
ZONE_HALF_WIDTH_FT = 17.0 / 24.0
DEFAULT_SZ_TOP = 3.4
DEFAULT_SZ_BOTTOM = 1.6

PHYSICS_COLUMNS = [
    "pitch_start_speed",
    "spin_rate",
    "break_vertical_induced",
    "break_horizontal",
    "x0",
    "z0",
]

_BASE_COLUMNS = [
    "game_pk",
    "season",
    "game_date",
    "at_bat_index",
    "pitch_number",
    "half_inning",
    "inning",
    "outs",
    "is_runner_on_first",
    "is_runner_on_second",
    "is_runner_on_third",
    "batter_id",
    "bat_side",
    "pitcher_id",
    "throw_side",
    "pitch_call_description",
    "event_type",
    "count_after_pitch",
    "pitch_type_code",
    "px",
    "pz",
    "pitch_strike_zone_top",
    "pitch_strike_zone_bottom",
    "away_score",
    "home_score",
    *PHYSICS_COLUMNS,
]

CATEGORICAL_FEATURES = [
    # NOTE: raw `pitcher_id_cat`/`batter_id_cat` are deliberately excluded.
    # CatBoost's ordered target statistics for those high-cardinality ids
    # produced a train/apply asymmetry: aggregate P(in_play) was ~2pp low on
    # training seasons and ~6pp high on every out-of-window season — and all
    # production inference is out-of-window. Player skill enters through the
    # numeric physics profiles and batter priors instead. The remaining
    # categoricals are low-cardinality and trained one-hot (no CTRs).
    "pitch_type",
    "throw_side",
    "bat_side",
]

NUMERIC_FEATURES = [
    "balls_before",
    "strikes_before",
    # NOTE: `outs` and `runner_on_*` are deliberately excluded. The current
    # pitches table carries a dead `outs` column (always 0) and mover-only
    # runner flags, which LEAK the outcome (a runner "on" often means someone
    # moved during this play — hits set the flags far more than outs).
    # Reinstate them only after the DB reload with reconstructed state
    # (src/data/base_state.py) and a retrain.
    "inning",
    "is_top_half",
    "score_diff",
    "season",
    "times_through_order",
    "px",
    "pz",
    "abs_px",
    "zone_norm_height",
    "zone_dist_center",
    "in_zone",
    # Pitcher physics profile deltas vs league, per pitch type.
    "profile_speed_delta",
    "profile_spin_delta",
    "profile_ivb_delta",
    "profile_hb_delta",
    "profile_release_x",
    "profile_release_z",
    "pitcher_whiff_rate",
    # Batter priors.
    "batter_swing_rate",
    "batter_whiff_rate",
    "batter_chase_rate",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# Exact dtypes for every loaded column (matches the pitches table schema).
# Row-wise construction with a full schema avoids dtype inference entirely:
# early seasons have all-null physics columns (e.g. induced break pre-2020)
# that break inference on the first rows.
_LOAD_SCHEMA: dict[str, type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "season": pl.Int64,
    "game_date": pl.String,
    "at_bat_index": pl.Int64,
    "pitch_number": pl.Int64,
    "half_inning": pl.String,
    "inning": pl.Int64,
    "outs": pl.Int64,
    "is_runner_on_first": pl.Boolean,
    "is_runner_on_second": pl.Boolean,
    "is_runner_on_third": pl.Boolean,
    "batter_id": pl.Int64,
    "bat_side": pl.String,
    "pitcher_id": pl.Int64,
    "throw_side": pl.String,
    "pitch_call_description": pl.String,
    "event_type": pl.String,
    "count_after_pitch": pl.String,
    "pitch_type_code": pl.String,
    "px": pl.Float64,
    "pz": pl.Float64,
    "pitch_strike_zone_top": pl.Float64,
    "pitch_strike_zone_bottom": pl.Float64,
    "away_score": pl.Int64,
    "home_score": pl.Int64,
    "pitch_start_speed": pl.Float64,
    "spin_rate": pl.Float64,
    "break_vertical_induced": pl.Float64,
    "break_horizontal": pl.Float64,
    "x0": pl.Float64,
    "z0": pl.Float64,
}

_LOAD_BATCH_ROWS = 500_000


def load_pitches(seasons: Sequence[int], attempts: int = 5) -> pl.DataFrame:
    """Load raw pitch rows for the given seasons from PostgreSQL.

    The database may live across a LAN (laptop training against the
    workstation), so transient connect/read failures are retried with a
    fixed backoff before giving up.
    """
    import time

    import psycopg

    from src.database import PostgresConfig

    assert list(_LOAD_SCHEMA) == _BASE_COLUMNS, "load schema drifted from columns"

    config = PostgresConfig.from_env()
    season_list = ", ".join(str(int(season)) for season in seasons)
    query = f"""
        SELECT {", ".join(_BASE_COLUMNS)}
        FROM {config.schema}.pitches
        WHERE season IN ({season_list})
          AND game_type = 'R'
        ORDER BY game_date, game_pk, at_bat_index, pitch_number
    """
    conninfo = {
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
        "host": config.host,
        "port": config.port,
        "connect_timeout": 15,
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with psycopg.connect(
                **{key: value for key, value in conninfo.items() if value is not None}
            ) as connection, connection.cursor() as cursor:
                cursor.execute(query.encode("utf-8"))
                frames: list[pl.DataFrame] = []
                while True:
                    rows = cursor.fetchmany(_LOAD_BATCH_ROWS)
                    if not rows:
                        break
                    frames.append(
                        pl.DataFrame(rows, schema=_LOAD_SCHEMA, orient="row")
                    )
                if not frames:
                    return pl.DataFrame(schema=_LOAD_SCHEMA)
                return pl.concat(frames)
        except psycopg.OperationalError as exc:
            last_error = exc
            if attempt == attempts:
                break
            wait_seconds = 20 * attempt
            print(
                f"Database load attempt {attempt}/{attempts} failed ({exc}); "
                f"retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Failed to load pitches after {attempts} attempts"
    ) from last_error


def _canonical_type_expr() -> pl.Expr:
    code = pl.col("pitch_type_code")
    return (
        pl.when(code.is_null() | code.is_in(["", "None", "UN", "IN", "PO", "AB"]))
        .then(pl.lit(None, dtype=pl.String))
        .when(code.is_in(CANONICAL_PITCH_TYPES))
        .then(code)
        .otherwise(pl.lit("OTHER"))
    )


def _expanding_mean_prev(value: pl.Expr, group: list[str]) -> pl.Expr:
    """Expanding mean over prior rows only (null-safe, leak-free)."""
    present = value.is_not_null()
    total = value.fill_null(0.0).cum_sum().over(group) - value.fill_null(0.0)
    count = present.cast(pl.Int64).cum_sum().over(group) - present.cast(pl.Int64)
    return pl.when(count > 0).then(total / count).otherwise(None)


def _expanding_rate_prev(
    numerator: pl.Expr, denominator: pl.Expr, group: list[str]
) -> pl.Expr:
    """Rate of prior events: sum(numerator)/sum(denominator), current row excluded."""
    num = numerator.cast(pl.Int64)
    den = denominator.cast(pl.Int64)
    num_prev = num.cum_sum().over(group) - num
    den_prev = den.cum_sum().over(group) - den
    return pl.when(den_prev > 0).then(num_prev / den_prev).otherwise(None)


def add_state_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Pre-pitch count, pre-at-bat score, zone geometry, situational flags."""
    frame = frame.sort(["game_date", "game_pk", "at_bat_index", "pitch_number"])

    counts = pl.col("count_after_pitch").str.split_exact("-", 1)
    frame = frame.with_columns(
        counts.struct.field("field_0").cast(pl.Int64, strict=False).alias("balls_after"),
        counts.struct.field("field_1").cast(pl.Int64, strict=False).alias("strikes_after"),
    )
    ab_key = ["game_pk", "at_bat_index"]
    frame = frame.with_columns(
        pl.col("balls_after").shift(1).over(ab_key).fill_null(0).alias("balls_before"),
        pl.col("strikes_after").shift(1).over(ab_key).fill_null(0).alias("strikes_before"),
    )

    # Pre-at-bat score: stored scores are post-play, so take the previous
    # at-bat's value within the game (0-0 before the first at-bat).
    ab_scores = (
        frame.group_by(["game_pk", "at_bat_index"], maintain_order=True)
        .agg(
            pl.col("away_score").first(),
            pl.col("home_score").first(),
        )
        .with_columns(
            pl.col("away_score").shift(1).over("game_pk").fill_null(0).alias("away_score_before"),
            pl.col("home_score").shift(1).over("game_pk").fill_null(0).alias("home_score_before"),
        )
        .select(["game_pk", "at_bat_index", "away_score_before", "home_score_before"])
    )
    frame = frame.join(ab_scores, on=["game_pk", "at_bat_index"], how="left")

    # Times through the order for the pitcher (1-indexed, 9 batters per turn).
    ab_sequence = (
        frame.group_by(["game_pk", "pitcher_id", "at_bat_index"], maintain_order=True)
        .agg(pl.len().alias("_n"))
        .with_columns(
            (
                ((pl.int_range(pl.len()).over(["game_pk", "pitcher_id"])) // 9) + 1
            ).alias("times_through_order")
        )
        .select(["game_pk", "pitcher_id", "at_bat_index", "times_through_order"])
    )
    frame = frame.join(
        ab_sequence, on=["game_pk", "pitcher_id", "at_bat_index"], how="left"
    )

    is_top = pl.col("half_inning").str.to_lowercase() == "top"
    sz_top = pl.col("pitch_strike_zone_top").fill_null(DEFAULT_SZ_TOP)
    sz_bottom = pl.col("pitch_strike_zone_bottom").fill_null(DEFAULT_SZ_BOTTOM)
    zone_center = (sz_top + sz_bottom) / 2.0
    zone_half_height = ((sz_top - sz_bottom) / 2.0).clip(lower_bound=1e-3)

    return frame.with_columns(
        _canonical_type_expr().alias("pitch_type"),
        is_top.cast(pl.Int8).alias("is_top_half"),
        # Batting team score minus fielding team score.
        pl.when(is_top)
        .then(pl.col("away_score_before") - pl.col("home_score_before"))
        .otherwise(pl.col("home_score_before") - pl.col("away_score_before"))
        .alias("score_diff"),
        pl.col("is_runner_on_first").cast(pl.Int8).alias("runner_on_first"),
        pl.col("is_runner_on_second").cast(pl.Int8).alias("runner_on_second"),
        pl.col("is_runner_on_third").cast(pl.Int8).alias("runner_on_third"),
        pl.col("px").abs().alias("abs_px"),
        ((pl.col("pz") - sz_bottom) / (sz_top - sz_bottom)).alias("zone_norm_height"),
        (
            ((pl.col("px") / ZONE_HALF_WIDTH_FT) ** 2
             + ((pl.col("pz") - zone_center) / zone_half_height) ** 2).sqrt()
        ).alias("zone_dist_center"),
        (
            (pl.col("px").abs() <= ZONE_HALF_WIDTH_FT)
            & (pl.col("pz") >= sz_bottom)
            & (pl.col("pz") <= sz_top)
        )
        .cast(pl.Int8)
        .alias("in_zone"),
        pl.col("pitcher_id").cast(pl.Int64, strict=False).cast(pl.String)
        .fill_null("unknown")
        .alias("pitcher_id_cat"),
        pl.col("batter_id").cast(pl.Int64, strict=False).cast(pl.String)
        .fill_null("unknown")
        .alias("batter_id_cat"),
    )


def add_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach Stage A and Stage B labels (nulls mean: not part of that stage)."""
    return frame.with_columns(
        pl.col("pitch_call_description")
        .map_elements(map_pitch_call, return_dtype=pl.String)
        .alias("label_result"),
        pl.col("event_type")
        .map_elements(map_event_type, return_dtype=pl.String)
        .alias("label_event"),
    )


def add_rolling_profiles(frame: pl.DataFrame) -> pl.DataFrame:
    """Leak-free pitcher physics profiles (delta vs league) and skill priors."""
    frame = frame.sort(["game_date", "game_pk", "at_bat_index", "pitch_number"])

    pitcher_type = ["pitcher_id", "pitch_type"]
    league_type = ["pitch_type"]
    profile_exprs = []
    for column, name in [
        ("pitch_start_speed", "speed"),
        ("spin_rate", "spin"),
        ("break_vertical_induced", "ivb"),
        ("break_horizontal", "hb"),
    ]:
        pitcher_mean = _expanding_mean_prev(pl.col(column), pitcher_type)
        league_mean = _expanding_mean_prev(pl.col(column), league_type)
        profile_exprs.append((pitcher_mean - league_mean).alias(f"profile_{name}_delta"))
    profile_exprs.append(
        _expanding_mean_prev(pl.col("x0"), pitcher_type).alias("profile_release_x")
    )
    profile_exprs.append(
        _expanding_mean_prev(pl.col("z0"), pitcher_type).alias("profile_release_z")
    )

    swung = pl.col("label_result").is_in(["swinging_strike", "foul", "in_play"])
    whiff = pl.col("label_result") == "swinging_strike"
    is_pitch = pl.col("label_result").is_not_null()
    out_of_zone = is_pitch & (pl.col("in_zone") == 0)

    return frame.with_columns(
        *profile_exprs,
        _expanding_rate_prev(whiff, swung, pitcher_type).alias("pitcher_whiff_rate"),
        _expanding_rate_prev(swung, is_pitch, ["batter_id"]).alias("batter_swing_rate"),
        _expanding_rate_prev(whiff, swung, ["batter_id"]).alias("batter_whiff_rate"),
        _expanding_rate_prev(swung & out_of_zone, out_of_zone, ["batter_id"]).alias(
            "batter_chase_rate"
        ),
    )


def build_training_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Full feature/label frame from raw pitch rows (all stages, unfiltered)."""
    frame = add_state_features(raw)
    frame = add_labels(frame)
    return add_rolling_profiles(frame)


def stage_a_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows eligible for Stage A training: labeled pitches with type+location."""
    return frame.filter(
        pl.col("label_result").is_not_null()
        & pl.col("pitch_type").is_not_null()
        & pl.col("px").is_not_null()
        & pl.col("pz").is_not_null()
    )


def stage_b_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows eligible for Stage B training: balls in play with a mapped event."""
    return frame.filter(
        (pl.col("label_result") == "in_play")
        & pl.col("label_event").is_not_null()
        & pl.col("pitch_type").is_not_null()
        & pl.col("px").is_not_null()
        & pl.col("pz").is_not_null()
    )
