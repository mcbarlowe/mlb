from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from mlb.data.game_feed_data import GameFeedData
from mlb.database import PostgresConfig, PostgresHandler

_PITCH_POLARS_DTYPES = {
    column: (
        pl.Int64
        if dtype is int
        else pl.Float64
        if dtype is float
        else pl.Boolean
        if dtype is bool
        else pl.String
    )
    for column, dtype in GameFeedData().data_types.items()
}



def discover_postgres_seasons(db_config: PostgresConfig | None = None) -> list[str]:
    with PostgresHandler(db_config) as db:
        seasons_df = db.query("SELECT DISTINCT season FROM mlb.pitches ORDER BY season")
    if seasons_df.empty:
        return []
    return [str(int(season)) for season in seasons_df["season"].tolist() if season is not None]


def load_pitches_from_postgres(
    seasons: Sequence[str] | None = None,
    db_config: PostgresConfig | None = None,
) -> pl.DataFrame:
    where_clause = ""
    if seasons:
        normalized = ", ".join(str(int(season)) for season in seasons)
        where_clause = f"WHERE season IN ({normalized})"

    query = f"""
    SELECT *
    FROM mlb.pitches
    {where_clause}
    ORDER BY season, game_pk, at_bat_index, pitch_number
    """

    with PostgresHandler(db_config) as db:
        pitches_df = db.query(query)

    if pitches_df.empty:
        return pl.DataFrame(schema={column: dtype for column, dtype in _PITCH_POLARS_DTYPES.items()})

    df = pl.from_pandas(pitches_df)
    return df.with_columns(
        [
            pl.col(column).cast(dtype, strict=False)
            for column, dtype in _PITCH_POLARS_DTYPES.items()
            if column in df.columns
        ]
    )
