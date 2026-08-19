"""Split historical odds parquet into open/close using STRICTLY PRE-GAME snapshots.

The historical odds endpoint returns every event present in a snapshot, including
games already in progress. Tagging the latest snapshot per game as "close" therefore
captures in-play prices for ~74% of games, which destroys both the edge computation
(model is pre-game) and CLV.

This script keeps only snapshots strictly before commence_time, then:
  open  = earliest surviving snapshot
  close = latest surviving snapshot
Games with fewer than two distinct pre-game snapshots are dropped: without two
prices there is no honest CLV, and a fabricated one is worse than a missing one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

SEASONS = (2020, 2021, 2022, 2023)
KEYS = ["game_id", "home_team", "away_team", "commence_time"]


def split_season(year: int) -> dict[str, int]:
    src = Path(f"data/odds_history/moneyline_{year}.parquet")
    if not src.exists():
        raise SystemExit(f"missing {src}")

    df = pl.read_parquet(src).with_columns(
        pl.col("snapshot_time").str.replace("Z", "+00:00").str.to_datetime().alias("_snap"),
        pl.col("commence_time").str.replace("Z", "+00:00").str.to_datetime().alias("_start"),
    )
    raw_games = df.select("game_id").n_unique()

    # Strictly pre-game only. This is the whole fix.
    pre = df.filter(pl.col("_snap") < pl.col("_start"))
    dropped_inplay = len(df) - len(pre)

    # Require two distinct pre-game snapshots so open != close.
    counts = pre.group_by("game_id").agg(pl.col("_snap").n_unique().alias("n"))
    keep = counts.filter(pl.col("n") >= 2).select("game_id")
    pre = pre.join(keep, on="game_id", how="inner")

    bounds = pre.group_by(KEYS).agg(
        pl.col("_snap").min().alias("_open_snap"),
        pl.col("_snap").max().alias("_close_snap"),
    )
    pre = pre.join(bounds, on=KEYS, how="inner")

    out: dict[str, int] = {}
    for line_type, col in (("open", "_open_snap"), ("close", "_close_snap")):
        subset = (
            pre.filter(pl.col("_snap") == pl.col(col))
            .drop("_snap", "_start", "_open_snap", "_close_snap")
        )
        dest = Path(f"data/odds_history/moneyline_{year}_{line_type}_pregame.parquet")
        subset.write_parquet(dest)
        out[line_type] = len(subset)

    kept_games = pre.select("game_id").n_unique()
    lead_open = (
        (pre.select((pl.col("_start") - pl.col("_open_snap")).dt.total_minutes() / 60.0))
        .to_series()
        .median()
    )
    lead_close = (
        (pre.select((pl.col("_start") - pl.col("_close_snap")).dt.total_minutes() / 60.0))
        .to_series()
        .median()
    )
    print(
        f"{year}: games {raw_games} -> {kept_games} kept | dropped {dropped_inplay:,} "
        f"in-play/post-start book-rows | open rows {out['open']:,} close rows {out['close']:,} "
        f"| median lead open {lead_open:.1f}h close {lead_close:.1f}h"
    )
    return out


def main() -> None:
    print("Splitting historical odds using strictly pre-game snapshots")
    print("=" * 78)
    for year in SEASONS:
        split_season(year)


if __name__ == "__main__":
    main()
