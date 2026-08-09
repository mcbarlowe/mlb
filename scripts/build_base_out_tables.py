"""Build empirical base-out transition tables from raw live feed JSONs.

The PostgreSQL pitches table currently carries broken base/out state (dead
``outs`` column, mover-only runner flags), so this reads the archived GUMBO
feeds directly and reconstructs true at-bat start state per half-inning.

Writes ``models/sim/base_out_tables.parquet`` (a local generated artifact,
like the trained models). Usage:

    uv run python scripts/build_base_out_tables.py --seasons 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sim.base_out import DEFAULT_TABLE_PATH, build_transition_frame

DEFAULT_SEASONS = list(range(2021, 2026))
LIVEFEED_ROOT = Path("data/raw/livefeeds")

AB_SCHEMA = {
    "game_pk": pl.Int64,
    "at_bat_index": pl.Int64,
    "inning": pl.Int64,
    "half_inning": pl.String,
    "event_type": pl.String,
    "outs_before": pl.Int64,
    "runners_before": pl.Int64,
    "outs_after": pl.Int64,
    "away_score": pl.Int64,
    "home_score": pl.Int64,
}


def extract_at_bat_rows(feed_path: Path) -> list[dict]:
    """At-bat state rows for one game feed; empty for non-regular games."""
    from src.data.base_state import compute_at_bat_states

    feed = json.loads(feed_path.read_text())
    game_data = feed.get("gameData", {})
    if game_data.get("game", {}).get("type") != "R":
        return []
    game_pk = game_data.get("game", {}).get("pk")
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    states = compute_at_bat_states(plays)

    rows = []
    for play, state in zip(plays, states):
        about = play.get("about", {})
        result = play.get("result", {})
        rows.append(
            {
                "game_pk": game_pk,
                "at_bat_index": about.get("atBatIndex"),
                "inning": about.get("inning"),
                "half_inning": about.get("halfInning"),
                "event_type": result.get("eventType"),
                "outs_before": state["outs_before"],
                "runners_before": int(state["is_runner_on_first"])
                | (int(state["is_runner_on_second"]) << 1)
                | (int(state["is_runner_on_third"]) << 2),
                "outs_after": state["outs_after"],
                "away_score": result.get("awayScore", 0),
                "home_score": result.get("homeScore", 0),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build base-out transition tables.")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--output", type=str, default=str(DEFAULT_TABLE_PATH))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for season in args.seasons:
        season_dir = LIVEFEED_ROOT / str(season)
        if not season_dir.is_dir():
            raise SystemExit(f"Missing live feed directory {season_dir}")
        files.extend(sorted(season_dir.glob("*.json")))
    print(f"Parsing {len(files):,} game feeds for seasons {args.seasons}...")

    start = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for game_rows in pool.map(extract_at_bat_rows, files, chunksize=25):
            rows.extend(game_rows)
    at_bats = pl.DataFrame(rows, schema=AB_SCHEMA)
    print(f"Reconstructed {at_bats.height:,} at-bats in {time.perf_counter() - start:.1f}s")

    table = build_transition_frame(at_bats)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output)

    n_states = table.select(["pa_outcome", "runners_before", "outs_before"]).unique().height
    print(f"Wrote {table.height:,} transition rows covering {n_states} states -> {output}")


if __name__ == "__main__":
    main()
