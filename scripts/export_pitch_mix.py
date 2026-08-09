"""Export count-conditioned pitch mix/location profiles from PostgreSQL.

Writes ``models/sim/pitch_mix.parquet`` and
``models/sim/pitch_locations.parquet`` (local generated artifacts) for the
game simulator's matchup providers. Recent seasons only, so the mixes
reflect current repertoires:

    uv run python scripts/export_pitch_mix.py --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.outcome.dataset import load_pitches
from src.sim.pitch_mix import (
    LOCATION_TABLE_PATH,
    MIX_TABLE_PATH,
    build_pitch_mix_tables,
)

DEFAULT_SEASONS = [2023, 2024, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pitch mix profiles.")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--mix-output", type=str, default=str(MIX_TABLE_PATH))
    parser.add_argument("--location-output", type=str, default=str(LOCATION_TABLE_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    print(f"Loading pitches for seasons {args.seasons}...")
    raw = load_pitches(args.seasons)
    print(f"Loaded {raw.height:,} rows in {time.perf_counter() - start:.1f}s")

    mix, locations = build_pitch_mix_tables(
        raw.select(
            [
                "pitcher_id",
                "game_pk",
                "at_bat_index",
                "pitch_number",
                "count_after_pitch",
                "pitch_type_code",
                "px",
                "pz",
            ]
        )
    )

    mix_path = Path(args.mix_output)
    location_path = Path(args.location_output)
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    mix.write_parquet(mix_path)
    locations.write_parquet(location_path)
    n_pitchers = mix["pitcher_id"].n_unique()
    print(f"Wrote mix for {n_pitchers} pitchers -> {mix_path}")
    print(f"Wrote {locations.height:,} location samples -> {location_path}")


if __name__ == "__main__":
    main()
