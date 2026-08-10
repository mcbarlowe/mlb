"""Export pitcher/batter profile stores for live outcome inference.

Usage:
    uv run python scripts/export_outcome_profiles.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.outcome.dataset import build_training_frame, load_pitches
from src.outcome.profiles import build_profile_stores, save_profile_stores


def main() -> None:
    parser = argparse.ArgumentParser(description="Export outcome profile stores.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026],
        help="Seasons used to compute current profiles (most recent last)",
    )
    parser.add_argument("--output-dir", type=str, default="models/outcome")
    args = parser.parse_args()

    print(f"Loading pitches for seasons {sorted(args.seasons)}...")
    raw = load_pitches(sorted(args.seasons))
    print(f"Loaded {raw.height:,} rows; building profiles...")
    frame = build_training_frame(raw)
    pitcher_profiles, batter_priors = build_profile_stores(frame)

    # Synthetic per-team bullpen arms: reliever rows relabeled to -team_id
    # flow through the same profile builder (drop its duplicate league row).
    from src.sim.bullpen import relabel_reliever_rows

    arm_frame = build_training_frame(relabel_reliever_rows(raw))
    arm_profiles, _ = build_profile_stores(arm_frame)
    arm_profiles = arm_profiles.filter(pl.col("pitcher_id") < 0)
    pitcher_profiles = pl.concat([pitcher_profiles, arm_profiles])

    output_dir = Path(args.output_dir)
    save_profile_stores(pitcher_profiles, batter_priors, output_dir)
    print(
        f"Saved {pitcher_profiles.height:,} pitcher×type profiles "
        f"({arm_profiles.height} team bullpen rows) and "
        f"{batter_priors.height:,} batter priors under {output_dir}"
    )


if __name__ == "__main__":
    main()
