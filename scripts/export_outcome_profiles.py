"""Export pitcher/batter profile stores for live outcome inference.

Usage:
    uv run python scripts/export_outcome_profiles.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    output_dir = Path(args.output_dir)
    save_profile_stores(pitcher_profiles, batter_priors, output_dir)
    print(
        f"Saved {pitcher_profiles.height:,} pitcher×type profiles and "
        f"{batter_priors.height:,} batter priors under {output_dir}"
    )


if __name__ == "__main__":
    main()
