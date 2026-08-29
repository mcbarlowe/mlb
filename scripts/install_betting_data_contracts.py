#!/usr/bin/env python3
"""Install additive read-only MLB result views consumed by betting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlb.data_contracts.result_views import install_result_views


def main() -> None:
    install_result_views()
    print("Installed mlb.betting_game_results_v1 and mlb.betting_player_results_v1.")


if __name__ == "__main__":
    main()
