#!/usr/bin/env python3
"""Write a futures edge report from projection and odds CSV files.

This is a paper-trade/reporting utility only. It does not touch databases,
networks, order-entry systems, or sportsbook APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.futures import generate_futures_edge_report_csv
from src.betting.futures_odds import FUTURES_MARKETS, normalize_market_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-csv",
        type=Path,
        required=True,
        help="Season projection CSV containing model probabilities by team.",
    )
    parser.add_argument(
        "--odds-csv",
        type=Path,
        required=True,
        help="Futures market CSV containing market_type, team, and odds/probability.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination CSV for the edge report.",
    )
    parser.add_argument(
        "--projection-type",
        default="model",
        help="projection_type row to use from projection CSV. Defaults to model.",
    )
    parser.add_argument(
        "--as-of-bucket",
        default=None,
        help=(
            "Optional as_of_bucket row to use when the projection CSV contains "
            "multiple as-of snapshots."
        ),
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=FUTURES_MARKETS,
        default=None,
        help="Optional subset of futures markets to report. Defaults to odds CSV markets.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=None,
        help="Optional minimum model-minus-no-vig-market edge to include.",
    )
    parser.add_argument(
        "--kelly-multiplier",
        type=float,
        default=None,
        help="Optional fractional Kelly multiplier used to populate stake_units.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.05,
        help="Maximum stake_units when --kelly-multiplier is provided. Defaults to 0.05.",
    )
    parser.add_argument(
        "--allow-market-source-leakage",
        action="store_true",
        help="Allow reporting when projection input_market_sources names a target market.",
    )
    parser.add_argument(
        "--target-total",
        action="append",
        default=[],
        metavar="MARKET=TOTAL",
        help=(
            "Override de-vig target total for a market, e.g. world_series=1 "
            "for separate AL/NL pennant groups. May be passed multiple times."
        ),
    )
    return parser.parse_args()


def parse_target_total_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--target-total must be MARKET=TOTAL, got {value!r}")
        market, total = value.split("=", 1)
        normalized_market = normalize_market_type(market)
        parsed_total = float(total)
        if parsed_total <= 0.0:
            raise ValueError(f"--target-total must be positive, got {value!r}")
        overrides[normalized_market] = parsed_total
    return overrides


def main() -> None:
    args = parse_args()
    rows = generate_futures_edge_report_csv(
        projection_csv=args.projection_csv,
        odds_csv=args.odds_csv,
        out_csv=args.out,
        projection_type=args.projection_type,
        as_of_bucket=args.as_of_bucket,
        markets=args.markets,
        edge_threshold=args.edge_threshold,
        kelly_multiplier=args.kelly_multiplier,
        kelly_cap=args.kelly_cap,
        target_total_overrides=parse_target_total_overrides(args.target_total),
        allow_market_source_leakage=args.allow_market_source_leakage,
    )
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
