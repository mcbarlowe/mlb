#!/usr/bin/env python3
"""Persist current MLB h2h boards as live close-line snapshots.

The Odds API current endpoint only returns pregame lines. Running this script
periodically during the day keeps ``mlb.odds`` ``line_type='close'`` at the
latest sampled board before each game disappears from the feed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.market_data.current_h2h import (
    PaperOddsLine,
    _fetch_current_odds,
    _h2h_odds_rows,
    _odds_by_game_pk,
    _parse_date,
)
from src.market_data.h2h_odds_store import upsert_h2h_odds_rows
from src.sim.slate import SlateGame, fetch_slate_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--regions", type=str, default="us")
    parser.add_argument("--max-match-hours", type=float, default=12.0)
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Match against all scheduled states instead of only MLB Preview games.",
    )
    parser.add_argument(
        "--odds-json",
        type=str,
        default=None,
        help="Use a saved current-odds payload instead of calling The Odds API.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-db-log", action="store_true")
    return parser.parse_args()


def _close_odds_rows(
    *,
    slate_games: Sequence[SlateGame],
    odds_by_game: Mapping[int, Sequence[PaperOddsLine]],
    snapshot_time: str,
) -> list[dict[str, object]]:
    return _h2h_odds_rows(
        slate_games=slate_games,
        odds_by_game=odds_by_game,
        snapshot_time=snapshot_time,
        line_type="close",
        source="the-odds-api-current-close",
    )


def _print_credit_headers(headers: Mapping[str, str]) -> None:
    if not headers:
        return
    print(
        "Odds API credits: "
        f"last={headers.get('x-requests-last', '')} "
        f"used={headers.get('x-requests-used', '')} "
        f"remaining={headers.get('x-requests-remaining', '')}"
    )


def main() -> None:
    args = parse_args()
    target_date = _parse_date(args.date)
    odds_date = target_date.isoformat()
    states = None if args.all_games else {"Preview"}
    slate_games = fetch_slate_games(target_date, abstract_states=states)
    if not slate_games:
        print(f"No slate games found for {odds_date}; no h2h close odds rows written")
        return

    if args.odds_json:
        odds_payload = json.loads(Path(args.odds_json).read_text())
        headers: dict[str, str] = {}
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            raise SystemExit("ODDS_API_KEY is required unless --odds-json is provided")
        odds_payload, headers = _fetch_current_odds(api_key, args.regions)

    snapshot_time = datetime.now(tz=UTC).isoformat()
    odds_by_game = _odds_by_game_pk(
        payload=odds_payload,
        slate_games=slate_games,
        max_match_hours=args.max_match_hours,
    )
    rows = _close_odds_rows(
        slate_games=slate_games,
        odds_by_game=odds_by_game,
        snapshot_time=snapshot_time,
    )
    print(
        f"H2H close saver {odds_date}: slate={len(slate_games)} "
        f"odds_matched={len(odds_by_game)} rows={len(rows)}"
    )
    _print_credit_headers(headers)

    if args.dry_run:
        print("dry-run: no h2h close odds rows written to DB")
        return
    if args.no_db_log:
        print("no-db-log: no h2h close odds rows written to DB")
        return

    upserted = upsert_h2h_odds_rows(rows)
    print(f"upserted {upserted} h2h close odds rows into odds")


if __name__ == "__main__":
    main()
