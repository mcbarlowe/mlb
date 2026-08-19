from __future__ import annotations

from typing import Any, cast

import pytest

from scripts.f5_clv_replay import build_replay_report, games_from_report_payload


def _payload() -> dict[str, object]:
    return {
        "games": [
            {
                "game_pk": 990001,
                "season": 2025,
                "take_line_type": "open",
                "actual_f5_total": 4.0,
                "simulated_totals": [3.0, 4.0, 4.0, 5.0, 6.0],
                "open_lines": [
                    {
                        "bookmaker": "book_a",
                        "point": 4.5,
                        "over_ml": -110,
                        "under_ml": -110,
                    },
                    {
                        "bookmaker": "book_b",
                        "point": 5.0,
                        "over_ml": -110,
                        "under_ml": -110,
                    },
                ],
                "close_lines": [
                    {
                        "bookmaker": "book_a",
                        "point": 5.0,
                        "over_ml": -110,
                        "under_ml": -110,
                    }
                ],
            }
        ]
    }


def test_games_from_report_payload_rebuilds_clv_games() -> None:
    games = games_from_report_payload(_payload())

    assert len(games) == 1
    game = games[0]
    assert game.game_pk == 990001
    assert game.season == 2025
    assert game.simulated_totals == (3.0, 4.0, 4.0, 5.0, 6.0)
    assert len(game.open_lines) == 2
    assert game.open_lines[1].bookmaker == "book_b"
    assert len(game.close_lines) == 1


def test_build_replay_report_can_recompute_raw_and_anchor() -> None:
    games = games_from_report_payload(_payload())

    report = build_replay_report(
        games,
        edges=(0.03,),
        market_anchor_lambda=0.5,
    )

    assert report["summaries"][0]["n_bets"] == 1
    assert report["bets"][0]["side"] == "under"
    assert report["anchor_blend"]["lambda"] == pytest.approx(0.5)
    assert report["anchor_blend"]["summaries"][0]["n_bets"] == 1
    assert report["anchor_blend"]["bets"][0]["side"] == "under"


def test_games_from_report_payload_rejects_legacy_non_replayable_json() -> None:
    payload = _payload()
    games = cast(list[dict[str, Any]], payload["games"])
    game = games[0]
    del game["simulated_totals"]

    with pytest.raises(ValueError, match="replayable JSON"):
        games_from_report_payload(payload)
