from __future__ import annotations

import argparse

from scripts import paper_trade_moneyline
from scripts import save_current_h2h_closing_lines as close_saver
from src.sim.slate import ProbablePitcher, SlateGame


def _slate_game() -> SlateGame:
    return SlateGame(
        game_pk=123,
        slate_date="2026-08-16",
        game_datetime="2026-08-16T17:05:00Z",
        status="Preview",
        away_team_id=1,
        home_team_id=2,
        away_abbrev="AWY",
        home_abbrev="HME",
        venue=None,
        away_probable=ProbablePitcher(player_id=10, full_name="Away Starter"),
        home_probable=ProbablePitcher(player_id=20, full_name="Home Starter"),
    )


def test_close_odds_rows_builds_db_ready_h2h_close_rows():
    rows = close_saver._close_odds_rows(
        slate_games=(_slate_game(),),
        odds_by_game={
            123: (
                paper_trade_moneyline.PaperOddsLine(
                    "book_a",
                    home_ml=-125,
                    away_ml=110,
                    last_update="2026-08-16T12:00:00Z",
                ),
            )
        },
        snapshot_time="2026-08-16T12:01:00+00:00",
    )

    assert rows == [
        {
            "game_pk": 123,
            "game_date": "2026-08-16",
            "away_team_id": 1,
            "home_team_id": 2,
            "bookmaker": "book_a",
            "market": "h2h",
            "line_type": "close",
            "home_ml": -125,
            "away_ml": 110,
            "snapshot_time": "2026-08-16T12:01:00+00:00",
            "source": "the-odds-api-current-close",
        }
    ]


def test_close_saver_main_dry_run_does_not_write_db(monkeypatch, capsys):
    game = _slate_game()
    db_calls: list[list[dict[str, object]]] = []
    monkeypatch.setenv("ODDS_API_KEY", "test")

    monkeypatch.setattr(
        close_saver,
        "parse_args",
        lambda: argparse.Namespace(
            date="2026-08-16",
            regions="us",
            max_match_hours=12.0,
            all_games=False,
            odds_json=None,
            dry_run=True,
            no_db_log=False,
        ),
    )
    monkeypatch.setattr(
        close_saver,
        "fetch_slate_games",
        lambda *_args, **_kwargs: [game],
    )
    monkeypatch.setattr(
        close_saver,
        "_fetch_current_odds",
        lambda *_args, **_kwargs: (
            [
                {
                    "id": "odds-game",
                    "commence_time": "2026-08-16T17:05:00Z",
                    "away_team": "AWY",
                    "home_team": "HME",
                    "bookmakers": [
                        {
                            "key": "book_a",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "AWY", "price": 110},
                                        {"name": "HME", "price": -125},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            {"x-requests-used": "1"},
        ),
    )
    monkeypatch.setattr(
        close_saver,
        "upsert_h2h_odds_rows",
        lambda rows: db_calls.append(list(rows)) or len(rows),
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "team_abbrev_to_id",
        lambda: {"AWY": 1, "HME": 2},
    )

    close_saver.main()

    captured = capsys.readouterr()
    assert "H2H close saver 2026-08-16: slate=1 odds_matched=1 rows=1" in captured.out
    assert "dry-run: no h2h close odds rows written to DB" in captured.out
    assert db_calls == []
