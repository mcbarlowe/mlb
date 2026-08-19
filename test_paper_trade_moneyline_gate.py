from __future__ import annotations

import argparse
import json
import sys

import pytest

from scripts import paper_trade_moneyline
from src.betting.gates import load_betting_gate, require_open_gate
from src.sim.slate import ProbablePitcher, SlateGame


def _write_gate(tmp_path, payload: dict[str, object]):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_betting_gate_accepts_wrapped_open_artifact(tmp_path):
    path = _write_gate(
        tmp_path,
        {
            "betting_gate": {
                "status": "open",
                "reason": "paper trading approved",
                "metrics": {"brier_delta": 0.01},
            }
        },
    )

    gate = load_betting_gate(path)

    assert gate.is_open is True
    assert gate.status == "open"
    assert gate.reason == "paper trading approved"
    assert gate.artifact == str(path)
    assert gate.metrics == {"brier_delta": 0.01}


def test_load_betting_gate_defaults_missing_status_to_closed(tmp_path):
    path = _write_gate(tmp_path, {"reason": "no current evidence"})

    gate = load_betting_gate(path)

    assert gate.is_open is False
    assert gate.status == "closed"
    assert gate.reason == "no current evidence"


def test_require_open_gate_exits_on_closed_artifact(tmp_path):
    path = _write_gate(
        tmp_path,
        {"betting_gate": {"status": "closed", "reason": "model gate failed"}},
    )

    with pytest.raises(SystemExit, match="Betting gate is CLOSED: model gate failed"):
        require_open_gate(path)


def test_paper_trade_parse_args_keeps_gate_json_optional(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["paper_trade_moneyline.py"])
    assert paper_trade_moneyline.parse_args().gate_json is None

    gate_path = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["paper_trade_moneyline.py", "--gate-json", str(gate_path)],
    )
    assert paper_trade_moneyline.parse_args().gate_json == str(gate_path)


def test_paper_trade_preflight_prints_open_gate_confirmation(tmp_path, capsys):
    path = _write_gate(
        tmp_path,
        {"betting_gate": {"status": "open", "reason": "research gate passed"}},
    )

    paper_trade_moneyline._preflight_gate(str(path))

    captured = capsys.readouterr()
    assert captured.out == f"Betting gate OPEN: research gate passed ({path})\n"


def test_paper_trade_main_closed_gate_exits_before_slate_or_odds(tmp_path, monkeypatch):
    path = _write_gate(
        tmp_path,
        {"betting_gate": {"status": "closed", "reason": "gate closed today"}},
    )

    def fail_fetch_slate_games(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("slate fetch should not run behind a closed gate")

    monkeypatch.setattr(
        paper_trade_moneyline,
        "parse_args",
        lambda: argparse.Namespace(gate_json=str(path)),
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "fetch_slate_games",
        fail_fetch_slate_games,
    )

    with pytest.raises(SystemExit, match="Betting gate is CLOSED: gate closed today"):
        paper_trade_moneyline.main()


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


def test_open_odds_rows_builds_db_ready_h2h_open_rows():
    rows = paper_trade_moneyline._open_odds_rows(
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
            "line_type": "open",
            "home_ml": -125,
            "away_ml": 110,
            "snapshot_time": "2026-08-16T12:01:00+00:00",
            "source": "the-odds-api-current",
        }
    ]


def test_paper_trade_main_persists_matched_open_odds(monkeypatch, tmp_path):
    odds_path = tmp_path / "odds.json"
    odds_path.write_text("[]")
    game = _slate_game()
    odds_calls: list[list[dict[str, object]]] = []
    paper_calls: list[list[dict[str, str]]] = []

    monkeypatch.setattr(
        paper_trade_moneyline,
        "parse_args",
        lambda: argparse.Namespace(
            gate_json=None,
            date="2026-08-16",
            all_games=False,
            odds_json=str(odds_path),
            regions="us",
            max_match_hours=12.0,
            skip_active_rosters=True,
            mlflow_tracking_uri="http://example.test",
            win_model_name="model",
            edge_threshold=0.05,
            staking="flat",
            bankroll_units=100.0,
            flat_stake_units=1.0,
            kelly_multiplier=0.25,
            kelly_cap=0.05,
            no_db_log=False,
            no_csv_log=True,
            no_odds_db_log=False,
            dry_run=False,
            out=str(tmp_path / "paper.csv"),
            replace_date=False,
        ),
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "fetch_slate_games",
        lambda *_args, **_kwargs: [game],
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "_odds_by_game_pk",
        lambda **_kwargs: {
            123: [
                paper_trade_moneyline.PaperOddsLine("book_a", home_ml=120, away_ml=-140)
            ]
        },
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "upsert_h2h_odds_rows",
        lambda rows: odds_calls.append(list(rows)) or len(rows),
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "build_live_strength_predictor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "_predict_home_probability",
        lambda **_kwargs: 0.60,
    )
    monkeypatch.setattr(
        paper_trade_moneyline,
        "upsert_paper_trade_rows",
        lambda rows: paper_calls.append(list(rows)) or len(rows),
    )

    paper_trade_moneyline.main()

    assert len(odds_calls) == 1
    assert odds_calls[0][0]["game_pk"] == 123
    assert odds_calls[0][0]["line_type"] == "open"
    assert odds_calls[0][0]["source"] == "the-odds-api-current"
    assert len(paper_calls) == 1
