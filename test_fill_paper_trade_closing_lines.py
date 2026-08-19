from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from scripts.fill_paper_trade_closing_lines import (
    MissingCloseDate,
    commands_for_date,
    fill_missing_close_dates,
    stage_path,
)


def test_commands_for_date_builds_fetch_and_load_steps():
    stage = Path("data/odds/paper_h2h_close_2026-08-14.parquet")

    fetch_cmd, load_cmd = commands_for_date(
        paper_date=date(2026, 8, 14),
        stage=stage,
        python="python",
    )

    assert fetch_cmd == [
        "python",
        "scripts/fetch_odds_history.py",
        "--start",
        "2026-08-14",
        "--end",
        "2026-08-14",
        "--markets",
        "h2h",
        "--keep",
        "latest",
        "--out",
        str(stage),
    ]
    assert load_cmd == [
        "python",
        "scripts/load_odds_to_db.py",
        "--stage",
        str(stage),
        "--season",
        "2026",
        "--line-type",
        "close",
    ]


def test_stage_path_is_stable_per_paper_date():
    assert stage_path(Path("data/odds"), date(2026, 8, 14)) == Path(
        "data/odds/paper_h2h_close_2026-08-14.parquet"
    )


def test_fill_missing_close_dates_runs_fetch_then_load(tmp_path, capsys):
    calls: list[list[str]] = []

    def fake_run(cmd, *, check):
        assert check is True
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    loaded = fill_missing_close_dates(
        [
            MissingCloseDate(
                date(2026, 8, 14),
                paper_games=4,
                missing_close_games=4,
                open_games=15,
            )
        ],
        out_dir=tmp_path,
        python="python",
        dry_run=False,
        run=fake_run,
        has_rows=lambda path: path == tmp_path / "paper_h2h_close_2026-08-14.parquet",
    )

    assert loaded == 1
    captured = capsys.readouterr()
    assert "open_games=15" in captured.out
    assert calls == [
        [
            "python",
            "scripts/fetch_odds_history.py",
            "--start",
            "2026-08-14",
            "--end",
            "2026-08-14",
            "--markets",
            "h2h",
            "--keep",
            "latest",
            "--out",
            str(tmp_path / "paper_h2h_close_2026-08-14.parquet"),
        ],
        [
            "python",
            "scripts/load_odds_to_db.py",
            "--stage",
            str(tmp_path / "paper_h2h_close_2026-08-14.parquet"),
            "--season",
            "2026",
            "--line-type",
            "close",
        ],
    ]


def test_fill_missing_close_dates_dry_run_does_not_execute(tmp_path):
    calls: list[list[str]] = []

    loaded = fill_missing_close_dates(
        [MissingCloseDate(date(2026, 8, 14), paper_games=4, missing_close_games=4)],
        out_dir=tmp_path,
        python="python",
        dry_run=True,
        run=lambda cmd, *, check: calls.append(cmd),
    )

    assert loaded == 0
    assert calls == []
