from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.sim_totals_eval import (
    TotalsEvalRow,
    make_eval_row,
    row_output,
    summarize_rows,
    write_outputs,
)


def test_make_eval_row_records_distribution_and_outcome() -> None:
    row = make_eval_row(
        season=2025,
        game_pk=1,
        point=8.0,
        market_prob_over=0.52,
        simulated_totals=[7, 8, 9, 9],
        actual_total=9,
    )

    assert row.sim_prob_over == pytest.approx(0.625)
    assert row.sim_prob_under == pytest.approx(0.375)
    assert row.sim_prob_push == pytest.approx(0.25)
    assert row.sim_mean_total == pytest.approx(8.25)
    assert row.outcome == "over"
    assert row.actual_over == 1


def test_summarize_rows_scores_non_push_games() -> None:
    rows = [
        TotalsEvalRow(
            season=2025,
            game_pk=1,
            point=8.5,
            sim_prob_over=0.7,
            sim_prob_under=0.3,
            sim_prob_push=0.0,
            market_prob_over=0.5,
            sim_mean_total=9.5,
            sim_total_stdev=2.0,
            actual_total=10.0,
            outcome="over",
        ),
        TotalsEvalRow(
            season=2025,
            game_pk=2,
            point=8.5,
            sim_prob_over=0.3,
            sim_prob_under=0.7,
            sim_prob_push=0.0,
            market_prob_over=0.5,
            sim_mean_total=7.5,
            sim_total_stdev=1.5,
            actual_total=7.0,
            outcome="under",
        ),
        TotalsEvalRow(
            season=2025,
            game_pk=3,
            point=8.0,
            sim_prob_over=0.5,
            sim_prob_under=0.5,
            sim_prob_push=0.2,
            market_prob_over=0.5,
            sim_mean_total=8.0,
            sim_total_stdev=1.0,
            actual_total=8.0,
            outcome="push",
        ),
    ]

    summary = summarize_rows(
        rows,
        run_id="test-run",
        season=2025,
        games_requested=3,
        sims=100,
        seed=7,
        pa_calibration_path=None,
        mlflow_tracking_uri="http://example.test:5001",
        outcome_run_dir="models/outcome/test",
        contact_environment_enabled=True,
    )

    assert summary["non_push_games"] == 2
    assert summary["push_games"] == 1
    assert summary["metrics"]["sim"]["brier"] == pytest.approx(0.09)
    assert summary["metrics"]["market"]["brier"] == pytest.approx(0.25)
    assert summary["totals"]["mean_sim_minus_actual_total"] == pytest.approx(0.0)
    assert "roi_by_edge" not in summary
    assert "edge_threshold" not in summary


def test_write_outputs_json_and_csv(tmp_path: Path) -> None:
    row = TotalsEvalRow(
        season=2025,
        game_pk=1,
        point=8.5,
        sim_prob_over=0.7,
        sim_prob_under=0.3,
        sim_prob_push=0.0,
        market_prob_over=0.5,
        sim_mean_total=9.5,
        sim_total_stdev=2.0,
        actual_total=10.0,
        outcome="over",
    )
    summary = summarize_rows(
        [row],
        run_id="test-run",
        season=2025,
        games_requested=1,
        sims=100,
        seed=7,
        pa_calibration_path="models/sim/example.json",
        contact_environment_enabled=False,
        mlflow_tracking_uri="http://example.test:5001",
        outcome_run_dir="models/outcome/test",
    )
    out_json = tmp_path / "totals.json"
    out_csv = tmp_path / "totals.csv"

    write_outputs(
        rows=[row],
        summary=summary,
        out_json=out_json,
        out_csv=out_csv,
    )

    payload = json.loads(out_json.read_text())
    assert payload["summary"]["run_id"] == "test-run"
    assert payload["rows"][0]["run_id"] == "test-run"
    assert "bet_side_at_threshold" not in payload["rows"][0]

    with out_csv.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["game_pk"] == "1"
    assert csv_rows[0]["actual_over"] == "1"


def test_row_output_marks_push_metrics_blank() -> None:
    row = TotalsEvalRow(
        season=2025,
        game_pk=1,
        point=8.0,
        sim_prob_over=0.5,
        sim_prob_under=0.5,
        sim_prob_push=0.25,
        market_prob_over=0.45,
        sim_mean_total=8.0,
        sim_total_stdev=1.0,
        actual_total=8.0,
        outcome="push",
    )

    output = row_output(row, run_id="run")

    assert output["actual_over"] is None
    assert output["sim_brier"] is None
    assert "bet_side_at_threshold" not in output
    assert "bet_result_at_threshold" not in output
