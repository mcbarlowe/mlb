from __future__ import annotations

import csv
import json
import math

from scripts.totals_clv_report import write_outputs
from src.betting.totals_clv import (
    TotalsBookLine,
    TotalsClvGame,
    consensus_totals_line,
    select_totals_clv_bet,
    summarize_totals_clv,
)


def _line(point: float, over_ml: float = -110, under_ml: float = -110) -> TotalsBookLine:
    return TotalsBookLine(
        bookmaker="draftkings",
        point=point,
        over_ml=over_ml,
        under_ml=under_ml,
    )


def test_consensus_totals_line_uses_median_point_and_devigged_prices() -> None:
    line = consensus_totals_line(
        [
            _line(8.0, over_ml=-110, under_ml=-110),
            _line(8.5, over_ml=-105, under_ml=-115),
            _line(8.0, over_ml=-115, under_ml=-105),
        ]
    )

    assert line is not None
    assert line.point == 8.0
    assert math.isclose(line.prob_over + line.prob_under, 1.0)
    assert 0.49 < line.prob_over < 0.51


def test_select_totals_clv_bet_records_open_over_with_positive_point_clv() -> None:
    game = TotalsClvGame(
        game_pk=1,
        season=2025,
        open_lines=[_line(8.0)],
        close_lines=[_line(8.5)],
        simulated_totals=[9, 9, 8, 7, 10],
        actual_total=10,
    )

    bet = select_totals_clv_bet(game, edge_threshold=0.05, staking="flat")

    assert bet is not None
    assert bet.side == "over"
    assert math.isclose(bet.model_prob, 0.7)
    assert math.isclose(bet.edge, 0.2)
    assert math.isclose(bet.point_clv, 0.5)
    assert bet.beat_close is True
    assert bet.result == "win"
    assert math.isclose(bet.profit, 100 / 110)


def test_select_totals_clv_bet_records_under_push_and_kelly_cap() -> None:
    game = TotalsClvGame(
        game_pk=2,
        season=2025,
        open_lines=[_line(8.5)],
        close_lines=[_line(8.0)],
        simulated_totals=[5, 7, 8, 9],
        actual_total=8.5,
    )

    bet = select_totals_clv_bet(game, edge_threshold=0.05, staking="kelly")

    assert bet is not None
    assert bet.side == "under"
    assert math.isclose(bet.model_prob, 0.75)
    assert math.isclose(bet.point_clv, 0.5)
    assert bet.result == "push"
    assert bet.profit == 0.0
    assert math.isclose(bet.stake, 0.05)


def test_summarize_totals_clv_reports_roi_and_beat_close_rate() -> None:
    games = [
        TotalsClvGame(
            game_pk=1,
            season=2025,
            open_lines=[_line(8.0)],
            close_lines=[_line(8.5)],
            simulated_totals=[9, 9, 8, 7, 10],
            actual_total=10,
        ),
        TotalsClvGame(
            game_pk=2,
            season=2025,
            open_lines=[_line(8.5)],
            close_lines=[_line(8.0)],
            simulated_totals=[5, 7, 8, 9],
            actual_total=8.5,
        ),
    ]

    summary, bets = summarize_totals_clv(games, edge_threshold=0.05, staking="flat")

    assert len(bets) == 2
    assert summary.n_games == 2
    assert summary.n_bets == 2
    assert summary.pushes == 1
    assert summary.wins == 1
    assert math.isclose(summary.total_staked, 2.0)
    assert math.isclose(summary.net_profit, 100 / 110)
    assert math.isclose(summary.roi, (100 / 110) / 2)
    assert summary.beat_close_rate == 1.0


def test_select_totals_clv_bet_returns_none_below_edge_threshold() -> None:
    game = TotalsClvGame(
        game_pk=3,
        season=2025,
        open_lines=[_line(8.0)],
        close_lines=[_line(8.0)],
        simulated_totals=[9, 8, 7, 8],
        actual_total=8,
    )

    assert select_totals_clv_bet(game, edge_threshold=0.2) is None


def test_write_outputs_exports_totals_clv_json_and_csv(tmp_path) -> None:
    game = TotalsClvGame(
        game_pk=1,
        season=2025,
        open_lines=[_line(8.0)],
        close_lines=[_line(8.5)],
        simulated_totals=[9, 9, 8, 7, 10],
        actual_total=10,
    )
    summary, bets = summarize_totals_clv([game], edge_threshold=0.05, staking="flat")
    out_json = tmp_path / "clv.json"
    out_csv = tmp_path / "clv.csv"

    write_outputs(
        games=[game],
        summaries=[summary],
        bets_by_key={("flat", 0.05): bets},
        out_json=out_json,
        out_csv=out_csv,
    )

    payload = json.loads(out_json.read_text())
    assert payload["summaries"][0]["n_bets"] == 1
    assert payload["games"][0]["open_point"] == 8.0
    assert payload["bets"][0]["side"] == "over"
    with out_csv.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["staking"] == "flat"
    assert csv_rows[0]["side"] == "over"
