from __future__ import annotations

import pytest

from src.betting.f5_clv import (
    F5BookLine,
    F5ClvGame,
    best_f5_execution_line,
    compare_f5_line_shopping,
    consensus_f5_totals_line,
    select_f5_clv_bet,
    summarize_f5_clv,
)


def test_select_f5_bet_uses_consensus_for_edge_and_best_book_for_execution() -> None:
    open_lines = (
        F5BookLine("book_a", 4.5, -110, -110),
        F5BookLine("book_b", 4.0, -120, 100),
        F5BookLine("book_c", 4.5, -105, -115),
    )
    close_lines = (
        F5BookLine("book_a", 5.0, -110, -110),
        F5BookLine("book_b", 5.0, -108, -112),
    )
    game = F5ClvGame(
        game_pk=1,
        season=2025,
        open_lines=open_lines,
        close_lines=close_lines,
        simulated_totals=(5, 6, 7, 4, 5),
        actual_total=5,
    )

    bet = select_f5_clv_bet(game, edge_threshold=0.10)

    assert bet is not None
    assert bet.side == "over"
    assert bet.open_point == pytest.approx(4.5)
    assert bet.take_bookmaker == "book_b"
    assert bet.take_point == pytest.approx(4.0)
    assert bet.model_prob == pytest.approx(0.8)
    assert bet.execution_model_prob == pytest.approx(0.9)
    assert bet.result == "win"
    assert bet.profit == pytest.approx(bet.take_decimal - 1.0)
    assert bet.point_clv == pytest.approx(1.0)
    assert bet.beat_close is True



def test_select_f5_bet_can_compare_consensus_and_best_execution() -> None:
    open_lines = (
        F5BookLine("book_a", 4.5, -110, -110),
        F5BookLine("book_b", 4.0, 120, -140),
        F5BookLine("book_c", 4.5, -110, -110),
    )
    game = F5ClvGame(
        game_pk=11,
        season=2025,
        open_lines=open_lines,
        close_lines=(F5BookLine("book_a", 4.5, -120, 100),),
        simulated_totals=(5, 5, 5, 4, 4),
        actual_total=5,
    )

    consensus_bet = select_f5_clv_bet(game, edge_threshold=0.05, execution="consensus")
    best_bet = select_f5_clv_bet(game, edge_threshold=0.05)

    assert consensus_bet is not None
    assert best_bet is not None
    assert consensus_bet.side == best_bet.side == "over"
    assert consensus_bet.edge == pytest.approx(best_bet.edge)
    assert consensus_bet.model_prob == pytest.approx(best_bet.model_prob)
    assert consensus_bet.take_bookmaker == "consensus"
    assert consensus_bet.execution == "consensus"
    assert consensus_bet.take_point == pytest.approx(4.5)
    assert best_bet.take_bookmaker == "book_b"
    assert best_bet.execution == "best"
    assert best_bet.take_point == pytest.approx(4.0)
    assert best_bet.execution_model_prob > consensus_bet.execution_model_prob


def test_f5_line_shopping_lift_uses_fixed_consensus_selection() -> None:
    game = F5ClvGame(
        game_pk=12,
        season=2025,
        open_lines=(
            F5BookLine("book_a", 4.5, -110, -110),
            F5BookLine("book_b", 4.0, 120, -140),
            F5BookLine("book_c", 4.5, -110, -110),
        ),
        close_lines=(F5BookLine("book_a", 4.5, -120, 100),),
        simulated_totals=(5, 5, 5, 4, 4),
        actual_total=5,
    )

    summary, bets = compare_f5_line_shopping((game,), edge_threshold=0.05)

    assert summary.n_bets == 1
    bet = bets[0]
    consensus_decimal = 1 + 100 / 110
    assert bet.side == "over"
    assert bet.edge == pytest.approx(0.10)
    assert bet.consensus_point == pytest.approx(4.5)
    assert bet.best_point == pytest.approx(4.0)
    assert bet.best_bookmaker == "book_b"
    assert bet.best_decimal == pytest.approx(2.2)
    assert bet.point_lift == pytest.approx(0.5)
    assert bet.decimal_lift == pytest.approx(2.2 - consensus_decimal)
    assert bet.best_profit - bet.consensus_profit == pytest.approx(bet.decimal_lift)
    assert bet.point_clv_lift == pytest.approx(0.5)
    assert summary.avg_point_lift == pytest.approx(0.5)
    assert summary.avg_decimal_lift == pytest.approx(2.2 - consensus_decimal)
    assert summary.roi_lift == pytest.approx(2.2 - consensus_decimal)
    assert summary.point_clv_lift == pytest.approx(0.5)


def test_f5_summary_reports_scores_roi_and_missing_close_as_na() -> None:
    game = F5ClvGame(
        game_pk=2,
        season=2025,
        open_lines=(F5BookLine("book_a", 3.5, -110, -110),),
        simulated_totals=(1, 2, 3, 4, 5),
        actual_total=2,
        take_line_type="current",
    )

    summary, bets = summarize_f5_clv((game,), edge_threshold=0.0)

    assert summary.n_games == 1
    assert summary.n_scored_games == 1
    assert summary.n_bets == 1
    assert summary.n_settled_bets == 1
    assert summary.total_staked == pytest.approx(1.0)
    assert summary.net_profit == pytest.approx(1 / 1.1)
    assert summary.roi == pytest.approx(1 / 1.1)
    assert summary.model_brier_all is not None
    assert summary.market_brier_all is not None
    assert summary.model_log_loss_all is not None
    assert summary.market_log_loss_all is not None
    assert summary.avg_point_clv is None
    assert summary.beat_close_rate is None
    assert bets[0].side == "under"
    assert bets[0].result == "win"


def test_f5_consensus_and_best_execution_validate_side() -> None:
    lines = (
        F5BookLine("low", 4.0, -115, -105),
        F5BookLine("high", 4.5, -105, -115),
    )

    consensus = consensus_f5_totals_line(lines)

    assert consensus is not None
    assert consensus.point == pytest.approx(4.25)
    assert best_f5_execution_line(lines, "over") == lines[0]
    assert best_f5_execution_line(lines, "under") == lines[1]
    with pytest.raises(ValueError, match="unknown F5 totals side"):
        best_f5_execution_line(lines, "middle")


def test_f5_summary_accepts_custom_model_probability() -> None:
    game = F5ClvGame(
        game_pk=21,
        season=2025,
        open_lines=(F5BookLine("book_a", 4.5, -110, -110),),
        simulated_totals=(1, 2, 3, 4, 4),
        actual_total=5,
    )

    def model_probability(
        _game: F5ClvGame,
        _point: float,
        side: str,
        _market_prob: float,
    ) -> float:
        return 0.62 if side == "over" else 0.38

    summary, bets = summarize_f5_clv(
        (game,),
        edge_threshold=0.05,
        model_probability=model_probability,
    )

    assert summary.settings["probability_source"] == "custom"
    assert summary.model_brier_all == pytest.approx((0.62 - 1.0) ** 2)
    assert len(bets) == 1
    assert bets[0].side == "over"
    assert bets[0].model_prob == pytest.approx(0.62)


def test_f5_line_shopping_accepts_custom_model_probability() -> None:
    game = F5ClvGame(
        game_pk=22,
        season=2025,
        open_lines=(
            F5BookLine("book_a", 4.5, -110, -110),
            F5BookLine("book_b", 4.0, 120, -140),
        ),
        simulated_totals=(1, 2, 3, 4, 4),
        actual_total=5,
    )

    def model_probability(
        _game: F5ClvGame,
        _point: float,
        side: str,
        _market_prob: float,
    ) -> float:
        return 0.62 if side == "over" else 0.38

    summary, bets = compare_f5_line_shopping(
        (game,),
        edge_threshold=0.05,
        model_probability=model_probability,
    )

    assert summary.settings["probability_source"] == "custom"
    assert len(bets) == 1
    assert bets[0].side == "over"
    assert bets[0].model_prob == pytest.approx(0.62)
    assert bets[0].best_point == pytest.approx(4.0)
