from __future__ import annotations

import math

from src.betting.backtest import (
    MoneylineGame,
    backtest_moneyline,
    kelly_fraction,
)
from src.betting.line_shopping import (
    BookLine,
    LineShoppingGame,
    line_shop_moneyline,
)
from src.betting.odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    devig_proportional,
    no_vig_two_way,
    prob_to_american,
    two_way_overround,
)
from src.betting.paper_settlement import (
    moneyline_profit,
    settle_paper_trade_row,
    summarize_paper_trade_rows,
)
from src.betting.paper_trade_store import normalize_paper_trade_row
from src.betting.paper_trading import PaperOddsLine, select_moneyline_paper_trade


def test_american_decimal_conversions():
    assert math.isclose(american_to_decimal(150), 2.5)
    assert math.isclose(american_to_decimal(-150), 1.0 + 100.0 / 150.0)
    assert math.isclose(american_to_decimal(100), 2.0)
    assert math.isclose(decimal_to_american(2.5), 150.0)
    assert math.isclose(decimal_to_american(1.0 + 100.0 / 150.0), -150.0)
    assert math.isclose(decimal_to_american(2.0), 100.0)


def test_american_prob_conversions():
    assert math.isclose(american_to_prob(100), 0.5)
    assert math.isclose(american_to_prob(-110), 110.0 / 210.0)
    assert math.isclose(american_to_prob(150), 100.0 / 250.0)
    # Round trip prob -> american -> prob.
    for p in (0.35, 0.5, 0.62):
        assert math.isclose(american_to_prob(prob_to_american(p)), p, rel_tol=1e-9)


def test_two_way_overround_and_proportional_devig():
    # A -110/-110 book: each side implies 0.5238, overround ~4.76%.
    p_home = american_to_prob(-110)
    p_away = american_to_prob(-110)
    assert math.isclose(two_way_overround(p_home, p_away), 2 * (110 / 210) - 1)
    fair_home, fair_away = devig_proportional(p_home, p_away)
    assert math.isclose(fair_home, 0.5)
    assert math.isclose(fair_away, 0.5)
    assert math.isclose(fair_home + fair_away, 1.0)


def test_no_vig_symmetric_is_half():
    for method in ("proportional", "shin"):
        home, away = no_vig_two_way(-110, -110, method=method)
        assert math.isclose(home, 0.5, abs_tol=1e-9)
        assert math.isclose(away, 0.5, abs_tol=1e-9)


def test_no_vig_favorite_orders_and_sums_to_one():
    for method in ("proportional", "shin"):
        home, away = no_vig_two_way(-200, 170, method=method)
        assert math.isclose(home + away, 1.0, abs_tol=1e-9)
        assert home > away  # -200 is the favorite
        assert 0.0 < away < home < 1.0


def test_shin_and_proportional_differ_on_lopsided_book():
    prop = no_vig_two_way(-350, 280, method="proportional")
    shin = no_vig_two_way(-350, 280, method="shin")
    # Shin corrects favorite-longshot bias, so it must not be identical.
    assert not math.isclose(prop[0], shin[0], abs_tol=1e-6)


def test_kelly_fraction():
    # p=0.6 at even money (decimal 2.0): f* = (0.6*1 - 0.4)/1 = 0.2
    assert math.isclose(kelly_fraction(0.6, 2.0), 0.2)
    # No edge at or below fair price -> zero.
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.4, 2.0) == 0.0


def _game(
    *,
    model_prob_home: float = 0.5,
    home_take: float = 100,
    away_take: float = 100,
    home_close: float = 100,
    away_close: float = 100,
    home_won: bool = True,
) -> MoneylineGame:
    return MoneylineGame(
        game_pk=1,
        model_prob_home=model_prob_home,
        home_take=home_take,
        away_take=away_take,
        home_close=home_close,
        away_close=away_close,
        home_won=home_won,
    )


def test_backtest_places_winning_flat_bet():
    # Even-money book -> fair 0.5/0.5; model 0.60 -> home edge 0.10 clears 0.02.
    games = [_game(model_prob_home=0.60, home_won=True)]
    summary, bets = backtest_moneyline(games, edge_threshold=0.02, staking="flat")
    assert summary.n_bets == 1
    assert bets[0].side == "home"
    assert math.isclose(bets[0].edge, 0.10)
    assert math.isclose(summary.net_profit, 1.0)  # +100 pays 1 unit on 1 staked
    assert math.isclose(summary.roi, 1.0)


def test_backtest_settles_losing_bet():
    games = [_game(model_prob_home=0.60, home_won=False)]
    summary, _ = backtest_moneyline(games, edge_threshold=0.02, staking="flat")
    assert summary.n_bets == 1
    assert math.isclose(summary.net_profit, -1.0)
    assert math.isclose(summary.roi, -1.0)


def test_backtest_skips_when_edge_below_threshold():
    games = [_game(model_prob_home=0.51, home_won=True)]
    summary, bets = backtest_moneyline(games, edge_threshold=0.02, staking="flat")
    assert summary.n_bets == 0
    assert bets == []
    assert summary.roi == 0.0
    assert summary.total_staked == 0.0


def test_backtest_kelly_staking_capped():
    games = [_game(model_prob_home=0.60, home_won=True)]
    summary, bets = backtest_moneyline(
        games, edge_threshold=0.02, staking="kelly",
        kelly_multiplier=0.25, kelly_cap=0.05,
    )
    # full kelly 0.2 * 0.25 = 0.05, equals cap.
    assert math.isclose(bets[0].stake, 0.05)
    assert math.isclose(summary.net_profit, 0.05)


def test_backtest_clv_beats_close():
    # Take +120 on home; closing book is -110/-110 (fair 0.5). Took a longer
    # price than the close, so CLV must be positive and beat_close True.
    game = MoneylineGame(
        game_pk=7,
        model_prob_home=0.55,
        home_take=120,
        away_take=-140,
        home_close=-110,
        away_close=-110,
        home_won=True,
    )
    summary, bets = backtest_moneyline([game], edge_threshold=0.02, staking="flat")
    assert summary.n_bets == 1
    bet = bets[0]
    assert bet.side == "home"
    assert bet.beat_close is True
    assert bet.clv_prob > 0.0
    # Take fair prob ~0.438; close fair 0.5; CLV ~ +0.062.
    assert math.isclose(bet.clv_prob, 0.5 - american_to_prob(120) /
                        (american_to_prob(120) + american_to_prob(-140)),
                        abs_tol=1e-9)


def test_closing_only_has_zero_clv():
    game = MoneylineGame.closing_only(
        game_pk=9, model_prob_home=0.60, home_close=100, away_close=100, home_won=True
    )
    _, bets = backtest_moneyline([game], edge_threshold=0.02, staking="flat")
    assert bets[0].clv_prob == 0.0
    assert bets[0].beat_close is False


def _line_shop_game() -> LineShoppingGame:
    return LineShoppingGame(
        game_pk=11,
        season=2025,
        model_prob_home=0.60,
        consensus_open_home=100,
        consensus_open_away=100,
        consensus_close_home=-110,
        consensus_close_away=-110,
        open_lines=(
            BookLine("book_a", home_ml=110, away_ml=-130),
            BookLine("book_b", home_ml=120, away_ml=-140),
        ),
        close_lines=(BookLine("book_b", home_ml=-110, away_ml=-110),),
        home_won=True,
    )


def test_line_shopping_uses_best_price_for_consensus_side():
    summary, bets = line_shop_moneyline([_line_shop_game()], edge_threshold=0.02)

    assert summary.n_bets == 1
    assert bets[0].side == "home"
    assert bets[0].source_book == "book_b"
    assert bets[0].source_books == ("book_b",)
    assert math.isclose(summary.consensus_net_profit, 1.0)
    assert math.isclose(summary.best_net_profit, 1.2)
    assert math.isclose(summary.roi_lift, 0.2)
    assert summary.best_avg_clv_vs_consensus_close > summary.consensus_avg_clv


def test_line_shopping_exposes_tied_best_sources():
    game = LineShoppingGame(
        game_pk=12,
        season=2025,
        model_prob_home=0.60,
        consensus_open_home=100,
        consensus_open_away=100,
        consensus_close_home=-110,
        consensus_close_away=-110,
        open_lines=(
            BookLine("book_a", home_ml=120, away_ml=-140),
            BookLine("book_b", home_ml=120, away_ml=-140),
        ),
        close_lines=(BookLine("book_b", home_ml=-110, away_ml=-110),),
        home_won=True,
    )

    _, bets = line_shop_moneyline([game], edge_threshold=0.02)

    assert bets[0].source_books == ("book_a", "book_b")
    assert bets[0].source_book == "book_b"


def test_line_shopping_reports_source_book_close_clv():
    summary, bets = line_shop_moneyline([_line_shop_game()], edge_threshold=0.02)

    assert bets[0].best_clv_vs_source_close is not None
    assert bets[0].best_beat_source_close is True
    assert summary.source_close_n == 1
    assert summary.best_avg_clv_vs_source_close > 0.0
    assert summary.best_pct_beat_source_close == 1.0


def test_line_shopping_skips_below_threshold():
    game = _line_shop_game()
    low_edge_game = LineShoppingGame(
        game_pk=game.game_pk,
        season=game.season,
        model_prob_home=0.51,
        consensus_open_home=game.consensus_open_home,
        consensus_open_away=game.consensus_open_away,
        consensus_close_home=game.consensus_close_home,
        consensus_close_away=game.consensus_close_away,
        open_lines=game.open_lines,
        close_lines=game.close_lines,
        home_won=game.home_won,
    )

    summary, bets = line_shop_moneyline([low_edge_game], edge_threshold=0.02)

    assert summary.n_bets == 0
    assert bets == []


def test_line_shopping_kelly_sizes_each_execution_price():
    summary, bets = line_shop_moneyline(
        [_line_shop_game()], edge_threshold=0.02, staking="kelly"
    )

    assert summary.n_bets == 1
    assert bets[0].best_stake > bets[0].consensus_stake
    assert 0.0 < bets[0].best_stake <= 0.05


def test_paper_trade_selects_best_price_and_kelly_stake():
    pick = select_moneyline_paper_trade(
        model_prob_home=0.60,
        odds_lines=(
            PaperOddsLine("book_a", home_ml=100, away_ml=100),
            PaperOddsLine("book_b", home_ml=120, away_ml=-140),
        ),
        edge_threshold=0.05,
        staking="kelly",
        bankroll_units=100.0,
    )

    assert pick is not None
    assert pick.side == "home"
    assert pick.best_books == ("book_b",)
    assert pick.best_ml == 120
    assert math.isclose(pick.best_decimal, 2.2)
    assert 0.0 < pick.stake_units <= 5.0


def test_paper_trade_flat_stake_and_tied_best_books():
    pick = select_moneyline_paper_trade(
        model_prob_home=0.60,
        odds_lines=(
            PaperOddsLine("book_a", home_ml=120, away_ml=-140),
            PaperOddsLine("book_b", home_ml=120, away_ml=-140),
        ),
        edge_threshold=0.05,
        staking="flat",
        bankroll_units=50.0,
        flat_stake_units=2.5,
    )

    assert pick is not None
    assert pick.best_books == ("book_a", "book_b")
    assert math.isclose(pick.stake_fraction, 0.05)
    assert math.isclose(pick.stake_units, 2.5)


def test_paper_trade_skips_when_edge_below_threshold():
    pick = select_moneyline_paper_trade(
        model_prob_home=0.51,
        odds_lines=(PaperOddsLine("book_a", home_ml=100, away_ml=100),),
        edge_threshold=0.05,
    )

    assert pick is None


def _paper_settlement_row(side: str = "home") -> dict[str, str]:
    return {
        "side": side,
        "best_decimal": "2.200000",
        "stake_units": "2.0000",
        "best_fair_prob": "0.500000",
        "status": "open",
    }


def test_moneyline_profit_settles_selected_side():
    assert math.isclose(
        moneyline_profit(
            side="away",
            best_decimal=2.5,
            stake_units=2.0,
            home_won=False,
        ),
        3.0,
    )
    assert math.isclose(
        moneyline_profit(
            side="away",
            best_decimal=2.5,
            stake_units=2.0,
            home_won=True,
        ),
        -2.0,
    )


def test_settle_paper_trade_row_fills_result_profit_and_clv():
    settled = settle_paper_trade_row(
        _paper_settlement_row(),
        home_won=True,
        close_home_ml=-120,
        close_away_ml=100,
    )

    assert settled["status"] == "settled"
    assert settled["result"] == "win"
    assert math.isclose(float(settled["profit_units"]), 2.4)
    assert settled["close_ml"] == "-120.0"
    assert float(settled["close_fair_prob"]) > 0.5
    assert float(settled["clv"]) > 0.0


def test_paper_trade_summary_aggregates_settled_rows():
    win = settle_paper_trade_row(
        _paper_settlement_row("home"),
        home_won=True,
        close_home_ml=-120,
        close_away_ml=100,
    )
    loss = settle_paper_trade_row(
        _paper_settlement_row("away"),
        home_won=True,
        close_home_ml=-120,
        close_away_ml=100,
    )

    summary = summarize_paper_trade_rows([win, loss, {"status": "open"}])

    assert summary.rows == 3
    assert summary.open_rows == 1
    assert summary.settled_rows == 2
    assert math.isclose(summary.total_staked, 4.0)
    assert math.isclose(summary.profit_units, 0.4)
    assert math.isclose(summary.roi, 0.1)
    assert summary.clv_rows == 2


def test_normalize_paper_trade_row_coerces_db_values():
    normalized = normalize_paper_trade_row(
        {
            "strategy_version": "strategy",
            "paper_date": "2026-08-14",
            "snapshot_time_utc": "2026-08-14T12:00:00+00:00",
            "game_pk": "123",
            "game_time": "",
            "away_team": "BAL",
            "home_team": "TB",
            "away_team_id": "110",
            "home_team_id": "139",
            "side": "away",
            "model_prob_home": "0.430000",
            "selected_model_prob": "0.570000",
            "edge": "0.070000",
            "best_books": "draftkings|fanduel",
            "best_ml": "154.0",
            "best_decimal": "2.540000",
            "best_fair_prob": "0.500000",
            "stake_units": "2.8647",
            "close_ml": "",
        }
    )

    assert normalized["game_pk"] == 123
    assert normalized["game_time"] is None
    assert normalized["away_team_id"] == 110
    assert normalized["best_books"] == ["draftkings", "fanduel"]
    assert math.isclose(normalized["best_ml"], 154.0)
    assert normalized["close_ml"] is None
    assert normalized["status"] == "open"
