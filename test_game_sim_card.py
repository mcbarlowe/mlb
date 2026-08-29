from __future__ import annotations

import pytest

from mlb.live.game_sim_card import (
    build_game_sim_card_html,
    card_data_from_results,
)
from mlb.sim.game import GameResult


def _results():
    # 6 home wins (4-2), 3 away wins (5-1), 1 tie -> p(home) = 6.5/10
    results = [GameResult(away_runs=2, home_runs=4, innings=9)] * 6
    results += [GameResult(away_runs=5, home_runs=1, innings=9)] * 3
    results += [GameResult(away_runs=3, home_runs=3, innings=12, tie=True)]
    return results


def _card_data():
    return card_data_from_results(
        _results(),
        away_abbrev="HOU",
        home_abbrev="LAA",
        away_team_id=117,
        home_team_id=108,
        away_starter="Away Starter",
        home_starter="Home Starter",
        game_date="2025-09-28",
        venue="Angel Stadium",
    )


def test_card_data_aggregates_probabilities_and_scores():
    data = _card_data()
    assert data.n_sims == 10
    assert data.home_win_probability == pytest.approx(0.65)
    assert data.top_scores[0] == (2, 4, pytest.approx(0.6))
    assert data.mean_away_runs == pytest.approx((2 * 6 + 5 * 3 + 3) / 10)
    # Total-run buckets sum to 1.
    assert sum(p for _, p in data.total_runs) == pytest.approx(1.0)


def test_card_data_rejects_empty_results():
    with pytest.raises(ValueError):
        card_data_from_results(
            [],
            away_abbrev="A",
            home_abbrev="B",
            away_team_id=None,
            home_team_id=None,
            away_starter="x",
            home_starter="y",
            game_date="2025-01-01",
            venue=None,
        )


def test_card_html_contains_teams_probability_and_winner_first_scores():
    html = build_game_sim_card_html(_card_data())
    assert "HOU" in html and "LAA" in html
    assert "65%" in html  # home side
    assert "35%" in html  # away side
    # Winner-first score ordering: home won 4-2, away won 5-1.
    assert "4&ndash;2" in html
    assert "5&ndash;1" in html
    assert "2&ndash;4" not in html
    assert "GAME <b>SIMULATION</b>" in html
    assert "2,000" not in html  # sims count comes from data (10 here)
    assert "10 SIMS" in html
