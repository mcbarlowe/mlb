from __future__ import annotations

from mlb.live.slate_sim_card import (
    SlateSimBoardData,
    SlateSimRow,
    board_columns,
    board_height,
    board_width,
    build_slate_sim_card_html,
)


def _row(game_pk: int, away: str, home: str, home_prob: float) -> SlateSimRow:
    return SlateSimRow(
        game_pk=game_pk,
        away_abbrev=away,
        home_abbrev=home,
        away_team_id=110,
        home_team_id=111,
        away_starter="Away Starter",
        home_starter="Home Starter",
        away_starter_id=None,
        home_starter_id=None,
        game_time="2026-08-09T18:35:00Z",
        venue="Fenway Park",
        home_win_probability=home_prob,
        mean_away_runs=3.6,
        mean_home_runs=4.2,
    )


def test_board_geometry_uses_narrow_layout_for_small_slates():
    assert board_columns(1) == 1
    assert board_width(1) == 900
    assert board_height(1) < board_height(3)
    assert board_height(0) == board_height(1)


def test_board_html_contains_matchups_probabilities_and_note():
    data = SlateSimBoardData(
        slate_date="2026-08-09",
        generated_at="2026-08-09 12:00 UTC",
        games_summary="2 preview games",
        n_sims=2000,
        rows=[_row(1, "BAL", "BOS", 0.62), _row(2, "NYY", "TOR", 0.41)],
        note="Monitoring preview games for probable-starter changes.",
    )

    html = build_slate_sim_card_html(data)

    assert "MLB <b>DAILY SIM BOARD</b>" in html
    assert "BAL" in html and "BOS" in html
    assert "NYY" in html and "TOR" in html
    assert "62%" in html
    assert "59%" in html  # away side for the second row
    assert "2,000 Monte Carlo sims per game" in html
    assert "Monitoring preview games for probable-starter changes." in html
    assert "Projected score <b>BAL 3.6</b> &ndash; <b>BOS 4.2</b>" in html
    assert "grid-template-columns: repeat(1, minmax(0, 1fr));" in html
    assert "width: 900px;" in html
    assert "2:35 PM ET · Fenway Park" in html
    assert "starter-fallback" in html
    assert "AS" in html and "HS" in html
