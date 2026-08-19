from __future__ import annotations

import csv
import math

import pytest

from src.betting.futures import (
    build_futures_edge_report,
    generate_futures_edge_report_csv,
)
from src.betting.futures_odds import FuturesOddsRow, parse_futures_odds_row
from src.betting.odds import american_to_prob, no_vig_outright


def _projection_row(
    team_id: int,
    abbreviation: str,
    team_name: str,
    championship_prob: float,
    *,
    input_market_sources: str = "",
    as_of_bucket: str = "opening_day",
) -> dict[str, str]:
    return {
        "projection_type": "model",
        "season": "2026",
        "as_of_bucket": as_of_bucket,
        "team_id": str(team_id),
        "abbreviation": abbreviation,
        "team_name": team_name,
        "league_name": "NL",
        "division_name": "NL West",
        "division_win_prob": "0.25",
        "playoff_prob": "0.50",
        "division_series_prob": "0.30",
        "league_championship_prob": "0.20",
        "world_series_prob": "0.12",
        "championship_prob": str(championship_prob),
        "input_market_sources": input_market_sources,
    }


def test_no_vig_outright_scales_multi_runner_market_to_target_total():
    no_vig = no_vig_outright([-110, -110, 400], target_total=1.0)

    assert math.isclose(sum(no_vig), 1.0, abs_tol=1e-12)
    assert math.isclose(no_vig[0], no_vig[1], abs_tol=1e-12)
    assert no_vig[0] > no_vig[2]


def test_parse_futures_odds_row_accepts_team_labels_and_implied_probability():
    row = parse_futures_odds_row(
        {
            "market": "World Series Winner",
            "team": "Los Angeles Dodgers",
            "implied_probability": "12.5%",
            "book": "example_book",
            "market_scope": "MLB",
        }
    )

    assert row.market_type == "championship"
    assert row.team_label == "Los Angeles Dodgers"
    assert row.bookmaker == "example_book"
    assert math.isclose(row.implied_probability, 0.125)
    assert row.american_odds is None


def test_parse_futures_odds_row_uses_american_odds_when_probability_absent():
    row = parse_futures_odds_row(
        {"market_type": "championship", "team_id": "119", "american_odds": "+500"}
    )

    assert row.team_id == 119
    assert row.american_odds == 500.0
    assert math.isclose(row.implied_probability, american_to_prob(500))


def test_build_futures_edge_report_joins_projection_and_prices_with_kelly_stake():
    projections = [
        _projection_row(1, "LAD", "Los Angeles Dodgers", 0.22),
        _projection_row(2, "SDP", "San Diego Padres", 0.10),
    ]
    odds = [
        FuturesOddsRow(
            market_type="championship",
            team_label="Los Angeles Dodgers",
            american_odds=1000,
            implied_probability=american_to_prob(1000),
            bookmaker="book_a",
            season="2026",
        ),
        FuturesOddsRow(
            market_type="championship",
            team_id=2,
            american_odds=100,
            implied_probability=american_to_prob(100),
            bookmaker="book_a",
            season="2026",
        ),
    ]

    report = build_futures_edge_report(
        projections,
        odds,
        markets=["championship"],
        kelly_multiplier=0.5,
        kelly_cap=0.05,
    )
    dodgers = next(row for row in report if row.team_id == 1)

    expected_no_vig = american_to_prob(1000) / (
        american_to_prob(1000) + american_to_prob(100)
    )
    assert math.isclose(dodgers.no_vig_market_probability, expected_no_vig)
    assert math.isclose(dodgers.model_probability, 0.22)
    assert math.isclose(dodgers.edge, 0.22 - expected_no_vig)
    assert math.isclose(dodgers.offered_decimal_payout, 11.0)
    assert math.isclose(dodgers.unit_ev, 0.22 * 11.0 - 1.0)
    assert math.isclose(dodgers.stake_units or 0.0, 0.05)


def test_futures_report_rejects_target_market_source_leakage():
    projections = [
        _projection_row(
            1,
            "LAD",
            "Los Angeles Dodgers",
            0.22,
            input_market_sources="team_priors; futures_championship",
        ),
        _projection_row(
            2,
            "SDP",
            "San Diego Padres",
            0.10,
            input_market_sources="team_priors; futures_championship",
        ),
    ]
    odds = [
        FuturesOddsRow(
            market_type="championship",
            team_id=1,
            american_odds=1000,
            implied_probability=american_to_prob(1000),
        ),
        FuturesOddsRow(
            market_type="championship",
            team_id=2,
            american_odds=100,
            implied_probability=american_to_prob(100),
        ),
    ]

    with pytest.raises(ValueError, match="input_market_sources"):
        build_futures_edge_report(projections, odds, markets=["championship"])

    report = build_futures_edge_report(
        projections,
        odds,
        markets=["championship"],
        allow_market_source_leakage=True,
    )
    assert len(report) == 2


def test_generate_futures_edge_report_csv_filters_as_of_bucket(tmp_path):
    projection_path = tmp_path / "projections.csv"
    odds_path = tmp_path / "odds.csv"
    out_path = tmp_path / "report.csv"

    projection_rows = [
        _projection_row(1, "LAD", "Los Angeles Dodgers", 0.25, as_of_bucket="opening_day"),
        _projection_row(1, "LAD", "Los Angeles Dodgers", 0.35, as_of_bucket="30_games"),
        _projection_row(2, "SDP", "San Diego Padres", 0.10, as_of_bucket="opening_day"),
        _projection_row(2, "SDP", "San Diego Padres", 0.15, as_of_bucket="30_games"),
    ]
    with projection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projection_rows[0]))
        writer.writeheader()
        writer.writerows(projection_rows)
    with odds_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["market_type", "team_id", "american_odds", "market_scope"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "market_type": "championship",
                    "team_id": "1",
                    "american_odds": "500",
                    "market_scope": "MLB",
                },
                {
                    "market_type": "championship",
                    "team_id": "2",
                    "american_odds": "800",
                    "market_scope": "MLB",
                },
            ]
        )

    report = generate_futures_edge_report_csv(
        projection_csv=projection_path,
        odds_csv=odds_path,
        out_csv=out_path,
        as_of_bucket="30_games",
    )

    assert len(report) == 2
    assert {row.model_probability for row in report} == {0.35, 0.15}
    with out_path.open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["model_probability"] in {"0.35", "0.15"}
