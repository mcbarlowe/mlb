from __future__ import annotations

from datetime import date

from src.sim.projection_charts import write_projection_graphics
from src.sim.season import SeasonProjection, TeamInfo, TeamProjection


def _team(team_id: int, abbreviation: str) -> TeamInfo:
    return TeamInfo(
        team_id=team_id,
        abbreviation=abbreviation,
        team_name=f"Team {abbreviation}",
        league_name="AL",
        division_name="East",
    )


def test_write_projection_graphics_creates_playoff_jpegs(tmp_path):
    teams = {
        1: _team(1, "AAA"),
        2: _team(2, "BBB"),
    }
    projection = SeasonProjection(
        season=2026,
        as_of_date=date(2026, 3, 29),
        trials=100,
        wild_cards_per_league=3,
        teams=(
            TeamProjection(1, 10, 91.2, 0.40, 0.82, 0.70, 0.42, 0.20, 0.11),
            TeamProjection(2, 8, 84.3, 0.20, 0.54, 0.31, 0.14, 0.06, 0.02),
        ),
    )

    playoff_path, stages_path = write_projection_graphics(
        projection,
        teams,
        tmp_path,
        projection_type="model",
    )

    assert playoff_path.name == "season_2026_model_playoff_probabilities.jpg"
    assert stages_path.name == "season_2026_model_playoff_stages.jpg"
    assert playoff_path.stat().st_size > 0
    assert stages_path.stat().st_size > 0
