import re
from pathlib import Path

from src.data_contracts.result_views import RESULT_VIEW_SQL

SCHEMA_SQL = (
    Path(__file__).parent / "src" / "database" / "schema.sql"
).read_text()


def _view_definition(name: str) -> str:
    match = re.search(
        rf"CREATE OR REPLACE VIEW {name} AS\s+(.*?);",
        SCHEMA_SQL,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _projection_names(view_sql: str) -> list[str]:
    projection = view_sql.split("FROM", maxsplit=1)[0].removeprefix("SELECT")
    names = []
    for expression in projection.split(","):
        expression = expression.strip()
        name = (
            expression.rsplit(" AS ", maxsplit=1)[-1]
            .rsplit(".", maxsplit=1)[-1]
        )
        names.append(name)
    return names


def test_game_results_v1_contract_uses_linescore_totals():
    view_sql = _view_definition("betting_game_results_v1")

    assert _projection_names(view_sql) == [
        "game_pk",
        "game_date",
        "abstract_game_state",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    ]
    assert (
        "SUM(runs) FILTER (WHERE team_type = 'home')::integer AS home_score"
        in view_sql
    )
    assert (
        "SUM(runs) FILTER (WHERE team_type = 'away')::integer AS away_score"
        in view_sql
    )
    assert "LEFT JOIN" in view_sql


def test_player_results_v1_contract_preserves_explicit_dnp_semantics():
    view_sql = _view_definition("betting_player_results_v1")

    assert _projection_names(view_sql) == [
        "game_pk",
        "game_date",
        "abstract_game_state",
        "player_id",
        "player_name",
        "appeared",
        "hits",
        "doubles",
        "triples",
        "home_runs",
        "total_bases",
        "rbi",
        "runs",
        "walks",
        "stolen_bases",
        "strikeouts",
    ]
    assert "WHEN b.gamesplayed = 0 THEN FALSE" in view_sql
    assert "WHEN b.gamesplayed > 0 THEN TRUE" in view_sql
    assert "ELSE NULL" in view_sql
    assert "b.homeruns AS home_runs" in view_sql
    assert "b.totalbases AS total_bases" in view_sql
    assert "b.baseonballs AS walks" in view_sql
    assert "b.stolenbases AS stolen_bases" in view_sql
    assert (
        "False is emitted only when the source explicitly stores gamesplayed=0"
        in SCHEMA_SQL
    )
    assert "null means appearance is unknown and must not void a bet" in SCHEMA_SQL


def test_existing_database_installer_uses_schema_qualified_additive_views():
    assert "CREATE OR REPLACE VIEW mlb.betting_game_results_v1" in RESULT_VIEW_SQL
    assert "CREATE OR REPLACE VIEW mlb.betting_player_results_v1" in RESULT_VIEW_SQL
    assert "FROM mlb.games AS g" in RESULT_VIEW_SQL
    assert "FROM mlb.batting AS b" in RESULT_VIEW_SQL
    assert "DROP " not in RESULT_VIEW_SQL.upper()
