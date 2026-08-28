"""Stable MLB team identity mapping for market-data ingestion."""

from __future__ import annotations

from src.database.postgres_handler import PostgresHandler


def team_abbrev_to_id() -> dict[str, int]:
    """Map MLB abbreviations and display aliases to stable team IDs."""
    handler = PostgresHandler()
    with handler.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT team_id, abbreviation, team_name, team_name_short, location_name
            FROM mlb.teams
            WHERE sport_id = 1
            """
        )
        rows = cursor.fetchall()
    mapping: dict[str, int] = {}
    for team_id, abbreviation, name, short_name, location in rows:
        for alias in (abbreviation, name, short_name, location):
            if alias:
                mapping[str(alias).strip().upper()] = int(team_id)
    return mapping
