"""Tests for futures betting infrastructure."""


from src.market_data.futures_odds_store import (
    FUTURES_ODDS_DDL,
    insert_futures_odds,
)


def test_futures_odds_ddl_valid():
    """Futures odds DDL should be valid SQL."""
    assert "CREATE TABLE" in FUTURES_ODDS_DDL
    assert "futures_odds" in FUTURES_ODDS_DDL
    assert "season" in FUTURES_ODDS_DDL
    assert "market_type" in FUTURES_ODDS_DDL
    assert "team_id" in FUTURES_ODDS_DDL
    assert "bookmaker" in FUTURES_ODDS_DDL




def test_insert_futures_odds_empty_rows():
    """Inserting empty list should return 0."""
    from unittest.mock import MagicMock
    
    pg = MagicMock()
    result = insert_futures_odds(pg, [])
    assert result == 0




def test_fetch_futures_odds_normalizes_market_types():
    """Market type normalization should map common aliases."""
    from scripts.fetch_futures_odds import _normalize_market_type
    
    assert _normalize_market_type("World Series Winner") == "championship"
    assert _normalize_market_type("championship") == "championship"
    assert _normalize_market_type("to win world series") == "championship"
    assert _normalize_market_type("To Make Playoffs") == "playoff"
    assert _normalize_market_type("Division Winner") == "division"
    assert _normalize_market_type("h2h") is None  # Not a futures market


def test_fetch_futures_odds_normalizes_team_names():
    """Team name normalization should apply aliases."""
    from scripts.fetch_futures_odds import _normalize_team_name
    
    assert _normalize_team_name("Cleveland Indians") == "Cleveland Guardians"
    assert _normalize_team_name("Cleveland Guardians") == "Cleveland Guardians"
    assert _normalize_team_name("Los Angeles Dodgers") == "Los Angeles Dodgers"



def test_parse_futures_odds_valid_payload():
    """Parse should extract futures odds from API response."""
    from scripts.fetch_futures_odds import _parse_futures_odds
    
    payload = [
        {
            "id": "test-event-1",
            "sport_key": "baseball_mlb",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "outrights",
                            "outcomes": [
                                {
                                    "name": "Los Angeles Dodgers",
                                    "price": 500,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    team_map = {"Los Angeles Dodgers": 119}
    
    rows = _parse_futures_odds(
        payload,
        season=2027,
        snapshot_time="2027-01-01T00:00:00Z",
        team_id_map=team_map,
    )
    
    assert len(rows) == 1
    assert rows[0]["season"] == 2027
    assert rows[0]["market_type"] == "championship"
    assert rows[0]["team_id"] == 119
    assert rows[0]["bookmaker"] == "draftkings"
    assert rows[0]["american_odds"] == 500


def test_parse_futures_odds_skips_unknown_teams():
    """Parse should skip teams not in the mapping."""
    from scripts.fetch_futures_odds import _parse_futures_odds
    
    payload = [
        {
            "id": "test-event-1",
            "sport_key": "baseball_mlb",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "outrights",
                            "outcomes": [
                                {
                                    "name": "Unknown Team",
                                    "price": 500,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    team_map = {"Los Angeles Dodgers": 119}
    
    rows = _parse_futures_odds(
        payload,
        season=2027,
        snapshot_time="2027-01-01T00:00:00Z",
        team_id_map=team_map,
    )
    
    # Should skip unknown team
    assert len(rows) == 0


def test_parse_futures_odds_handles_multiple_bookmakers():
    """Parse should extract odds from all bookmakers."""
    from scripts.fetch_futures_odds import _parse_futures_odds
    
    payload = [
        {
            "id": "test-event-1",
            "sport_key": "baseball_mlb",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "outrights",
                            "outcomes": [
                                {"name": "Los Angeles Dodgers", "price": 500}
                            ]
                        }
                    ]
                },
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "outrights",
                            "outcomes": [
                                {"name": "Los Angeles Dodgers", "price": 450}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    team_map = {"Los Angeles Dodgers": 119}
    
    rows = _parse_futures_odds(
        payload,
        season=2027,
        snapshot_time="2027-01-01T00:00:00Z",
        team_id_map=team_map,
    )
    
    assert len(rows) == 2
    bookmakers = {row["bookmaker"] for row in rows}
    assert bookmakers == {"draftkings", "fanduel"}
