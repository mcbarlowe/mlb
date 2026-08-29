import pandas as pd

from mlb.data.game_data import GameData
from mlb.data.game_feed_data import GameFeedData
from mlb.data.player_data import PlayerData
from mlb.data.team_data import TeamData
from mlb.data.venue_data import VenueData


def test_game_feed_data_handles_missing_result_keys_and_empty_pitch_events():
    payload = {
        "gameData": {
            "game": {"pk": 1, "season": "2009", "type": "R", "gameNumber": 1},
            "datetime": {"dateTime": "2009-04-01T00:00:00Z", "dayNight": "night"},
            "teams": {
                "away": {"id": 1, "name": "Away"},
                "home": {"id": 2, "name": "Home"},
            },
            "venue": {"id": 10, "name": "Venue"},
            "weather": {},
        },
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "result": {"type": "atBat", "rbi": 0, "awayScore": 0, "homeScore": 0, "isOut": False},
                        "about": {"atBatIndex": 0, "halfInning": "top", "inning": 1},
                        "matchup": {
                            "batter": {"id": 101, "fullName": "Batter"},
                            "batSide": {"code": "R"},
                            "pitcher": {"id": 202, "fullName": "Pitcher"},
                            "pitchHand": {"code": "L"},
                        },
                    }
                ]
            }
        },
    }

    df = GameFeedData().transform(payload, game_id=1, season=2009)

    assert df.empty
    assert set(df.columns) == set(GameFeedData().data_types.keys())



def test_team_data_allows_missing_nullable_integer_fields():
    payload = {
        "gameData": {
            "teams": {
                "away": {
                    "id": 1,
                    "name": "Away",
                    "league": {"id": 103, "name": "AL"},
                    "division": {"id": None, "name": None},
                    "sport": {"id": 1, "name": "MLB"},
                    "venue": {"id": 10, "name": "Venue"},
                },
                "home": {
                    "id": 2,
                    "name": "Home",
                    "league": {"id": 104, "name": "NL"},
                    "division": {"id": None, "name": None},
                    "sport": {"id": 1, "name": "MLB"},
                    "venue": {"id": 10, "name": "Venue"},
                },
            }
        }
    }

    df = TeamData().transform(payload)

    assert len(df) == 2
    assert df["division_id"].isna().all()



def test_player_data_allows_missing_nullable_integer_fields():
    payload = {
        "gameData": {
            "players": {
                "ID123": {
                    "id": 123,
                    "fullName": "Player",
                    "currentAge": None,
                    "weight": None,
                    "active": True,
                    "primaryPosition": {},
                    "batSide": {},
                    "pitchHand": {},
                }
            }
        }
    }

    df = PlayerData().transform(payload)

    assert len(df) == 1
    assert pd.isna(df.loc[0, "current_age"])
    assert pd.isna(df.loc[0, "weight"])



def test_game_and_venue_data_allow_missing_nullable_fields():
    payload = {
        "gameData": {
            "game": {"pk": 1, "id": "gid", "season": "2009"},
            "datetime": {},
            "status": {},
            "venue": {"id": None, "name": "Unknown Venue", "location": {}, "timeZone": {}, "fieldInfo": {}},
            "weather": {},
            "gameInfo": {},
            "teams": {"away": {"id": 1, "record": {}}, "home": {"id": 2, "record": {}}},
            "probablePitchers": {},
            "review": {},
            "flags": {},
        }
    }

    game_df = GameData().transform(payload)
    venue_df = VenueData().transform(payload)

    assert len(game_df) == 1
    assert pd.isna(game_df.loc[0, "venue_id"])
    assert len(venue_df) == 1
    assert pd.isna(venue_df.loc[0, "venue_id"])
