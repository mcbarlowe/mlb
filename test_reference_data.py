from dataclasses import replace

from mlb.data.reference_data import ReferenceData
from mlb.database import PostgresConfig, PostgresHandler

TEST_SCHEMA = "mlb_test_reference_data"


def test_reference_data_save_to_db_normalizes_reference_columns():
    db_config = replace(PostgresConfig.from_env(), schema=TEST_SCHEMA)

    positions_df = ReferenceData().transform(
        [
            {
                "shortName": "Designated Hitter",
                "fullName": "Designated Hitter",
                "abbrev": "DH",
                "code": "10",
                "type": "Hitter",
                "formalName": "Designated Hitter",
                "displayName": "Designated Hitter",
            },
            {
                "shortName": "Batter",
                "fullName": "Batter",
                "abbrev": "B",
                "code": "10",
                "type": "Batter",
                "formalName": "Batter",
                "displayName": "Batter",
            },
        ]
    )
    event_types_df = ReferenceData().transform(
        [
            {
                "code": "strikeout",
                "description": "Strikeout",
                "plateAppearance": True,
                "hit": False,
                "baseRunningEvent": False,
            }
        ]
    )

    with PostgresHandler(db_config) as db:
        db.reset_schema()
        db.create_reference_tables()
        db.connection.execute(
            """
            INSERT INTO positions (code, name, type, abbreviation)
            VALUES ('stale', 'stale', 'stale', 'S')
            """
        )

        ref = ReferenceData()
        ref.save_to_db(positions_df, "positions", db, if_exists="replace")
        ref.save_to_db(event_types_df, "event_types", db, if_exists="replace")

        positions_rows = db.query(
            "SELECT code, name, type, abbreviation FROM positions ORDER BY code"
        )
        event_type_rows = db.query(
            "SELECT code, description FROM event_types ORDER BY code"
        )

    assert positions_rows.to_dict("records") == [
        {
            "code": "10",
            "name": "Designated Hitter",
            "type": "Hitter",
            "abbreviation": "DH",
        }
    ]
    assert event_type_rows.to_dict("records") == [
        {
            "code": "strikeout",
            "description": "Strikeout",
        }
    ]
