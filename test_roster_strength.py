from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import src.sim.team_strength as team_strength_module
from src.sim.roster_strength import (
    BatterGameLine,
    RelieverGameLine,
    RosterFeatureBuilder,
)
from src.sim.team_strength import load_completed_games


def batter(
    player_id: int,
    *,
    batting_order: int = 100,
    hits: int = 1,
    doubles: int = 0,
    triples: int = 0,
    home_runs: int = 0,
    walks: int = 0,
    birth_date: str | None = None,
) -> BatterGameLine:
    return BatterGameLine(
        player_id=player_id,
        batting_order=batting_order,
        at_bats=4,
        hits=hits,
        doubles=doubles,
        triples=triples,
        home_runs=home_runs,
        walks=walks,
        intentional_walks=0,
        hit_by_pitch=0,
        sacrifice_flies=0,
        birth_date=birth_date,
    )


def reliever(
    player_id: int,
    *,
    outs: int = 3,
    strikeouts: int = 1,
    walks: int = 0,
    home_runs: int = 0,
    pitches: int = 15,
) -> RelieverGameLine:
    return RelieverGameLine(
        player_id=player_id,
        outs=outs,
        strikeouts=strikeouts,
        walks=walks,
        home_runs=home_runs,
        hit_batters=0,
        pitches=pitches,
    )


def matchup(
    builder: RosterFeatureBuilder,
    as_of: date,
    *,
    away_batter_ids: tuple[int, ...] = (10,),
    home_batter_ids: tuple[int, ...] = (20,),
    away_reliever_ids: tuple[int, ...] = (100,),
    home_reliever_ids: tuple[int, ...] = (200,),
):
    return builder.matchup_features(
        season=as_of.year,
        prediction_date=as_of,
        away_team_id=1,
        home_team_id=2,
        away_starter_id=1000,
        home_starter_id=2000,
        away_batter_ids=away_batter_ids,
        home_batter_ids=home_batter_ids,
        away_reliever_ids=away_reliever_ids,
        home_reliever_ids=home_reliever_ids,
    )


def test_current_game_player_results_do_not_leak_into_pregame_features() -> None:
    game_date = date(2025, 4, 1)
    builder = RosterFeatureBuilder()

    first = builder.observe(
        season=2025,
        game_date=game_date,
        away_team_id=1,
        home_team_id=2,
        away_starter_id=1000,
        home_starter_id=2000,
        away_batters=(batter(10, hits=0),),
        home_batters=(batter(20, hits=4, home_runs=2),),
        away_relievers=(reliever(100, strikeouts=0, walks=3, home_runs=1),),
        home_relievers=(reliever(200, strikeouts=3),),
    )

    assert first.lineup_woba_edge == pytest.approx(0.0)
    assert first.bullpen_fip_edge == pytest.approx(0.0)
    assert first.bullpen_availability_edge == pytest.approx(0.0)

    second = matchup(builder, game_date + timedelta(days=1))
    assert second.lineup_woba_edge > 0.0
    assert second.bullpen_fip_edge > 0.0


def test_batter_projection_decays_toward_prior_with_inactivity() -> None:
    observed_on = date(2025, 4, 1)
    builder = RosterFeatureBuilder()
    builder.update(
        season=2025,
        game_date=observed_on,
        away_team_id=1,
        home_team_id=2,
        away_batters=(batter(10, hits=0),),
        home_batters=(batter(20, hits=4, home_runs=2),),
        away_relievers=(),
        home_relievers=(),
    )

    recent = matchup(builder, observed_on + timedelta(days=1))
    stale = matchup(builder, observed_on + timedelta(days=180))

    assert recent.lineup_woba_edge > 0.0
    assert 0.0 < stale.lineup_woba_edge < recent.lineup_woba_edge


def test_age_curve_regresses_old_batter_below_young_batter() -> None:
    observed_on = date(2025, 4, 1)
    builder = RosterFeatureBuilder()
    builder.update(
        season=2025,
        game_date=observed_on,
        away_team_id=1,
        home_team_id=2,
        away_batters=(batter(10, hits=2, birth_date="2002-04-01"),),
        home_batters=(batter(20, hits=2, birth_date="1988-04-01"),),
        away_relievers=(),
        home_relievers=(),
    )

    features = matchup(builder, date(2026, 4, 1))

    assert features.lineup_woba_edge < 0.0
    builder.update(
        season=2026,
        game_date=date(2026, 4, 1),
        away_team_id=1,
        home_team_id=2,
        away_batters=(batter(10, hits=2, birth_date="2002-04-01"),),
        home_batters=(batter(20, hits=2, birth_date="1988-04-01"),),
        away_relievers=(),
        home_relievers=(),
    )
    after_new_observations = matchup(builder, date(2026, 4, 2))

    assert after_new_observations.lineup_woba_edge < 0.0


def test_active_roster_replaces_inactive_lineup_members_with_rookie_prior() -> None:
    observed_on = date(2025, 4, 1)
    builder = RosterFeatureBuilder()
    builder.update(
        season=2025,
        game_date=observed_on,
        away_team_id=1,
        home_team_id=2,
        away_batters=(batter(10, hits=1),),
        home_batters=(batter(20, hits=4, home_runs=2),),
        away_relievers=(),
        home_relievers=(),
    )

    with_inactive_star = builder.matchup_features(
        season=2025,
        prediction_date=observed_on + timedelta(days=1),
        away_team_id=1,
        home_team_id=2,
        away_starter_id=1000,
        home_starter_id=2000,
        away_batter_ids=(10,),
        home_batter_ids=(20,),
        home_active_batter_ids=(21,),
        away_active_batter_ids=(11,),
    )

    assert with_inactive_star.lineup_woba_edge == pytest.approx(0.0, abs=0.001)


def test_recent_workload_penalizes_home_bullpen_availability() -> None:
    observed_on = date(2025, 4, 1)
    builder = RosterFeatureBuilder()
    builder.update(
        season=2025,
        game_date=observed_on,
        away_team_id=1,
        home_team_id=2,
        away_batters=(),
        home_batters=(),
        away_relievers=(reliever(100, pitches=0),),
        home_relievers=(reliever(200, pitches=70),),
    )

    features = matchup(builder, observed_on + timedelta(days=1))

    assert features.bullpen_fip_edge == pytest.approx(0.0)
    assert features.bullpen_availability_edge < 0.0


def test_active_rotation_pitcher_is_excluded_from_bullpen_projection() -> None:
    observed_on = date(2025, 4, 1)
    builder = RosterFeatureBuilder()
    builder.update(
        season=2025,
        game_date=observed_on,
        away_team_id=1,
        home_team_id=2,
        away_batters=(),
        home_batters=(),
        away_relievers=(reliever(100),),
        home_relievers=(
            reliever(200, walks=2, home_runs=1),
            reliever(300, strikeouts=3),
        ),
    )
    builder.update(
        season=2025,
        game_date=observed_on + timedelta(days=1),
        away_team_id=1,
        home_team_id=2,
        away_batters=(),
        home_batters=(),
        away_relievers=(reliever(100),),
        home_relievers=(reliever(200, walks=2, home_runs=1),),
        home_starter_id=300,
    )

    with_rotation_arm = matchup(
        builder,
        observed_on + timedelta(days=2),
        home_reliever_ids=(200, 300),
    )
    reliever_only = matchup(
        builder,
        observed_on + timedelta(days=2),
        home_reliever_ids=(200,),
    )

    assert with_rotation_arm.bullpen_fip_edge == pytest.approx(
        reliever_only.bullpen_fip_edge
    )
    assert with_rotation_arm.bullpen_availability_edge == pytest.approx(
        reliever_only.bullpen_availability_edge
    )


def test_reliever_trade_removes_pitcher_from_former_team_fallback() -> None:
    builder = RosterFeatureBuilder()
    first_date = date(2025, 4, 1)
    builder.update(
        season=2025,
        game_date=first_date,
        away_team_id=1,
        home_team_id=3,
        away_batters=(),
        home_batters=(),
        away_relievers=(reliever(100, strikeouts=3),),
        home_relievers=(),
    )
    builder.update(
        season=2025,
        game_date=first_date + timedelta(days=1),
        away_team_id=3,
        home_team_id=2,
        away_batters=(),
        home_batters=(),
        away_relievers=(),
        home_relievers=(reliever(100, strikeouts=3),),
    )

    features = builder.matchup_features(
        season=2025,
        prediction_date=first_date + timedelta(days=2),
        away_team_id=1,
        home_team_id=2,
        away_starter_id=1000,
        home_starter_id=2000,
    )

    assert features.bullpen_fip_edge > 0.0


def test_completed_game_loader_attaches_batter_and_reliever_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_frame = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "season": 2025,
                "game_datetime": "2025-04-01T17:00:00Z",
                "away_team_id": 10,
                "home_team_id": 20,
                "away_runs": 2,
                "home_runs": 3,
                "away_id": 1000,
                "away_outs": 18,
                "away_er": 2,
                "away_k": 6,
                "away_bb": 1,
                "away_hr": 1,
                "away_hbp": 0,
                "home_id": 2000,
                "home_outs": 18,
                "home_er": 2,
                "home_k": 6,
                "home_bb": 1,
                "home_hr": 1,
                "home_hbp": 0,
            }
        ]
    )
    batting_frame = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "team_type": "away",
                "player_id": 10,
                "batting_order": 100,
                "at_bats": 4,
                "hits": 2,
                "doubles": 1,
                "triples": 0,
                "home_runs": 0,
                "walks": 1,
                "intentional_walks": 0,
                "hit_by_pitch": 0,
                "sacrifice_flies": 0,
                "birth_date": "1998-01-02",
            }
        ]
    )
    reliever_frame = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "team_type": "home",
                "player_id": 200,
                "outs": 3,
                "strikeouts": 2,
                "walks": 0,
                "home_runs": 0,
                "hit_batters": 0,
                "pitches": 14,
            }
        ]
    )

    class StubPostgresHandler:
        def __init__(self, _config: object) -> None:
            self._frames = iter((game_frame, batting_frame, reliever_frame))

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def query(self, _query: str) -> pd.DataFrame:
            return next(self._frames)

    monkeypatch.setattr(
        team_strength_module,
        "PostgresHandler",
        StubPostgresHandler,
    )

    games = load_completed_games(start_season=2025, end_season=2025)

    assert len(games) == 1
    assert games[0].away_batters[0].player_id == 10
    assert games[0].away_batters[0].birth_date == "1998-01-02"
    assert games[0].home_relievers[0].player_id == 200
    assert games[0].home_relievers[0].pitches == 14
