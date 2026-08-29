"""Reliever deployment policy in the game simulator's pitching staff."""

from __future__ import annotations

from mlb.sim.game import Pitcher, _PitchingStaff

STARTER = Pitcher(1, "R")
AGG = Pitcher(-99, "R")
# closer, setup, middle (highest-leverage first)
POOL = (Pitcher(10, "R"), Pitcher(11, "L"), Pitcher(12, "R"))


def _staff(relievers=()) -> _PitchingStaff:
    return _PitchingStaff(STARTER, AGG, relievers)


def test_starter_pitches_until_hook():
    staff = _staff()
    assert staff.take(1, 9, 90) is STARTER
    assert staff.take(4, 9, 90) is STARTER
    staff.pitches = 90
    # over the pitch limit -> no longer the starter
    assert staff.take(5, 9, 90).player_id != STARTER.player_id


def test_aggregate_arm_when_no_individual_pool():
    staff = _staff()
    staff.pitches = 95
    assert staff.take(6, 9, 90) is AGG
    assert staff.take(7, 9, 90) is AGG  # stays on the aggregate arm


def test_closer_reserved_for_the_ninth():
    # Mid-game relief should not burn the closer...
    early = _staff(POOL)
    early.is_starter = False
    early._entered_inning = 0
    assert early.take(7, 9, 90).player_id != 10
    # ...but the ninth gets the top-leverage arm.
    late = _staff(POOL)
    late.is_starter = False
    late._entered_inning = 0
    assert late.take(9, 9, 90).player_id == 10


def test_escalates_toward_better_arms_late():
    staff = _staff(POOL)
    staff.is_starter = False
    staff._entered_inning = 0
    seventh = staff.take(7, 9, 90).player_id
    eighth = staff.take(8, 9, 90).player_id
    ninth = staff.take(9, 9, 90).player_id
    assert (seventh, eighth, ninth) == (12, 11, 10)


def test_reliever_outing_capped_then_aggregate():
    staff = _staff((Pitcher(10, "R"),))
    staff.is_starter = False
    staff._entered_inning = 0
    first = staff.take(3, 9, 90).player_id
    second = staff.take(4, 9, 90).player_id
    third = staff.take(5, 9, 90).player_id  # 3rd inning exceeds the 2-inning cap
    assert (first, second, third) == (10, 10, AGG.player_id)


def test_one_reliever_per_inning_no_midinning_churn():
    staff = _staff(POOL)
    staff.is_starter = False
    staff._entered_inning = 0
    first = staff.take(8, 9, 90)
    # same inning, another PA -> same arm
    assert staff.take(8, 9, 90) is first


def test_closer_only_in_save_spots():
    blowout = _staff(POOL)
    blowout.is_starter = False
    blowout._entered_inning = 0
    assert blowout.take(9, 9, 90, lead=10).player_id != 10  # not a save spot
    trailing = _staff(POOL)
    trailing.is_starter = False
    trailing._entered_inning = 0
    assert trailing.take(9, 9, 90, lead=-2).player_id != 10  # trailing
    save = _staff(POOL)
    save.is_starter = False
    save._entered_inning = 0
    assert save.take(9, 9, 90, lead=1).player_id == 10  # one-run save


def test_platoon_tiebreak_within_leverage_tier():
    # closer(R), two R middles, one L middle
    pool = (Pitcher(10, "R"), Pitcher(11, "R"), Pitcher(12, "R"), Pitcher(13, "L"))
    left = _staff(pool)
    left.is_starter = False
    left._entered_inning = 0
    left._reliever_innings[12] = 2  # exhaust the exact-target arm -> tier {1,3}
    assert left.take(7, 9, 90, lead=0, next_bat_side="L").player_id == 13
    right = _staff(pool)
    right.is_starter = False
    right._entered_inning = 0
    right._reliever_innings[12] = 2
    assert right.take(7, 9, 90, lead=0, next_bat_side="R").player_id == 11
