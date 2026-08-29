"""Extract simulation lineups from archived GUMBO feeds."""

from __future__ import annotations

from mlb.sim.game import Batter, Lineup, Pitcher


def starting_batter_ids_from_feed(feed: dict, side: str) -> list[int]:
    """Return the nine original starters, excluding later substitutions."""
    box = feed["liveData"]["boxscore"]["teams"][side]
    starters: list[tuple[int, int]] = []
    ordered_entries = 0
    for entry in box.get("players", {}).values():
        raw_order = entry.get("battingOrder")
        if raw_order is None:
            continue
        ordered_entries += 1
        batting_order = int(raw_order)
        if batting_order % 100 != 0:
            continue
        person = entry.get("person", {})
        if person.get("id") is not None:
            starters.append((batting_order // 100, int(person["id"])))
    starters.sort()
    if len(starters) == 9:
        return [player_id for _, player_id in starters]
    if ordered_entries:
        raise ValueError(f"Feed has an incomplete {side} starting lineup")
    order = [int(player_id) for player_id in box.get("battingOrder", [])]
    if len(order) < 9:
        raise ValueError(f"Feed has no usable {side} lineup")
    return order[:9]


def starting_batters_from_feed(feed: dict, side: str) -> list[Batter]:
    players = feed["gameData"]["players"]
    return [
        Batter(player_id, players[f"ID{player_id}"]["batSide"]["code"])
        for player_id in starting_batter_ids_from_feed(feed, side)
    ]


def lineup_from_feed(feed: dict, side: str) -> Lineup:
    """Batting order + starting pitcher with handedness for one side."""
    from mlb.sim.bullpen import bullpen_for_team

    box = feed["liveData"]["boxscore"]["teams"][side]
    players = feed["gameData"]["players"]
    batters = starting_batters_from_feed(feed, side)
    pitchers = box.get("pitchers", [])
    if not pitchers:
        raise ValueError(f"Feed has no usable {side} starting pitcher")
    starter_id = pitchers[0]
    starter = Pitcher(starter_id, players[f"ID{starter_id}"]["pitchHand"]["code"])
    team_id = feed["gameData"]["teams"][side].get("id")
    return Lineup(batters=batters, starter=starter, bullpen=bullpen_for_team(team_id))


def describe_game(feed: dict) -> str:
    teams = feed["gameData"]["teams"]
    return f"{teams['away']['abbreviation']} @ {teams['home']['abbreviation']}"


def actual_final(feed: dict) -> tuple[int, int]:
    """(away, home) final score from the linescore."""
    lines = feed["liveData"]["linescore"]["teams"]
    return int(lines["away"].get("runs", 0)), int(lines["home"].get("runs", 0))
