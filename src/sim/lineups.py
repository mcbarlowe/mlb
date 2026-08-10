"""Extract simulation lineups from archived GUMBO feeds."""

from __future__ import annotations

from src.sim.game import Batter, Lineup, Pitcher


def lineup_from_feed(feed: dict, side: str) -> Lineup:
    """Batting order + starting pitcher with handedness for one side."""
    from src.sim.bullpen import bullpen_for_team

    box = feed["liveData"]["boxscore"]["teams"][side]
    players = feed["gameData"]["players"]
    order = box.get("battingOrder", [])
    pitchers = box.get("pitchers", [])
    if len(order) < 9 or not pitchers:
        raise ValueError(f"Feed has no usable {side} lineup")
    batters = [
        Batter(pid, players[f"ID{pid}"]["batSide"]["code"]) for pid in order[:9]
    ]
    starter_id = pitchers[0]
    starter = Pitcher(starter_id, players[f"ID{starter_id}"]["pitchHand"]["code"])
    team_id = feed["gameData"]["teams"][side].get("id")
    return Lineup(
        batters=batters, starter=starter, bullpen=bullpen_for_team(team_id)
    )


def describe_game(feed: dict) -> str:
    teams = feed["gameData"]["teams"]
    return f"{teams['away']['abbreviation']} @ {teams['home']['abbreviation']}"


def actual_final(feed: dict) -> tuple[int, int]:
    """(away, home) final score from the linescore."""
    lines = feed["liveData"]["linescore"]["teams"]
    return int(lines["away"].get("runs", 0)), int(lines["home"].get("runs", 0))
