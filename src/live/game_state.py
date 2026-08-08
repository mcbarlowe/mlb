"""Live-feed parsing for next-pitch prediction.

Builds a model-ready at-bat frame for the *upcoming* pitch from a live
GUMBO feed: all completed pitches in the game (for cumulative context)
plus one synthetic "pending" row representing the pitch that has not yet
been thrown.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.data.game_feed_data import GameFeedData
from src.etl.daily_pipeline import GameState, extract_game_status
from src.ml.pitch_predictor import GameContext

_transformer = GameFeedData()


@dataclass(frozen=True)
class LiveSnapshot:
    """State of one live game at the moment before the next pitch."""

    game_pk: int
    state: GameState
    at_bat_index: int
    next_pitch_number: int
    balls: int
    strikes: int
    outs: int
    pitch_key: tuple[int, int, int, int]
    frame: pl.DataFrame
    context: GameContext
    inning_key: tuple[str, int]


def _team_label(team_node: dict) -> str:
    return (
        team_node.get("abbreviation")
        or team_node.get("teamName")
        or team_node.get("name")
        or "?"
    )


def _parse_temperature(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _runners_from_linescore(linescore: dict) -> tuple[bool, bool, bool]:
    offense = linescore.get("offense", {})
    return ("first" in offense, "second" in offense, "third" in offense)


def build_pending_row(
    feed: dict,
    game_info: dict,
    completed: pl.DataFrame,
) -> dict:
    """Build the raw-column row for the upcoming (not yet thrown) pitch."""
    live = feed.get("liveData", {})
    current_play = live.get("plays", {}).get("currentPlay", {})
    linescore = live.get("linescore", {})
    matchup = current_play.get("matchup", {})
    about = current_play.get("about", {})
    count = current_play.get("count", {})

    at_bat_index = about.get("atBatIndex", 0)
    on_first, on_second, on_third = _runners_from_linescore(linescore)

    ab_pitches = (
        completed.filter(pl.col("at_bat_index") == at_bat_index)
        if not completed.is_empty()
        else completed
    )
    if ab_pitches.is_empty():
        next_pitch_number = 1
        last_pitch_speed = None
    else:
        ordered = ab_pitches.sort("pitch_number")
        max_pitch_number = ordered["pitch_number"].max()
        next_pitch_number = int(str(max_pitch_number)) + 1
        raw_speed = ordered["pitch_start_speed"][-1]
        last_pitch_speed = float(raw_speed) if raw_speed is not None else None

    teams = linescore.get("teams", {})
    home_score = teams.get("home", {}).get("runs", 0) or 0
    away_score = teams.get("away", {}).get("runs", 0) or 0

    balls = int(count.get("balls", 0) or 0)
    strikes = int(count.get("strikes", 0) or 0)
    outs = int(count.get("outs", 0) or 0)

    return {
        # Game-level context
        "game_pk": game_info.get("game_pk"),
        "season": game_info.get("season"),
        "game_date": game_info.get("game_date"),
        "day_night": game_info.get("day_night"),
        "weather_temp": _parse_temperature(game_info.get("weather_temp")),
        "weather_wind": game_info.get("weather_wind"),
        # At-bat context
        "at_bat_index": at_bat_index,
        "half_inning": (about.get("halfInning") or "top").lower(),
        "inning": about.get("inning") or linescore.get("currentInning") or 1,
        "batter_id": matchup.get("batter", {}).get("id"),
        "batter_name": matchup.get("batter", {}).get("fullName"),
        "bat_side": matchup.get("batSide", {}).get("code"),
        "pitcher_id": matchup.get("pitcher", {}).get("id"),
        "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
        "throw_side": matchup.get("pitchHand", {}).get("code"),
        "is_runner_on_first": on_first,
        "is_runner_on_second": on_second,
        "is_runner_on_third": on_third,
        "away_score": away_score,
        "home_score": home_score,
        "description": None,
        # Upcoming pitch: state going in, measurements unknown
        "pitch_number": next_pitch_number,
        "count_after_pitch": f"{balls}-{strikes}",
        "outs": outs,
        "pitch_type_code": None,
        "pitch_type": None,
        "px": None,
        "pz": None,
        # Carry the previous pitch speed so velocity_delta stays neutral (0)
        # instead of implying a phantom speed drop from the 90 mph fill.
        "pitch_start_speed": last_pitch_speed,
        "pitch_strike_zone_top": None,
        "pitch_strike_zone_bottom": None,
        "is_strike": False,
        "is_ball": False,
        "is_in_play": False,
    }


def build_context(feed: dict, pending: dict) -> GameContext:
    """Build the rendering context for the upcoming pitch."""
    game_data = feed.get("gameData", {})
    teams = game_data.get("teams", {})
    half = str(pending.get("half_inning") or "top")

    return GameContext(
        pitcher_name=pending.get("pitcher_name") or "Unknown Pitcher",
        batter_name=pending.get("batter_name") or "Unknown Batter",
        pitcher_hand=pending.get("throw_side") or "R",
        batter_hand=pending.get("bat_side") or "R",
        home_team=_team_label(teams.get("home", {})),
        away_team=_team_label(teams.get("away", {})),
        inning=int(pending.get("inning") or 1),
        inning_half="Top" if half == "top" else "Bot",
        balls=int(str(pending.get("count_after_pitch", "0-0")).split("-")[0]),
        strikes=int(str(pending.get("count_after_pitch", "0-0")).split("-")[1]),
        outs=int(pending.get("outs") or 0),
        date=str(pending.get("game_date") or "")[:10] or None,
        runner_on_1b=bool(pending.get("is_runner_on_first")),
        runner_on_2b=bool(pending.get("is_runner_on_second")),
        runner_on_3b=bool(pending.get("is_runner_on_third")),
        score_home=int(pending.get("home_score") or 0),
        score_away=int(pending.get("away_score") or 0),
        pitch_number=int(pending.get("pitch_number") or 1),
        pitcher_id=pending.get("pitcher_id"),
        batter_id=pending.get("batter_id"),
    )


def build_live_snapshot(feed: dict) -> LiveSnapshot | None:
    """Parse one live feed poll into a prediction-ready snapshot.

    Returns None when the game is not live or has no current play yet.
    """
    _, state = extract_game_status(feed)
    if state != GameState.LIVE:
        return None

    current_play = feed.get("liveData", {}).get("plays", {}).get("currentPlay")
    if not current_play:
        return None
    if current_play.get("about", {}).get("isComplete"):
        # The at-bat just ended; there is no upcoming pitch to predict until
        # the feed rolls over to the next batter.
        return None

    game_info = _transformer._process_game_data(feed)
    game_pk = game_info.get("game_pk")
    if game_pk is None:
        return None

    completed_pd = _transformer.transform(feed, game_pk, game_info.get("season"))
    completed = (
        pl.from_pandas(completed_pd) if len(completed_pd) else pl.DataFrame()
    )

    pending = build_pending_row(feed, game_info, completed)
    pending_frame = pl.DataFrame([pending])

    if completed.is_empty():
        frame = pending_frame
    else:
        frame = pl.concat([completed, pending_frame], how="diagonal_relaxed")

    at_bat_index = int(pending["at_bat_index"])
    completed_in_ab = (
        int(completed.filter(pl.col("at_bat_index") == at_bat_index).height)
        if not completed.is_empty()
        else 0
    )

    return LiveSnapshot(
        game_pk=int(game_pk),
        state=state,
        at_bat_index=at_bat_index,
        next_pitch_number=int(pending["pitch_number"]),
        balls=int(str(pending["count_after_pitch"]).split("-")[0]),
        strikes=int(str(pending["count_after_pitch"]).split("-")[1]),
        outs=int(pending["outs"]),
        pitch_key=(
            at_bat_index,
            completed_in_ab,
            int(str(pending["count_after_pitch"]).split("-")[0]),
            int(str(pending["count_after_pitch"]).split("-")[1]),
        ),
        frame=frame,
        context=build_context(feed, pending),
        inning_key=(str(build_context(feed, pending).inning_half), int(pending["inning"])),
    )
