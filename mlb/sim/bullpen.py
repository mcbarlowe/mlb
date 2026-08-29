"""Synthetic per-team bullpen arms.

Each team's relief corps is aggregated into one synthetic pitcher whose id
is ``-team_id``. Reliever pitch rows (everything after the game's starting
pitcher for that side) are relabeled to the synthetic id and flow through
the SAME builders as real pitchers — pitch mixes, location pools, and
profile stores — so the simulator can hand the bullpen a team-specific arm
instead of one league-average stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mlb.sim.game import BULLPEN_ARM, Pitcher

TEAM_BULLPENS_PATH = Path("models/sim/team_bullpens.json")


def bullpen_arm_id(team_id: int) -> int:
    return -int(team_id)


def relabel_reliever_rows(raw: pl.DataFrame) -> pl.DataFrame:
    """Reliever pitch rows with ``pitcher_id`` set to the team's arm id.

    The starter is the first pitcher (by at-bat/pitch order) each game for
    the pitching side; every other pitcher that game is a reliever.
    """
    is_top = pl.col("half_inning").str.to_lowercase() == "top"
    frame = raw.with_columns(
        pl.when(is_top)
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("pitching_team_id")
    )
    starters = (
        frame.sort(["game_pk", "at_bat_index", "pitch_number"])
        .group_by(["game_pk", "pitching_team_id"], maintain_order=True)
        .agg(pl.col("pitcher_id").first().alias("starter_id"))
    )
    relievers = frame.join(
        starters, on=["game_pk", "pitching_team_id"], how="left"
    ).filter(pl.col("pitcher_id") != pl.col("starter_id"))
    return relievers.with_columns(
        (-pl.col("pitching_team_id")).cast(pl.Int64).alias("pitcher_id")
    ).drop(["pitching_team_id", "starter_id"])


def build_team_bullpen_hands(relabeled: pl.DataFrame) -> dict[int, str]:
    """Majority throwing hand per team from relabeled reliever rows."""
    counts = (
        relabeled.filter(pl.col("throw_side").is_in(["L", "R"]))
        .group_by(["pitcher_id", "throw_side"])
        .len()
        .sort(["pitcher_id", "len"], descending=[False, True])
        .group_by("pitcher_id", maintain_order=True)
        .agg(pl.col("throw_side").first())
    )
    return {
        -int(arm_id): str(hand)
        for arm_id, hand in zip(
            counts["pitcher_id"].to_list(), counts["throw_side"].to_list()
        )
    }


def save_team_bullpens(hands: dict[int, str], path: Path = TEAM_BULLPENS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(team_id): {"pitcher_id": bullpen_arm_id(team_id), "throw_side": hand}
        for team_id, hand in sorted(hands.items())
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


_TEAM_BULLPENS_CACHE: dict[int, Pitcher] | None = None


def bullpen_for_team(
    team_id: int | None, path: Path = TEAM_BULLPENS_PATH
) -> Pitcher:
    """Team's synthetic bullpen arm; league-average arm when unknown."""
    global _TEAM_BULLPENS_CACHE
    if team_id is None:
        return BULLPEN_ARM
    if _TEAM_BULLPENS_CACHE is None:
        if not path.exists():
            return BULLPEN_ARM
        payload = json.loads(path.read_text())
        _TEAM_BULLPENS_CACHE = {
            int(team): Pitcher(entry["pitcher_id"], entry["throw_side"])
            for team, entry in payload.items()
        }
    return _TEAM_BULLPENS_CACHE.get(int(team_id), BULLPEN_ARM)
