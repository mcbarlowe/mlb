"""Leak-free team talent from player performance in *other* seasons.

Purpose: give a regression a control for how good a club's players are that is
not built from the club's results in the season being explained. Each player's
quality for season ``s`` is computed from every other season he appears in,
shrunk toward the league mean, then aggregated to the club.

Two aggregation weightings, and the choice matters enormously for what a
manager coefficient means:

``playing_time``
    Weight each player by the plate appearances / innings he actually got with
    that club that season. This treats the manager's own allocation decisions
    as part of "talent", so allocation skill is subtracted out of the manager
    term. Conservative.

``allocation_neutral``
    Weight each player by his *typical* usage rate in other seasons times the
    number of that club's games he appeared in. Roster composition and
    availability are kept (the front office and the training staff own those),
    but the per-game volume decision — how many trips to the plate a given bat
    gets, how many batters a given arm faces — is removed from the control and
    therefore stays in the manager term.

The gap between the two is an estimate of playing-time allocation effects.

Caveat that survives both: whether a player appears in a game at all is still
partly a managerial decision, so ``allocation_neutral`` is closer to, but not
exactly, allocation-free.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

__all__ = [
    "BATTING_SQL",
    "CLUB_SEASON_SQL",
    "PITCHING_SQL",
    "Weighting",
    "leave_one_season_out",
    "read_player_seasons",
    "team_talent",
]

Weighting = Literal["playing_time", "allocation_neutral"]

# Linear weights per plate appearance (wOBA-style scale).
WEIGHT_BB, WEIGHT_HBP, WEIGHT_1B, WEIGHT_2B, WEIGHT_3B, WEIGHT_HR = (
    0.69, 0.72, 0.89, 1.27, 1.62, 2.10,
)
# Prior weight for empirical-Bayes shrinkage of other-season quality.
BAT_SHRINK_PA = 400.0
PIT_SHRINK_IP = 120.0

BATTING_SQL = """
    select g.season,
           t.abbreviation as team_abbrev,
           b.player_id,
           sum(b.plateappearances) as pa,
           sum(b.hits) as h,
           sum(b.doubles) as d2,
           sum(b.triples) as d3,
           sum(b.homeruns) as hr,
           sum(b.baseonballs) as bb,
           sum(b.hitbypitch) as hbp,
           count(distinct b.game_pk) as games_appeared
    from mlb.batting b
    join mlb.games g on g.game_pk = b.game_pk
    join lateral (
        values (g.home_team_id, 'home'), (g.away_team_id, 'away')
    ) as s(team_id, side) on s.side = b.team_type
    join mlb.teams t on t.team_id = s.team_id
    where g.game_type = 'R' and g.abstract_game_state = 'Final'
    group by 1, 2, 3
    having sum(b.plateappearances) > 0
"""

PITCHING_SQL = """
    select g.season,
           t.abbreviation as team_abbrev,
           p.player_id,
           sum(p.battersfaced) as bf,
           sum(p.outs) as outs,
           sum(p.homeruns) as hr,
           sum(p.baseonballs) as bb,
           sum(p.hitbatsmen) as hbp,
           sum(p.strikeouts) as k,
           count(distinct p.game_pk) as games_appeared
    from mlb.pitching p
    join mlb.games g on g.game_pk = p.game_pk
    join lateral (
        values (g.home_team_id, 'home'), (g.away_team_id, 'away')
    ) as s(team_id, side) on s.side = p.team_type
    join mlb.teams t on t.team_id = s.team_id
    where g.game_type = 'R' and g.abstract_game_state = 'Final'
    group by 1, 2, 3
    having sum(p.outs) > 0
"""

CLUB_SEASON_SQL = """
    with team_game as (
        select g.season,
               g.game_pk,
               t.abbreviation as team_abbrev,
               sum(case when l.team_type = s.side then l.runs else 0 end) as rs,
               sum(case when l.team_type <> s.side then l.runs else 0 end) as ra
        from mlb.games g
        join lateral (
            values (g.home_team_id, 'home'), (g.away_team_id, 'away')
        ) as s(team_id, side) on true
        join mlb.teams t on t.team_id = s.team_id
        join mlb.linescore l on l.game_pk = g.game_pk
        where g.game_type = 'R' and g.abstract_game_state = 'Final'
        group by 1, 2, 3
    )
    select season,
           team_abbrev,
           count(*) as games,
           sum(case when rs > ra then 1 else 0 end) as wins,
           sum(rs) as runs_scored,
           sum(ra) as runs_allowed
    from team_game
    where rs <> ra
    group by 1, 2
"""


def read_player_seasons() -> dict[str, pl.DataFrame]:
    """Player-club-season batting and pitching lines plus club-season totals."""

    import psycopg

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    frames: dict[str, pl.DataFrame] = {}
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    ) as connection:
        for name, query in (
            ("batting", BATTING_SQL),
            ("pitching", PITCHING_SQL),
            ("club", CLUB_SEASON_SQL),
        ):
            with connection.cursor() as cursor:
                cursor.execute(query)  # type: ignore[arg-type]
                columns = [c.name for c in cursor.description or ()]
                rows = cursor.fetchall()
            # Postgres sum() over integers returns numeric, which arrives as
            # Decimal and will not join against Int64 keys.
            frames[name] = pl.DataFrame(
                {col: [r[i] for r in rows] for i, col in enumerate(columns)}
            ).with_columns((pl.selectors.integer() | pl.selectors.decimal()).cast(pl.Int64))
    return frames


def leave_one_season_out(
    frame: pl.DataFrame,
    value_numerator: pl.Expr,
    weight: str,
    shrink: float,
) -> pl.DataFrame:
    """Per-season player quality computed from that player's other seasons.

    Career totals minus the focal season, shrunk toward the league mean by
    ``shrink`` units of prior weight. A player with no other season falls back
    entirely to the league mean, the correct no-information default.
    """

    per = frame.group_by("season", "player_id").agg(
        value_numerator.sum().alias("num"), pl.col(weight).sum().alias("den")
    )
    league = float(per["num"].sum() / per["den"].sum())
    career = per.group_by("player_id").agg(
        pl.col("num").sum().alias("career_num"), pl.col("den").sum().alias("career_den")
    )
    return (
        per.join(career, on="player_id", how="left")
        .with_columns(
            (pl.col("career_num") - pl.col("num")).alias("other_num"),
            (pl.col("career_den") - pl.col("den")).alias("other_den"),
        )
        .with_columns(
            (
                (pl.col("other_num") + shrink * league) / (pl.col("other_den") + shrink)
            ).alias("quality"),
            (pl.col("other_den") > 0).alias("has_other_season"),
            # Typical usage per game in other seasons, for allocation-neutral
            # weighting. Falls back to this club-season's own rate when the
            # player has no other season to speak for him.
            pl.col("other_den").alias("other_usage"),
        )
        .select("season", "player_id", "quality", "has_other_season", "other_usage")
    )


def _other_season_games(frame: pl.DataFrame) -> pl.DataFrame:
    """Games appeared summed over a player's other seasons, for usage rates."""

    per = frame.group_by("season", "player_id").agg(
        pl.col("games_appeared").sum().alias("games_appeared")
    )
    career = per.group_by("player_id").agg(pl.col("games_appeared").sum().alias("career_games"))
    return (
        per.join(career, on="player_id", how="left")
        .with_columns((pl.col("career_games") - pl.col("games_appeared")).alias("other_games"))
        .select("season", "player_id", "other_games")
    )


def _aggregate(
    lines: pl.DataFrame,
    quality: pl.DataFrame,
    other_games: pl.DataFrame,
    usage_column: str,
    weighting: Weighting,
    prefix: str,
) -> pl.DataFrame:
    joined = lines.join(quality, on=["season", "player_id"], how="left").join(
        other_games, on=["season", "player_id"], how="left"
    )
    if weighting == "playing_time":
        weight = pl.col(usage_column).cast(pl.Float64)
    else:
        # Typical per-game usage in other seasons, applied to the games he was
        # actually present for with this club. Players with no other season keep
        # their observed usage, which is the only information available.
        per_game = pl.when(pl.col("other_games") > 0).then(
            pl.col("other_usage") / pl.col("other_games")
        ).otherwise(pl.col(usage_column) / pl.col("games_appeared"))
        weight = per_game * pl.col("games_appeared")
    return (
        joined.with_columns(weight.alias("w"))
        .group_by("season", "team_abbrev")
        .agg(
            ((pl.col("quality") * pl.col("w")).sum() / pl.col("w").sum()).alias(f"{prefix}_talent"),
            (
                (pl.col(usage_column) * pl.col("has_other_season")).sum()
                / pl.col(usage_column).sum()
            ).alias(f"{prefix}_coverage"),
        )
    )


def team_talent(
    frames: dict[str, pl.DataFrame], weighting: Weighting = "playing_time"
) -> tuple[pl.DataFrame, dict]:
    """Per club-season batting and pitching talent, plus coverage diagnostics."""

    batting = frames["batting"].with_columns(
        (pl.col("h") - pl.col("d2") - pl.col("d3") - pl.col("hr")).alias("d1")
    )
    bat_numerator = (
        WEIGHT_BB * pl.col("bb")
        + WEIGHT_HBP * pl.col("hbp")
        + WEIGHT_1B * pl.col("d1")
        + WEIGHT_2B * pl.col("d2")
        + WEIGHT_3B * pl.col("d3")
        + WEIGHT_HR * pl.col("hr")
    )
    bat_quality = leave_one_season_out(batting, bat_numerator, "pa", BAT_SHRINK_PA)
    bat_team = _aggregate(
        batting, bat_quality, _other_season_games(batting), "pa", weighting, "bat"
    )

    pitching = frames["pitching"].with_columns((pl.col("outs") / 3.0).alias("ip"))
    pit_numerator = 13 * pl.col("hr") + 3 * (pl.col("bb") + pl.col("hbp")) - 2 * pl.col("k")
    pit_quality = leave_one_season_out(pitching, pit_numerator, "ip", PIT_SHRINK_IP)
    pit_team = _aggregate(
        pitching, pit_quality, _other_season_games(pitching), "ip", weighting, "pit"
    )

    talent = bat_team.join(pit_team, on=["season", "team_abbrev"], how="inner")
    meta = {
        "weighting": weighting,
        "team_seasons": talent.height,
        "batter_pa_coverage": float(talent["bat_coverage"].mean() or 0.0),
        "pitcher_ip_coverage": float(talent["pit_coverage"].mean() or 0.0),
    }
    return talent, meta
