"""Postgres-backed game data for simulation validation.

Sources starting lineups, starting pitchers, final scores, and — for the
realistic bullpen model — a leak-free, as-of reliever pool straight from the
``mlb`` schema, so validation needs no archived GUMBO feeds.

Reliever roles are inferred only from a team's appearances *before* the game
being scored (trailing saves / holds / games-finished), so nothing about the
game under evaluation leaks into its own bullpen deployment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from mlb.sim.game import Batter, Lineup, Pitcher

_SIDE_TO_TEAM_COL = {"away": "away_team_id", "home": "home_team_id"}
# Trailing window (days) used to infer a team's current bullpen and roles.
_BULLPEN_WINDOW_DAYS = 45
_MAX_RELIEVERS = 8


def _connect():
    import psycopg

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    conninfo = {
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
        "host": config.host,
        "port": config.port,
        "connect_timeout": 15,
    }
    conn = psycopg.connect(
        **{key: value for key, value in conninfo.items() if value is not None}
    )
    return conn, config.schema


def _query(conn, sql: str, params: tuple) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@dataclass
class GameDataStore:
    """In-memory snapshot of one season's game/lineup/pitching rows."""

    season: int
    games: pl.DataFrame  # game_pk, game_date, away_team_id, home_team_id
    finals: pl.DataFrame  # game_pk, away_runs, home_runs
    batting: pl.DataFrame  # game_pk, team_type, player_id, batting_order
    pitching: pl.DataFrame  # game_pk, team_type, player_id, gs, gp, bf, sv, hld, gf
    bat_side: dict[int, str] = field(default_factory=dict)
    throw_side: dict[int, str] = field(default_factory=dict)
    _game_row: dict[int, dict] = field(default_factory=dict)

    # ---- loading ---------------------------------------------------------
    @classmethod
    def load(cls, season: int) -> GameDataStore:
        conn, schema = _connect()
        try:
            games = pl.DataFrame(
                _query(
                    conn,
                    f"""
                    SELECT game_pk, game_date, away_team_id, home_team_id
                    FROM {schema}.games
                    WHERE season::int=%s AND game_type='R'
                      AND abstract_game_state='Final'
                    """,
                    (season,),
                ),
                schema=["game_pk", "game_date", "away_team_id", "home_team_id"],
                orient="row",
            )
            finals = pl.DataFrame(
                _query(
                    conn,
                    f"""
                    SELECT game_pk, team_type, SUM(runs) AS runs
                    FROM {schema}.linescore
                    WHERE runs IS NOT NULL
                    GROUP BY game_pk, team_type
                    """,
                    (),
                ),
                schema=["game_pk", "team_type", "runs"],
                orient="row",
            )
            batting = pl.DataFrame(
                _query(
                    conn,
                    f"""
                    SELECT game_pk, team_type, player_id, batting_order::int AS bo
                    FROM {schema}.batting
                    WHERE batting_order ~ '^[0-9]+$'
                      AND (batting_order::int %% 100)=0
                    """,
                    (),
                ),
                schema=["game_pk", "team_type", "player_id", "batting_order"],
                orient="row",
            )
            pitching = pl.DataFrame(
                _query(
                    conn,
                    f"""
                    SELECT p.game_pk, p.team_type, p.player_id,
                           COALESCE(p.gamesstarted,0), COALESCE(p.gamespitched,0),
                           COALESCE(p.battersfaced,0), COALESCE(p.saves,0),
                           COALESCE(p.holds,0), COALESCE(p.gamesfinished,0)
                    FROM {schema}.pitching p
                    JOIN {schema}.games g ON g.game_pk=p.game_pk
                    WHERE g.season::int=%s AND g.game_type='R'
                    """,
                    (season,),
                ),
                schema=[
                    "game_pk",
                    "team_type",
                    "player_id",
                    "gs",
                    "gp",
                    "bf",
                    "sv",
                    "hld",
                    "gf",
                ],
                orient="row",
            )
            players = _query(
                conn,
                f"""
                SELECT player_id, bat_side_code, pitch_hand_code
                FROM {schema}.players
                """,
                (),
            )
        finally:
            conn.close()

        # game_date can arrive as date or str; normalize to datetime.date.
        games = games.with_columns(pl.col("game_date").cast(pl.Date, strict=False))
        # attach team_id per game to pitching rows for team-level trailing pools
        game_teams = games.select(
            ["game_pk", "game_date", "away_team_id", "home_team_id"]
        )
        pitching = pitching.join(game_teams, on="game_pk", how="left").with_columns(
            pl.when(pl.col("team_type") == "away")
            .then(pl.col("away_team_id"))
            .otherwise(pl.col("home_team_id"))
            .alias("team_id")
        )

        bat_side = {int(pid): (code or "R") for pid, code, _ in players}
        throw_side = {int(pid): (code or "R") for pid, _, code in players}
        store = cls(
            season=season,
            games=games,
            finals=finals,
            batting=batting,
            pitching=pitching,
            bat_side=bat_side,
            throw_side=throw_side,
        )
        for row in games.iter_rows(named=True):
            store._game_row[int(row["game_pk"])] = row
        return store

    # ---- sampling --------------------------------------------------------
    def final_game_pks(self, seed: int, limit: int) -> list[int]:
        """Deterministic sample of completed regular-season games with a
        final score and both starting lineups on file."""
        finals_wide = self.finals.pivot(values="runs", index="game_pk", on="team_type")
        scored = {int(g) for g in finals_wide["game_pk"].to_list()}
        # games with a full 9-man lineup for both sides
        lineup_counts = (
            self.batting.group_by(["game_pk", "team_type"])
            .len()
            .filter(pl.col("len") >= 9)
            .group_by("game_pk")
            .len()
            .filter(pl.col("len") == 2)
        )
        have_lineups = {int(g) for g in lineup_counts["game_pk"].to_list()}
        eligible = sorted(
            pk for pk in self._game_row if pk in scored and pk in have_lineups
        )
        rng = random.Random(seed)
        rng.shuffle(eligible)
        return eligible[:limit]

    def final(self, game_pk: int) -> tuple[int, int]:
        rows = self.finals.filter(pl.col("game_pk") == game_pk)
        runs = {r["team_type"]: int(r["runs"]) for r in rows.iter_rows(named=True)}
        return runs.get("away", 0), runs.get("home", 0)

    # ---- lineup construction --------------------------------------------
    def lineup(self, game_pk: int, side: str, *, individual_bullpen: bool) -> Lineup:
        row = self._game_row[int(game_pk)]
        team_id = int(row[_SIDE_TO_TEAM_COL[side]])
        batters = self._batters(game_pk, side)
        starter = self._starter(game_pk, side)
        if individual_bullpen:
            relievers = self._reliever_pool(team_id, row["game_date"], starter)
            return Lineup(batters=batters, starter=starter, relievers=relievers)
        from mlb.sim.bullpen import bullpen_for_team

        return Lineup(
            batters=batters, starter=starter, bullpen=bullpen_for_team(team_id)
        )

    def _batters(self, game_pk: int, side: str) -> list[Batter]:
        rows = (
            self.batting.filter(
                (pl.col("game_pk") == game_pk) & (pl.col("team_type") == side)
            )
            .sort("batting_order")
            .head(9)
        )
        batters = [
            Batter(int(pid), self.bat_side.get(int(pid), "R"))
            for pid in rows["player_id"].to_list()
        ]
        if len(batters) != 9:
            raise ValueError(f"Game {game_pk} {side} lineup has {len(batters)} batters")
        return batters

    def _starter(self, game_pk: int, side: str) -> Pitcher:
        rows = self.pitching.filter(
            (pl.col("game_pk") == game_pk)
            & (pl.col("team_type") == side)
            & (pl.col("gs") == 1)
        )
        if rows.height == 0:
            raise ValueError(f"Game {game_pk} {side} has no starting pitcher")
        pid = int(rows["player_id"].to_list()[0])
        return Pitcher(pid, self.throw_side.get(pid, "R"))

    def _reliever_pool(
        self, team_id: int, game_date: date, starter: Pitcher
    ) -> tuple[Pitcher, ...]:
        """Leak-free bullpen: relievers used by this team in the trailing
        window before ``game_date``, ranked by leverage role (closer first)."""
        window_start = game_date - _timedelta_days(_BULLPEN_WINDOW_DAYS)
        trailing = self.pitching.filter(
            (pl.col("team_id") == team_id)
            & (pl.col("gs") == 0)
            & (pl.col("gp") >= 1)
            & (pl.col("game_date") < game_date)
            & (pl.col("game_date") >= window_start)
            & (pl.col("player_id") != starter.player_id)
        )
        if trailing.height == 0:
            return ()
        agg = (
            trailing.group_by("player_id")
            .agg(
                pl.len().alias("apps"),
                pl.col("sv").sum().alias("sv"),
                pl.col("hld").sum().alias("hld"),
                pl.col("gf").sum().alias("gf"),
                pl.col("bf").sum().alias("bf"),
            )
            # Leverage proxy: closers save/finish, setup men hold; break ties
            # toward the more heavily used arm.
            .with_columns(
                (pl.col("sv") * 3 + pl.col("gf") * 2 + pl.col("hld") * 1).alias(
                    "leverage"
                )
            )
            .sort(["leverage", "bf"], descending=[True, True])
            .head(_MAX_RELIEVERS)
        )
        return tuple(
            Pitcher(int(pid), self.throw_side.get(int(pid), "R"))
            for pid in agg["player_id"].to_list()
        )


def _timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=days)


def trailing_reliever_pool(
    team_id: int,
    slate_date: str,
    season: int,
    *,
    window_days: int = _BULLPEN_WINDOW_DAYS,
    max_arms: int = _MAX_RELIEVERS,
) -> tuple[Pitcher, ...]:
    """Live production bullpen pool for ``team_id`` as of ``slate_date``.

    One targeted Postgres query (no whole-season load): the team's relievers
    over the trailing window before the slate date, ranked closer-first by the
    same leverage proxy the validation store uses. Leak-free by construction
    (only games strictly before the slate date). Returns ``()`` on any failure
    so callers fall back to the aggregate arm.
    """
    from datetime import date as date_cls
    from datetime import timedelta

    as_of = date_cls.fromisoformat(slate_date)
    window_start = as_of - timedelta(days=window_days)
    try:
        conn, schema = _connect()
    except Exception:
        return ()
    try:
        rows = _query(
            conn,
            f"""
            SELECT p.player_id,
                   SUM(COALESCE(p.saves,0)) AS sv,
                   SUM(COALESCE(p.holds,0)) AS hld,
                   SUM(COALESCE(p.gamesfinished,0)) AS gf,
                   SUM(COALESCE(p.battersfaced,0)) AS bf,
                   MAX(pl.pitch_hand_code) AS hand
            FROM {schema}.pitching p
            JOIN {schema}.games g ON g.game_pk = p.game_pk
            JOIN {schema}.players pl ON pl.player_id = p.player_id
            WHERE g.season::int = %s AND g.game_type = 'R'
              AND g.game_date >= %s AND g.game_date < %s
              AND COALESCE(p.gamesstarted,0) = 0 AND COALESCE(p.gamespitched,0) >= 1
              AND (
                    (p.team_type = 'away' AND g.away_team_id = %s)
                    OR (p.team_type = 'home' AND g.home_team_id = %s)
                  )
            GROUP BY p.player_id
            """,
            (season, window_start.isoformat(), as_of.isoformat(), team_id, team_id),
        )
    except Exception:
        return ()
    finally:
        conn.close()
    # leverage = saves*3 + games-finished*2 + holds; tie-break on batters faced
    ranked = sorted(
        rows,
        key=lambda r: (
            -(int(r[1]) * 3 + int(r[3]) * 2 + int(r[2])),
            -int(r[4]),
        ),
    )[:max_arms]
    return tuple(Pitcher(int(r[0]), (r[5] or "R")) for r in ranked)
