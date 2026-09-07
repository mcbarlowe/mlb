"""Build a team-game log with the manager of record for every game.

Source: Retrosheet game logs (``https://www.retrosheet.org/gamelogs/``), one zip
per season, 161 comma-separated fields per game. Fields used (1-indexed as
documented in ``glfields.txt``): 1 date, 2 game number, 4/7 visiting/home team,
5/8 leagues, 10/11 visiting/home score, 17 park, 90/92 visiting/home manager id,
91/93 visiting/home manager name.

Retrosheet is the only source that carries the *manager of record per game*.
The StatsAPI ``/teams/{id}/coaches?date=`` endpoint resolves by date but files
interim managers under a separate ``NTRM`` job and returns no ``MNGR`` row for
those stretches (verified: Angels 2022-06-20 returns only Phil Nevin as
"Interim Manager"), so mid-season changes are silently lost there.

Output: one row per team-game (both teams of a game appear), written to
``data/managers/manager_game_log.parquet``.

Franchise continuity comes from ``CurrentNames.csv``, which maps each
contemporary team id to its franchise id over an explicit date range. That
matters because team ids are reused across franchises (``BAL`` is the 1901-02
Orioles that became the Yankees franchise, and separately the 1954+ Orioles).

Data © Retrosheet (https://www.retrosheet.org/), used per their terms; not
redistributed by this repository.

Usage::

    uv run python scripts/build_manager_game_log.py
    uv run python scripts/build_manager_game_log.py --seasons 1901-2024 --verify-postgres
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "retrosheet" / "gamelogs"
OUTPUT_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"

GAMELOG_URL = "https://www.retrosheet.org/gamelogs/gl{season}.zip"
CURRENT_NAMES_URL = "https://www.retrosheet.org/CurrentNames.csv"

# 0-indexed positions in the 161-field game log record.
F_DATE = 0
F_GAME_NUMBER = 1
F_VIS_TEAM = 3
F_VIS_LEAGUE = 4
F_HOME_TEAM = 6
F_HOME_LEAGUE = 7
F_VIS_SCORE = 9
F_HOME_SCORE = 10
F_OUTS = 11
F_PARK = 16
F_VIS_MANAGER_ID = 89
F_VIS_MANAGER_NAME = 90
F_HOME_MANAGER_ID = 91
F_HOME_MANAGER_NAME = 92
N_FIELDS = 161

# 1901 is the start of the two-league modern era. Earlier seasons have short
# schedules, unstable franchises, and a different run environment, so they are
# excluded by default rather than pooled with modern baseball.
DEFAULT_FIRST_SEASON = 1901
DEFAULT_LAST_SEASON = date.today().year


def parse_seasons(spec: str) -> list[int]:
    seasons: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seasons.update(range(int(lo), int(hi) + 1))
        else:
            seasons.add(int(part))
    return sorted(seasons)


def fetch_gamelog(season: int, client: httpx.Client, refresh: bool) -> bytes | None:
    """Return the raw game-log text for ``season``, or ``None`` if unpublished."""

    cached = CACHE_DIR / f"gl{season}.zip"
    if cached.exists() and not refresh:
        payload = cached.read_bytes()
    else:
        response = client.get(GAMELOG_URL.format(season=season))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.content
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"gl{season}.zip holds {names}, expected one .txt")
        return archive.read(names[0])


def load_franchise_map(client: httpx.Client, refresh: bool) -> dict[str, list[tuple[date, date, str]]]:
    """Map contemporary team id -> [(start, end, franchise_id)] intervals."""

    cached = CACHE_DIR.parent / "CurrentNames.csv"
    if cached.exists() and not refresh:
        payload = cached.read_bytes()
    else:
        response = client.get(CURRENT_NAMES_URL)
        response.raise_for_status()
        payload = response.content
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)

    intervals: dict[str, list[tuple[date, date, str]]] = defaultdict(list)
    for row in csv.reader(io.StringIO(payload.decode("latin-1"))):
        if len(row) < 9 or not row[0]:
            continue
        franchise_id, team_id = row[0].strip(), row[1].strip()
        start = datetime.strptime(row[7].strip(), "%m/%d/%Y").date()
        end_raw = row[8].strip()
        end = datetime.strptime(end_raw, "%m/%d/%Y").date() if end_raw else date(9999, 12, 31)
        intervals[team_id].append((start, end, franchise_id))
    for spans in intervals.values():
        spans.sort()
    return dict(intervals)


def resolve_franchise(
    franchise_map: dict[str, list[tuple[date, date, str]]], team_id: str, game_date: date
) -> str:
    for start, end, franchise_id in franchise_map.get(team_id, ()):
        if start <= game_date <= end:
            return franchise_id
    # Unmapped ids fall back to the contemporary id so nothing is dropped; the
    # franchise random effect then treats that club as its own group.
    return team_id


def parse_season(
    season: int,
    raw: bytes,
    franchise_map: dict[str, list[tuple[date, date, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader = csv.reader(io.StringIO(raw.decode("latin-1")))
    for record in reader:
        if not record or not record[F_DATE]:
            continue
        if len(record) != N_FIELDS:
            raise ValueError(f"gl{season}: record has {len(record)} fields, expected {N_FIELDS}")
        game_date = datetime.strptime(record[F_DATE], "%Y%m%d").date()
        vis_score = int(record[F_VIS_SCORE])
        home_score = int(record[F_HOME_SCORE])
        game_id = f"{record[F_HOME_TEAM]}{record[F_DATE]}{record[F_GAME_NUMBER]}"
        sides = (
            (
                record[F_VIS_TEAM],
                record[F_VIS_LEAGUE],
                record[F_VIS_MANAGER_ID],
                record[F_VIS_MANAGER_NAME],
                vis_score,
                record[F_HOME_TEAM],
                record[F_HOME_MANAGER_ID],
                home_score,
                False,
            ),
            (
                record[F_HOME_TEAM],
                record[F_HOME_LEAGUE],
                record[F_HOME_MANAGER_ID],
                record[F_HOME_MANAGER_NAME],
                home_score,
                record[F_VIS_TEAM],
                record[F_VIS_MANAGER_ID],
                vis_score,
                True,
            ),
        )
        for team, league, mgr_id, mgr_name, scored, opp, opp_mgr_id, allowed, is_home in sides:
            rows.append(
                {
                    "season": season,
                    "game_date": game_date,
                    "game_id": game_id,
                    "game_number": record[F_GAME_NUMBER],
                    "park_id": record[F_PARK],
                    "team_id": team,
                    "franchise_id": resolve_franchise(franchise_map, team, game_date),
                    "league": league,
                    "opponent_id": opp,
                    "opponent_franchise_id": resolve_franchise(franchise_map, opp, game_date),
                    "is_home": is_home,
                    "runs_scored": scored,
                    "runs_allowed": allowed,
                    "won": scored > allowed,
                    "tied": scored == allowed,
                    "outs": int(record[F_OUTS]) if record[F_OUTS] else None,
                    "manager_id": mgr_id or None,
                    "manager_name": mgr_name or None,
                    "opponent_manager_id": opp_mgr_id or None,
                }
            )
    return rows


def verify_against_postgres(frame: pl.DataFrame) -> None:
    """Compare team-season W/RS/RA against ``mlb.games`` + ``mlb.linescore``."""

    import psycopg

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    query = """
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
            group by g.season, g.game_pk, t.abbreviation
        )
        select season,
               team_abbrev,
               count(*) as games,
               sum(case when rs > ra then 1 else 0 end) as wins,
               sum(rs) as runs_scored,
               sum(ra) as runs_allowed
        from team_game
        group by season, team_abbrev
        order by season, team_abbrev
    """
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(query)
        db_rows = cursor.fetchall()
    db = pl.DataFrame(
        {
            "season": [r[0] for r in db_rows],
            "team_abbrev": [r[1] for r in db_rows],
            "db_games": [r[2] for r in db_rows],
            "db_wins": [r[3] for r in db_rows],
            "db_rs": [int(r[4]) for r in db_rows],
            "db_ra": [int(r[5]) for r in db_rows],
        }
    )

    rs_season = (
        frame.group_by("season", "franchise_id")
        .agg(
            pl.len().alias("gl_games"),
            pl.col("won").sum().alias("gl_wins"),
            pl.col("runs_scored").sum().alias("gl_rs"),
            pl.col("runs_allowed").sum().alias("gl_ra"),
        )
    )
    seasons = sorted(set(rs_season["season"].to_list()) & set(db["season"].to_list()))
    if not seasons:
        print("verify: no overlapping seasons between game logs and Postgres")
        return

    # Franchise ids and StatsAPI abbreviations differ; match on the season
    # totals themselves, which is the check that matters.
    print(f"verify: {len(seasons)} overlapping seasons ({seasons[0]}-{seasons[-1]})")
    for season in seasons:
        gl = rs_season.filter(pl.col("season") == season)
        pg = db.filter(pl.col("season") == season)
        gl_totals = (gl["gl_games"].sum(), gl["gl_wins"].sum(), gl["gl_rs"].sum(), gl["gl_ra"].sum())
        pg_totals = (pg["db_games"].sum(), pg["db_wins"].sum(), pg["db_rs"].sum(), pg["db_ra"].sum())
        gl_sorted = sorted(zip(gl["gl_wins"].to_list(), gl["gl_rs"].to_list(), gl["gl_ra"].to_list()))
        pg_sorted = sorted(zip(pg["db_wins"].to_list(), pg["db_rs"].to_list(), pg["db_ra"].to_list()))
        status = "OK " if gl_totals == pg_totals and gl_sorted == pg_sorted else "DIFF"
        print(
            f"  {status} {season}: gamelog games/W/RS/RA={gl_totals} postgres={pg_totals} "
            f"clubs={gl.height}/{pg.height}"
        )


STATSAPI_COACHES_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/coaches?date={date}"
# MNGR is the manager of record; NTRM is an interim manager. A stretch run by
# an interim has only the NTRM row, so accepting just MNGR silently loses it.
MANAGER_JOB_IDS = {"MNGR", "NTRM"}

CURRENT_SEASON_SQL = """
    select g.game_pk,
           g.game_date,
           g.game_number,
           g.venue_id,
           s.team_id,
           t.abbreviation as team_abbrev,
           t.league_name,
           s.side,
           o.team_id as opponent_id,
           ot.abbreviation as opponent_abbrev,
           sum(case when l.team_type = s.side then l.runs else 0 end) as runs_scored,
           sum(case when l.team_type <> s.side then l.runs else 0 end) as runs_allowed
    from mlb.games g
    join lateral (
        values (g.home_team_id, 'home', g.away_team_id),
               (g.away_team_id, 'away', g.home_team_id)
    ) as s(team_id, side, opp_id) on true
    join lateral (values (s.opp_id)) as o(team_id) on true
    join mlb.teams t on t.team_id = s.team_id
    join mlb.teams ot on ot.team_id = o.team_id
    join mlb.linescore l on l.game_pk = g.game_pk
    where g.season = %(season)s
      and g.game_type = 'R'
      and g.abstract_game_state = 'Final'
    group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    order by 2, 1
"""


def fetch_manager(team_id: int, day: date, client: httpx.Client) -> tuple[str, str] | None:
    """Manager of record for ``team_id`` on ``day``, or ``None`` if unlisted."""

    response = client.get(STATSAPI_COACHES_URL.format(team_id=team_id, date=day.isoformat()))
    response.raise_for_status()
    for entry in response.json().get("roster", []):
        if entry.get("jobId") in MANAGER_JOB_IDS:
            person = entry.get("person", {})
            return str(person.get("id")), str(person.get("fullName"))
    return None


def resolve_manager_by_date(
    team_id: int, dates: list[date], client: httpx.Client, probe_days: int = 10
) -> dict[date, tuple[str, str] | None]:
    """Map every game date to its manager using a coarse probe plus bisection.

    Probing every date would be ~160 requests per club. Managers change at most
    a couple of times a season, so probe every ``probe_days`` and binary-search
    only the intervals where the answer actually changed. That is exact to the
    day at a fraction of the requests.
    """

    if not dates:
        return {}
    probe_index = sorted({0, *range(0, len(dates), probe_days), len(dates) - 1})
    cache: dict[int, tuple[str, str] | None] = {
        i: fetch_manager(team_id, dates[i], client) for i in probe_index
    }

    def value(i: int) -> tuple[str, str] | None:
        if i not in cache:
            cache[i] = fetch_manager(team_id, dates[i], client)
        return cache[i]

    boundaries: list[int] = []
    for left, right in itertools.pairwise(probe_index):
        if value(left) == value(right):
            continue
        lo, hi = left, right
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if value(mid) == value(left):
                lo = mid
            else:
                hi = mid
        boundaries.append(hi)

    resolved: dict[date, tuple[str, str] | None] = {}
    current = value(0)
    cuts = sorted(boundaries)
    for i, day in enumerate(dates):
        while cuts and i >= cuts[0]:
            current = value(cuts.pop(0))
        resolved[day] = current
    return resolved


def build_from_statsapi(season: int, workers: int) -> list[dict[str, object]]:
    """Team-game rows for a season Retrosheet has not published yet.

    Games and scores come from Postgres; managers come from the StatsAPI
    date-resolved coaches endpoint. Club identifiers are StatsAPI
    abbreviations, NOT Retrosheet franchise ids, so this output is meant to be
    analyzed on its own rather than concatenated with the historical log.
    """

    import psycopg
    from psycopg.rows import dict_row

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        row_factory=dict_row,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(CURRENT_SEASON_SQL, {"season": season})
        games = cast("list[dict[str, Any]]", cursor.fetchall())
    if not games:
        raise SystemExit(f"no final regular-season games in mlb.games for {season}")

    dates_by_team: dict[int, list[date]] = {}
    for row in games:
        dates_by_team.setdefault(row["team_id"], []).append(row["game_date"])
    dates_by_team = {team: sorted(set(days)) for team, days in dates_by_team.items()}

    with (
        httpx.Client(timeout=30.0, follow_redirects=True) as client,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        resolved = dict(
            pool.map(
                lambda item: (item[0], resolve_manager_by_date(item[0], item[1], client)),
                dates_by_team.items(),
            )
        )

    by_game: dict[tuple[int, int], tuple[str, str] | None] = {
        (row["game_pk"], row["team_id"]): resolved[row["team_id"]][row["game_date"]]
        for row in games
    }
    rows: list[dict[str, object]] = []
    for row in games:
        manager = by_game[(row["game_pk"], row["team_id"])]
        opponent_manager = by_game.get((row["game_pk"], row["opponent_id"]))
        scored, allowed = int(row["runs_scored"]), int(row["runs_allowed"])
        rows.append(
            {
                "season": season,
                "game_date": row["game_date"],
                "game_id": str(row["game_pk"]),
                "game_number": str(row["game_number"]),
                "park_id": str(row["venue_id"]),
                "team_id": row["team_abbrev"],
                "franchise_id": row["team_abbrev"],
                "league": row["league_name"],
                "opponent_id": row["opponent_abbrev"],
                "opponent_franchise_id": row["opponent_abbrev"],
                "is_home": row["side"] == "home",
                "runs_scored": scored,
                "runs_allowed": allowed,
                "won": scored > allowed,
                "tied": scored == allowed,
                "outs": None,
                "manager_id": manager[0] if manager else None,
                "manager_name": manager[1] if manager else None,
                "opponent_manager_id": opponent_manager[0] if opponent_manager else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default=f"{DEFAULT_FIRST_SEASON}-{DEFAULT_LAST_SEASON}",
        help="season spec, e.g. '1901-2024' or '2015,2016'",
    )
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    parser.add_argument("--workers", type=int, default=8, help="parallel download workers")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--verify-postgres",
        action="store_true",
        help="cross-check overlapping seasons against mlb.games + mlb.linescore",
    )
    parser.add_argument(
        "--statsapi-season",
        type=int,
        help=(
            "build a single season from Postgres games + the StatsAPI "
            "date-resolved coaches endpoint, for a season Retrosheet has not "
            "published; club ids are StatsAPI abbreviations, not franchise ids"
        ),
    )
    args = parser.parse_args()

    if args.statsapi_season:
        rows = build_from_statsapi(args.statsapi_season, args.workers)
        frame = pl.DataFrame(rows).sort("game_date", "game_id", "is_home")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(args.output)
        stints = frame.group_by("franchise_id", "manager_id").len()
        print(
            f"wrote {frame.height:,} team-game rows to {args.output} "
            f"(season {args.statsapi_season}, "
            f"{frame['manager_id'].n_unique()} managers, "
            f"{stints.height} club-manager stints, "
            f"{frame['manager_id'].null_count()} rows missing a manager, "
            f"through {frame['game_date'].max()})"
        )
        return

    seasons = parse_seasons(args.seasons)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        franchise_map = load_franchise_map(client, args.refresh)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            payloads = list(pool.map(lambda s: (s, fetch_gamelog(s, client, args.refresh)), seasons))

    rows: list[dict[str, object]] = []
    missing: list[int] = []
    for season, raw in payloads:
        if raw is None:
            missing.append(season)
            continue
        rows.extend(parse_season(season, raw, franchise_map))
    if missing:
        print(f"no game log published for: {missing}")
    if not rows:
        raise SystemExit("no game-log rows parsed")

    frame = pl.DataFrame(rows).sort("season", "game_date", "game_id", "is_home")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output)

    covered = frame["season"].unique().sort()
    print(
        f"wrote {frame.height:,} team-game rows to {args.output} "
        f"({covered[0]}-{covered[-1]}, {covered.len()} seasons, "
        f"{frame['manager_id'].n_unique()} managers, "
        f"{frame['manager_id'].null_count()} rows missing a manager)"
    )

    if args.verify_postgres:
        verify_against_postgres(frame)


if __name__ == "__main__":
    main()
