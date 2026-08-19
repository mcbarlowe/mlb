"""Derive futures outcomes from actual game results rather than hardcoded sets.

``backtest_futures._load_actual_outcomes`` previously carried literal team-id sets, one of them
annotated with a question mark, and several were wrong: 2023 AL East was Baltimore not Tampa
Bay, Seattle is listed as a 2023 division winner having won nothing, and 2022 NL Central was
St. Louis not Milwaukee. A backtest graded against guessed outcomes measures nothing.

Records come from ``mlb.linescore`` aggregated per game, counting each team once per game.
Division membership comes from ``mlb.teams`` with one correction: Houston is recorded in the
National League Central, which leaves the American League West with four teams. Houston has
been in the AL West since 2013. The override is applied here rather than by mutating the
reference table.

Ties for a division lead are broken on head-to-head record between the tied teams, which is
MLB's first tiebreaker, and any tie that head-to-head cannot resolve is reported rather than
silently assigned.

Playoff field size varies by era and is applied per season:
  2012-2019 and 2021: three division winners plus two wild cards per league, ten teams
  2020:               top two per division plus two per league, sixteen teams
  2022 onward:        three division winners plus three wild cards per league, twelve teams

    uv run python scripts/futures_outcomes.py --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

# team_id -> (division_id, league_id). Houston sits in NL Central in mlb.teams, which is wrong.
TEAM_DIVISION_OVERRIDE: dict[int, tuple[int, int]] = {117: (200, 103)}


def _wildcards_per_league(season: int) -> int:
    if season == 2020:
        return 2  # alongside two qualifiers per division
    return 3 if season >= 2022 else 2


def team_divisions(conn, schema: str) -> dict[int, tuple[int, int]]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT team_id, division_id, league_id
            FROM {schema}.teams
            WHERE sport_id = 1 AND division_id IS NOT NULL
            """
        )
        out = {int(t): (int(d), int(lg)) for t, d, lg in cur.fetchall()}
    out.update(TEAM_DIVISION_OVERRIDE)
    return out


def season_records(conn, schema: str, season: int):
    """(wins, losses) per team and head-to-head wins, from aggregated linescore runs."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH scored AS (
                SELECT game_pk,
                       SUM(CASE WHEN team_type = 'home' THEN runs ELSE 0 END) AS home_r,
                       SUM(CASE WHEN team_type = 'away' THEN runs ELSE 0 END) AS away_r
                FROM {schema}.linescore
                GROUP BY game_pk
                HAVING COUNT(DISTINCT team_type) = 2
            )
            SELECT g.home_team_id, g.away_team_id, s.home_r, s.away_r
            FROM {schema}.games g
            JOIN scored s ON s.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R'
              AND s.home_r IS NOT NULL AND s.away_r IS NOT NULL
              AND s.home_r <> s.away_r
            """,
            (season,),
        )
        rows = cur.fetchall()

    wins: dict[int, int] = defaultdict(int)
    losses: dict[int, int] = defaultdict(int)
    h2h: dict[tuple[int, int], int] = defaultdict(int)
    for home, away, hr, ar in rows:
        home, away = int(home), int(away)
        winner, loser = (home, away) if hr > ar else (away, home)
        wins[winner] += 1
        losses[loser] += 1
        h2h[(winner, loser)] += 1
    return wins, losses, h2h


def derive_outcomes(conn, schema: str, season: int) -> tuple[dict[str, set[int]], list[str]]:
    divisions = team_divisions(conn, schema)
    wins, losses, h2h = season_records(conn, schema, season)
    notes: list[str] = []

    played = {t for t in wins} | {t for t in losses}
    teams = [t for t in played if t in divisions]

    def pct(t: int) -> float:
        g = wins[t] + losses[t]
        return wins[t] / g if g else 0.0

    by_div: dict[int, list[int]] = defaultdict(list)
    by_league: dict[int, list[int]] = defaultdict(list)
    for t in teams:
        div, lg = divisions[t]
        by_div[div].append(t)
        by_league[lg].append(t)

    division_winners: set[int] = set()
    for div, members in sorted(by_div.items()):
        best = max(pct(t) for t in members)
        tied = [t for t in members if pct(t) == best]
        if len(tied) == 1:
            division_winners.add(tied[0])
            continue
        # MLB's first tiebreaker is head-to-head record between the tied clubs.
        scored = {
            t: sum(h2h[(t, o)] for o in tied if o != t)
            - sum(h2h[(o, t)] for o in tied if o != t)
            for t in tied
        }
        top = max(scored.values())
        finalists = [t for t in tied if scored[t] == top]
        division_winners.add(finalists[0])
        notes.append(
            f"{season} division {div}: tie at {best:.4f} among {sorted(tied)}, "
            f"head-to-head {'resolved to ' + str(finalists[0]) if len(finalists) == 1 else 'UNRESOLVED, took ' + str(finalists[0])}"
        )

    playoff_teams: set[int] = set()
    n_wc = _wildcards_per_league(season)
    for lg, members in sorted(by_league.items()):
        if season == 2020:
            for div in {divisions[t][0] for t in members}:
                ranked = sorted(
                    (t for t in members if divisions[t][0] == div), key=pct, reverse=True
                )
                playoff_teams.update(ranked[:2])
            rest = sorted(
                (t for t in members if t not in playoff_teams), key=pct, reverse=True
            )
            playoff_teams.update(rest[:n_wc])
            continue
        div_winners_lg = {t for t in members if t in division_winners}
        playoff_teams.update(div_winners_lg)
        rest = sorted(
            (t for t in members if t not in div_winners_lg), key=pct, reverse=True
        )
        playoff_teams.update(rest[:n_wc])

    outcomes = {
        "division": division_winners,
        "make_playoffs": playoff_teams,
        "miss_playoffs": {t for t in teams if t not in playoff_teams},
    }
    expected = {"division": 6, "make_playoffs": 16 if season == 2020 else (12 if season >= 2022 else 10)}
    for key, want in expected.items():
        got = len(outcomes[key])
        if got != want:
            notes.append(f"{season} {key}: derived {got} teams, expected {want}")
    return outcomes, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(f"SELECT team_id, abbreviation FROM {c.schema}.teams WHERE sport_id = 1")
        abbr = {int(t): a for t, a in cur.fetchall()}

    # What the backtest used to assert, for comparison.
    hardcoded = {
        2022: {147, 114, 117, 144, 158, 119},
        2023: {141, 136, 117, 144, 158, 119},
        2024: {147, 114, 117, 143, 158, 119},
    }

    for season in (int(s) for s in args.seasons.split(",")):
        outcomes, notes = derive_outcomes(conn, c.schema, season)
        dw = sorted(outcomes["division"])
        print(f"{season} division winners ({len(dw)}): "
              f"{', '.join(abbr.get(t, str(t)) for t in dw)}")
        print(f"{season} playoff teams  ({len(outcomes['make_playoffs'])}): "
              f"{', '.join(sorted(abbr.get(t, str(t)) for t in outcomes['make_playoffs']))}")
        if season in hardcoded:
            old = hardcoded[season]
            wrong = old - outcomes["division"]
            missed = outcomes["division"] - old
            print(f"  hardcoded had wrong: {', '.join(abbr.get(t, str(t)) for t in sorted(wrong)) or 'none'}"
                  f" | missed: {', '.join(abbr.get(t, str(t)) for t in sorted(missed)) or 'none'}")
        for n in notes:
            print(f"  NOTE {n}")
        print()
    conn.close()


if __name__ == "__main__":
    main()
