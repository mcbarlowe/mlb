"""Compute leak-free per-venue park factors for the MLB Monte Carlo sim.

Methodology: classic home/road balanced park factors (Baseball-Reference style),
which isolate the park by comparing a team at its own home park against the same
team on the road. For each team ``T`` with home venue ``V``::

    PF(outcome) = (outcome rate per PA in T's HOME games)
                / (outcome rate per PA in T's ROAD games)

Both buckets count every PA in the game (both teams batting), so the numerator
and denominator share the same set of teams (T plus its opponents); the only
systematic difference is the ballpark. This controls for roster quality far
better than a raw venue-vs-league rate, which is why parks like Coors surface
correctly despite the Rockies' park-suppressed road roster. Rate is
``outcome_count / total_PA``; factors are clamped to ``[0.70, 1.60]``.

Leak-free guarantee: factors used to simulate season ``S`` are computed strictly
from seasons before ``S`` (prior seasons only, trailing window from 2021).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import PostgresConfig

FIRST_SEASON = 2021
TARGET_SEASONS = (2024, 2025)
OUTCOME_CLASSES = ("home_run", "single", "double", "triple")
CLAMP = (0.70, 1.60)
# Minimum home + road PA for a team to receive a park factor. Real MLB teams
# accumulate ~15k+ PA per trailing window on each side; teams/venues that only
# appear at temporary or neutral sites fall far below this and are dropped so
# their small samples do not distort the factors.
MIN_PA = 3000

# event_type values that are pure baserunning / non-batting plays. They occupy
# their own at_bat_index but are not plate appearances, so they are excluded
# from the PA denominator.
NON_PA_EVENTS = (
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "stolen_base_2b",
    "stolen_base_3b",
    "stolen_base_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "pickoff_error_1b",
    "pickoff_error_2b",
    "pickoff_error_3b",
    "wild_pitch",
    "passed_ball",
    "balk",
    "other_advance",
    "defensive_indiff",
    "stolen_base",
    "runner_double_play",
    "cs_double_play",
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "models" / "sim" / "park_factors.json"


def _connect(cfg: PostgresConfig) -> psycopg.Connection:
    return psycopg.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        connect_timeout=10,
    )


def fetch_game_side_counts(
    conn: psycopg.Connection, schema: str
) -> list[dict[str, int]]:
    """Return per-game PA/outcome tallies for regular-season completed games.

    Each row is one game: ``{season, game_pk, venue_id, home_team_id,
    away_team_id, pa, home_run, single, double, triple}``. A plate appearance is
    a distinct ``(game_pk, at_bat_index)`` whose ``event_type`` is not a pure
    baserunning play. Counts cover both teams batting in the game.
    """
    outcome_case = " ".join(
        f"COUNT(*) FILTER (WHERE event_type = '{cls}') AS {cls}," for cls in OUTCOME_CLASSES
    )
    query = f"""
        WITH pa AS (
            SELECT DISTINCT
                game_pk, at_bat_index, season, venue_id,
                home_team_id, away_team_id, event_type
            FROM {schema}.pitches
            WHERE game_type = 'R'
              AND season BETWEEN %s AND %s
              AND venue_id IS NOT NULL
              AND home_team_id IS NOT NULL
              AND away_team_id IS NOT NULL
              AND event_type IS NOT NULL
              AND event_type NOT IN ({",".join(["%s"] * len(NON_PA_EVENTS))})
        )
        SELECT
            season, game_pk, venue_id, home_team_id, away_team_id,
            COUNT(*) AS pa,
            {outcome_case}
            0 AS _pad
        FROM pa
        GROUP BY season, game_pk, venue_id, home_team_id, away_team_id
    """
    params: list[object] = [FIRST_SEASON, max(TARGET_SEASONS) - 1, *NON_PA_EVENTS]
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    games: list[dict[str, int]] = []
    for row in rows:
        rec = {
            "season": row[0],
            "game_pk": row[1],
            "venue_id": row[2],
            "home_team_id": row[3],
            "away_team_id": row[4],
            "pa": row[5],
        }
        for i, cls in enumerate(OUTCOME_CLASSES):
            rec[cls] = row[6 + i]
        games.append(rec)
    return games


def fetch_venue_names(conn: psycopg.Connection, schema: str) -> dict[int, str]:
    """Return the most recent (max season) name per venue_id."""
    query = f"""
        SELECT DISTINCT ON (venue_id) venue_id, venue_name
        FROM {schema}.venues
        ORDER BY venue_id, season DESC
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return {vid: name for vid, name in cur.fetchall()}


def _empty_bucket() -> dict[str, int]:
    return {"pa": 0, **{c: 0 for c in OUTCOME_CLASSES}}


def compute_park_factors(
    games: list[dict[str, int]], train_seasons: set[int]
) -> tuple[dict[str, dict[str, float]], int]:
    """Home/road balanced park factors keyed by each team's home venue.

    Returns ``({venue_id: {outcome: pf}}, total_pa)`` where ``total_pa`` is the
    total PA (home side) pooled over ``train_seasons``.
    """
    home: dict[int, dict[str, int]] = {}
    road: dict[int, dict[str, int]] = {}
    # Home games per team per venue, to pick each team's primary home park.
    home_venue_pa: dict[int, dict[int, int]] = {}

    for g in games:
        if g["season"] not in train_seasons:
            continue
        h, a = g["home_team_id"], g["away_team_id"]
        hb = home.setdefault(h, _empty_bucket())
        rb = road.setdefault(a, _empty_bucket())
        hb["pa"] += g["pa"]
        rb["pa"] += g["pa"]
        for cls in OUTCOME_CLASSES:
            hb[cls] += g[cls]
            rb[cls] += g[cls]
        venue_pa = home_venue_pa.setdefault(h, {})
        venue_pa[g["venue_id"]] = venue_pa.get(g["venue_id"], 0) + g["pa"]

    total_pa = sum(b["pa"] for b in home.values())

    factors: dict[str, dict[str, float]] = {}
    for team, hb in home.items():
        rb = road.get(team)
        if rb is None:
            continue
        if hb["pa"] < MIN_PA or rb["pa"] < MIN_PA:
            continue
        # Primary home venue = venue where the team logged the most home PA.
        venue_id = max(home_venue_pa[team].items(), key=lambda kv: kv[1])[0]
        pf: dict[str, float] = {}
        for cls in OUTCOME_CLASSES:
            home_rate = hb[cls] / hb["pa"]
            road_rate = rb[cls] / rb["pa"]
            value = home_rate / road_rate if road_rate > 0 else 1.0
            pf[cls] = round(min(CLAMP[1], max(CLAMP[0], value)), 4)
        factors[str(venue_id)] = pf
    return factors, total_pa


def main() -> None:
    cfg = PostgresConfig.from_env()
    schema = cfg.schema
    conn = _connect(cfg)
    try:
        games = fetch_game_side_counts(conn, schema)
        venue_names = fetch_venue_names(conn, schema)
    finally:
        conn.close()

    factors_by_season: dict[str, dict[str, dict[str, float]]] = {}
    total_pa_by_season: dict[int, int] = {}
    for target in TARGET_SEASONS:
        train = set(range(FIRST_SEASON, target))
        factors, total_pa = compute_park_factors(games, train)
        factors_by_season[str(target)] = factors
        total_pa_by_season[target] = total_pa

    payload = {
        "meta": {
            "method": (
                "Home/road balanced park factors (Baseball-Reference style): "
                "PF(outcome) = (outcome rate per PA in a team's home games) / "
                "(outcome rate per PA in the same team's road games), pooled over "
                "all prior seasons from 2021 (leak-free trailing window), keyed by "
                "each team's primary home venue."
            ),
            "created": datetime.now(UTC).isoformat(),
            "clamp": [CLAMP[0], CLAMP[1]],
            "outcome_classes": list(OUTCOME_CLASSES),
            "first_season": FIRST_SEASON,
            "min_pa": MIN_PA,
            "train_windows": {
                str(t): list(range(FIRST_SEASON, t)) for t in TARGET_SEASONS
            },
        },
        "factors": factors_by_season,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    # --- Sanity summary ---------------------------------------------------
    print(f"Wrote {OUTPUT_PATH}")
    for target in TARGET_SEASONS:
        train = list(range(FIRST_SEASON, target))
        n_venues = len(factors_by_season[str(target)])
        print(
            f"season {target}: train={train} venues={n_venues} "
            f"total_PA={total_pa_by_season[target]:,}"
        )

    def _name(vid: str) -> str:
        return venue_names.get(int(vid), f"venue {vid}")

    hr_2024 = [(vid, pf["home_run"]) for vid, pf in factors_by_season["2024"].items()]
    hr_2024.sort(key=lambda x: x[1], reverse=True)

    print("\n2024 home_run park factor - 5 HIGHEST:")
    for vid, f in hr_2024[:5]:
        print(f"  {vid:>6}  {f:5.3f}  {_name(vid)}")
    print("2024 home_run park factor - 5 LOWEST:")
    for vid, f in hr_2024[-5:]:
        print(f"  {vid:>6}  {f:5.3f}  {_name(vid)}")


if __name__ == "__main__":
    main()
