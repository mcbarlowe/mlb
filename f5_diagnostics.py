#!/usr/bin/env python3
"""F5 odds and linescore diagnostics: coverage, line movement, and calibration.

This read-only diagnostic queries the PostgreSQL database to understand:
- Coverage: how many games have open/close F5 totals lines
- Line movement: open-to-close shifts in point and probability
- Actual distribution: F5 total runs (inning 1-5) distribution
- Market calibration: over/under rate vs implied probability
- Slices: by pitcher handedness, team strength, venue

Run:
    uv run f5_diagnostics.py --season 2025 --limit 50

Output: JSON artifacts in output/diagnostics/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent))

from src.database import PostgresConfig

# Configuration
PRECISION = 4
MIN_SAMPLE_SIZE = 20


@dataclass
class F5Line:
    """Single F5 totals line snapshot."""
    game_pk: int
    game_date: str
    away_team: str
    home_team: str
    bookmaker: str
    line_type: str
    total_point: float
    over_ml: int
    under_ml: int


@dataclass
class F5Actual:
    """Actual F5 total runs from linescore."""
    game_pk: int
    away_runs: float
    home_runs: float
    total_runs: float


def _connect(config: PostgresConfig):
    return psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    )


def ml_to_probability(ml: int) -> float:
    """Convert American odds to decimal probability."""
    if ml == 0:
        return 0.5  # Even money
    if ml > 0:
        # Positive: +150 means 100/250 = 0.40
        return 100.0 / (ml + 100.0)
    else:
        # Negative: -150 means 150/50 = 0.75
        return abs(ml) / (abs(ml) + 100.0)


def load_f5_lines(
    config: PostgresConfig,
    seasons: tuple[int, ...],
    line_types: tuple[str, ...] | None = None,
) -> dict[str, list[F5Line]]:
    """Load F5 totals lines by game_pk, keyed by line_type."""
    if line_types is None:
        line_types = ("open", "close", "current")
    
    by_type: dict[str, list[F5Line]] = {lt: [] for lt in line_types}
    
    with _connect(config) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT f.game_pk, g.game_date, f.away_team, f.home_team, 
                   f.bookmaker, f.line_type, f.total_point, f.over_ml, f.under_ml
            FROM {}.f5_odds f
            JOIN {}.games g ON f.game_pk = g.game_pk
            WHERE g.season::int = ANY(%s)
              AND g.game_type = 'R'
              AND f.total_point IS NOT NULL
              AND f.over_ml IS NOT NULL
              AND f.under_ml IS NOT NULL
              AND f.line_type = ANY(%s)
            ORDER BY f.game_pk, f.line_type, f.bookmaker
            """
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
        cursor.execute(query, (list(seasons), list(line_types)))
        for pk, game_date, away, home, bm, ltype, point, over_ml, under_ml in cursor.fetchall():
            line = F5Line(
                game_pk=int(pk),
                game_date=str(game_date),
                away_team=str(away),
                home_team=str(home),
                bookmaker=str(bm),
                line_type=str(ltype),
                total_point=float(point),
                over_ml=int(over_ml),
                under_ml=int(under_ml),
            )
            by_type[str(ltype)].append(line)
    
    return by_type


def load_f5_actuals(
    config: PostgresConfig,
    game_pks: list[int],
    prefix_innings: int = 5,
) -> dict[int, F5Actual]:
    """Load actual F5 totals from linescore."""
    if not game_pks:
        return {}
    
    actuals: dict[int, F5Actual] = {}
    with _connect(config) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT game_pk,
                   SUM(runs) FILTER (WHERE team_type = 'away')::float AS away_runs,
                   SUM(runs) FILTER (WHERE team_type = 'home')::float AS home_runs,
                   COUNT(*) FILTER (WHERE team_type = 'away' AND runs IS NOT NULL) AS away_rows,
                   COUNT(*) FILTER (WHERE team_type = 'home' AND runs IS NOT NULL) AS home_rows
            FROM {}.linescore
            WHERE game_pk = ANY(%s)
              AND inning BETWEEN 1 AND %s
            GROUP BY game_pk
            """
        ).format(sql.Identifier(config.schema))
        cursor.execute(query, (game_pks, prefix_innings))
        for pk, away_runs, home_runs, away_rows, home_rows in cursor.fetchall():
            pk = int(pk)
            # Skip if incomplete inning data
            if away_rows < prefix_innings or home_rows < prefix_innings:
                continue
            if away_runs is None or home_runs is None:
                continue
            actuals[pk] = F5Actual(
                game_pk=pk,
                away_runs=float(away_runs),
                home_runs=float(home_runs),
                total_runs=float(away_runs) + float(home_runs),
            )
    return actuals


def analyze_coverage(
    by_type: dict[str, list[F5Line]],
) -> dict[str, Any]:
    """Analyze coverage by line type."""
    results = {}
    unique_games: dict[str, set[int]] = {}
    
    for line_type, lines in by_type.items():
        unique_games[line_type] = {l.game_pk for l in lines}
        results[line_type] = {
            "total_rows": len(lines),
            "unique_games": len(unique_games[line_type]),
            "unique_bookmakers": len({l.bookmaker for l in lines}),
            "avg_bookmakers_per_game": round(len(lines) / len(unique_games[line_type]), PRECISION) if unique_games[line_type] else 0,
        }
    
    # Pair coverage
    open_games = unique_games.get("open", set())
    close_games = unique_games.get("close", set())
    current_games = unique_games.get("current", set())
    
    if open_games and close_games:
        paired_oc = len(open_games & close_games)
    else:
        paired_oc = 0
    
    if current_games and close_games:
        paired_cc = len(current_games & close_games)
    else:
        paired_cc = 0
    
    results["pairing"] = {
        "open_close_paired": paired_oc,
        "current_close_paired": paired_cc,
    }
    
    return results


def analyze_line_movement(
    by_type: dict[str, list[F5Line]],
) -> dict[str, Any]:
    """Analyze open-to-close line movement."""
    open_lines = {l.game_pk: l for l in by_type.get("open", []) if l.bookmaker == "draftkings"}
    close_lines = {l.game_pk: l for l in by_type.get("close", []) if l.bookmaker == "draftkings"}
    
    paired_games = set(open_lines.keys()) & set(close_lines.keys())
    
    if not paired_games:
        return {"status": "no paired open-close data"}
    
    movements = []
    for game_pk in paired_games:
        open_line = open_lines[game_pk]
        close_line = close_lines[game_pk]
        
        point_move = close_line.total_point - open_line.total_point
        open_over_prob = ml_to_probability(open_line.over_ml)
        close_over_prob = ml_to_probability(close_line.over_ml)
        prob_move = close_over_prob - open_over_prob
        
        movements.append({
            "game_pk": game_pk,
            "game_date": open_line.game_date,
            "point_move": round(point_move, PRECISION),
            "prob_move": round(prob_move, PRECISION),
            "open_point": round(open_line.total_point, PRECISION),
            "close_point": round(close_line.total_point, PRECISION),
            "open_over_prob": round(open_over_prob, PRECISION),
            "close_over_prob": round(close_over_prob, PRECISION),
        })
    
    if movements:
        point_moves = [m["point_move"] for m in movements]
        prob_moves = [m["prob_move"] for m in movements]
        
        return {
            "paired_games": len(movements),
            "avg_point_move": round(sum(point_moves) / len(point_moves), PRECISION),
            "median_point_move": round(sorted(point_moves)[len(point_moves) // 2], PRECISION),
            "avg_prob_move": round(sum(prob_moves) / len(prob_moves), PRECISION),
            "moves_up_count": sum(1 for m in point_moves if m > 0),
            "sample_movements": movements[:10],
        }
    
    return {"status": "no movements computed"}


def analyze_actuals_distribution(
    actuals: dict[int, F5Actual],
) -> dict[str, Any]:
    """Analyze distribution of actual F5 totals."""
    if not actuals:
        return {"status": "no actual data"}
    
    totals = [a.total_runs for a in actuals.values()]
    totals_sorted = sorted(totals)
    
    return {
        "games_with_data": len(actuals),
        "mean": round(sum(totals) / len(totals), PRECISION),
        "median": round(totals_sorted[len(totals_sorted) // 2], PRECISION),
        "min": round(min(totals), PRECISION),
        "max": round(max(totals), PRECISION),
        "stdev": round((sum((x - sum(totals) / len(totals)) ** 2 for x in totals) / len(totals)) ** 0.5, PRECISION),
        "distribution": dict(Counter(totals)),
    }


def analyze_over_rate(
    by_type: dict[str, list[F5Line]],
    actuals: dict[int, F5Actual],
    line_type: str = "open",
) -> dict[str, Any]:
    """Analyze over/under rate and calibration vs market."""
    lines_by_game = {l.game_pk: l for l in by_type.get(line_type, []) if l.bookmaker == "draftkings"}
    
    # Get games with both line and actual
    paired = []
    for game_pk, line in lines_by_game.items():
        if game_pk not in actuals:
            continue
        actual = actuals[game_pk]
        paired.append({
            "game_pk": game_pk,
            "line": line,
            "actual": actual,
        })
    
    if not paired:
        return {"status": f"no paired {line_type} lines with actuals"}
    
    over_count = sum(1 for p in paired if p["actual"].total_runs > p["line"].total_point)
    over_rate = over_count / len(paired)
    
    # Market probability
    avg_over_ml = sum(p["line"].over_ml for p in paired) / len(paired)
    market_over_prob = ml_to_probability(int(avg_over_ml))
    
    # Calibration: does the market over rate match probability?
    calibration_error = over_rate - market_over_prob
    
    return {
        "line_type": line_type,
        "games": len(paired),
        "over_count": over_count,
        "over_rate": round(over_rate, PRECISION),
        "market_over_prob": round(market_over_prob, PRECISION),
        "calibration_error": round(calibration_error, PRECISION),
        "vig_pct": round(100.0 * calibration_error, PRECISION),
    }


def analyze_by_slice(
    by_type: dict[str, list[F5Line]],
    actuals: dict[int, F5Actual],
    config: PostgresConfig,
) -> dict[str, Any]:
    """Analyze calibration by slices: venue, pitcher handedness, etc."""
    # Load pitcher handedness by game
    game_to_pitchers: dict[int, tuple[str, str]] = {}
    
    with _connect(config) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT DISTINCT g.game_pk,
                   (SELECT throw_side FROM {}.pitches p WHERE p.game_pk = g.game_pk 
                    AND p.inning >= 1 AND p.inning <= 5 AND p.half_inning = 'top' 
                    LIMIT 1) AS away_starter_hand,
                   (SELECT throw_side FROM {}.pitches p WHERE p.game_pk = g.game_pk 
                    AND p.inning >= 1 AND p.inning <= 5 AND p.half_inning = 'bottom'
                    LIMIT 1) AS home_starter_hand
            FROM {}.games g
            WHERE g.game_pk = ANY(%s)
            """
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema), sql.Identifier(config.schema))
        
        game_pks = list(actuals.keys())
        try:
            cursor.execute(query, (game_pks,))
            for pk, away_hand, home_hand in cursor.fetchall():
                if away_hand and home_hand:
                    game_to_pitchers[int(pk)] = (str(away_hand), str(home_hand))
        except Exception as exc:
            print(f"Pitcher handedness lookup failed: {exc}")
    
    # Analyze by matchup handedness
    lines_by_game = {l.game_pk: l for l in by_type.get("open", []) if l.bookmaker == "draftkings"}
    
    by_matchup: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    
    for game_pk, actual in actuals.items():
        if game_pk not in lines_by_game:
            continue
        line = lines_by_game[game_pk]
        over_result = actual.total_runs > line.total_point
        
        if game_pk in game_to_pitchers:
            away_hand, home_hand = game_to_pitchers[game_pk]
            matchup = f"{away_hand}L_vs_{home_hand}L"
        else:
            matchup = "unknown"
        
        by_matchup[matchup].append((line.over_ml, over_result))
    
    slice_results = {}
    for matchup, results in by_matchup.items():
        if len(results) < MIN_SAMPLE_SIZE:
            continue
        over_count = sum(1 for _, result in results if result)
        over_rate = over_count / len(results)
        avg_ml = sum(ml for ml, _ in results) / len(results)
        market_prob = ml_to_probability(int(avg_ml))
        
        slice_results[matchup] = {
            "n": len(results),
            "over_rate": round(over_rate, PRECISION),
            "market_prob": round(market_prob, PRECISION),
            "cal_error": round(over_rate - market_prob, PRECISION),
        }
    
    return {"by_matchup": slice_results}


def main():
    parser = argparse.ArgumentParser(
        description="F5 odds diagnostics: coverage, movement, calibration"
    )
    parser.add_argument("--season", type=int, default=2025, help="MLB season (default 2025)")
    parser.add_argument("--db-host", help="Postgres host (from env if not set)")
    parser.add_argument("--db-user", help="Postgres user (from env if not set)")
    parser.add_argument("--db-password", help="Postgres password (from env if not set)")
    parser.add_argument("--db-schema", default="mlb", help="Postgres schema (default mlb)")
    parser.add_argument("--limit", type=int, default=0, help="Limit games (0=all)")
    args = parser.parse_args()
    
    # Build config from args or env
    env_config = PostgresConfig.from_env()
    config = PostgresConfig(
        dbname=env_config.dbname,
        user=args.db_user or env_config.user,
        password=args.db_password or env_config.password,
        host=args.db_host or env_config.host,
        port=env_config.port,
        schema=args.db_schema,
    )
    
    print(f"Connected to: {config.describe()}", flush=True)
    
    # Load data
    print(f"Loading F5 odds for season {args.season}...", flush=True)
    by_type = load_f5_lines(config, (args.season,))
    
    # Get all unique game pks
    all_game_pks = set()
    for lines in by_type.values():
        all_game_pks.update(l.game_pk for l in lines)
    all_game_pks = sorted(all_game_pks)
    
    if args.limit > 0:
        all_game_pks = all_game_pks[:args.limit]
    
    print(f"Loading actual F5 totals for {len(all_game_pks)} games...", flush=True)
    actuals = load_f5_actuals(config, all_game_pks)
    
    # Diagnostics
    print("Computing diagnostics...", flush=True)
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "season": args.season,
            "schema": config.schema,
        },
        "coverage": analyze_coverage(by_type),
        "line_movement": analyze_line_movement(by_type),
        "actuals_distribution": analyze_actuals_distribution(actuals),
        "calibration": {
            "open": analyze_over_rate(by_type, actuals, "open"),
            "close": analyze_over_rate(by_type, actuals, "close"),
            "current": analyze_over_rate(by_type, actuals, "current"),
        },
        "slices": analyze_by_slice(by_type, actuals, config),
    }
    
    # Output
    output_dir = Path("output") / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"f5_diagnostics_{args.season}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Diagnostics written to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("F5 ODDS DIAGNOSTICS SUMMARY")
    print("=" * 70)
    
    cov = results["coverage"]
    print("\nCOVERAGE:")
    print(f"  Open:                {cov['open']['unique_games']:>6} games, {cov['open']['total_rows']:>6} rows")
    print(f"  Close:               {cov['close']['unique_games']:>6} games, {cov['close']['total_rows']:>6} rows")
    print(f"  Paired (open+close): {cov['pairing']['open_close_paired']:>6} games")
    
    dist = results["actuals_distribution"]
    if "games_with_data" in dist:
        print(f"\nACTUAL F5 DISTRIBUTION ({dist.get('games_with_data', 'N/A')} games):")
        print(f"  Mean:   {dist.get('mean', 'N/A')}")
        print(f"  Median: {dist.get('median', 'N/A')}")
        print(f"  Range:  {dist.get('min', 'N/A')} - {dist.get('max', 'N/A')}")
    
    for line_type in ["open", "close"]:
        cal = results["calibration"].get(line_type, {})
        if "games" in cal:
            print(f"\nCALIBRATION ({line_type}):")
            print(f"  Games:           {cal['games']:>6}")
            print(f"  Over rate:       {cal['over_rate']:>6.1%}")
            print(f"  Market prob:     {cal['market_over_prob']:>6.1%}")
            print(f"  Cal error:       {cal['calibration_error']:>6.1%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
