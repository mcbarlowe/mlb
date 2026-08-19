#!/usr/bin/env python3
"""Settle completed futures paper trades based on actual season outcomes.

Reads open futures paper trades from mlb.futures_paper_trades and settles
them based on actual playoff/championship outcomes from the database.

Usage:
    uv run python scripts/settle_futures_paper_trades.py --season 2026 --dry-run
    uv run python scripts/settle_futures_paper_trades.py --season 2026 --db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.futures_paper_trade_store import (
    load_open_futures_paper_trades,
    settle_futures_paper_trade,
)
from src.database import PostgresConfig, PostgresHandler


def _load_actual_outcomes(
    pg: PostgresHandler, season: int
) -> dict[str, dict[int, bool]]:
    """Load actual season outcomes from database.
    
    Returns:
        {market_type: {team_id: won_or_made}}
    """
    # Query for playoff teams
    playoff_sql = """
        SELECT DISTINCT team_id
        FROM games
        WHERE season = %s
          AND game_type = 'P'
          AND (home_team_id = team_id OR away_team_id = team_id)
    """
    
    # Query for division winners (first place in division at season end)
    division_sql = """
        WITH final_standings AS (
            SELECT 
                team_id,
                division_id,
                COUNT(*) FILTER (WHERE home_won) as wins
            FROM (
                SELECT home_team_id as team_id, home_division_id as division_id,
                       home_runs > away_runs as home_won
                FROM games
                WHERE season = %s AND game_type = 'R'
                UNION ALL
                SELECT away_team_id as team_id, away_division_id as division_id,
                       away_runs > home_runs as home_won
                FROM games
                WHERE season = %s AND game_type = 'R'
            ) team_games
            GROUP BY team_id, division_id
        ),
        division_leaders AS (
            SELECT team_id, division_id,
                   ROW_NUMBER() OVER (PARTITION BY division_id ORDER BY wins DESC) as rank
            FROM final_standings
        )
        SELECT team_id FROM division_leaders WHERE rank = 1
    """
    
    # Query for World Series champion
    champion_sql = """
        SELECT winner_id as team_id
        FROM postseason_results
        WHERE season = %s
          AND series_code = 'WS'
        LIMIT 1
    """
    
    outcomes: dict[str, dict[int, bool]] = {
        "playoff": {},
        "division": {},
        "championship": {},
    }
    
    with pg.cursor() as cursor:
        # Playoff teams
        cursor.execute(playoff_sql, (season,))
        playoff_teams = {row[0] for row in cursor.fetchall()}
        for team_id in playoff_teams:
            outcomes["playoff"][team_id] = True
        
        # Division winners
        cursor.execute(division_sql, (season, season))
        division_winners = {row[0] for row in cursor.fetchall()}
        for team_id in division_winners:
            outcomes["division"][team_id] = True
        
        # Champion
        cursor.execute(champion_sql, (season,))
        champion_row = cursor.fetchone()
        if champion_row:
            champion_id = champion_row[0]
            outcomes["championship"][champion_id] = True
    
    return outcomes


def _calculate_profit(
    american_odds: int, stake_units: float, won: bool
) -> float:
    """Calculate profit in units for a futures bet.
    
    Args:
        american_odds: American odds (e.g. +500, -150)
        stake_units: Stake size in units
        won: Whether the bet won
    
    Returns:
        Profit in units (negative if lost)
    """
    if not won:
        return -stake_units
    
    # Win profit calculation
    if american_odds > 0:
        # Positive odds: profit = stake * (odds / 100)
        profit = stake_units * (american_odds / 100.0)
    else:
        # Negative odds: profit = stake * (100 / abs(odds))
        profit = stake_units * (100.0 / abs(american_odds))
    
    return profit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year to settle",
    )
    parser.add_argument(
        "--strategy-version",
        default="futures_baseline_model_v1",
        help="Strategy version to settle (default: futures_baseline_model_v1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show settlement results but don't write to database",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write settlement results to mlb.futures_paper_trades",
    )
    args = parser.parse_args()
    
    db_config = PostgresConfig.from_env()
    
    with PostgresHandler(db_config) as pg:
        # Load open trades
        open_trades = load_open_futures_paper_trades(
            pg,
            strategy_version=args.strategy_version,
            season=args.season,
        )
    
    if not open_trades:
        print(f"No open futures paper trades found for {args.season}")
        return
    
    print(f"Found {len(open_trades)} open futures paper trades for {args.season}")
    
    # Load actual outcomes
    with PostgresHandler(db_config) as pg:
        outcomes = _load_actual_outcomes(pg, args.season)
    
    # Settle each trade
    settlements = []
    
    for trade in open_trades:
        market_type = trade["market_type"]
        team_id = int(trade["team_id"])
        team_name = trade["team_name"]
        stake_units = float(trade["stake_units"])
        american_odds = int(trade["best_american_odds"])
        
        # Check if team won
        won = outcomes.get(market_type, {}).get(team_id, False)
        result = "won" if won else "lost"
        
        # Calculate profit
        profit_units = _calculate_profit(american_odds, stake_units, won)
        
        settlements.append({
            "market_type": market_type,
            "team_id": team_id,
            "team_name": team_name,
            "stake_units": stake_units,
            "american_odds": american_odds,
            "result": result,
            "profit_units": profit_units,
        })
    
    # Display results
    print(f"\nSettlement summary:")
    
    total_staked = sum(s["stake_units"] for s in settlements)
    total_profit = sum(s["profit_units"] for s in settlements)
    wins = sum(1 for s in settlements if s["result"] == "won")
    losses = len(settlements) - wins
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    
    print(f"  Total bets: {len(settlements)}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Total staked: {total_staked:.4f} units")
    print(f"  Total profit: {total_profit:+.4f} units")
    print(f"  ROI: {roi:+.1f}%")
    
    print(f"\nIndividual results:")
    for s in settlements:
        result_str = "✓ WON " if s["result"] == "won" else "✗ LOST"
        print(
            f"  {s['market_type']:20s} {s['team_name']:25s} "
            f"{s['american_odds']:>6.0f} {result_str} "
            f"stake={s['stake_units']:.4f} profit={s['profit_units']:+.4f}"
        )
    
    if args.dry_run:
        print("\n--dry-run: not writing to database")
        return
    
    if not args.db:
        print("Use --db to write to database or --dry-run to test")
        return
    
    # Write settlements
    with PostgresHandler(db_config) as pg:
        for settlement in settlements:
            settled = settle_futures_paper_trade(
                pg,
                strategy_version=args.strategy_version,
                season=args.season,
                market_type=settlement["market_type"],
                team_id=settlement["team_id"],
                result=settlement["result"],
                profit_units=settlement["profit_units"],
            )
            if settled:
                print(
                    f"Settled {settlement['market_type']} "
                    f"{settlement['team_name']} → {settlement['result']}"
                )
    
    print(f"\nSettled {len(settlements)} futures paper trades")


if __name__ == "__main__":
    main()
