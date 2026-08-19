#!/usr/bin/env python3
"""Generate futures paper trades from database odds and model projections.

Reads the latest futures odds from mlb.futures_odds, generates season
projections with the champion team-strength model, calculates edges,
and writes paper trade recommendations to mlb.futures_paper_trades.

Usage:
    uv run python scripts/paper_trade_futures_from_db.py --season 2027 --dry-run
    uv run python scripts/paper_trade_futures_from_db.py --season 2027 --db
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.futures_odds_store import load_latest_futures_odds
from src.betting.futures_paper_trade_store import (
    ensure_futures_paper_trades_table,
    insert_futures_paper_trades,
)
from src.betting.odds import american_to_decimal, no_vig_outright
from src.database import PostgresConfig, PostgresHandler
from src.sim.season import (
    SeasonProjection,
    build_baseline_projection,
    simulate_season,
)
from src.sim.team_strength import (
    fit_strength_predictor,
    load_completed_games,
    load_team_info,
)

STRATEGY_VERSION = "futures_baseline_model_v1"


def _generate_season_projection(
    season: int,
    *,
    as_of_date: date,
    trials: int = 5000,
) -> SeasonProjection:
    """Generate season projection using baseline team-strength model.
    
    Args:
        season: Season year to project
        as_of_date: Projection as-of date
        trials: Monte Carlo trials
    
    Returns:
        Season projection with playoff/championship probabilities
    """
    # Load team info
    teams = load_team_info()
    
    # Load completed games through prior season
    # For preseason projection, we'd use prior seasons only
    train_start = season - 5
    games = load_completed_games(start_season=train_start, end_season=season - 1)
    
    # Fit predictor on prior seasons
    train_seasons = list(range(season - 4, season))
    predictor, _ = fit_strength_predictor(
        games,
        prediction_season=season,
        train_seasons=train_seasons,
    )
    
    # Build baseline projection (no priors)
    projection = build_baseline_projection(
        season,
        teams=teams,
        as_of_date=as_of_date,
    )
    
    # Run simulation
    projection = simulate_season(
        projection,
        trials=trials,
        predictor=predictor,
    )
    
    return projection


def _calculate_futures_edges(
    projection: SeasonProjection,
    odds_rows: list[dict[str, object]],
    *,
    market_type: str,
    edge_threshold: float = 0.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
) -> list[dict[str, object]]:
    """Calculate edges and generate paper trade rows.
    
    Args:
        projection: Season projection with team probabilities
        odds_rows: Raw odds from database
        market_type: Market to analyze (championship, division, playoff)
        edge_threshold: Minimum edge to include
        kelly_multiplier: Fractional Kelly multiplier
        kelly_cap: Maximum stake in units
    
    Returns:
        List of paper trade row dicts
    """
    # Extract model probabilities
    model_probs = {}
    for team_proj in projection.teams:
        team_id = team_proj.team.team_id
        
        if market_type == "championship":
            prob = team_proj.championship_prob
        elif market_type == "playoff":
            prob = team_proj.playoff_prob
        elif market_type == "division":
            prob = team_proj.division_win_prob
        else:
            raise ValueError(f"Unknown market_type: {market_type}")
        
        model_probs[team_id] = prob
    
    # Group odds by team and find best price
    best_odds_by_team: dict[int, dict[str, object]] = {}
    
    for odds_row in odds_rows:
        team_id = int(odds_row["team_id"])
        american_odds = int(odds_row["american_odds"])
        bookmaker = str(odds_row["bookmaker"])
        
        if team_id not in best_odds_by_team:
            best_odds_by_team[team_id] = {
                "team_id": team_id,
                "team_name": odds_row["team_name"],
                "best_bookmaker": bookmaker,
                "best_american_odds": american_odds,
                "best_decimal_odds": american_to_decimal(american_odds),
                "all_odds": [],
            }
        
        best_odds_by_team[team_id]["all_odds"].append({
            "bookmaker": bookmaker,
            "american_odds": american_odds,
            "decimal_odds": american_to_decimal(american_odds),
        })
        
        # Update if this is better (higher decimal odds = better for bettor)
        current_decimal = american_to_decimal(american_odds)
        best_decimal = best_odds_by_team[team_id]["best_decimal_odds"]
        if current_decimal > best_decimal:
            best_odds_by_team[team_id]["best_bookmaker"] = bookmaker
            best_odds_by_team[team_id]["best_american_odds"] = american_odds
            best_odds_by_team[team_id]["best_decimal_odds"] = current_decimal
    
    # Calculate no-vig consensus probability
    all_decimal_odds = []
    for team_data in best_odds_by_team.values():
        for odds_data in team_data["all_odds"]:
            all_decimal_odds.append(odds_data["decimal_odds"])
    
    # De-vig using the best-book method for outrights
    consensus_probs = {}
    if all_decimal_odds:
        # Simple approach: use best odds per team as fair probability proxy
        for team_id, team_data in best_odds_by_team.items():
            decimal = team_data["best_decimal_odds"]
            raw_prob = 1.0 / decimal
            consensus_probs[team_id] = raw_prob
        
        # Normalize to sum to 1.0 (removes vig)
        total = sum(consensus_probs.values())
        if total > 0:
            consensus_probs = {tid: p / total for tid, p in consensus_probs.items()}
    
    # Generate paper trade rows
    paper_trades = []
    
    for team_id, team_data in best_odds_by_team.items():
        model_prob = model_probs.get(team_id, 0.0)
        consensus_prob = consensus_probs.get(team_id, 0.0)
        edge = model_prob - consensus_prob
        
        if edge < edge_threshold:
            continue
        
        # Kelly stake
        decimal_odds = team_data["best_decimal_odds"]
        if decimal_odds > 1.0:
            # Kelly fraction = (prob * (decimal - 1) - (1 - prob)) / (decimal - 1)
            kelly_full = (model_prob * (decimal_odds - 1) - (1 - model_prob)) / (decimal_odds - 1)
            kelly_fraction = max(0.0, kelly_full * kelly_multiplier)
            stake_units = min(kelly_fraction, kelly_cap)
        else:
            stake_units = 0.0
        
        if stake_units <= 0:
            continue
        
        paper_trades.append({
            "team_id": team_id,
            "team_name": team_data["team_name"],
            "model_probability": model_prob,
            "consensus_market_prob": consensus_prob,
            "edge": edge,
            "best_bookmaker": team_data["best_bookmaker"],
            "best_american_odds": team_data["best_american_odds"],
            "best_decimal_odds": decimal_odds,
            "best_fair_prob": 1.0 / decimal_odds if decimal_odds > 0 else 0.0,
            "staking": f"{kelly_multiplier:.2f}-kelly",
            "stake_fraction": kelly_fraction if stake_units > 0 else 0.0,
            "stake_units": stake_units,
        })
    
    # Sort by edge descending
    paper_trades.sort(key=lambda x: x["edge"], reverse=True)
    
    return paper_trades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year for futures bets",
    )
    parser.add_argument(
        "--market-type",
        choices=["championship", "division", "playoff"],
        default="championship",
        help="Futures market to analyze (default: championship)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum edge to generate a paper trade (default: 0.05)",
    )
    parser.add_argument(
        "--kelly-multiplier",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (default: 0.25)",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.05,
        help="Maximum stake in units (default: 0.05)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5000,
        help="Monte Carlo trials for projection (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate paper trades but don't write to database",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write paper trades to mlb.futures_paper_trades",
    )
    args = parser.parse_args()
    
    # Load odds from database
    db_config = PostgresConfig.from_env()
    
    with PostgresHandler(db_config) as pg:
        odds_rows = load_latest_futures_odds(
            pg,
            season=args.season,
            market_type=args.market_type,
        )
    
    if not odds_rows:
        print(f"No futures odds found for season {args.season} market {args.market_type}")
        print("Run fetch_futures_odds.py first to populate mlb.futures_odds")
        return
    
    print(f"Loaded {len(odds_rows)} odds rows for {args.market_type}")
    
    # Generate projection
    print(f"Generating {args.season} season projection ({args.trials} trials)...")
    projection = _generate_season_projection(
        args.season,
        as_of_date=datetime.now(UTC).date(),
        trials=args.trials,
    )
    
    # Calculate edges
    print("Calculating edges...")
    paper_trades = _calculate_futures_edges(
        projection,
        odds_rows,
        market_type=args.market_type,
        edge_threshold=args.edge_threshold,
        kelly_multiplier=args.kelly_multiplier,
        kelly_cap=args.kelly_cap,
    )
    
    print(f"\nGenerated {len(paper_trades)} paper trades (edge >= {args.edge_threshold:.2%})")
    
    if paper_trades:
        print("\nTop 10 edges:")
        for i, trade in enumerate(paper_trades[:10], 1):
            print(
                f"  {i:2d}. {trade['team_name']:25s} "
                f"model={trade['model_probability']:.3f} "
                f"market={trade['consensus_market_prob']:.3f} "
                f"edge={trade['edge']:+.3f} "
                f"odds={trade['best_american_odds']:>6.0f} "
                f"stake={trade['stake_units']:.4f}"
            )
    
    if args.dry_run:
        print("\n--dry-run: not writing to database")
        return
    
    if not args.db:
        print("Use --db to write to database or --dry-run to test")
        return
    
    # Prepare rows for insertion
    snapshot_time = datetime.now(UTC).isoformat()
    paper_date = datetime.now(UTC).date()
    
    insert_rows = []
    for trade in paper_trades:
        insert_rows.append({
            "strategy_version": STRATEGY_VERSION,
            "season": args.season,
            "market_type": args.market_type,
            "team_id": trade["team_id"],
            "team_name": trade["team_name"],
            "paper_date": paper_date,
            "snapshot_time": snapshot_time,
            "model_probability": trade["model_probability"],
            "consensus_market_prob": trade["consensus_market_prob"],
            "edge": trade["edge"],
            "best_bookmaker": trade["best_bookmaker"],
            "best_american_odds": trade["best_american_odds"],
            "best_decimal_odds": trade["best_decimal_odds"],
            "best_fair_prob": trade["best_fair_prob"],
            "staking": trade["staking"],
            "stake_fraction": trade["stake_fraction"],
            "stake_units": trade["stake_units"],
            "status": "open",
        })
    
    # Write to database
    with PostgresHandler(db_config) as pg:
        ensure_futures_paper_trades_table(pg)
        inserted = insert_futures_paper_trades(pg, insert_rows)
        print(f"\nInserted/updated {inserted} paper trades in mlb.futures_paper_trades")


if __name__ == "__main__":
    main()
