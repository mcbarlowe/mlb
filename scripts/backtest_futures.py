#!/usr/bin/env python3
"""Backtest futures betting strategy on historical seasons.

Compares model championship/division/playoff probabilities against historical
preseason futures odds to calculate hypothetical ROI with Kelly-fraction sizing.

Usage:
    # Single season backtest
    uv run python scripts/backtest_futures.py --season 2024 --market championship

    # Multi-season backtest
    uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 --market championship

    # All markets
    uv run python scripts/backtest_futures.py --seasons 2022 2023 2024 2025 --all-markets
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.futures_odds_store import load_latest_futures_odds
from src.betting.odds import american_to_decimal
from src.database import PostgresConfig, PostgresHandler
from src.sim.season import (
    load_season_schedule,
    load_team_info,
    simulate_season,
)
from src.sim.team_strength import fit_strength_predictor, load_completed_games

# Number of winning slots each market settles. A market's implied probabilities sum to this
# figure, not to 1, so it is the correct de-vig target. Getting it wrong scales every fair
# probability by the slot count and fabricates edge in that proportion.
#
# Playoff-field markets are era-dependent: MLB carried two wild cards per league through 2021
# (10 berths) and three from 2022 (12 berths). Using 12 for every season overstates the field by
# two teams before 2022, which inflates every team's fair make-playoffs probability and deflates
# its fair miss-playoffs probability, fabricating edge on the fade side in exactly that
# proportion. ``scripts/futures_outcomes.py`` already derives outcomes era-aware; these targets
# must agree with it.
MARKET_SLOTS: dict[str, int] = {
    "championship": 1,
    "world_series": 1,
    "al_pennant": 1,
    "nl_pennant": 1,
    "league_championship": 1,
    "division": 6,
    "division_series": 8,
}


def playoff_berths(season: int) -> int:
    """Teams qualifying for the playoffs in a given season.

    Three formats, not two. Six division winners plus wild cards is the shape for every season
    except 2020, when the pandemic format seeded the top two in each division plus two wild cards
    per league, giving sixteen. ``6 + 2 * wild_cards`` does not describe that season, so it is
    rejected rather than silently mis-sized. 2020 carries no futures odds and is excluded from
    every backtest season list, so this is a guard rather than a live branch.
    """
    if season == 2020:
        raise SystemExit(
            "2020 used a 16-team pandemic playoff format that the division-plus-wildcard "
            "formula does not describe; exclude 2020 rather than backtesting it"
        )
    return 6 + 2 * (3 if season >= 2022 else 2)


def wildcards_per_league(season: int) -> int:
    """Wild cards per league. Must match scripts/futures_outcomes.py."""
    if season == 2020:
        return 2
    return 3 if season >= 2022 else 2


def market_slots(market_type: str, season: int) -> int | None:
    """Winning-slot count for a market in a given season."""
    if market_type in ("playoff", "make_playoffs"):
        return playoff_berths(season)
    if market_type == "miss_playoffs":
        return 30 - playoff_berths(season)
    return MARKET_SLOTS.get(market_type)


@dataclass
class BacktestBet:
    """Single hypothetical futures bet."""
    season: int
    market_type: str
    team_id: int
    team_name: str
    model_prob: float
    best_odds: int
    decimal_odds: float
    edge: float
    kelly_fraction: float
    stake: float
    actual_win: bool
    profit: float


def _load_actual_outcomes(season: int, pg) -> dict[str, set[int]]:
    """Load actual outcomes for a completed season.
    
    Returns dict of market_type -> set of winning team_ids.
    """
    outcomes: dict[str, set[int]] = {}
    
    # Hardcoded outcomes for backtest (2022-2025)
    # In production, query mlb.postseason_results table
    
    if season == 2022:
        outcomes["championship"] = {117}  # Astros
        outcomes["al_pennant"] = {117}  # Astros
        outcomes["nl_pennant"] = {143}  # Phillies
        # Division winners 2022
        outcomes["division"] = {147, 114, 117, 144, 158, 119}  # NYY, CLE, HOU, ATL, MIL, LAD
        # Playoff teams 2022 (12 teams: 6 AL, 6 NL)
        outcomes["make_playoffs"] = {147, 117, 136, 114, 141, 142, 144, 121, 119, 158, 143, 135}
        # NYY, HOU, SEA, CLE, TB, TOR, ATL, NYM, LAD, MIL, PHI, SD
    elif season == 2023:
        outcomes["championship"] = {140}  # Rangers
        outcomes["al_pennant"] = {140}  # Rangers
        outcomes["nl_pennant"] = {109}  # Diamondbacks
        # Division winners 2023
        outcomes["division"] = {141, 136, 117, 144, 158, 119}  # TB, SEA, HOU, ATL, MIL, LAD (?)
        # Playoff teams 2023
        outcomes["make_playoffs"] = {141, 110, 136, 117, 140, 142, 144, 143, 119, 158, 109, 121}
        # TB, BAL, SEA, HOU, TEX, TOR, ATL, PHI, LAD, MIL, ARI, NYM
    elif season == 2024:
        outcomes["championship"] = {119}  # Dodgers
        outcomes["al_pennant"] = {147}  # Yankees
        outcomes["nl_pennant"] = {119}  # Dodgers
        # Division winners 2024
        outcomes["division"] = {147, 114, 117, 143, 158, 119}  # NYY, CLE, HOU, PHI, MIL, LAD
        # Playoff teams 2024 (12 teams)
        outcomes["make_playoffs"] = {147, 110, 114, 118, 117, 116, 143, 144, 158, 119, 135, 121}
        # NYY, BAL, CLE, KC, HOU, DET, PHI, ATL, MIL, LAD, SD, NYM
    elif season == 2025:
        outcomes["championship"] = {119}  # Dodgers
        outcomes["al_pennant"] = {147}  # Yankees
        outcomes["nl_pennant"] = {119}  # Dodgers
        # Division winners 2025 (placeholder - update with actual)
        outcomes["division"] = {142, 114, 136, 143, 158, 119}  # TOR, CLE, SEA, PHI, MIL, LAD
        # Playoff teams 2025 (placeholder - update with actual)
        outcomes["make_playoffs"] = {142, 147, 114, 136, 117, 110, 143, 144, 158, 119, 135, 109}
        # TOR, NYY, CLE, SEA, HOU, BAL, PHI, ATL, MIL, LAD, SD, ARI
    # Multi-winner markets were hardcoded and several were wrong: 2022 NL Central was
    # St. Louis not Milwaukee, and 2023 had Seattle and Toronto in place of Baltimore and
    # Minnesota. Derive them from actual game results instead. Championship and pennant are
    # single postseason outcomes that cannot come from regular-season records, so those stay
    # as declared above.
    from scripts.futures_outcomes import derive_outcomes

    derived, notes = derive_outcomes(pg.connection, PostgresConfig.from_env().schema, season)
    for note in notes:
        print(f"  outcome note: {note}")
    outcomes.update(derived)
    return outcomes



def _run_season_backtest(
    season: int,
    market_type: str,
    edge_threshold: float,
    kelly_multiplier: float,
    max_stake_pct: float,
    pg,
) -> list[BacktestBet]:
    """Run backtest for one season and market type."""
    
    # Load historical odds
    odds_rows = load_latest_futures_odds(
        pg,
        season=season,
        market_type=market_type,
    )
    
    if not odds_rows:
        print(f"Warning: no historical odds found for {season} {market_type}")
        return []
    
    # Generate model projection
    print(f"Generating model projection for {season}...")
    
    # Load team info
    teams = load_team_info()
    
    # Load season schedule
    games = load_season_schedule(season)
    
    # Load prior seasons for training predictor
    train_start = season - 5
    completed_games = load_completed_games(start_season=train_start, end_season=season - 1)
    
    # Fit predictor
    train_seasons = list(range(season - 4, season))
    predictor, _ = fit_strength_predictor(
        completed_games,
        prediction_season=season,
        train_seasons=train_seasons,
    )
    
    # Run simulation (this builds and simulates the projection)
    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=datetime(season, 3, 15, tzinfo=UTC).date(),
        trials=10_000,
        predictor=predictor,
        wild_cards_per_league=wildcards_per_league(season),
    )
    team_odds: dict[int, list[int]] = defaultdict(list)
    for row in odds_rows:
        team_odds[row["team_id"]].append(row["american_odds"])
    
    # Best available price per team: highest American odds is the longest price, whether the
    # team is a favourite or a longshot.
    team_best_odds: dict[int, int] = {}
    for team_id, odds_list in team_odds.items():
        team_best_odds[team_id] = max(odds_list)

    # De-vig target. A market's implied probabilities sum to the number of winning slots it
    # settles, not to 1. Division has six winners; the playoff field is era-dependent, ten
    # berths through 2021 and twelve from 2022. Normalising to the wrong count scales every fair
    # probability and fabricates edge in exactly that proportion.
    slots = market_slots(market_type, season)
    if slots is None:
        raise SystemExit(
            f"market_type {market_type!r} has no declared winning-slot count; add it to "
            "MARKET_SLOTS or market_slots() rather than assuming a single winner"
        )
    all_implied = [1.0 / american_to_decimal(o) for o in team_best_odds.values()]
    overround = sum(all_implied) / slots
    if overround <= 0:
        raise SystemExit(f"non-positive overround for {market_type} {season}")
    
    # Load actual outcomes
    outcomes = _load_actual_outcomes(season, pg)
    winners = outcomes.get(market_type, set())
    
    # Generate bets
    bets: list[BacktestBet] = []
    
    for team_id, best_odds in team_best_odds.items():
        # Get model probability from projection
        team_proj = next((t for t in projection.teams if t.team_id == team_id), None)
        if team_proj is None:
            continue
        
        if market_type == "championship":
            model_prob = team_proj.championship_prob
        elif market_type in ("al_pennant", "nl_pennant"):
            model_prob = team_proj.league_championship_prob
        elif market_type == "division":
            model_prob = team_proj.division_win_prob
        elif market_type in ("playoff", "make_playoffs"):
            model_prob = team_proj.playoff_prob
        elif market_type == "miss_playoffs":
            model_prob = 1.0 - team_proj.playoff_prob
        else:
            print(f"Warning: {market_type} not yet implemented, skipping")
            continue
        
        if model_prob == 0:
            continue
        
        # Calculate edge
        decimal_odds = american_to_decimal(best_odds)
        implied_prob = (1.0 / decimal_odds) / overround
        edge = model_prob - implied_prob
        
        if edge < edge_threshold:
            continue
        
        # Kelly sizing
        kelly_fraction = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
        kelly_fraction = max(0, kelly_fraction)  # Don't bet negative edge
        stake_pct = kelly_multiplier * kelly_fraction
        stake_pct = min(stake_pct, max_stake_pct)  # Cap
        
        # Check if won
        actual_win = team_id in winners
        
        # Calculate profit (stake = 1.0 for percentage terms)
        if actual_win:
            profit = stake_pct * (decimal_odds - 1)
        else:
            profit = -stake_pct
        
        # Get team name
        team_name = next(
            (row["team_name"] for row in odds_rows if row["team_id"] == team_id),
            f"Team {team_id}",
        )
        
        bets.append(BacktestBet(
            season=season,
            market_type=market_type,
            team_id=team_id,
            team_name=team_name,
            model_prob=model_prob,
            best_odds=best_odds,
            decimal_odds=decimal_odds,
            edge=edge,
            kelly_fraction=kelly_fraction,
            stake=stake_pct,
            actual_win=actual_win,
            profit=profit,
        ))
    
    return bets


def _print_backtest_results(
    bets: list[BacktestBet],
    *,
    seasons: list[int] | None = None,
    market_type: str | None = None,
) -> None:
    """Print formatted backtest results."""
    
    if not bets:
        print("No bets generated (no edges found)")
        return
    
    total_stake = sum(bet.stake for bet in bets)
    total_profit = sum(bet.profit for bet in bets)
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
    
    winners = [b for b in bets if b.actual_win]
    losers = [b for b in bets if not b.actual_win]
    win_rate = len(winners) / len(bets) * 100 if bets else 0
    
    print(f"\n{'='*80}")
    if seasons:
        print(f"BACKTEST RESULTS - Seasons {min(seasons)}-{max(seasons)}")
    if market_type:
        print(f"Market: {market_type}")
    print(f"{'='*80}")
    
    print("\nOverall Performance:")
    print(f"  Total bets: {len(bets)}")
    print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers: {len(losers)}")
    print(f"  Total stake: {total_stake:.3f} units")
    print(f"  Total profit: {total_profit:+.3f} units")
    print(f"  ROI: {roi:+.2f}%")
    
    # Breakdown by season
    if len({b.season for b in bets}) > 1:
        print("\nBy Season:")
        for season in sorted({b.season for b in bets}):
            season_bets = [b for b in bets if b.season == season]
            season_stake = sum(b.stake for b in season_bets)
            season_profit = sum(b.profit for b in season_bets)
            season_roi = (season_profit / season_stake * 100) if season_stake > 0 else 0
            season_winners = sum(1 for b in season_bets if b.actual_win)
            
            print(
                f"  {season}: {len(season_bets)} bets, "
                f"{season_winners} wins, "
                f"{season_roi:+.1f}% ROI, "
                f"{season_profit:+.3f} profit"
            )
    
    # Show all bets
    print("\nAll Bets:")
    print(f"{'Season':<8} {'Team':<25} {'Model%':>7} {'Odds':>6} {'Edge%':>6} "
          f"{'Stake':>6} {'Result':<8} {'Profit':>8}")
    print("-" * 80)
    
    for bet in sorted(bets, key=lambda b: (b.season, -b.edge)):
        result = "WIN ✓" if bet.actual_win else "LOSS"
        print(
            f"{bet.season:<8} {bet.team_name:<25} "
            f"{bet.model_prob*100:>6.2f}% {bet.best_odds:>6d} {bet.edge*100:>5.1f}% "
            f"{bet.stake:>6.3f} {result:<8} {bet.profit:>+8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        help="Single season to backtest",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Multiple seasons to backtest",
    )
    parser.add_argument(
        "--market",
        default="championship",
        help="Market type (championship, playoffs, al_pennant, nl_pennant)",
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Run backtest for all available markets",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum edge to bet (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--kelly-multiplier",
        type=float,
        default=0.25,
        help="Kelly fraction multiplier (default: 0.25 = quarter Kelly)",
    )
    parser.add_argument(
        "--max-stake",
        type=float,
        default=0.05,
        help="Maximum stake per bet as fraction (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()
    
    # Determine seasons
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = sorted(args.seasons)
    else:
        raise SystemExit("Must specify --season or --seasons")
    
    # Determine markets
    if args.all_markets:
        markets = ["championship", "al_pennant", "nl_pennant", "playoffs"]
    else:
        markets = [args.market]
    
    db_config = PostgresConfig.from_env()
    all_bets = []
    
    with PostgresHandler(db_config) as pg:
        for market_type in markets:
            print(f"\n{'='*80}")
            print(f"Backtesting {market_type} - Seasons {seasons}")
            print(f"{'='*80}")
            
            market_bets = []
            
            for season in seasons:
                print(f"\nProcessing {season} {market_type}...")
                
                season_bets = _run_season_backtest(
                    season=season,
                    market_type=market_type,
                    edge_threshold=args.edge_threshold,
                    kelly_multiplier=args.kelly_multiplier,
                    max_stake_pct=args.max_stake,
                    pg=pg,
                )
                
                market_bets.extend(season_bets)
                all_bets.extend(season_bets)
            
            if market_bets:
                _print_backtest_results(
                    market_bets,
                    seasons=seasons,
                    market_type=market_type,
                )
    
    # Overall summary if multiple markets
    if len(markets) > 1 and all_bets:
        _print_backtest_results(
            all_bets,
            seasons=seasons,
        )


if __name__ == "__main__":
    main()
