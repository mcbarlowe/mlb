#!/usr/bin/env python3
"""Evaluate selection-fixed moneyline line shopping.

The consensus opening market still chooses the side and edge threshold. The
simulation then executes that side at the best available sportsbook opening
price, so the report measures price-shopping lift separately from model changes.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.ingest import champion_home_probs, load_finals
from src.betting.line_shopping import (
    BookLine,
    LineShoppingBet,
    LineShoppingGame,
    LineShoppingSummary,
    line_shop_moneyline,
    summarize_line_shopping,
)
from src.betting.odds import american_to_decimal, decimal_to_american
from src.database import PostgresConfig


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _consensus_american(lines: Sequence[BookLine]) -> tuple[float, float] | None:
    if not lines:
        return None
    home_decimal = statistics.median(american_to_decimal(line.home_ml) for line in lines)
    away_decimal = statistics.median(american_to_decimal(line.away_ml) for line in lines)
    return decimal_to_american(home_decimal), decimal_to_american(away_decimal)


def load_line_shopping_games(
    seasons: Sequence[int],
) -> tuple[list[LineShoppingGame], dict[int, int]]:
    probs_df = champion_home_probs(seasons)
    probs = {
        int(cast(Any, row.game_pk)): float(cast(Any, row.model_prob_home))
        for row in probs_df.itertuples()
    }
    finals = load_finals(seasons).set_index("game_pk")
    config = PostgresConfig.from_env()

    by_game: dict[int, dict[str, dict[str, BookLine]]] = defaultdict(
        lambda: {"open": {}, "close": {}}
    )
    game_seasons: dict[int, int] = {}
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    ) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT g.season::int, o.game_pk, o.bookmaker, o.line_type,
                   o.home_ml, o.away_ml
            FROM {}.odds o JOIN {}.games g USING(game_pk)
            WHERE g.season::int = ANY(%s)
              AND g.game_type = 'R'
              AND o.market = 'h2h'
              AND o.line_type IN ('open', 'close')
              AND o.home_ml IS NOT NULL
              AND o.away_ml IS NOT NULL
            """
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
        cursor.execute(query, (list(seasons),))
        for season, game_pk, bookmaker, line_type, home_ml, away_ml in cursor.fetchall():
            pk = int(game_pk)
            game_seasons[pk] = int(season)
            by_game[pk][str(line_type)][str(bookmaker)] = BookLine(
                bookmaker=str(bookmaker),
                home_ml=float(home_ml),
                away_ml=float(away_ml),
            )

    games: list[LineShoppingGame] = []
    coverage: Counter[int] = Counter()
    for game_pk in sorted(by_game):
        if game_pk not in probs or game_pk not in finals.index:
            continue
        open_lines = tuple(by_game[game_pk]["open"].values())
        close_lines = tuple(by_game[game_pk]["close"].values())
        consensus_open = _consensus_american(open_lines)
        consensus_close = _consensus_american(close_lines)
        if consensus_open is None or consensus_close is None:
            continue
        season = game_seasons[game_pk]
        games.append(
            LineShoppingGame(
                game_pk=game_pk,
                season=season,
                model_prob_home=probs[game_pk],
                consensus_open_home=consensus_open[0],
                consensus_open_away=consensus_open[1],
                consensus_close_home=consensus_close[0],
                consensus_close_away=consensus_close[1],
                open_lines=open_lines,
                close_lines=close_lines,
                home_won=bool(finals.loc[game_pk, "home_won"]),
            )
        )
        coverage[season] += 1
    return games, dict(sorted(coverage.items()))


def _bootstrap_lift(
    bets: Sequence[LineShoppingBet], *, reps: int, seed: int = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not bets or reps <= 0:
        return (0.0, 0.0), (0.0, 0.0)
    rng = random.Random(seed)
    roi_values: list[float] = []
    clv_values: list[float] = []
    for _ in range(reps):
        sample = [bets[rng.randrange(len(bets))] for _ in bets]
        summary = summarize_line_shopping(n_games=len(sample), bets=sample)
        roi_values.append(summary.roi_lift)
        clv_values.append(summary.clv_lift_vs_consensus_close)
    roi_values.sort()
    clv_values.sort()
    lo = int(0.025 * reps)
    hi = int(0.975 * reps) - 1
    return (roi_values[lo], roi_values[hi]), (clv_values[lo], clv_values[hi])


def _staking_plans(value: str) -> tuple[str, ...]:
    if value == "both":
        return "flat", "kelly"
    return (value,)


def _report_summary(
    edge: float,
    summary: LineShoppingSummary,
    bets: Sequence[LineShoppingBet],
    boot: int,
) -> None:
    roi_ci, clv_ci = _bootstrap_lift(bets, reps=boot)
    print(
        f"edge>={edge:.2f}: bets {summary.n_bets:4d} | "
        f"ROI consensus {summary.consensus_roi:+.2%} -> "
        f"best {summary.best_roi:+.2%} "
        f"lift {summary.roi_lift:+.2%} "
        f"CI [{roi_ci[0]:+.2%}, {roi_ci[1]:+.2%}] | "
        f"CLV consensus {summary.consensus_avg_clv:+.4f} -> "
        f"best {summary.best_avg_clv_vs_consensus_close:+.4f} "
        f"lift {summary.clv_lift_vs_consensus_close:+.4f} "
        f"CI [{clv_ci[0]:+.4f}, {clv_ci[1]:+.4f}] | "
        f"beat close {summary.consensus_pct_beat_close:.1%} -> "
        f"{summary.best_pct_beat_consensus_close:.1%} | "
        f"staked {summary.consensus_total_staked:.2f}u -> "
        f"{summary.best_total_staked:.2f}u | "
        f"net lift {summary.best_net_profit - summary.consensus_net_profit:+.1f}u"
    )


def _report_source_breakdown(
    *, edge: float, bets: Sequence[LineShoppingBet], rows: int
) -> None:
    print(f"\nBest-price source books for flat edge>={edge:.2f} (ties fractional)")
    grouped: dict[str, list[tuple[float, LineShoppingBet]]] = defaultdict(list)
    for bet in bets:
        sources = bet.source_books or (bet.source_book,)
        weight = 1.0 / len(sources)
        for source in sources:
            grouped[source].append((weight, bet))
    source_rows = []
    for source, weighted_bets in grouped.items():
        bet_count = sum(weight for weight, _bet in weighted_bets)
        best_staked = sum(weight * bet.best_stake for weight, bet in weighted_bets)
        best_profit = sum(weight * bet.best_profit for weight, bet in weighted_bets)
        best_roi = best_profit / best_staked if best_staked else 0.0
        best_clv = (
            sum(weight * bet.best_clv_vs_consensus_close for weight, bet in weighted_bets)
            / bet_count
        )
        decimal_lift = (
            sum(
                weight * (bet.best_decimal - bet.consensus_decimal)
                for weight, bet in weighted_bets
            )
            / bet_count
        )
        net_lift = sum(
            weight * (bet.best_profit - bet.consensus_profit)
            for weight, bet in weighted_bets
        )
        source_rows.append(
            (bet_count, source, best_roi, best_clv, decimal_lift, net_lift)
        )
    source_rows.sort(reverse=True)
    print("book                         bets* bestROI  bestCLV  avgDecLift  netLift")
    for (
        bet_count,
        source,
        best_roi,
        best_clv,
        decimal_lift,
        net_lift,
    ) in source_rows[:rows]:
        print(
            f"{source[:28]:28s} {bet_count:5.1f} {best_roi:+8.2%} {best_clv:+8.4f} "
            f"{decimal_lift:+10.4f} {net_lift:+8.1f}u"
        )


def _report_walkforward(
    *,
    games: Sequence[LineShoppingGame],
    edges: Sequence[float],
    train: int,
    test: int,
    staking: str,
) -> None:
    train_games = [game for game in games if game.season == train]
    test_games = [game for game in games if game.season == test]
    train_rows = []
    for edge in edges:
        summary, _ = line_shop_moneyline(train_games, edge_threshold=edge, staking=staking)
        train_rows.append((summary.best_roi, summary.best_avg_clv_vs_consensus_close, summary.n_bets, edge, summary))
    train_rows.sort(reverse=True)
    chosen_edge = train_rows[0][3]
    chosen_train = train_rows[0][4]
    chosen_test, _ = line_shop_moneyline(
        test_games,
        edge_threshold=chosen_edge,
        staking=staking,
    )
    print(f"\nWalk-forward threshold selection ({staking})")
    print(
        f"objective=max train best-price ROI over supplied edges | "
        f"train={train} test={test} chosen_edge={chosen_edge:.2f}"
    )
    print(
        f"train {train}: bets {chosen_train.n_bets:4d} bestROI "
        f"{chosen_train.best_roi:+.2%} CLV "
        f"{chosen_train.best_avg_clv_vs_consensus_close:+.4f} beat-close "
        f"{chosen_train.best_pct_beat_consensus_close:.1%}"
    )
    print(
        f"test  {test}: bets {chosen_test.n_bets:4d} bestROI "
        f"{chosen_test.best_roi:+.2%} CLV "
        f"{chosen_test.best_avg_clv_vs_consensus_close:+.4f} beat-close "
        f"{chosen_test.best_pct_beat_consensus_close:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2024,2025")
    parser.add_argument("--edges", default="0,0.02,0.03,0.05")
    parser.add_argument("--source-edges", default="0.03,0.05")
    parser.add_argument("--source-rows", type=int, default=12)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--staking", choices=("flat", "kelly", "both"), default="both")
    parser.add_argument("--walkforward-train", type=int, default=2024)
    parser.add_argument("--walkforward-test", type=int, default=2025)
    args = parser.parse_args()

    seasons = _parse_ints(args.seasons)
    edges = _parse_floats(args.edges)
    source_edges = _parse_floats(args.source_edges)
    games, coverage = load_line_shopping_games(seasons)

    print("Line-shopping moneyline report")
    print(f"seasons: {seasons}")
    print(f"coverage: {coverage} total={len(games)}")
    print("selection: consensus-open side/edge; execution: best book open for that side")
    print("Kelly staking: quarter Kelly, 5% cap, fixed bankroll/no compounding")

    flat_by_edge: dict[float, list[LineShoppingBet]] = {}
    for staking in _staking_plans(args.staking):
        print(f"\n=== {staking.upper()} staking ===")
        by_edge: dict[float, list[LineShoppingBet]] = {}
        for edge in edges:
            summary, bets = line_shop_moneyline(games, edge_threshold=edge, staking=staking)
            by_edge[edge] = bets
            _report_summary(edge, summary, bets, args.boot)
        if staking == "flat":
            flat_by_edge = by_edge

        print(f"\nSame-source close CLV check ({staking})")
        for edge in edges:
            summary, _ = line_shop_moneyline(games, edge_threshold=edge, staking=staking)
            print(
                f"edge>={edge:.2f}: source close n={summary.source_close_n:4d} "
                f"CLV {summary.best_avg_clv_vs_source_close:+.4f} "
                f"beat-close {summary.best_pct_beat_source_close:.1%}"
            )

        if args.walkforward_train in seasons and args.walkforward_test in seasons:
            _report_walkforward(
                games=games,
                edges=edges,
                train=args.walkforward_train,
                test=args.walkforward_test,
                staking=staking,
            )

    for edge in source_edges:
        if edge in flat_by_edge:
            _report_source_breakdown(edge=edge, bets=flat_by_edge[edge], rows=args.source_rows)


if __name__ == "__main__":
    main()
