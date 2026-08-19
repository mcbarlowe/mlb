#!/usr/bin/env python3
"""Replay first-five totals CLV summaries from a saved F5 report JSON.

This script requires report JSONs produced by ``scripts/f5_clv_report.py`` after
replay fields were added: each game must include ``simulated_totals``,
``open_lines``, and ``close_lines``. It recalculates raw sim summaries and,
optionally, market-anchored blend summaries without rerunning simulations or
querying Postgres.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.f5_clv_report import (
    _bet_row,
    _line_shopping_bet_row,
    _line_shopping_summary_line,
    _market_anchor_probability,
    _parse_floats,
    _staking_plans,
    _summary_line,
)
from src.betting.f5_clv import (
    F5BookLine,
    F5ClvBet,
    F5ClvGame,
    F5ClvSummary,
    F5LineShoppingBet,
    F5LineShoppingSummary,
    compare_f5_line_shopping,
    summarize_f5_clv,
)


def games_from_report_payload(payload: Mapping[str, Any]) -> list[F5ClvGame]:
    raw_games = payload.get("games")
    if not isinstance(raw_games, list):
        raise TypeError("F5 report JSON must contain a games list")
    games: list[F5ClvGame] = []
    for index, raw_game in enumerate(raw_games):
        if not isinstance(raw_game, Mapping):
            raise TypeError(f"games[{index}] must be an object")
        games.append(_game_from_payload(raw_game, index=index))
    return games


def build_replay_report(
    games: Sequence[F5ClvGame],
    *,
    edges: Sequence[float],
    staking: str = "flat",
    execution: str = "best",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
    market_anchor_lambda: float | None = None,
) -> dict[str, Any]:
    summaries, bets_by_key, line_summaries, line_bets_by_key = _evaluate(
        games,
        edges=edges,
        staking=staking,
        execution=execution,
        flat_stake=flat_stake,
        kelly_multiplier=kelly_multiplier,
        kelly_cap=kelly_cap,
    )
    report: dict[str, Any] = {
        "report_type": "f5_clv_replay",
        "settings": {
            "edges": list(edges),
            "staking": staking,
            "execution": execution,
            "flat_stake": flat_stake,
            "kelly_multiplier": kelly_multiplier,
            "kelly_cap": kelly_cap,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "line_shopping_summaries": [asdict(summary) for summary in line_summaries],
        "bets": [
            _bet_row(edge, plan, bet)
            for (plan, edge), bets in bets_by_key.items()
            for bet in bets
        ],
        "line_shopping_bets": [
            _line_shopping_bet_row(edge, plan, bet)
            for (plan, edge), bets in line_bets_by_key.items()
            for bet in bets
        ],
    }
    if market_anchor_lambda is not None:
        model_probability = _market_anchor_probability(market_anchor_lambda)
        (
            anchor_summaries,
            anchor_bets_by_key,
            anchor_line_summaries,
            anchor_line_bets_by_key,
        ) = _evaluate(
            games,
            edges=edges,
            staking=staking,
            execution=execution,
            flat_stake=flat_stake,
            kelly_multiplier=kelly_multiplier,
            kelly_cap=kelly_cap,
            model_probability=model_probability,
        )
        report["anchor_blend"] = {
            "lambda": market_anchor_lambda,
            "summaries": [asdict(summary) for summary in anchor_summaries],
            "line_shopping_summaries": [
                asdict(summary) for summary in anchor_line_summaries
            ],
            "bets": [
                _bet_row(edge, plan, bet)
                for (plan, edge), bets in anchor_bets_by_key.items()
                for bet in bets
            ],
            "line_shopping_bets": [
                _line_shopping_bet_row(edge, plan, bet)
                for (plan, edge), bets in anchor_line_bets_by_key.items()
                for bet in bets
            ],
        }
    return report


def _evaluate(
    games: Sequence[F5ClvGame],
    *,
    edges: Sequence[float],
    staking: str,
    execution: str,
    flat_stake: float,
    kelly_multiplier: float,
    kelly_cap: float,
    model_probability=None,
) -> tuple[
    list[F5ClvSummary],
    dict[tuple[str, float], list[F5ClvBet]],
    list[F5LineShoppingSummary],
    dict[tuple[str, float], list[F5LineShoppingBet]],
]:
    summaries: list[F5ClvSummary] = []
    bets_by_key: dict[tuple[str, float], list[F5ClvBet]] = {}
    line_summaries: list[F5LineShoppingSummary] = []
    line_bets_by_key: dict[tuple[str, float], list[F5LineShoppingBet]] = {}
    for plan in _staking_plans(staking):
        for edge in edges:
            summary, bets = summarize_f5_clv(
                games,
                edge_threshold=edge,
                execution=execution,
                staking=plan,
                flat_stake=flat_stake,
                kelly_multiplier=kelly_multiplier,
                kelly_cap=kelly_cap,
                model_probability=model_probability,
            )
            summaries.append(summary)
            bets_by_key[(plan, edge)] = bets
            line_summary, line_bets = compare_f5_line_shopping(
                games,
                edge_threshold=edge,
                staking=plan,
                flat_stake=flat_stake,
                kelly_multiplier=kelly_multiplier,
                kelly_cap=kelly_cap,
                model_probability=model_probability,
            )
            line_summaries.append(line_summary)
            line_bets_by_key[(plan, edge)] = line_bets
    return summaries, bets_by_key, line_summaries, line_bets_by_key


def _game_from_payload(raw_game: Mapping[str, Any], *, index: int) -> F5ClvGame:
    simulated_totals = _float_list(raw_game, "simulated_totals", index=index)
    open_lines = _book_lines(raw_game, "open_lines", index=index)
    close_lines = _book_lines(raw_game, "close_lines", index=index)
    actual = raw_game.get("actual_f5_total")
    return F5ClvGame(
        game_pk=int(_required(raw_game, "game_pk", index=index)),
        season=int(_required(raw_game, "season", index=index)),
        open_lines=open_lines,
        simulated_totals=simulated_totals,
        actual_total=None if actual is None else float(actual),
        close_lines=close_lines,
        take_line_type=str(raw_game.get("take_line_type", "open")),
    )


def _book_lines(
    raw_game: Mapping[str, Any], key: str, *, index: int
) -> tuple[F5BookLine, ...]:
    raw_lines = _required(raw_game, key, index=index)
    if not isinstance(raw_lines, list):
        raise TypeError(f"games[{index}].{key} must be a list")
    return tuple(_book_line(raw_line, game_index=index, line_key=key) for raw_line in raw_lines)


def _book_line(
    raw_line: object, *, game_index: int, line_key: str
) -> F5BookLine:
    if not isinstance(raw_line, Mapping):
        raise TypeError(f"games[{game_index}].{line_key} entries must be objects")
    return F5BookLine(
        bookmaker=str(_required(raw_line, "bookmaker", index=game_index)),
        point=float(_required(raw_line, "point", index=game_index)),
        over_ml=float(_required(raw_line, "over_ml", index=game_index)),
        under_ml=float(_required(raw_line, "under_ml", index=game_index)),
    )


def _float_list(raw_game: Mapping[str, Any], key: str, *, index: int) -> tuple[float, ...]:
    value = _required(raw_game, key, index=index)
    if not isinstance(value, list):
        raise TypeError(f"games[{index}].{key} must be a list")
    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"games[{index}].{key} must not be empty")
    return values


def _required(raw: Mapping[str, Any], key: str, *, index: int) -> Any:
    if key not in raw:
        raise ValueError(
            f"games[{index}] is missing {key!r}; rerun scripts/f5_clv_report.py "
            "to produce replayable JSON"
        )
    return raw[key]


def _print_report(report: Mapping[str, Any], *, market_anchor_lambda: float | None) -> None:
    for summary in report["summaries"]:
        edge = float(summary["settings"]["edge_threshold"])
        print(_summary_line(edge, F5ClvSummary(**summary)))
    for summary in report["line_shopping_summaries"]:
        edge = float(summary["settings"]["edge_threshold"])
        print(_line_shopping_summary_line(edge, F5LineShoppingSummary(**summary)))
    anchor = report.get("anchor_blend")
    if market_anchor_lambda is None or not isinstance(anchor, Mapping):
        return
    print(f"\n=== MARKET-ANCHOR lambda={market_anchor_lambda:g} ===")
    for summary in anchor["summaries"]:
        edge = float(summary["settings"]["edge_threshold"])
        print(f"anchor lambda={market_anchor_lambda:g} " + _summary_line(edge, F5ClvSummary(**summary)))
    for summary in anchor["line_shopping_summaries"]:
        edge = float(summary["settings"]["edge_threshold"])
        print(
            f"anchor lambda={market_anchor_lambda:g} "
            + _line_shopping_summary_line(edge, F5LineShoppingSummary(**summary))
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--edges", type=_parse_floats, default=(0.0, 0.03, 0.05))
    parser.add_argument("--staking", choices=("flat", "kelly", "both"), default="flat")
    parser.add_argument("--execution", choices=("best", "consensus"), default="best")
    parser.add_argument("--flat-stake", type=float, default=1.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--kelly-cap", type=float, default=0.05)
    parser.add_argument("--market-anchor-lambda", type=float, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    source_payload = json.loads(args.report_json.read_text())
    if not isinstance(source_payload, Mapping):
        raise TypeError("F5 report JSON must contain an object")
    games = games_from_report_payload(source_payload)
    report = build_replay_report(
        games,
        edges=args.edges,
        staking=args.staking,
        execution=args.execution,
        flat_stake=args.flat_stake,
        kelly_multiplier=args.kelly_multiplier,
        kelly_cap=args.kelly_cap,
        market_anchor_lambda=args.market_anchor_lambda,
    )
    report["source_report_json"] = str(args.report_json)
    print(f"F5 CLV replay report from {args.report_json}")
    print(f"replayed games: {len(games)}")
    _print_report(report, market_anchor_lambda=args.market_anchor_lambda)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote replay report to {args.out_json}")


if __name__ == "__main__":
    main()
