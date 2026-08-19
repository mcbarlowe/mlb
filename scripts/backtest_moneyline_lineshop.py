"""Moneyline backtest with genuine line shopping.

Design notes, because the naive version is wrong in a way that flatters the result:

*Edge* and *execution* must come from different quantities.

  - Edge is measured against a fair probability built by de-vigging **each book's own
    two-sided pair** and taking the median of those fair probabilities. Pairing is never
    broken. De-vigging a cross-book aggregate (or taking the best price on both sides)
    yields sub-1% or negative overround, i.e. fabricated arbitrage, which inflates every
    apparent edge.
  - Execution is the best decimal price available on the **one** side selected, among a
    fixed panel of books. Only one side of a game is ever backed.

Because the de-vig normalises the two outcomes to sum to 1, the away edge is the exact
negative of the home edge, so side selection reduces to ``model > fair_home``.

The identical bet list is settled twice, at consensus price and at best price. The
difference isolates execution value with selection held constant.

Bets are placed at the last strictly-pre-game snapshot (``line_type='close'``, ~2.5h
before first pitch). That is the only market state whose book coverage is stable across
2020-2025; the earlier ``open`` bucket sits 19-29h out and in 2023 carries almost no book
but DraftKings, so cross-season comparison there is confounded by panel depth.

    uv run python scripts/backtest_moneyline_lineshop.py --season 2025
    uv run python scripts/backtest_moneyline_lineshop.py --season 2025 --panel all
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from src.betting.backtest import kelly_fraction
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

# Books with >=80% coverage of the last-pre-game snapshot in every season 2020-2025.
PANEL_TOP5 = ("betonlineag", "betrivers", "bovada", "draftkings", "fanduel")
# Adds williamhill_us, which dips to 78% in 2020.
PANEL_TOP6 = PANEL_TOP5 + ("williamhill_us",)
# Priority order for best-of-K, most consistently available first.
PANEL_PRIORITY = (
    "betonlineag",
    "draftkings",
    "fanduel",
    "bovada",
    "betrivers",
    "williamhill_us",
)

PANELS = {"top5": PANEL_TOP5, "top6": PANEL_TOP6, "all": None}


@dataclass(frozen=True)
class Quote:
    """One game's per-book prices at the bet point, plus derived fair probability."""

    game_pk: int
    fair_home: float
    best_home_dec: float
    best_away_dec: float
    cons_home_dec: float
    cons_away_dec: float
    n_books: int


def load_quotes(
    season: int, panel: tuple[str, ...] | None, line_type: str, devig: str
) -> dict[int, Quote]:
    """Per-game fair probability and prices at ``line_type``.

    ``fair_home`` is the median over books of that book's own de-vigged home probability.
    """
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.bookmaker, o.home_ml, o.away_ml
            FROM {c.schema}.odds o JOIN {c.schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type),
        )
        for pk, book, home, away in cur.fetchall():
            if panel is not None and book not in panel:
                continue
            rows[int(pk)].append((int(home), int(away)))
    conn.close()

    quotes: dict[int, Quote] = {}
    for pk, prices in rows.items():
        if not prices:
            continue  # with one book, shop gain is 0 by construction: a valid baseline
        fair = statistics.median(
            no_vig_two_way(h, a, method=devig)[0] for h, a in prices
        )
        home_decs = [american_to_decimal(h) for h, _ in prices]
        away_decs = [american_to_decimal(a) for _, a in prices]
        quotes[pk] = Quote(
            game_pk=pk,
            fair_home=fair,
            best_home_dec=max(home_decs),
            best_away_dec=max(away_decs),
            cons_home_dec=statistics.median(home_decs),
            cons_away_dec=statistics.median(away_decs),
            n_books=len(prices),
        )
    return quotes


@dataclass
class Result:
    bets: int = 0
    staked: float = 0.0
    profit_best: float = 0.0
    profit_cons: float = 0.0
    staked_cons: float = 0.0
    wins: int = 0
    edge_sum: float = 0.0
    improvement_sum: float = 0.0
    # (stake, profit) per bet at best-price execution, for interval estimation.
    settled: list[tuple[float, float]] = field(default_factory=list)

    def report(self, label: str, threshold: float, n_games: int) -> str:
        if not self.bets:
            return f"  edge>={threshold:.2f}: no bets"
        roi_best = self.profit_best / self.staked
        roi_cons = self.profit_cons / self.staked_cons
        return (
            f"  edge>={threshold:.2f}: bets {self.bets:4d} ({self.bets / n_games:4.0%})  "
            f"win {self.wins / self.bets:5.1%}  "
            f"ROI best {roi_best:+7.2%}  ROI consensus {roi_cons:+7.2%}  "
            f"shop gain {(roi_best - roi_cons) * 100:+5.2f}pp  "
            f"avg edge {self.edge_sum / self.bets:+.3f}  "
            f"avg price impr {self.improvement_sum / self.bets:+.3%}"
        )


def run(
    quotes: dict[int, Quote],
    probs,
    finals,
    threshold: float,
    staking: str,
    kelly_multiplier: float,
    kelly_cap: float,
) -> tuple[Result, int]:
    """Settle one bet list twice: at best price and at consensus price."""
    res = Result()
    n_games = 0
    for pk, q in quotes.items():
        if pk not in probs.index or pk not in finals.index:
            continue
        n_games += 1
        model_home = float(probs.loc[pk])
        # De-vig normalises to a two-outcome distribution, so the away edge is the exact
        # negative of the home edge. Side selection is therefore a single comparison.
        edge = model_home - q.fair_home
        if edge >= 0.0:
            side_home, model_p = True, model_home
            best_dec, cons_dec = q.best_home_dec, q.cons_home_dec
        else:
            side_home, model_p = False, 1.0 - model_home
            best_dec, cons_dec = q.best_away_dec, q.cons_away_dec
        edge = abs(edge)
        if edge < threshold:
            continue

        if staking == "flat":
            stake_best = stake_cons = 1.0
        else:
            stake_best = min(
                kelly_fraction(model_p, best_dec) * kelly_multiplier, kelly_cap
            )
            stake_cons = min(
                kelly_fraction(model_p, cons_dec) * kelly_multiplier, kelly_cap
            )
            if stake_best <= 0.0:
                continue

        won = side_home == bool(finals.loc[pk, "home_won"])
        res.bets += 1
        res.wins += int(won)
        res.edge_sum += edge
        res.staked += stake_best
        res.staked_cons += stake_cons
        res.profit_best += stake_best * (best_dec - 1.0) if won else -stake_best
        res.profit_cons += stake_cons * (cons_dec - 1.0) if won else -stake_cons
        # Price improvement expressed in implied-probability terms: how much cheaper the
        # best price is than the consensus price for the same outcome.
        res.improvement_sum += (1.0 / cons_dec) - (1.0 / best_dec)
        res.settled.append(
            (stake_best, stake_best * (best_dec - 1.0) if won else -stake_best)
        )
    return res, n_games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--panel", default="top5", choices=sorted(PANELS))
    ap.add_argument("--best-of", type=int, default=None,
                    help="restrict to the first K books of the priority order")
    ap.add_argument("--line-type", default="close", choices=("open", "close"))
    ap.add_argument("--edges", default="0.0,0.02,0.03,0.05")
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    ap.add_argument("--kelly-multiplier", type=float, default=0.25)
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    args = ap.parse_args()

    panel = PANELS[args.panel]
    if args.best_of is not None:
        panel = PANEL_PRIORITY[: args.best_of]

    quotes = load_quotes(args.season, panel, args.line_type, args.devig)
    train = list(range(2015, args.season))
    probs = walkforward_home_probs(args.season, train).set_index("game_pk")[
        "model_prob_home"
    ]
    finals = load_finals([args.season]).set_index("game_pk")

    depth = statistics.median(q.n_books for q in quotes.values()) if quotes else 0
    label = f"best-of-{args.best_of}" if args.best_of else f"panel={args.panel}"
    print(f"Line-shop backtest {args.season} | {label} | bet at {args.line_type} "
          f"({len(quotes)} games, median {depth:.0f} books/game) | "
          f"train {min(train)}-{max(train)} | de-vig {args.devig}")
    print("Edge vs per-book de-vigged median fair prob; execution at best price, one side.")

    for staking in ("flat", "kelly"):
        head = "flat 1u" if staking == "flat" else (
            f"{args.kelly_multiplier:g}x Kelly, cap {args.kelly_cap:.0%}"
        )
        print(f"\n=== {head} ===")
        for threshold in (float(x) for x in args.edges.split(",")):
            res, n_games = run(
                quotes, probs, finals, threshold, staking,
                args.kelly_multiplier, args.kelly_cap,
            )
            print(res.report(staking, threshold, max(n_games, 1)))


if __name__ == "__main__":
    main()
