"""Benchmark the champion team-strength win model against the betting market.

Answers "how much headroom is left" by comparing, on the same games:
  * model home-win probability (champion contract, reproduced), vs
  * consensus de-vigged market home-win probability (from mlb.odds),
against realized outcomes, using Brier + log loss (plus baselines and a paired
bootstrap CI on the model-minus-market Brier gap).

    uv run python scripts/benchmark_model_vs_market.py --season 2024
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.ingest import champion_home_probs, load_finals
from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig

LEAGUE_HOME_RATE = 0.543
EPS = 1e-9


def market_probs(season: int, devig: str) -> dict[int, float]:
    """Consensus de-vigged home-win prob per game_pk from mlb.odds."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT o.game_pk, o.home_ml, o.away_ml
                    FROM {c.schema}.odds o JOIN {c.schema}.games g USING (game_pk)
                    WHERE g.season::int=%s AND o.market='h2h' AND o.line_type='close'""",
                (season,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    per_game: dict[int, list[float]] = {}
    for game_pk, home_ml, away_ml in rows:
        fair_home, _ = no_vig_two_way(
            float(home_ml), float(away_ml), method=devig
        )
        per_game.setdefault(int(game_pk), []).append(fair_home)
    return {pk: sum(v) / len(v) for pk, v in per_game.items()}


def brier(p: float, y: float) -> float:
    return (p - y) ** 2


def logloss(p: float, y: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--devig", choices=("proportional", "shin"), default="proportional")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    model = champion_home_probs([args.season])
    model_p = {int(r.game_pk): float(r.model_prob_home) for r in model.itertuples()}
    finals = load_finals([args.season])
    outcome = {int(r.game_pk): (1.0 if r.home_won else 0.0) for r in finals.itertuples()}
    market_p = market_probs(args.season, args.devig)

    pks = sorted(set(model_p) & set(outcome) & set(market_p))
    n = len(pks)
    if n == 0:
        raise SystemExit("no overlapping games (model x finals x odds)")

    y = [outcome[pk] for pk in pks]
    m = [model_p[pk] for pk in pks]
    k = [market_p[pk] for pk in pks]
    home_rate = sum(y) / n

    def agg(fn, probs):
        return sum(fn(pi, yi) for pi, yi in zip(probs, y)) / n

    rows = {
        "model": (agg(brier, m), agg(logloss, m)),
        "market": (agg(brier, k), agg(logloss, k)),
        "league (p=0.543)": (agg(brier, [LEAGUE_HOME_RATE] * n), agg(logloss, [LEAGUE_HOME_RATE] * n)),
        "coin (p=0.5)": (agg(brier, [0.5] * n), agg(logloss, [0.5] * n)),
    }

    # paired bootstrap: model Brier - market Brier (positive => market better)
    diffs = [brier(mi, yi) - brier(ki, yi) for mi, ki, yi in zip(m, k, y)]
    mean_diff = sum(diffs) / n
    rng = random.Random(0)
    boots = []
    for _ in range(args.boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * args.boot)]
    hi = boots[int(0.975 * args.boot)]

    print(f"\nSeason {args.season}: {n} games (model x finals x market), home rate {home_rate:.3f}")
    print(f"de-vig: {args.devig}\n")
    print(f"{'':20}{'Brier':>10}{'log loss':>10}")
    for name, (b, ll) in rows.items():
        print(f"{name:20}{b:10.4f}{ll:10.4f}")
    print(
        f"\nBrier gap (model - market): {mean_diff:+.4f}  "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    )
    verdict = (
        "market significantly better -> real headroom" if lo > 0
        else "model significantly better" if hi < 0
        else "no significant gap -> model ~= market"
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
