"""Opening-line / CLV analysis for the champion win model.

Compares the model to the de-vigged OPENING and CLOSING market probabilities,
and measures closing-line value: does the market move from open->close in the
direction the model already pointed?

    uv run python scripts/clv_analysis.py --season 2025
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.ingest import champion_home_probs, load_finals
from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig

EPS = 1e-9


def market_probs(season: int, line_type: str, devig: str = "proportional") -> dict[int, float]:
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
                    WHERE g.season::int=%s AND o.market='h2h' AND o.line_type=%s""",
                (season, line_type),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    per: dict[int, list[float]] = {}
    for game_pk, home_ml, away_ml in rows:
        fair_home, _ = no_vig_two_way(float(home_ml), float(away_ml), method=devig)
        per.setdefault(int(game_pk), []).append(fair_home)
    return {pk: sum(v) / len(v) for pk, v in per.items()}


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    model = champion_home_probs([args.season])
    model_p = {int(r.game_pk): float(r.model_prob_home) for r in model.itertuples()}
    finals = load_finals([args.season])
    y = {int(r.game_pk): (1.0 if r.home_won else 0.0) for r in finals.itertuples()}
    op = market_probs(args.season, "open")
    cp = market_probs(args.season, "close")

    pks = sorted(set(model_p) & set(y) & set(op) & set(cp))
    n = len(pks)
    if n == 0:
        raise SystemExit("no overlap")

    def agg(fn, get):
        return sum(fn(get(pk), y[pk]) for pk in pks) / n

    print(f"\nSeason {args.season}: {n} games with model + open + close + result\n")
    print(f"{'':10}{'Brier':>10}{'log loss':>10}")
    for name, get in (
        ("model", lambda pk: model_p[pk]),
        ("open", lambda pk: op[pk]),
        ("close", lambda pk: cp[pk]),
    ):
        print(f"{name:10}{agg(brier, get):10.4f}{agg(logloss, get):10.4f}")

    # CLV: did the line move open->close toward the model?
    move = [cp[pk] - op[pk] for pk in pks]              # market drift
    edge = [model_p[pk] - op[pk] for pk in pks]         # model vs open
    toward = sum(1 for e, mv in zip(edge, move) if e * mv > 0) / n
    # correlation of model edge-at-open with subsequent market move
    if statistics.pstdev(edge) > 0 and statistics.pstdev(move) > 0:
        mean_e, mean_m = statistics.mean(edge), statistics.mean(move)
        cov = sum((e - mean_e) * (mv - mean_m) for e, mv in zip(edge, move)) / n
        corr = cov / (statistics.pstdev(edge) * statistics.pstdev(move))
    else:
        corr = float("nan")
    avg_abs_move = sum(abs(mv) for mv in move) / n

    print(
        f"\nCLV: line moved toward the model in {toward:.1%} of games "
        f"(50% = no signal)"
    )
    print(f"corr(model_edge_at_open, open->close move): {corr:+.3f}")
    print(f"avg |open->close| move: {avg_abs_move * 100:.2f} pp")


if __name__ == "__main__":
    main()
