"""Where does line movement actually live, and what is left to win at each entry point?

The oracle ceiling of +6.14% is measured betting at the *open* with perfect foresight of the close.
That is not the strategy a movement model would run. Any feature describing how a price is
behaving - which book moved first, whether books are converging, how fast it is moving - requires
watching some of the move happen, so the bet lands later than the open. Entering at time k only
captures the movement remaining after k.

So the ceiling has to be recomputed per entry point, and that is free with the three price points
already in the database:

    open        19-29h before first pitch
    close       a fixed-cadence pull, median 2.5h before first pitch
    true_close  targeted per game, median 4 minutes before first pitch

This reports, for each entry point, how much movement remains, and the oracle ROI available from
perfect foresight of ``true_close`` when transacting at that entry point's best price. It also
splits the total move into the early window (open -> close) and the late window (close ->
true_close), because that split decides whether observable-movement features can be used at all.

If almost all movement happens before the 2.5h mark, then by the time a dynamics feature exists the
money is gone, and no amount of ladder data helps. That would close the thesis for free.

    uv run python scripts/test_entry_point_ceiling.py
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

PANEL = PANEL_PRIORITY[:5]
BOOT = 4000
SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
LINE_TYPES = ("open", "close", "true_close")


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def load_prices(conn, schema: str, season: int, line_type: str):
    """Per game: de-vigged median home probability, plus best price each side."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type, list(PANEL)),
        )
        rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pk, home, away in cur.fetchall():
            rows[int(pk)].append((int(home), int(away)))
    fair, best_home, best_away = {}, {}, {}
    for pk, quotes in rows.items():
        if len(quotes) < 2:
            continue
        fair[pk] = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in quotes
        )
        best_home[pk] = max(american_to_decimal(h) for h, _ in quotes)
        best_away[pk] = max(american_to_decimal(a) for _, a in quotes)
    return fair, best_home, best_away


def boot_ci(arr: np.ndarray, seed: int = 17):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", default="0.01,0.02,0.03,0.05")
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    data: dict[str, dict] = {lt: {"fair": {}, "bh": {}, "ba": {}} for lt in LINE_TYPES}
    finals: dict[int, bool] = {}
    for season in SEASONS:
        f = load_finals([season]).set_index("game_pk")
        for pk in f.index:
            finals[int(pk)] = bool(f.loc[pk, "home_won"])
        for lt in LINE_TYPES:
            fair, bh, ba = load_prices(conn, c.schema, season, lt)
            data[lt]["fair"].update(fair)
            data[lt]["bh"].update(bh)
            data[lt]["ba"].update(ba)
    conn.close()

    keys = sorted(
        set(data["open"]["fair"])
        & set(data["close"]["fair"])
        & set(data["true_close"]["fair"])
        & set(finals)
    )
    print(f"{len(keys)} games with all three price points and a result")
    print()

    early = np.array([
        logit(data["close"]["fair"][k]) - logit(data["open"]["fair"][k]) for k in keys
    ])
    late = np.array([
        logit(data["true_close"]["fair"][k]) - logit(data["close"]["fair"][k]) for k in keys
    ])
    total = np.array([
        logit(data["true_close"]["fair"][k]) - logit(data["open"]["fair"][k]) for k in keys
    ])

    print("Where the movement lives, in logits:")
    print(f"  {'window':>26} | {'mean |move|':>11} | {'sd':>8} | {'share of |total|':>16}")
    print("  " + "-" * 72)
    denom = float(np.abs(early).mean() + np.abs(late).mean())
    for label, arr in (
        ("open -> close (early)", early),
        ("close -> true_close (late)", late),
        ("open -> true_close (total)", total),
    ):
        share = f"{np.abs(arr).mean() / denom:.1%}" if label != "open -> true_close (total)" else "-"
        print(f"  {label:>26} | {np.abs(arr).mean():11.4f} | {arr.std():8.4f} | {share:>16}")
    print()
    print(f"  correlation(early move, late move) = {np.corrcoef(early, late)[0, 1]:+.4f}")
    print("  A positive correlation would mean early movement predicts further movement in the")
    print("  same direction, which is the whole premise of a dynamics feature.")
    print()

    print("Oracle ceiling by entry point. Perfect foresight of true_close, bet at that entry's")
    print("best price, flat 1u.")
    print(f"  {'entry':>12} | {'thr':>5} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22}")
    print("  " + "-" * 76)
    for entry in ("open", "close"):
        for thr in thresholds:
            settled = []
            for k in keys:
                edge = data["true_close"]["fair"][k] - data[entry]["fair"][k]
                if abs(edge) < thr:
                    continue
                home = edge >= 0
                dec = data[entry]["bh"][k] if home else data[entry]["ba"][k]
                won = finals[k] == home
                settled.append((dec - 1.0) if won else -1.0)
            if len(settled) < 30:
                print(f"  {entry:>12} | {thr:5.0%} | {len(settled):5d} | too few")
                continue
            arr = np.array(settled)
            m, lo, hi = boot_ci(arr)
            flag = "  <-- excludes zero" if lo > 0 else ""
            print(f"  {entry:>12} | {thr:5.0%} | {len(arr):5d} | "
                  f"{float((arr > 0).mean()):5.1%} | {m:+7.2%} | "
                  f"[{lo:+7.2%}, {hi:+7.2%}]{flag}")
    print()
    print("Read the 'close' rows as the ceiling for any strategy whose features require watching")
    print("the early move. If they are flat, observable-movement features cannot pay regardless")
    print("of how good the model gets, and no ladder data changes that.")


if __name__ == "__main__":
    main()
