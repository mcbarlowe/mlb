"""Can the open-to-close line move be predicted from information available at the open?

The oracle test established the prize: knowing the closing line and betting the best opening
price returns +6.14% (95% CI +1.14 to +10.94) restricted to games whose line moves at least 3
points, and +11.45% at 5 points. Our outcome model captures none of it, flagging 1,780 games at
the 5% threshold when only 337 actually move that far. It selects on its own noise instead of on
games the market will reprice.

This targets the move directly:

    target = logit(close_fair) - logit(open_fair)

A continuous target carries far more information per game than a binary win, so the screen has
much better power than the outcome screens that failed. Every feature must be observable at the
opening snapshot; anything dated later would leak.

Candidates, all derivable from what is already stored:

  book_dispersion   cross-book spread of de-vigged probability at the open. Books disagreeing
                    means the price is unsettled.
  n_books           how many panel books had posted. Thin coverage means an unsettled market.
  lead_hours        hours from the opening snapshot to first pitch. More time, more scope to move.
  model_disagree    our model minus the opening fair price. Known to carry some signal, with a
                    coefficient of +0.1156 in a movement regression.
  fav_extremity     |open_fair - 0.5|. Lopsided games may reprice differently.
  is_night, is_doubleheader, month_late, is_early_season   schedule context.

Pass one screens each candidate on the move after conditioning on the opening price. Pass two
fits a movement model walk-forward and reports whether its selections beat the outcome model at
opening prices.

    uv run python scripts/build_movement_frame.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig
from src.market_data.moneyline_quotes import MONEYLINE_PANEL
from src.market_data.pricing import american_to_decimal, no_vig_two_way
from src.model_evaluation.market_inputs import load_finals
from src.model_evaluation.moneyline_inputs import walkforward_home_probs

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
OUT = Path("data/analysis/line_movement.parquet")
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_book_level(conn, schema: str, season: int, line_type: str):
    """Per game: fair prob per book, best prices, snapshot time, book count."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.bookmaker, o.home_ml, o.away_ml, o.snapshot_time,
                   g.game_datetime
            FROM {schema}.odds o JOIN {schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type, list(MONEYLINE_PANEL)),
        )
        rows = cur.fetchall()

    fair: dict[int, list[float]] = defaultdict(list)
    best_home: dict[int, float] = {}
    best_away: dict[int, float] = {}
    snap: dict[int, object] = {}
    start: dict[int, object] = {}
    for pk, _book, home, away, snapshot, gdt in rows:
        pk = int(pk)
        fair[pk].append(no_vig_two_way(int(home), int(away), method="proportional")[0])
        dh, da = american_to_decimal(int(home)), american_to_decimal(int(away))
        best_home[pk] = max(best_home.get(pk, 0.0), dh)
        best_away[pk] = max(best_away.get(pk, 0.0), da)
        if snapshot is not None and (pk not in snap or snapshot < snap[pk]):
            snap[pk] = snapshot
        start[pk] = gdt
    return fair, best_home, best_away, snap, start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )

    frames = []
    for season in SEASONS:
        of, bh, ba, snap, start = load_book_level(conn, c.schema, season, "open")
        cf, _, _, _, _ = load_book_level(conn, c.schema, season, "true_close")
        finals = load_finals([season]).set_index("game_pk")
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT game_pk, day_night, double_header,
                       EXTRACT(MONTH FROM game_datetime)::int
                FROM {c.schema}.games
                WHERE season::int = %s AND game_type = 'R'
                """,
                (season,),
            )
            ctx = {int(r[0]): (r[1], r[2], r[3]) for r in cur.fetchall()}

        rows = []
        for pk, book_fairs in of.items():
            if pk not in cf or len(book_fairs) < 2 or len(cf[pk]) < 2:
                continue
            if pk not in finals.index or pk not in probs.index or pk not in ctx:
                continue
            open_fair = statistics.median(book_fairs)
            close_fair = statistics.median(cf[pk])
            lead = None
            if snap.get(pk) is not None and start.get(pk) is not None:
                lead = (start[pk] - snap[pk]).total_seconds() / 3600.0
            dn, dh_flag, month = ctx[pk]
            rows.append({
                "game_pk": pk,
                "season": season,
                "open_fair": open_fair,
                "close_fair": close_fair,
                "model_p": float(probs.loc[pk]),
                "home_won": bool(finals.loc[pk, "home_won"]),
                "best_home_dec": bh[pk],
                "best_away_dec": ba[pk],
                "book_dispersion": max(book_fairs) - min(book_fairs),
                "n_books": len(book_fairs),
                "lead_hours": lead,
                "is_night": 1 if dn == "night" else 0,
                "is_doubleheader": 0 if dh_flag in (None, "N") else 1,
                "month": month,
            })
        frames.append(pl.DataFrame(rows, strict=False))
        print(f"{season}: {len(rows)} games with open and true_close")
    conn.close()

    frame = pl.concat(frames, how="vertical_relaxed").with_columns(
        (
            pl.col("close_fair").log() - (1 - pl.col("close_fair")).log()
            - (pl.col("open_fair").log() - (1 - pl.col("open_fair")).log())
        ).alias("move"),
        (pl.col("model_p") - pl.col("open_fair")).alias("model_disagree"),
        (pl.col("open_fair") - 0.5).abs().alias("fav_extremity"),
        (pl.col("month") >= 9).cast(pl.Int8).alias("month_late"),
        (pl.col("month") <= 4).cast(pl.Int8).alias("is_early_season"),
    ).with_columns(pl.col("move").abs().alias("abs_move"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    print(f"\nwrote {len(frame):,} rows to {args.out}")
    print()
    print("move distribution (logit points):")
    d = frame.select(
        pl.col("move").mean().alias("mean"),
        pl.col("move").std().alias("sd"),
        pl.col("abs_move").median().alias("median_abs"),
        pl.col("abs_move").quantile(0.9).alias("p90_abs"),
    )
    print(d)
    big = frame.filter(pl.col("abs_move") >= 0.12)
    print(f"games with |move| >= 0.12 logits (about 3 probability points): "
          f"{len(big):,} of {len(frame):,} ({len(big) / len(frame):.1%})")


if __name__ == "__main__":
    main()
