"""Where is the market weakest? Condition on the price, then test candidate observables.

The only well-posed version of "beat the market in segment X" is: after conditioning on
``logit(market_p)``, does feature X still predict the outcome? A nonzero residual coefficient
is a market inefficiency by construction. Testing features without conditioning on the price
instead rediscovers whatever the price already knows.

Two passes:

  1. Univariate screen, each candidate added to the price alone, reported with z and a
     Bonferroni-adjusted threshold. With k candidates at alpha=0.05 roughly k/20 spurious
     hits are expected, so the unadjusted p-value is not evidence on its own.
  2. Out-of-sample confirmation of any screened hit, fit on earlier seasons and evaluated on
     later ones. A coefficient that does not survive this is noise regardless of its z.

    uv run python scripts/find_market_residual_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

FRAME = Path("data/analysis/model_vs_market.parquet")
EPS = 1e-6

# Candidate proxies for market weakness, all observable strictly pre-game.
CANDIDATES = {
    "model_disagreement": "model_p - market_p, the current edge filter",
    "dispersion": "cross-book spread of fair prob: market uncertainty",
    "line_move": "open-to-close fair-prob drift: steam",
    "abs_line_move": "magnitude of drift regardless of direction",
    "is_night": "night game",
    "is_doubleheader": "either half of a doubleheader",
    "month_late": "September or later: callups, tanking, rest",
    "fav_extremity": "|market_p - 0.5|: how lopsided the game is priced",
    "cold": "game temperature below 55F",
    "windy": "wind above 12 mph",
}


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def irls(x: np.ndarray, y: np.ndarray, iters: int = 60):
    """Unpenalised logistic fit by Newton-Raphson, returning coefficients and standard errors.

    Written out rather than taken from sklearn because the standard errors are the point and
    sklearn does not expose them.
    """
    xd = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xd.shape[1])
    for _ in range(iters):
        eta = xd @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        hess = xd.T @ (xd * w[:, None])
        grad = xd.T @ (y - p)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = xd @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv(xd.T @ (xd * w[:, None]))
    return beta, np.sqrt(np.diag(cov))


def features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("disagreement").alias("model_disagreement"),
        pl.col("line_move").abs().alias("abs_line_move"),
        (pl.col("day_night") == "night").cast(pl.Int8).alias("is_night"),
        (pl.col("double_header") != "N").cast(pl.Int8).alias("is_doubleheader"),
        (pl.col("month") >= 9).cast(pl.Int8).alias("month_late"),
        (pl.col("market_p") - 0.5).abs().alias("fav_extremity"),
        (pl.col("weather_temp") < 55).cast(pl.Int8).alias("cold"),
        (
            pl.col("weather_wind")
            .str.extract(r"(\d+)")
            .cast(pl.Float64, strict=False)
            > 12
        )
        .cast(pl.Int8)
        .alias("windy"),
    )


def main() -> None:
    frame = features(
        pl.read_parquet(FRAME).drop_nulls(["model_p", "market_p", "home_won"])
    )
    k = len(CANDIDATES)
    z_bonf = 2.807  # two-sided alpha=0.05/10, normal approximation

    print("Pass 1: univariate screen, each candidate added to logit(market_p)")
    print(f"Bonferroni threshold for {k} candidates: |z| > {z_bonf:.2f}")
    print()
    print(f"{'feature':>20} | {'n':>5} | {'coef':>8} | {'se':>6} | {'z':>6} | "
          f"{'survives':>8} | description")
    print("-" * 104)

    hits = []
    for name, desc in CANDIDATES.items():
        sub = frame.drop_nulls([name])
        if len(sub) < 500:
            print(f"{name:>20} | {len(sub):5d} | insufficient rows")
            continue
        x = np.column_stack(
            [logit(sub["market_p"].to_numpy()), sub[name].to_numpy().astype(float)]
        )
        y = sub["home_won"].to_numpy().astype(int)
        beta, se = irls(x, y)
        coef, s = beta[2], se[2]
        z = coef / s
        survives = abs(z) > z_bonf
        if survives:
            hits.append(name)
        print(f"{name:>20} | {len(sub):5d} | {coef:+8.4f} | {s:6.4f} | {z:+6.2f} | "
              f"{'YES' if survives else 'no':>8} | {desc}")

    print()
    if not hits:
        print("No candidate survives correction. On this data the price is not beaten by")
        print("any of these observables, so there is no located weakness to exploit.")
    else:
        print(f"Pass 2: out-of-sample confirmation for {hits}")
        seasons = sorted(frame["season"].unique().to_list())
        split = seasons[: len(seasons) // 2 + 1]
        train = frame.filter(pl.col("season").is_in(split))
        hold = frame.filter(~pl.col("season").is_in(split))
        print(f"  fit on {split}, hold out {[s for s in seasons if s not in split]}")
        for name in hits:
            tr, ho = train.drop_nulls([name]), hold.drop_nulls([name])
            xtr = np.column_stack(
                [logit(tr["market_p"].to_numpy()), tr[name].to_numpy().astype(float)]
            )
            beta, _ = irls(xtr, tr["home_won"].to_numpy().astype(int))
            xho = np.column_stack(
                [logit(ho["market_p"].to_numpy()), ho[name].to_numpy().astype(float)]
            )
            beta_ho, se_ho = irls(xho, ho["home_won"].to_numpy().astype(int))
            same_sign = np.sign(beta[2]) == np.sign(beta_ho[2])
            print(f"  {name}: train coef {beta[2]:+.4f}, holdout coef "
                  f"{beta_ho[2]:+.4f} +/- {se_ho[2]:.4f} "
                  f"-> {'CONFIRMED' if same_sign and abs(beta_ho[2] / se_ho[2]) > 2 else 'not confirmed'}")

    # Reference: the market coefficient itself, to show the price is already ~sufficient.
    x = logit(frame["market_p"].to_numpy()).reshape(-1, 1)
    beta, se = irls(x, frame["home_won"].to_numpy().astype(int))
    print()
    print(f"Reference: logit(market_p) alone, coef {beta[1]:+.3f} +/- {se[1]:.3f} "
          f"(1.000 would be a perfectly calibrated price), intercept {beta[0]:+.3f}")


if __name__ == "__main__":
    main()
