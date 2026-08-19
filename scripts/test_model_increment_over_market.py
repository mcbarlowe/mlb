"""Does the model carry information the market price does not?

Fits ``home_won ~ logit(market_p) + logit(model_p)`` with nested walk-forward: for each test
season, the blend is fit only on prior seasons, whose model probabilities are themselves
walk-forward. Compares held-out Brier and log loss for three forecasters:

  market  - the de-vigged price alone
  model   - the team-strength model alone
  blend   - the fitted combination

The blend's coefficient on the model is the quantity of interest. If it is ~0 the model has
no incremental information over the price and no segment or threshold will produce an edge,
because the edge filter is then selecting on noise. If it is positive, its magnitude says how
far a forecast should deviate from the price, which is strictly more useful than the raw
disagreement the backtest currently thresholds on.

    uv run python scripts/test_model_increment_over_market.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

FRAME = Path("data/analysis/model_vs_market.parquet")
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--price", default="market_p", choices=("market_p", "open_p"),
        help="market_p = last strictly pre-game snapshot (~2.5h out); "
             "open_p = earliest pre-game snapshot (19-29h out)",
    )
    args = ap.parse_args()
    price = args.price

    frame = pl.read_parquet(FRAME).drop_nulls(["model_p", price, "home_won"])
    seasons = sorted(frame["season"].unique().to_list())

    label = "2.5h pre-game" if price == "market_p" else "19-29h pre-game (open)"
    print(f"Nested walk-forward blend: home_won ~ logit({price}) + logit(model)")
    print(f"Price tested against: {price} ({label}), n={len(frame)}")
    print("Blend fit on prior seasons only; model probs already walk-forward.")
    print()
    print(f"{'Test':>5} | {'n':>5} | {'coef mkt':>8} | {'coef mdl':>8} | "
          f"{'Brier mkt':>9} | {'Brier mdl':>9} | {'Brier bld':>9} | {'best':>6}")
    print("-" * 84)

    pooled = {"market": [], "model": [], "blend": [], "y": []}
    for i, test in enumerate(seasons):
        if i == 0:
            continue  # no prior season to fit the blend on
        train = frame.filter(pl.col("season").is_in(seasons[:i]))
        hold = frame.filter(pl.col("season") == test)
        if len(train) < 500 or len(hold) < 100:
            continue

        xtr = np.column_stack(
            [logit(train[price].to_numpy()), logit(train["model_p"].to_numpy())]
        )
        ytr = train["home_won"].to_numpy().astype(int)
        xho = np.column_stack(
            [logit(hold[price].to_numpy()), logit(hold["model_p"].to_numpy())]
        )
        yho = hold["home_won"].to_numpy().astype(int)

        # No regularisation: the question is the unshrunk weight the data supports.
        fit = LogisticRegression(C=np.inf, max_iter=2000).fit(xtr, ytr)
        c_mkt, c_mdl = fit.coef_[0]

        p_mkt = hold[price].to_numpy()
        p_mdl = hold["model_p"].to_numpy()
        p_bld = fit.predict_proba(xho)[:, 1]

        b_mkt, b_mdl, b_bld = brier(p_mkt, yho), brier(p_mdl, yho), brier(p_bld, yho)
        best = min(
            (("market", b_mkt), ("model", b_mdl), ("blend", b_bld)), key=lambda t: t[1]
        )[0]
        print(f"{test:5d} | {len(hold):5d} | {c_mkt:+8.3f} | {c_mdl:+8.3f} | "
              f"{b_mkt:9.4f} | {b_mdl:9.4f} | {b_bld:9.4f} | {best:>6}")

        pooled["market"].append(p_mkt)
        pooled["model"].append(p_mdl)
        pooled["blend"].append(p_bld)
        pooled["y"].append(yho)

    y = np.concatenate(pooled["y"])
    print("-" * 84)
    print(f"{'POOL':>5} | {len(y):5d} | {'':>8} | {'':>8} | "
          f"{brier(np.concatenate(pooled['market']), y):9.4f} | "
          f"{brier(np.concatenate(pooled['model']), y):9.4f} | "
          f"{brier(np.concatenate(pooled['blend']), y):9.4f} |")
    print()
    for name in ("market", "model", "blend"):
        p = np.concatenate(pooled[name])
        print(f"  {name:6s}: Brier {brier(p, y):.4f}  log loss {logloss(p, y):.4f}")

    # Full-sample coefficient with a normal-approximation interval, for magnitude only.
    x = np.column_stack(
        [logit(frame[price].to_numpy()), logit(frame["model_p"].to_numpy())]
    )
    yy = frame["home_won"].to_numpy().astype(int)
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(x, yy)
    p = fit.predict_proba(x)[:, 1]
    w = p * (1.0 - p)
    xd = np.column_stack([np.ones(len(x)), x])
    cov = np.linalg.inv(xd.T @ (xd * w[:, None]))
    se = np.sqrt(np.diag(cov))[1:]
    print()
    print(f"Full-sample fit (n={len(frame)}), magnitude only, in-sample:")
    for name, coef, s in zip(("market", "model"), fit.coef_[0], se, strict=True):
        z = coef / s
        print(f"  logit({name:6s}) coef {coef:+.3f} +/- {s:.3f} (z={z:+.1f}) "
              f"95% CI [{coef - 1.96 * s:+.3f}, {coef + 1.96 * s:+.3f}]")


if __name__ == "__main__":
    main()
