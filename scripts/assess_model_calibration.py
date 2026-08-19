"""Is the model well calibrated? Slope, intercept, reliability, and resolution.

"Well calibrated" and "useful" are separate properties and this model has been described as
having the first. A forecast that always returns the base rate is perfectly calibrated and
worthless, so calibration is reported alongside resolution rather than on its own.

Three instruments, each applied to the model and to the de-vigged market price on the same
games, so every number has a benchmark:

  slope / intercept  Fit ``outcome ~ logit(p)``. Perfect calibration is slope 1.0 with
                     intercept 0.0. Slope below 1 means the forecasts are too extreme
                     (overconfident); above 1 means too timid.

  reliability table  Observed frequency against mean forecast, by decile of forecast.

  Murphy decomposition  Brier = reliability - resolution + uncertainty. Reliability is
                     calibration error, lower is better. Resolution is how far the
                     conditional outcome rates move away from the base rate, higher is
                     better. Uncertainty is a property of the outcome, identical for both
                     forecasters, and sets the Brier of always predicting the base rate.

Also reports calibration restricted to the games the strategy actually backs, because
aggregate calibration says nothing about a selected subset.

    uv run python scripts/assess_model_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

FRAME = Path("data/analysis/model_vs_market.parquet")
EPS = 1e-6
BINS = 10


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def irls(x: np.ndarray, y: np.ndarray, iters: int = 60):
    xd = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xd.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xd @ beta)))
        w = np.clip(p * (1 - p), 1e-10, None)
        step = np.linalg.solve(xd.T @ (xd * w[:, None]), xd.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(xd @ beta)))
    w = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv(xd.T @ (xd * w[:, None]))
    return beta, np.sqrt(np.diag(cov))


def murphy(p: np.ndarray, y: np.ndarray, bins: int = BINS):
    """Brier = reliability - resolution + uncertainty, via equal-count bins."""
    base = y.mean()
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(p)
    rel = res = 0.0
    rows = []
    for idx in chunks:
        if len(idx) == 0:
            continue
        pk, ok = p[idx].mean(), y[idx].mean()
        rel += len(idx) * (pk - ok) ** 2
        res += len(idx) * (ok - base) ** 2
        rows.append((len(idx), p[idx].min(), p[idx].max(), pk, ok))
    return rel / n, res / n, base * (1 - base), rows


def summarise(name: str, p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    beta, se = irls(logit(p).reshape(-1, 1), y)
    intercept, slope = beta[0], beta[1]
    rel, res, unc, _ = murphy(p, y)
    brier = float(np.mean((p - y) ** 2))
    z_slope = (slope - 1.0) / se[1]
    z_int = intercept / se[0]
    print(f"{name}:")
    print(f"  slope     {slope:+.3f} +/- {se[1]:.3f}   (1.000 = calibrated, "
          f"z vs 1 = {z_slope:+.2f})")
    print(f"  intercept {intercept:+.3f} +/- {se[0]:.3f}   (0.000 = calibrated, "
          f"z vs 0 = {z_int:+.2f})")
    print(f"  spread of forecasts: sd {p.std():.4f}, range [{p.min():.3f}, {p.max():.3f}]")
    print(f"  Brier {brier:.4f} = reliability {rel:.4f} - resolution {res:.4f} "
          f"+ uncertainty {unc:.4f}")
    verdict = (
        "calibrated" if abs(z_slope) < 1.96 and abs(z_int) < 1.96
        else "MISCALIBRATED"
    )
    print(f"  verdict: {verdict}")
    return {"slope": slope, "rel": rel, "res": res, "brier": brier, "unc": unc}


def reliability_table(name: str, p: np.ndarray, y: np.ndarray) -> None:
    _, _, _, rows = murphy(p, y)
    print(f"\n{name} reliability by decile of forecast:")
    print(f"  {'n':>5} | {'range':>15} | {'mean fcst':>9} | {'observed':>9} | {'err':>7}")
    print("  " + "-" * 58)
    for n, lo, hi, pk, ok in rows:
        print(f"  {n:5d} | [{lo:.3f}, {hi:.3f}] | {pk:8.1%} | {ok:8.1%} | {pk - ok:+6.1%}")


def main() -> None:
    frame = pl.read_parquet(FRAME).drop_nulls(["model_p", "market_p", "home_won"])
    y = frame["home_won"].to_numpy().astype(int)
    pm = frame["model_p"].to_numpy()
    pk = frame["market_p"].to_numpy()

    print(f"Walk-forward out-of-sample predictions, {len(frame):,} games, 2020-2025")
    print(f"Base rate (home wins): {y.mean():.2%}")
    print("=" * 74)
    print()
    m = summarise("MODEL", pm, y)
    print()
    k = summarise("MARKET (de-vigged close)", pk, y)

    print()
    print("=" * 74)
    print("Comparison")
    print(f"  resolution: model {m['res']:.4f} vs market {k['res']:.4f}  "
          f"-> market carries {k['res'] / max(m['res'], 1e-9):.1f}x the discriminating signal")
    print(f"  reliability: model {m['rel']:.4f} vs market {k['rel']:.4f}  "
          f"(lower is better)")
    print(f"  always-predict-base-rate Brier would be {m['unc']:.4f}; "
          f"model {m['brier']:.4f}, market {k['brier']:.4f}")

    reliability_table("MODEL", pm, y)
    reliability_table("MARKET", pk, y)

    # Calibration restricted to games the strategy backs.
    print()
    print("=" * 74)
    print("Calibration on the SELECTED subset (|model - market| >= 5%), backed side only")
    dis = frame["disagreement"].to_numpy()
    sel = np.abs(dis) >= 0.05
    back_home = dis[sel] >= 0
    p_sel = np.where(back_home, pm[sel], 1 - pm[sel])
    k_sel = np.where(back_home, pk[sel], 1 - pk[sel])
    y_sel = np.where(back_home, y[sel], 1 - y[sel])
    print(f"  {sel.sum():,} of {len(frame):,} games ({sel.mean():.0%})")
    print()
    summarise("MODEL on selected", p_sel, y_sel)
    print()
    summarise("MARKET on selected", k_sel, y_sel)


if __name__ == "__main__":
    main()
