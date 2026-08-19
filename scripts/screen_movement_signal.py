"""Screen candidate features against the open-to-close line move, then trade the prediction.

Target is the signed move, ``logit(close_fair) - logit(open_fair)``, conditioned on the opening
price. Direction matters as much as magnitude: filtering to games that move a lot is useless
without knowing which way, so the screen is on signed move rather than absolute move.

Pass one is a univariate screen with a Bonferroni threshold for the candidate count. Pass two
fits the surviving set walk-forward and asks the only question that pays: do bets selected by
predicted movement beat bets selected by raw model disagreement, at opening prices?

The benchmark to beat is the oracle, which knows the close exactly and returns +6.14% restricted
to moves of at least three points. The benchmark to beat on the downside is our outcome model,
which returns -1.43% at the same threshold because it selects on its own noise.

    uv run python scripts/screen_movement_signal.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

FRAME = Path("data/analysis/line_movement.parquet")
EPS = 1e-6
BOOT = 4000

CANDIDATES = {
    "model_disagree": "our outcome model minus the opening fair price",
    "book_dispersion": "cross-book spread of fair probability at the open",
    "n_books": "panel books posted at the open",
    "lead_hours": "hours from opening snapshot to first pitch",
    "fav_extremity": "|open_fair - 0.5|, how lopsided the opener is",
    "is_night": "night game",
    "is_doubleheader": "either half of a doubleheader",
    "month_late": "September or later",
    "is_early_season": "April or earlier",
}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ols(x: np.ndarray, y: np.ndarray):
    xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    resid = y - xd @ beta
    dof = len(y) - xd.shape[1]
    s2 = float((resid**2).sum() / dof)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xd.T @ xd)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot else 0.0
    return beta, se, r2


def roi_ci(settled, seed=5):
    n = len(settled)
    if not n:
        return None, None, None
    staked = sum(s for s, _ in settled)
    roi = sum(p for _, p in settled) / staked
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            sum(settled[i][1] for i in idx) / sum(settled[i][0] for i in idx)
        )
    draws.sort()
    return roi, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def settle(frame: pl.DataFrame, signal: np.ndarray, threshold: float):
    """Bet the side the signal says the line will move toward, at the best opening price."""
    home = signal >= 0
    take = np.where(home, frame["best_home_dec"].to_numpy(), frame["best_away_dec"].to_numpy())
    won = np.where(home, frame["home_won"].to_numpy(), ~frame["home_won"].to_numpy())
    keep = np.abs(signal) >= threshold
    return [
        (1.0, (t - 1.0) if w else -1.0)
        for t, w, k in zip(take, won, keep, strict=True) if k
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--move-threshold", type=float, default=0.12,
                    help="logit points of predicted movement required to bet")
    args = ap.parse_args()

    frame = pl.read_parquet(FRAME).drop_nulls(["move", "open_fair", "model_p"])
    y = frame["move"].to_numpy()
    base = logit(frame["open_fair"].to_numpy())

    k = len(CANDIDATES)
    t_bonf = 2.77  # two-sided alpha = 0.05 / 9, normal approximation
    print("Pass 1: univariate screen on signed move, conditioned on logit(open_fair).")
    print(f"n = {len(frame):,}. Bonferroni threshold for {k} candidates: |t| > {t_bonf:.2f}")
    print()
    print(f"{'feature':>17} | {'coef':>10} | {'se':>8} | {'t':>7} | {'dR2':>7} | "
          f"{'survives':>8} | description")
    print("-" * 108)

    _, _, r2_base = ols(base.reshape(-1, 1), y)
    survivors = []
    for name, desc in CANDIDATES.items():
        sub = frame.drop_nulls([name])
        if len(sub) < 500:
            print(f"{name:>17} | insufficient rows ({len(sub)})")
            continue
        yy = sub["move"].to_numpy()
        bb = logit(sub["open_fair"].to_numpy())
        x = np.column_stack([bb, sub[name].to_numpy().astype(float)])
        beta, se, r2 = ols(x, yy)
        _, _, r2b = ols(bb.reshape(-1, 1), yy)
        t = beta[2] / se[2]
        ok = abs(t) > t_bonf
        if ok:
            survivors.append(name)
        print(f"{name:>17} | {beta[2]:+10.5f} | {se[2]:8.5f} | {t:+7.2f} | "
              f"{r2 - r2b:+7.4f} | {'YES' if ok else 'no':>8} | {desc}")

    print()
    print(f"baseline R2 from the opening price alone: {r2_base:.4f}")
    print(f"survivors: {survivors or 'none'}")

    if not survivors:
        print("\nNo candidate predicts the move after conditioning on the opening price.")
        return

    print()
    print("Pass 2: walk-forward movement model, traded at opening prices.")
    print(f"Bet when predicted |move| >= {args.move_threshold:.2f} logits.")
    print()
    seasons = sorted(frame["season"].unique().to_list())
    pred_settled, model_settled = [], []
    print(f"{'test':>5} | {'n':>5} | {'OOS R2':>7} | {'bets':>5} | {'move-pred ROI':>13} | "
          f"{'model-sel ROI':>13}")
    print("-" * 68)
    for i, test in enumerate(seasons):
        if i < 2:
            continue
        tr = frame.filter(pl.col("season").is_in(seasons[:i])).drop_nulls(survivors)
        ho = frame.filter(pl.col("season") == test).drop_nulls(survivors)
        if len(tr) < 500 or len(ho) < 100:
            continue
        xtr = np.column_stack(
            [logit(tr["open_fair"].to_numpy())]
            + [tr[c].to_numpy().astype(float) for c in survivors]
        )
        xho = np.column_stack(
            [logit(ho["open_fair"].to_numpy())]
            + [ho[c].to_numpy().astype(float) for c in survivors]
        )
        beta, _, _ = ols(xtr, tr["move"].to_numpy())
        pred = np.column_stack([np.ones(len(xho)), xho]) @ beta
        yho = ho["move"].to_numpy()
        ss = float(((yho - yho.mean()) ** 2).sum())
        oos_r2 = 1.0 - float(((yho - pred) ** 2).sum()) / ss if ss else 0.0

        s_pred = settle(ho, pred, args.move_threshold)
        s_model = settle(
            ho, (ho["model_p"] - ho["open_fair"]).to_numpy(), 0.05
        )
        pred_settled += s_pred
        model_settled += s_model
        rp, _, _ = roi_ci(s_pred)
        rm, _, _ = roi_ci(s_model)
        print(f"{test:5d} | {len(ho):5d} | {oos_r2:+7.4f} | {len(s_pred):5d} | "
              f"{(rp if rp is not None else 0):+12.2%} | "
              f"{(rm if rm is not None else 0):+12.2%}")

    print("-" * 68)
    for label, settled in (("movement-selected", pred_settled),
                           ("model-selected", model_settled)):
        roi, lo, hi = roi_ci(settled)
        if roi is None:
            print(f"{label:>18}: no bets")
            continue
        print(f"{label:>18}: {len(settled):5d} bets  ROI {roi:+7.2%}  "
              f"95% CI [{lo:+7.2%}, {hi:+7.2%}]")
    print()
    print("Oracle ceiling for reference: +6.14% (95% CI +1.14 to +10.94) on 1,566 bets,")
    print("restricted to games the line actually moved at least three points.")


if __name__ == "__main__":
    main()
