"""Does adding Kalshi as a model feature help? Cross-validated logistic stack
on 2026 games combining our walk-forward model, the sharp sportsbook consensus,
and Kalshi's pre-game price. Reports out-of-fold Brier for each feature set and
the full-sample coefficients (in logit space, so ~1.0 = 'just be that market').
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import walkforward_home_probs
from scripts.kalshi_analysis import kalshi_home_probs, sharp_home_probs
from src.betting.ingest import load_finals


def logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def main() -> None:
    finals = load_finals([2026]).set_index("game_pk")
    won = finals["home_won"].to_dict()
    kal = kalshi_home_probs()
    sharp = sharp_home_probs(2026)
    model = walkforward_home_probs(2026, [2021, 2022, 2023, 2024, 2025])
    model = model.set_index("game_pk")["model_prob_home"].to_dict()

    pks = [pk for pk in kal if pk in won and pk in sharp and pk in model]
    y = np.array([1.0 if won[pk] else 0.0 for pk in pks])
    feats = {
        "model": np.array([model[pk] for pk in pks]),
        "sharp": np.array([sharp[pk] for pk in pks]),
        "kalshi": np.array([kal[pk] for pk in pks]),
    }
    n = len(pks)
    print(f"common games (model+sharp+kalshi+result): {n}, home-win rate {y.mean():.3f}\n")

    def brier(p):
        return float(np.mean((p - y) ** 2))

    print("raw (each source as-is):")
    for name, p in feats.items():
        print(f"  {name:20s} Brier {brier(p):.4f}")

    def cv_stack(cols):
        X = np.column_stack([logit(feats[c]) for c in cols])
        oof = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=7).split(X):
            m = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        full = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        return brier(oof), full.intercept_[0], dict(zip(cols, full.coef_[0]))

    print("\nout-of-fold logistic stacks (5-fold CV):")
    for cols in (["model"], ["sharp"], ["kalshi"], ["model", "sharp"],
                 ["sharp", "kalshi"], ["model", "sharp", "kalshi"]):
        b, _ic, coef = cv_stack(cols)
        cs = " ".join(f"{k}={v:+.2f}" for k, v in coef.items())
        print(f"  {'+'.join(cols):26s} OOF Brier {b:.4f} | coef: {cs}")


if __name__ == "__main__":
    main()
