"""Leak-free calibration + lineage metrics for the batter prop estimator.

Reproduces the estimator lineage and evaluates every version on held-out
2024 and 2025 starts (PA >= 3, matching live alert semantics):

  v1 props-raw-shrink-v1      uniform window, no aging, EB shrink k=50
  v2 props-decay400-age-k50   + recency decay (H=400 games) + aging curves
  v3 props-cond-v3            + hits-family conditioning: starts-only stream,
                              expected-PA rescale, park offsets (HR stays on
                              the v2 estimator - calibration shows conditioning
                              does not help HR)

Metrics per (test season, stat in {hr1, h1}): max-pick and top-decile
realized/estimated, UNDER-side top-decile realized/estimated, Brier, log
loss, and flat-1u ROI at the production text-gate price (HR +15%, H +11%,
price = (1+gate)/estimate: "book is gate-cheap vs our number").

Used by scripts/register_prop_model_mlflow.py; run directly for the table.
"""

from __future__ import annotations

import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

STATS = ("hr1", "h1")
TESTS = (2024, 2025)
START_PA = 3
MIN_GP = 150
K = 50.0
HALF_LIFE = 400.0
K_PARK = 2000.0
EXP_PA_WINDOW = 30
CURVE_MIN_GP = 60
AGE_LO, AGE_HI = 21, 38
GATE = {"hr1": 0.15, "h1": 0.11}
VERSIONS = ("props-raw-shrink-v1", "props-decay400-age-k50-mkt", "props-cond-v3")


def logit(p):
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_lines() -> pd.DataFrame:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    frame = pd.read_sql(
        """
        SELECT b.player_id, g.season::int AS season,
               COALESCE(g.game_datetime, g.game_date::timestamptz) AS gdt,
               g.game_date::date AS game_date, g.home_team_id,
               p.birth_date::date AS birth_date,
               COALESCE(b.plateappearances, 0)::int AS pa,
               (COALESCE(b.homeruns, 0) >= 1)::int AS hr1,
               (COALESCE(b.hits, 0) >= 1)::int AS h1
        FROM mlb.batting b
        JOIN mlb.games g USING (game_pk)
        JOIN mlb.players p USING (player_id)
        WHERE g.game_type = 'R' AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN 2015 AND 2025
          AND COALESCE(b.plateappearances, 0) > 0
          AND p.birth_date IS NOT NULL
        ORDER BY player_id, gdt, game_pk
        """,
        conn,
    )
    conn.close()
    frame["age"] = (
        pd.to_datetime(frame["game_date"]) - pd.to_datetime(frame["birth_date"])
    ).dt.days / 365.25
    return frame


def build_curve(frame: pd.DataFrame, stat: str, max_season: int) -> dict[int, float]:
    hist = frame[(frame["season"] <= max_season) & (frame["pa"] >= START_PA)]
    agg = (
        hist.assign(iage=hist["age"].round().astype(int))
        .groupby(["player_id", "season", "iage"])[stat]
        .agg(succ="sum", n="count")
        .reset_index()
    )
    agg = agg[agg["n"] >= CURVE_MIN_GP]
    agg["lo"] = logit((agg["succ"] + 0.5) / (agg["n"] + 1.0))
    deltas: dict[int, list[tuple[float, float]]] = {}
    for _, g in agg.groupby("player_id"):
        g = g.sort_values("season")
        rows = list(g.itertuples(index=False))
        for a, b in pairwise(rows):
            if b.season == a.season + 1 and b.iage == a.iage + 1:
                w = 2.0 / (1.0 / a.n + 1.0 / b.n)
                deltas.setdefault(int(a.iage), []).append((float(b.lo - a.lo), w))
    step = {age: sum(d * w for d, w in ps) / sum(w for _, w in ps)
            for age, ps in deltas.items() if ps}
    curve = {27: 0.0}
    for age in range(27, AGE_HI):
        curve[age + 1] = curve[age] + step.get(age, 0.0)
    for age in range(27, AGE_LO, -1):
        curve[age - 1] = curve[age] - step.get(age - 1, 0.0)
    return curve


def curve_at(curve: dict[int, float], age: float) -> float:
    a = min(max(age, AGE_LO), AGE_HI)
    lo = math.floor(a)
    hi = min(lo + 1, AGE_HI)
    frac = a - lo
    return curve[lo] * (1 - frac) + curve.get(hi, curve[lo]) * frac


def park_deltas(frame: pd.DataFrame, stat: str, max_season: int) -> dict[int, float]:
    hist = frame[
        (frame["season"].between(max_season - 1, max_season)) & (frame["pa"] >= START_PA)
    ]
    overall = hist[stat].mean()
    lo_all = math.log(overall / (1 - overall))
    out: dict[int, float] = {}
    for venue, g in hist.groupby("home_team_id"):
        rate = (g[stat].sum() + 0.5) / (len(g) + 1.0)
        out[int(venue)] = (math.log(rate / (1 - rate)) - lo_all) * len(g) / (
            len(g) + K_PARK
        )
    return out


def trailing(sub: pd.DataFrame, stat: str, lam: float) -> pd.DataFrame:
    """Decayed rate/age/PA streams + start counts + last-N mean PA, per player."""
    outs = {k: [] for k in ("s_w", "n_w", "age_w", "gp", "exp_pa", "mean_pa")}
    for _, g in sub.groupby("player_id", sort=False):
        x = g[stat].to_numpy(float)
        age = g["age"].to_numpy(float)
        pa = g["pa"].to_numpy(float)
        n = len(x)
        s = w = a = paw = 0.0
        s_arr = np.empty(n); w_arr = np.empty(n); a_arr = np.empty(n)
        exp_arr = np.empty(n); mean_arr = np.empty(n)
        for i in range(n):
            s_arr[i], w_arr[i], a_arr[i] = s, w, a
            lo = max(0, i - EXP_PA_WINDOW)
            exp_arr[i] = pa[lo:i].mean() if i > lo else float("nan")
            mean_arr[i] = paw / w if w > 0 else float("nan")
            s = lam * s + x[i]
            w = lam * w + 1.0
            a = lam * a + age[i]
            paw = lam * paw + pa[i]
        outs["s_w"].append(s_arr); outs["n_w"].append(w_arr)
        outs["age_w"].append(a_arr); outs["gp"].append(np.arange(n, dtype=float))
        outs["exp_pa"].append(exp_arr); outs["mean_pa"].append(mean_arr)
    res = sub.copy()
    for k, v in outs.items():
        res[k] = np.concatenate(v)
    return res


def _estimate(f: pd.DataFrame, mu: float, curve: dict[int, float] | None,
              k: float) -> np.ndarray:
    p_w = ((f["s_w"] + 0.5) / (f["n_w"] + 1.0)).to_numpy()
    if curve is not None:
        mean_age = (f["age_w"] / f["n_w"]).to_numpy()
        delta = np.clip(
            np.array([curve_at(curve, an) - curve_at(curve, am)
                      for an, am in zip(f["age"].to_numpy(), mean_age)]),
            -0.75, 0.75,
        )
        p_w = sigmoid(logit(np.clip(p_w, 1e-6, 1 - 1e-6)) + delta)
    return (f["n_w"].to_numpy() * p_w + k * mu) / (f["n_w"].to_numpy() + k)


def _metrics(rows: pd.DataFrame, stat: str, col: str) -> dict[str, float]:
    y = rows[stat].to_numpy(float)
    p = rows[col].to_numpy(float)
    picks = rows.loc[rows.groupby("game_date")[col].idxmax()]
    top = rows[rows[col] >= rows[col].quantile(0.9)]
    bot = rows[rows[col] <= rows[col].quantile(0.1)]
    dec = (1.0 + GATE[stat]) / picks[col].to_numpy()
    profit = picks[stat].to_numpy(float) * (dec - 1.0) - (1.0 - picks[stat].to_numpy(float))
    return {
        "maxpick_ratio": float(picks[stat].mean() / picks[col].mean()),
        "top10_ratio": float(top[stat].mean() / top[col].mean()),
        "under10_ratio": float((1 - bot[stat]).mean() / (1 - bot[col]).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "logloss": float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))),
        "gate_roi": float(profit.mean()),
    }


def lineage_metrics(frame: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    """version -> {"<test>_<stat>_<metric>": value} on held-out 2024+2025 starts."""
    if frame is None:
        frame = load_lines()
    lam = 0.5 ** (1.0 / HALF_LIFE)
    out: dict[str, dict[str, float]] = {v: {} for v in VERSIONS}
    for test in TESTS:
        window = frame[frame["season"].between(test - 2, test)].sort_values(
            ["player_id", "gdt"]
        )
        starts = window[window["pa"] >= START_PA]
        for stat in STATS:
            curve = build_curve(frame, stat, test - 1)
            parks = park_deltas(frame, stat, test - 1)
            mu_all = frame.loc[frame["season"].between(test - 2, test - 1), stat].mean()
            mu_starts = frame.loc[
                frame["season"].between(test - 2, test - 1)
                & (frame["pa"] >= START_PA), stat
            ].mean()

            fa_uni = trailing(window, stat, 1.0)
            fa_uni = fa_uni[(fa_uni["season"] == test) & (fa_uni["pa"] >= START_PA)
                            & (fa_uni["gp"] >= MIN_GP)]
            fa = trailing(window, stat, lam)
            fa = fa[(fa["season"] == test) & (fa["pa"] >= START_PA)
                    & (fa["gp"] >= MIN_GP)]
            fb = trailing(starts, stat, lam)
            fb = fb[(fb["season"] == test) & (fb["gp"] >= MIN_GP)]

            fa_uni = fa_uni.assign(p=_estimate(fa_uni, mu_all, None, K))
            fa = fa.assign(p=_estimate(fa, mu_all, curve, K))
            # v3 hybrid: HR uses the v2 estimator; hits-family conditioned
            if stat == "hr1":
                fc = fa
            else:
                p_b = _estimate(fb, mu_starts, curve, K)
                q = 1.0 - (1.0 - p_b) ** (1.0 / fb["mean_pa"].clip(lower=1.0))
                exp_pa = fb["exp_pa"].fillna(fb["mean_pa"]).clip(lower=1.0, upper=6.0)
                p_pa = 1.0 - (1.0 - q) ** exp_pa
                park = fb["home_team_id"].map(parks).fillna(0.0)
                fc = fb.assign(p=sigmoid(logit(np.clip(p_pa, 1e-4, 1 - 1e-4)) + park))

            for version, f in zip(VERSIONS, (fa_uni, fa, fc)):
                for key, val in _metrics(f, stat, "p").items():
                    out[version][f"{test}_{stat}_{key}"] = round(val, 5)
    return out


def main() -> None:
    metrics = lineage_metrics()
    for version, m in metrics.items():
        print(f"\n=== {version} ===")
        for key in sorted(m):
            print(f"  {key:<28} {m[key]:+.4f}" if "roi" in key or "ratio" in key
                  else f"  {key:<28} {m[key]:.4f}")


if __name__ == "__main__":
    main()
