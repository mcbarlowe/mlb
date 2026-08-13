"""Evaluate a market+sim blend for totals (over/under) betting.

Reads per-game rows of (season, point, sim_over, mkt_over, actual_total) from
``sim_totals_eval`` log files and reports discrimination (AUC) and calibration
*separately*, plus a flat-bet ROI, using a leak-free rolling walk-forward blend
(each season is predicted only from strictly earlier seasons).

    uv run python scripts/totals_blend_eval.py \
        --logs 2024:/tmp/sim_totals_2024.log \
        --logs 2025:/tmp/sim_totals_2025.log

Each log line is parsed with the regex::

    ^(\\d+) pt=([0-9.]+) sim_over=([0-9.]+) mkt_over=([0-9.]+) actual=([0-9.]+)

Pushes (``actual == point``) are dropped; the label is ``y = 1`` when
``actual > point`` else ``0``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-9
LINE_RE = re.compile(
    r"^(\d+) pt=([0-9.]+) sim_over=([0-9.]+) mkt_over=([0-9.]+) actual=([0-9.]+)"
)


@dataclass(frozen=True)
class Row:
    game_pk: int
    season: int
    point: float
    sim: float
    mkt: float
    actual: float
    y: int


def parse_logs(specs: list[str]) -> dict[int, list[Row]]:
    """Parse ``season:path`` specs into ``{season: [Row, ...]}`` (pushes dropped)."""
    by_season: dict[int, list[Row]] = {}
    for spec in specs:
        season_str, _, path_str = spec.partition(":")
        if not path_str:
            raise SystemExit(f"--logs expects 'season:path', got: {spec!r}")
        season = int(season_str)
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"log file not found: {path}")
        rows: list[Row] = []
        for line in path.read_text().splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            game_pk = int(m.group(1))
            point = float(m.group(2))
            sim = float(m.group(3))
            mkt = float(m.group(4))
            actual = float(m.group(5))
            if actual == point:  # push
                continue
            y = 1 if actual > point else 0
            rows.append(Row(game_pk, season, point, sim, mkt, actual, y))
        by_season[season] = rows
    return by_season


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def fmt_auc(auc: float | None) -> str:
    return f"{auc:.4f}" if auc is not None else "  n/a "


def arrays(rows: list[Row]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = np.array([r.sim for r in rows], dtype=float)
    mkt = np.array([r.mkt for r in rows], dtype=float)
    y = np.array([r.y for r in rows], dtype=int)
    return sim, mkt, y


# ---------------------------------------------------------------------------
# Reporting: discrimination + calibration
# ---------------------------------------------------------------------------


def report_metrics(label: str, rows: list[Row]) -> None:
    sim, mkt, y = arrays(rows)
    n = len(rows)
    over_rate = float(np.mean(y))
    print(f"\n=== {label}: {n} games, actual over-rate {over_rate:.3f} ===")

    print("            Brier   log loss     AUC")
    for name, p in (("sim", sim), ("market", mkt)):
        print(
            f"  {name:<8}{brier(p, y):8.4f}{logloss(p, y):10.4f}"
            f"{'   ' + fmt_auc(safe_auc(y, p))}"
        )

    print("  calibration (mean predicted P(over) vs actual over-rate):")
    for name, p in (("sim", sim), ("market", mkt)):
        print(
            f"    {name:<8} mean pred {np.mean(p):.3f}   actual {over_rate:.3f}"
            f"   diff {np.mean(p) - over_rate:+.3f}"
        )

    print("  sim reliability (5 equal-width bins on P(over)):")
    print("    bin           n   mean_pred   actual")
    edges = np.linspace(0.0, 1.0, 6)
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        if i == 4:
            mask = (sim >= lo) & (sim <= hi)
        else:
            mask = (sim >= lo) & (sim < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"    [{lo:.2f},{hi:.2f})   {0:4d}      n/a      n/a")
            continue
        print(
            f"    [{lo:.2f},{hi:.2f})   {cnt:4d}    {np.mean(sim[mask]):.4f}"
            f"   {np.mean(y[mask]):.4f}"
        )


# ---------------------------------------------------------------------------
# Rolling walk-forward blend
# ---------------------------------------------------------------------------


@dataclass
class BlendResult:
    oos_rows: list[Row]
    blend_p: np.ndarray  # mkt+sim OOS predictions, aligned with oos_rows
    mktonly_p: np.ndarray  # market-only OOS predictions, aligned with oos_rows
    fold_coefs: list[tuple[int, int, float, float, float]]
    # (season, n_train, intercept, coef_mkt, coef_sim)
    mktonly_coefs: list[tuple[int, int, float, float]]
    # (season, n_train, intercept, coef_mkt)
    in_sample: bool


def fit_predict(train: list[Row], test: list[Row], features: list[str]):
    s_tr, m_tr, y_tr = arrays(train)
    s_te, m_te, _ = arrays(test)
    cols_tr = {"mkt": m_tr, "sim": s_tr}
    cols_te = {"mkt": m_te, "sim": s_te}
    X_tr = np.column_stack([cols_tr[f] for f in features])
    X_te = np.column_stack([cols_te[f] for f in features])
    clf = LogisticRegression()
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    return clf, p


def rolling_blend(by_season: dict[int, list[Row]]) -> BlendResult:
    seasons = sorted(by_season)
    oos_rows: list[Row] = []
    blend_chunks: list[np.ndarray] = []
    mktonly_chunks: list[np.ndarray] = []
    fold_coefs: list[tuple[int, int, float, float, float]] = []
    mktonly_coefs: list[tuple[int, int, float, float]] = []

    if len(seasons) == 1:
        s = seasons[0]
        rows = by_season[s]
        clf, p = fit_predict(rows, rows, ["mkt", "sim"])
        clf_m, pm = fit_predict(rows, rows, ["mkt"])
        oos_rows.extend(rows)
        blend_chunks.append(p)
        mktonly_chunks.append(pm)
        fold_coefs.append(
            (s, len(rows), float(clf.intercept_[0]), float(clf.coef_[0][0]),
             float(clf.coef_[0][1]))
        )
        mktonly_coefs.append(
            (s, len(rows), float(clf_m.intercept_[0]), float(clf_m.coef_[0][0]))
        )
        return BlendResult(
            oos_rows,
            np.concatenate(blend_chunks),
            np.concatenate(mktonly_chunks),
            fold_coefs,
            mktonly_coefs,
            in_sample=True,
        )

    for s in seasons[1:]:
        train = [r for prev in seasons if prev < s for r in by_season[prev]]
        test = by_season[s]
        if not train or not test:
            continue
        clf, p = fit_predict(train, test, ["mkt", "sim"])
        clf_m, pm = fit_predict(train, test, ["mkt"])
        oos_rows.extend(test)
        blend_chunks.append(p)
        mktonly_chunks.append(pm)
        fold_coefs.append(
            (s, len(train), float(clf.intercept_[0]), float(clf.coef_[0][0]),
             float(clf.coef_[0][1]))
        )
        mktonly_coefs.append(
            (s, len(train), float(clf_m.intercept_[0]), float(clf_m.coef_[0][0]))
        )

    return BlendResult(
        oos_rows,
        np.concatenate(blend_chunks) if blend_chunks else np.array([]),
        np.concatenate(mktonly_chunks) if mktonly_chunks else np.array([]),
        fold_coefs,
        mktonly_coefs,
        in_sample=False,
    )


def report_blend(res: BlendResult) -> None:
    print("\n=== Rolling walk-forward blend (leak-free) ===")
    if res.in_sample:
        print("  IN-SAMPLE (no OOS possible): only one season available.")
    else:
        print("  Each season predicted from LogisticRegression fit on strictly")
        print("  earlier seasons only. Metrics pooled over held-out seasons.")

    if len(res.oos_rows) == 0:
        print("  no held-out games to evaluate.")
        return

    y = np.array([r.y for r in res.oos_rows], dtype=int)
    print("\n  fitted coefficients per fold (features: market, sim):")
    print("    season  n_train  intercept   coef_mkt   coef_sim")
    for s, n_tr, b0, cm, cs in res.fold_coefs:
        print(f"    {s:<7} {n_tr:>7}  {b0:+.4f}   {cm:+.4f}   {cs:+.4f}")
    print("  fitted coefficients per fold (market-only baseline):")
    print("    season  n_train  intercept   coef_mkt")
    for s, n_tr, b0, cm in res.mktonly_coefs:
        print(f"    {s:<7} {n_tr:>7}  {b0:+.4f}   {cm:+.4f}")

    print("\n  pooled held-out metrics:")
    print("                     Brier   log loss     AUC")
    print(
        f"    blend(mkt+sim) {brier(res.blend_p, y):8.4f}"
        f"{logloss(res.blend_p, y):10.4f}   {fmt_auc(safe_auc(y, res.blend_p))}"
    )
    print(
        f"    blend(mkt-only){brier(res.mktonly_p, y):8.4f}"
        f"{logloss(res.mktonly_p, y):10.4f}   {fmt_auc(safe_auc(y, res.mktonly_p))}"
    )


# ---------------------------------------------------------------------------
# Flat-bet ROI at -110
# ---------------------------------------------------------------------------

EDGES = (0.03, 0.05, 0.08)
WIN_PROFIT = 100.0 / 110.0  # net win units on a -110 bet


def roi_for_signal(
    signal_over: np.ndarray, mkt: np.ndarray, y: np.ndarray, edge: float
) -> tuple[int, float, float]:
    """Bet the side ``signal`` favors vs market by > edge. Returns (bets, wins, roi)."""
    bet_over = signal_over - mkt > edge
    bet_under = mkt - signal_over > edge
    bets = int(bet_over.sum() + bet_under.sum())
    if bets == 0:
        return 0, 0.0, 0.0
    wins = float(np.sum((bet_over & (y == 1)) | (bet_under & (y == 0))))
    roi = (wins * WIN_PROFIT - (bets - wins)) / bets
    return bets, wins, roi


def report_roi(pooled: list[Row], res: BlendResult) -> None:
    print("\n=== Flat-bet ROI @ -110 ===")
    sim, mkt, y = arrays(pooled)

    print("  sim-only (bet where sim favors a side vs market by > edge):")
    print("    edge    bets    wins    win%      ROI")
    for edge in EDGES:
        bets, wins, roi = roi_for_signal(sim, mkt, y, edge)
        if bets == 0:
            print(f"    {edge:.2f}       0     0.0     n/a      n/a")
        else:
            print(
                f"    {edge:.2f}   {bets:5d}  {wins:6.1f}  {wins / bets:6.1%}"
                f"  {roi:+7.1%}"
            )

    print("  blend-vs-market (bet where blend favors a side vs market by > edge):")
    if len(res.oos_rows) == 0:
        print("    no held-out blend predictions available.")
        return
    tag = " [IN-SAMPLE]" if res.in_sample else ""
    b_mkt = np.array([r.mkt for r in res.oos_rows], dtype=float)
    b_y = np.array([r.y for r in res.oos_rows], dtype=int)
    print(f"    edge    bets    wins    win%      ROI{tag}")
    for edge in EDGES:
        bets, wins, roi = roi_for_signal(res.blend_p, b_mkt, b_y, edge)
        if bets == 0:
            print(f"    {edge:.2f}       0     0.0     n/a      n/a")
        else:
            print(
                f"    {edge:.2f}   {bets:5d}  {wins:6.1f}  {wins / bets:6.1%}"
                f"  {roi:+7.1%}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logs",
        action="append",
        required=True,
        metavar="SEASON:PATH",
        help="repeatable season:path to a sim_totals_eval log file",
    )
    args = ap.parse_args()

    by_season = parse_logs(args.logs)
    seasons = sorted(by_season)
    pooled = [r for s in seasons for r in by_season[s]]
    if not pooled:
        raise SystemExit("no usable rows parsed from the provided logs")

    print("Totals market+sim blend evaluation")
    print(f"seasons: {seasons}   total non-push games: {len(pooled)}")

    for s in seasons:
        report_metrics(f"Season {s}", by_season[s])
    if len(seasons) > 1:
        report_metrics("POOLED", pooled)

    res = rolling_blend(by_season)
    report_blend(res)
    report_roi(pooled, res)


if __name__ == "__main__":
    main()
