"""Pythagorean regression with managers as a random effect.

Question: once a team's run scoring and run prevention are accounted for, is
there a persistent manager-level component to winning?

Design
------
Unit of analysis is a *manager stint*: one (season, franchise, manager) cell
aggregated to games ``G``, wins ``W``, runs scored ``RS``, runs allowed ``RA``.
Tied games are dropped from every total so ``W/G`` is a true win percentage.

The Pythagorean expectation uses an exponent ``k`` fitted on the data::

    pythag = RS^k / (RS^k + RA^k)

Response specifications:

``resid``    y = W/G - pythag   (offset form: slope on pythag pinned at 1)
``winpct``   y = W/G with pythag as a covariate — the literal "pythagorean
             regression", free intercept and slope
``rundiff``  y = (RS - RA)/G    (no pythag conditioning; upper bound only)

Mixed model, fitted by exact REML::

    y_i = x_i'beta + sum_g a_{group_g(i)} + e_i
    a_g ~ N(0, sigma^2_g)        e_i ~ N(0, sigma^2_e / G_i)

The ``1/G_i`` residual scaling is the sampling noise of a win percentage over
``G`` games; ``--report`` includes the empirical check of that scaling rather
than assuming it. The likelihood is evaluated through the
Woodbury/matrix-determinant identities on the ``q x q`` random-effect space
(``q`` = managers + franchises + seasons, a few hundred) instead of the
``n x n`` observation space, and every term that does not depend on the
variance parameters is precomputed once.

Identification: ``sigma^2_m`` is identified *only* by managers with more than
one stint — with a single stint per manager the manager effect and the stint
residual are the same quantity. Franchise and season effects are included so
``sigma^2_m`` cannot simply absorb organizational or era differences; they
enter as random effects in the primary spec and as fixed effects in a
robustness spec.

Estimand and its limits
-----------------------
Conditioning on ``pythag`` removes everything a manager does that lands in run
totals (lineup construction, pinch hitting, base stealing, which arms pitch).
What remains is *run-to-win conversion*: sequencing and leverage — bullpen
deployment in close games, bench usage, one-run-game tactics. The ``rundiff``
spec reports the same model without that conditioning as an upper bound on
total manager association; that quantity is confounded with roster talent and
is not a causal manager effect.

Inference
---------
* REML point estimate and profile-likelihood 95% CI for ``sigma_m``.
* Likelihood-ratio test against ``sigma^2_m = 0`` with the boundary correction
  ``p = 0.5 * P(chi^2_1 > LRT)``.
* Permutation test on a weighted between-manager statistic (manager labels
  shuffled globally and within franchise), which assumes no normality.
* Split-half (odd/even season) and consecutive-season persistence of manager
  residuals.
* Simulated power for the observed design, so a null reads as "no effect
  larger than X wins per season" rather than "no effect".

Usage::

    uv run python scripts/analyze_manager_effects.py
    uv run python scripts/analyze_manager_effects.py --first-season 1969 --permutations 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb.analysis.mixed import (
    MixedFit,
    REMLProblem,
    between_group_statistic,
    fit_without_block,
    moment_variance_from_permutation,
    permutation_test,
    profile_ci_lower,
    profile_ci_upper,
    residualize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_LOG_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"
REPORT_PATH = REPO_ROOT / "output" / "manager_effects" / "manager_effects_report.json"

# A stint counts as a full season when it covers at least this share of the
# season's typical club schedule. Season length comes from the data, so 1901
# (140 games), 1981, 1994, and 2020 (60 games) need no hardcoding.
FULL_SEASON_SHARE = 0.90
GAMES_SCALE = 162.0  # every variance component is reported per 162 games


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------


def load_stints(path: Path, first_season: int, last_season: int) -> tuple[pl.DataFrame, dict]:
    log = pl.read_parquet(path).filter(pl.col("season").is_between(first_season, last_season))
    n_rows = log.height
    ties = int(log["tied"].sum())
    log = log.filter(~pl.col("tied"))

    stints = (
        log.group_by("season", "franchise_id", "manager_id")
        .agg(
            pl.col("manager_name").first().alias("manager_name"),
            pl.len().alias("games"),
            pl.col("won").sum().alias("wins"),
            pl.col("runs_scored").sum().alias("runs_scored"),
            pl.col("runs_allowed").sum().alias("runs_allowed"),
            pl.col("game_date").min().alias("first_game"),
        )
        .sort("season", "franchise_id", "first_game")
    )
    season_length = (
        log.group_by("season", "franchise_id")
        .agg(pl.len().alias("club_games"))
        .group_by("season")
        .agg(pl.col("club_games").median().alias("season_games"))
    )
    stints = stints.join(season_length, on="season", how="left").with_columns(
        (pl.col("games") >= FULL_SEASON_SHARE * pl.col("season_games")).alias("full_season"),
        (pl.col("wins") / pl.col("games")).alias("win_pct"),
    )
    meta = {
        "team_game_rows": n_rows,
        "tied_team_games_dropped": ties,
        "stints": stints.height,
        "seasons": [int(stints["season"].min()), int(stints["season"].max())],
        "managers": int(stints["manager_id"].n_unique()),
        "franchises": int(stints["franchise_id"].n_unique()),
        "full_season_stints": int(stints["full_season"].sum()),
    }
    return stints, meta


def fit_pythagorean_exponent(stints: pl.DataFrame) -> float:
    """Fit ``k`` by games-weighted least squares on full-season stints."""

    full = stints.filter(pl.col("full_season"))
    rs = full["runs_scored"].to_numpy().astype(float)
    ra = full["runs_allowed"].to_numpy().astype(float)
    wp = full["win_pct"].to_numpy()
    g = full["games"].to_numpy().astype(float)

    def loss(k: float) -> float:
        pred = rs**k / (rs**k + ra**k)
        return float(np.sum(g * (wp - pred) ** 2))

    result = optimize.minimize_scalar(loss, bounds=(1.0, 3.0), method="bounded")
    return float(np.asarray(result.x).item())


def pythagorean(rs: np.ndarray, ra: np.ndarray, k: float) -> np.ndarray:
    return rs**k / (rs**k + ra**k)




def variance_scaling_diagnostic(frame: pl.DataFrame, k: float) -> list[dict[str, float]]:
    """Empirical check that Var(residual win pct) scales like 1/G."""

    rs = frame["runs_scored"].to_numpy().astype(float)
    ra = frame["runs_allowed"].to_numpy().astype(float)
    y = frame["win_pct"].to_numpy() - pythagorean(rs, ra, k)
    g = frame["games"].to_numpy().astype(float)
    out = []
    for lo, hi in ((1, 10), (11, 25), (26, 50), (51, 100), (101, 140), (141, 200)):
        mask = (g >= lo) & (g <= hi)
        if mask.sum() < 20:
            continue
        out.append(
            {
                "games_lo": lo,
                "games_hi": hi,
                "n": int(mask.sum()),
                "mean_games": float(g[mask].mean()),
                "var_times_games": float(np.var(y[mask], ddof=1) * g[mask].mean()),
                "sd_wins_per_162": float(np.std(y[mask], ddof=1) * GAMES_SCALE),
            }
        )
    return out


def split_half_reliability(
    frame: pl.DataFrame, resid: np.ndarray, min_seasons: int
) -> dict[str, float]:
    work = frame.with_columns(pl.Series("resid", resid))
    eligible = (
        work.group_by("manager_id")
        .agg(pl.len().alias("n_stints"))
        .filter(pl.col("n_stints") >= min_seasons)
        .select("manager_id")
    )
    work = work.join(eligible, on="manager_id", how="semi").with_columns(
        (pl.col("season") % 2).alias("half")
    )
    halves = work.group_by("manager_id", "half").agg(
        ((pl.col("resid") * pl.col("games")).sum() / pl.col("games").sum()).alias("mean_resid")
    )
    even = halves.filter(pl.col("half") == 0).select("manager_id", pl.col("mean_resid").alias("even"))
    odd = halves.filter(pl.col("half") == 1).select("manager_id", pl.col("mean_resid").alias("odd"))
    paired = even.join(odd, on="manager_id", how="inner")
    if paired.height < 10:
        return {"min_seasons": min_seasons, "n_managers": paired.height}
    a = paired["even"].to_numpy()
    b = paired["odd"].to_numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    fisher = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(a.size - 3, 1))
    return {
        "min_seasons": min_seasons,
        "n_managers": int(a.size),
        "odd_even_r": r,
        "ci95": [float(np.tanh(fisher - 1.96 * se)), float(np.tanh(fisher + 1.96 * se))],
    }


def consecutive_season_correlation(frame: pl.DataFrame, resid: np.ndarray) -> dict[str, float]:
    work = (
        frame.with_columns(pl.Series("resid", resid))
        .filter(pl.col("full_season"))
        .select("manager_id", "franchise_id", "season", "resid")
    )
    nxt = work.with_columns(pl.col("season") - 1).rename({"resid": "resid_next"})
    pairs = work.join(nxt, on=["manager_id", "franchise_id", "season"], how="inner")
    if pairs.height < 20:
        return {"n_pairs": pairs.height}
    a = pairs["resid"].to_numpy()
    b = pairs["resid_next"].to_numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    fisher = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(a.size - 3, 1))
    return {
        "n_pairs": int(a.size),
        "r": r,
        "ci95": [float(np.tanh(fisher - 1.96 * se)), float(np.tanh(fisher + 1.96 * se))],
    }


def power_curve(
    X: np.ndarray,
    codes: dict[str, np.ndarray],
    weights: np.ndarray,
    fit: MixedFit,
    effects_wins: Sequence[float],
    n_sims: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Simulated power of the permutation statistic for the observed design."""

    manager_codes = codes["manager"]
    n_levels = int(manager_codes.max()) + 1
    sizes = {name: int(c.max()) + 1 for name, c in codes.items()}

    def simulate(sigma_m: float) -> np.ndarray:
        y = rng.normal(0.0, sigma_m, sizes["manager"])[manager_codes]
        for name in ("franchise", "season"):
            var = fit.variances.get(name, 0.0)
            if var > 0 and name in codes:
                y = y + rng.normal(0.0, np.sqrt(var), sizes[name])[codes[name]]
        return y + rng.normal(0.0, np.sqrt(fit.resid_variance / weights))

    null_stats = np.array(
        [
            between_group_statistic(
                residualize(simulate(0.0), X, weights), weights, manager_codes, n_levels
            )
            for _ in range(n_sims)
        ]
    )
    crit = float(np.quantile(null_stats, 0.95))
    rows = []
    for wins in effects_wins:
        sigma_m = wins / GAMES_SCALE
        hits = sum(
            between_group_statistic(
                residualize(simulate(sigma_m), X, weights), weights, manager_codes, n_levels
            )
            > crit
            for _ in range(n_sims)
        )
        rows.append({"sigma_m_wins_per_162": wins, "power": hits / n_sims})
    return rows


# ---------------------------------------------------------------------------
# spec runner
# ---------------------------------------------------------------------------


def build_codes(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    codes: dict[str, np.ndarray] = {}
    for name, column in (
        ("manager", "manager_id"),
        ("franchise", "franchise_id"),
        ("season", "season"),
    ):
        _, inverse = np.unique(frame[column].to_numpy(), return_inverse=True)
        codes[name] = inverse.astype(np.int64).ravel()
    return codes


def dummy_design(codes: np.ndarray) -> np.ndarray:
    levels = int(codes.max()) + 1
    Z = np.zeros((codes.size, levels))
    Z[np.arange(codes.size), codes] = 1.0
    return Z[:, 1:]


@dataclass
class SpecResult:
    payload: dict
    frame: pl.DataFrame
    y: np.ndarray
    X: np.ndarray
    codes: dict[str, np.ndarray]
    weights: np.ndarray
    fit: MixedFit
    resid_adjusted: np.ndarray


def career_residual_table(
    frame: pl.DataFrame,
    resid: np.ndarray,
    weights: np.ndarray,
    resid_variance: float,
    min_games: int,
    top: int,
) -> list[dict[str, object]]:
    """Unshrunken career mean residual per manager, with its standard error.

    Reported alongside the BLUPs because when the REML estimate of
    ``sigma^2_m`` sits on the zero boundary every BLUP collapses to 0, which
    hides the raw spread and the noise it has to be judged against.
    """

    work = frame.with_columns(pl.Series("resid", resid), pl.Series("w", weights))
    career = (
        work.group_by("manager_id")
        .agg(
            pl.col("manager_name").first().alias("manager"),
            pl.len().alias("seasons"),
            pl.col("w").sum().alias("games"),
            ((pl.col("resid") * pl.col("w")).sum() / pl.col("w").sum()).alias("mean_resid"),
        )
        .filter(pl.col("games") >= min_games)
        .with_columns(
            (pl.col("mean_resid") * GAMES_SCALE).alias("wins_per_162"),
            ((resid_variance / pl.col("games")).sqrt() * GAMES_SCALE).alias("se_per_162"),
        )
        .sort("wins_per_162", descending=True)
    )
    head = career.head(top)
    tail = career.tail(top)
    return [
        {
            "manager": row["manager"],
            "seasons": int(row["seasons"]),
            "games": int(row["games"]),
            "wins_per_162": float(row["wins_per_162"]),
            "se_per_162": float(row["se_per_162"]),
            "z": float(row["wins_per_162"] / row["se_per_162"]),
        }
        for row in list(head.iter_rows(named=True)) + list(tail.iter_rows(named=True))
    ]


def run_spec(
    label: str,
    frame: pl.DataFrame,
    k: float,
    response: str,
    random_effects: Sequence[str],
    fixed_effects: Sequence[str],
    permutations: int,
    rng: np.random.Generator,
    want_ci: bool,
) -> SpecResult:
    rs = frame["runs_scored"].to_numpy().astype(float)
    ra = frame["runs_allowed"].to_numpy().astype(float)
    win_pct = frame["win_pct"].to_numpy()
    games = frame["games"].to_numpy().astype(float)
    pyth = pythagorean(rs, ra, k)
    codes_all = build_codes(frame)

    columns: list[np.ndarray] = [np.ones((frame.height, 1))]
    if response == "resid":
        y = win_pct - pyth
        units = "wins per 162"
    elif response == "winpct":
        y = win_pct
        columns.append(pyth[:, None])
        units = "wins per 162"
    elif response == "rundiff":
        y = (rs - ra) / games
        units = "run differential per 162"
    else:
        raise ValueError(f"unknown response {response}")
    for name in fixed_effects:
        columns.append(dummy_design(codes_all[name]))
    X = np.column_stack(columns)

    blocks = {name: codes_all[name] for name in random_effects}
    problem = REMLProblem(y, X, games, blocks)
    fit = problem.fit()
    null_fit = fit_without_block(y, X, games, blocks, "manager")
    lrt = max(null_fit.neg2_reml - fit.neg2_reml, 0.0)

    # The permutation and persistence statistics run on residuals that always
    # absorb franchise and season, whatever the spec's random-effect
    # structure. Otherwise a manager with a long tenure at one club inherits
    # that club's persistent conversion tendency and its era.
    adjust_columns = list(columns[: 2 if response == "winpct" else 1])
    adjust_columns.append(dummy_design(codes_all["franchise"]))
    adjust_columns.append(dummy_design(codes_all["season"]))
    X_adjust = np.column_stack(adjust_columns)
    resid_adjusted = residualize(y, X_adjust, games)
    perm_global = permutation_test(
        resid_adjusted, games, codes_all["manager"], None, permutations, rng
    )
    perm_block = permutation_test(
        resid_adjusted, games, codes_all["manager"], codes_all["franchise"], permutations, rng
    )
    # resid_adjusted has franchise and season absorbed, which removes
    # between-manager variance the analytic degrees-of-freedom term does not
    # know about and biases that estimator low. Take the null expectation from
    # the permutation, which passes through the same projection.
    moment_var = moment_variance_from_permutation(perm_block, games, codes_all["manager"])

    payload: dict = {
        "label": label,
        "response": response,
        "units": units,
        "n_stints": frame.height,
        "n_managers": int(codes_all["manager"].max() + 1),
        "n_managers_multi_stint": int(
            np.sum(np.bincount(codes_all["manager"]) > 1)
        ),
        "random_effects": list(random_effects),
        "fixed_effects": list(fixed_effects),
        "beta_intercept_and_pythag": [float(b) for b in fit.beta[: 2 if response == "winpct" else 1]],
        "sigma_per_162": {
            name: float(np.sqrt(max(var, 0.0)) * GAMES_SCALE) for name, var in fit.variances.items()
        },
        "residual_sd_per_162": float(np.sqrt(fit.resid_variance / GAMES_SCALE) * GAMES_SCALE),
        "lrt_statistic": lrt,
        "lrt_p_value": 0.5 * float(stats.chi2.sf(lrt, 1)) if lrt > 0 else 0.5,
        "moment_sigma2_manager": moment_var,
        "moment_sigma_manager_signed_per_162": float(
            np.sign(moment_var) * np.sqrt(abs(moment_var)) * GAMES_SCALE
        ),
        "permutation_global": perm_global,
        "permutation_within_franchise": perm_block,
        "converged": fit.converged,
    }
    if want_ci:
        threshold = fit.neg2_reml + float(stats.chi2.ppf(0.95, 1))
        lower = profile_ci_lower(problem, y, X, games, blocks, fit, "manager", threshold)
        upper = profile_ci_upper(problem, fit, "manager", threshold)
        payload["sigma_manager_ci95_per_162"] = [
            float(np.sqrt(lower) * GAMES_SCALE),
            float(np.sqrt(upper) * GAMES_SCALE),
        ]

    if "manager" in fit.blups:
        ids = np.unique(frame["manager_id"].to_numpy())
        info = (
            frame.group_by("manager_id")
            .agg(
                pl.col("manager_name").first().alias("manager_name"),
                pl.col("games").sum().alias("games"),
                pl.len().alias("stints"),
            )
            .sort("manager_id")
        )
        blup = fit.blups["manager"] * GAMES_SCALE
        blup_sd = fit.blup_sd["manager"] * GAMES_SCALE
        order = np.argsort(-blup)
        keep = list(order[:10]) + list(order[-10:])
        payload["manager_blup_extremes"] = [
            {
                "manager": info["manager_name"][int(i)],
                "manager_id": str(ids[int(i)]),
                "seasons": int(info["stints"][int(i)]),
                "games": int(info["games"][int(i)]),
                "blup_per_162": float(blup[int(i)]),
                "blup_sd": float(blup_sd[int(i)]),
            }
            for i in keep
        ]
        payload["blup_spread_sd"] = float(np.std(blup, ddof=1))

    payload["career_residual_extremes"] = career_residual_table(
        frame, resid_adjusted, games, fit.resid_variance, min_games=1000, top=10
    )
    return SpecResult(payload, frame, y, X, codes_all, games, fit, resid_adjusted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-log", type=Path, default=GAME_LOG_PATH)
    parser.add_argument("--first-season", type=int, default=1901)
    parser.add_argument("--last-season", type=int, default=2100)
    parser.add_argument("--min-games", type=int, default=20, help="min games for the all-stint spec")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--power-sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--skip-ci", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    stints, meta = load_stints(args.game_log, args.first_season, args.last_season)
    k = fit_pythagorean_exponent(stints)
    print("data:", json.dumps(meta))
    print(f"fitted pythagorean exponent k = {k:.4f}")

    diagnostics = variance_scaling_diagnostic(stints, k)
    print("\nresidual variance scaling (Var x mean games is flat iff 1/G holds):")
    for row in diagnostics:
        print(
            f"  G {row['games_lo']:>4}-{row['games_hi']:<4} n={row['n']:>5} "
            f"meanG={row['mean_games']:6.1f} Var*G={row['var_times_games']:.4f} "
            f"sd={row['sd_wins_per_162']:6.2f} wins/162"
        )

    full = stints.filter(pl.col("full_season"))
    all_stints = stints.filter(pl.col("games") >= args.min_games)
    report: dict = {
        "meta": meta,
        "pythagorean_k": k,
        "variance_scaling": diagnostics,
        "specs": [],
    }

    specs = [
        (
            "primary: full-season stints, pythag offset, mgr+franchise+season random",
            full, "resid", ("manager", "franchise", "season"), (), True,
        ),
        (
            "literal pythagorean regression: winpct ~ pythag + (1|mgr)+(1|fr)+(1|season)",
            full, "winpct", ("manager", "franchise", "season"), (), False,
        ),
        (
            "robustness: franchise+season absorbed as fixed effects",
            full, "resid", ("manager",), ("franchise", "season"), False,
        ),
        (
            "robustness: manager random effect only",
            full, "resid", ("manager",), (), False,
        ),
        (
            f"robustness: every stint with >= {args.min_games} games",
            all_stints, "resid", ("manager", "franchise", "season"), (), False,
        ),
        (
            "upper bound, confounded with roster: run differential per game",
            full, "rundiff", ("manager", "franchise", "season"), (), False,
        ),
    ]

    primary: SpecResult | None = None
    for label, frame, response, random_effects, fixed_effects, want_ci in specs:
        print(f"\n=== {label} ===")
        result = run_spec(
            label, frame, k, response, random_effects, fixed_effects,
            args.permutations, rng, want_ci and not args.skip_ci,
        )
        report["specs"].append(result.payload)
        p = result.payload
        print(
            f"  n={p['n_stints']} managers={p['n_managers']} "
            f"(multi-season {p['n_managers_multi_stint']}) converged={p['converged']}"
        )
        for name, value in p["sigma_per_162"].items():
            print(f"  sigma_{name:<10} = {value:6.3f} {p['units']}")
        print(f"  residual sd      = {p['residual_sd_per_162']:6.3f} {p['units']}")
        if response == "winpct":
            b = p["beta_intercept_and_pythag"]
            print(f"  intercept={b[0]:+.4f} pythag slope={b[1]:.4f}")
        print(f"  LRT sigma_m=0: chi2={p['lrt_statistic']:.3f} p={p['lrt_p_value']:.4f}")
        print(
            f"  moment estimate of sigma_manager (may be negative) = "
            f"{p['moment_sigma_manager_signed_per_162']:+.3f} {p['units']}"
        )
        print(
            f"  permutation p: global={p['permutation_global']['p_value']:.4f} "
            f"within-franchise={p['permutation_within_franchise']['p_value']:.4f}"
        )
        if "sigma_manager_ci95_per_162" in p:
            lo, hi = p["sigma_manager_ci95_per_162"]
            print(f"  sigma_manager 95% profile CI = [{lo:.3f}, {hi:.3f}] {p['units']}")
        if primary is None:
            primary = result

    assert primary is not None
    resid = primary.resid_adjusted

    print("\n=== persistence of the pythagorean residual within manager ===")
    print("    (residuals already net of franchise and season)")
    report["split_half"] = []
    for min_seasons in (2, 4, 8):
        rel = split_half_reliability(primary.frame, resid, min_seasons)
        report["split_half"].append(rel)
        if rel.get("n_managers", 0) >= 10:
            lo, hi = rel["ci95"]
            print(
                f"  odd/even season r = {rel['odd_even_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                f"(>= {min_seasons} seasons, n={rel['n_managers']})"
            )
    yoy = consecutive_season_correlation(primary.frame, resid)
    report["consecutive_season"] = yoy
    if "r" in yoy:
        lo, hi = yoy["ci95"]
        print(
            f"  same manager+club, consecutive seasons r = {yoy['r']:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] (n={yoy['n_pairs']})"
        )

    print("\n=== era splits (primary spec refit within era) ===")
    report["eras"] = []
    for lo_season, hi_season in ((1901, 1946), (1947, 1976), (1977, 2000), (2001, 2100)):
        era = full.filter(pl.col("season").is_between(lo_season, hi_season))
        if era.height < 100:
            continue
        era_result = run_spec(
            f"era {lo_season}-{min(hi_season, int(era['season'].max()))}",
            era, k, "resid", ("manager", "franchise", "season"), (),
            args.permutations, rng, False,
        )
        report["eras"].append(era_result.payload)
        ep = era_result.payload
        print(
            f"  {ep['label']:<16} n={ep['n_stints']:>4} mgrs={ep['n_managers']:>3} "
            f"sigma_mgr={ep['sigma_per_162']['manager']:6.3f} "
            f"resid_sd={ep['residual_sd_per_162']:5.2f} "
            f"perm_p={ep['permutation_within_franchise']['p_value']:.3f}"
        )

    print("\n=== power of this design (permutation test, alpha=0.05) ===")
    power = power_curve(
        primary.X, primary.codes, primary.weights, primary.fit,
        (0.5, 1.0, 1.5, 2.0, 3.0), args.power_sims, rng,
    )
    report["power"] = power
    for row in power:
        print(
            f"  true sigma_m = {row['sigma_m_wins_per_162']:.1f} wins/162 "
            f"-> power {row['power']:.2f}"
        )

    primary_payload = report["specs"][0]
    blups = primary_payload.get("manager_blup_extremes", [])
    # Below a thousandth of a win per season the estimate is the zero boundary,
    # not a small effect, and every BLUP is numerically 0.
    if blups and primary_payload["sigma_per_162"].get("manager", 0.0) > 1e-3:
        print("\n=== top / bottom shrunken (BLUP) manager estimates ===")
        for row in blups:
            print(
                f"  {row['blup_per_162']:+.3f} +/- {row['blup_sd']:.3f} wins/162  "
                f"{row['manager']:<22} {row['seasons']:>2} seasons {row['games']:>5} games"
            )
    else:
        print("\n(BLUPs are all zero: the REML estimate of sigma_m sits on the 0 boundary,")
        print(" so the model shrinks every manager fully to the league mean)")

    print("\n=== raw career pythagorean residual, best and worst (>= 1000 games) ===")
    for row in primary_payload.get("career_residual_extremes", []):
        print(
            f"  {row['wins_per_162']:+.3f} +/- {row['se_per_162']:.3f} wins/162 "
            f"(z={row['z']:+.2f})  {row['manager']:<22} "
            f"{row['seasons']:>2} seasons {row['games']:>5} games"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
