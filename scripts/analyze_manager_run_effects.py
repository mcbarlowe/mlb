"""Manager effects on RUN TOTALS — scoring and preventing runs, not converting.

Why this exists: a Pythagorean regression conditions on the club's own runs
scored and allowed, which makes every managerial decision that shows up in run
totals invisible by construction. Lineup construction, pinch hitting, base
stealing, who bats where, which arm faces which hitter, when the bullpen comes
in — all of that moves RS and RA, so it is *inside* the Pythagorean offset and
cannot appear in the residual. The Pythagorean model can only ever see
sequencing and leverage.

This script targets the run totals directly.

Part 1 — assumption-light bound, 1901-2025 (game log only, no talent model)
    For consecutive seasons of the same franchise with exactly one manager in
    each::

        Var(delta | manager changed) - Var(delta | manager stayed) = 2 sigma^2_m

    applied to runs scored per game, runs allowed per game, and run
    differential. The manager term cancels in the "stayed" difference and
    appears twice in the "changed" difference. Roster churn and post-firing
    regression both inflate the changed set, so this reads as an upper bound.

Part 2 — talent-controlled models, 2009-2025
    Response is runs scored per game, runs allowed per game, or run
    differential. Team talent is leave-one-season-out player quality, so the
    control is never built from the season being explained. Manager and
    franchise enter as crossed random effects with season fixed effects, so a
    manager is measured against the other managers of his own club.

    The talent control is built two ways, and this is the crux of the
    estimand:

    ``playing_time``       weights players by the PA/IP they actually got, so
                           the manager's own allocation counts as talent and is
                           subtracted OUT of the manager term. Conservative.
    ``allocation_neutral`` weights players by their typical usage in other
                           seasons times games appeared, so roster composition
                           and availability are controlled but the per-game
                           volume decision stays IN the manager term.

    The gap between the two estimates playing-time allocation effects.

Every fit is checked with the contiguity placebo: manager tenures are
contiguous runs of seasons, so any franchise-era effect the talent proxy misses
will masquerade as a manager effect. Re-cutting each club's seasons into random
contiguous blocks with the same tenure-length multiset is the test that
separates a manager from an arbitrary era.

Usage::

    uv run python scripts/analyze_manager_run_effects.py
    uv run python scripts/analyze_manager_run_effects.py --bootstrap 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_manager_effects import (
    GAMES_SCALE,
    build_codes,
    dummy_design,
    fit_pythagorean_exponent,
    load_stints,
)
from analyze_manager_win_effects import (
    MIN_FULL_SEASON_GAMES,
    build_transitions,
    contiguity_placebo,
    contrast_bound,
    crosswalk_franchises,
    out_of_sample_slope,
)

from mlb.analysis.mixed import (
    REMLProblem,
    fit_without_block,
    moment_variance_from_permutation,
    permutation_test,
    residualize,
)
from mlb.analysis.talent import read_player_seasons, team_talent

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_LOG_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"
REPORT_PATH = REPO_ROOT / "output" / "manager_effects" / "manager_run_effects_report.json"

RESPONSES = {
    "runs_scored": "runs scored per game",
    "runs_allowed": "runs allowed per game",
    "run_diff": "run differential per game",
}


def fit_run_model(
    panel: pl.DataFrame,
    response: str,
    controls: list[str],
    label: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    games = panel["games"].to_numpy().astype(float)
    rs = panel["runs_scored"].to_numpy().astype(float)
    ra = panel["runs_allowed"].to_numpy().astype(float)
    if response == "runs_scored":
        y = rs / games
    elif response == "runs_allowed":
        y = ra / games
    elif response == "run_diff":
        y = (rs - ra) / games
    else:
        raise ValueError(f"unknown response {response}")

    codes = build_codes(panel)
    columns: list[np.ndarray] = [np.ones((panel.height, 1))]
    for name in controls:
        column = panel[name].to_numpy().astype(float)
        columns.append(((column - column.mean()) / column.std())[:, None])
    columns.append(dummy_design(codes["season"]))
    X = np.column_stack(columns)

    blocks = {"manager": codes["manager"], "franchise": codes["franchise"]}
    fit = REMLProblem(y, X, games, blocks).fit()
    lrt = max(fit_without_block(y, X, games, blocks, "manager").neg2_reml - fit.neg2_reml, 0.0)

    resid = residualize(y, X, games)
    baseline = residualize(y, np.column_stack([np.ones((panel.height, 1)),
                                               dummy_design(codes["season"])]), games)
    control_r2 = 1.0 - float(np.var(resid) / np.var(baseline)) if controls else 0.0

    resid_franchise = residualize(
        y, np.column_stack([X, dummy_design(codes["franchise"])]), games
    )
    perm_block = permutation_test(
        resid_franchise, games, codes["manager"], codes["franchise"], permutations, rng
    )
    moment = moment_variance_from_permutation(perm_block, games, codes["manager"])
    placebo = contiguity_placebo(panel, resid_franchise, permutations, rng)
    # Absorbing franchise means forces a manager's own seasons to sum to ~0
    # within his club, which mechanically drags the odd/even slope negative
    # whenever he is most of his club's sample. Report both: franchise-absorbed
    # (biased low, no franchise confound) and season-only (unbiased slope, but
    # franchise quality still in it). The truth is bracketed by the pair.
    oos = out_of_sample_slope(panel, resid_franchise, 4, f"{label} / {response}")
    oos_no_franchise = out_of_sample_slope(panel, resid, 4, f"{label} / {response} (season-only)")

    return {
        "label": label,
        "response": response,
        "units": "runs per 162",
        "controls": controls,
        "n_team_seasons": panel.height,
        "n_managers": int(codes["manager"].max() + 1),
        "sigma_manager_per_162": float(np.sqrt(fit.variances["manager"]) * GAMES_SCALE),
        "sigma_franchise_per_162": float(np.sqrt(fit.variances["franchise"]) * GAMES_SCALE),
        "residual_sd_per_162": float(np.sqrt(fit.resid_variance / GAMES_SCALE) * GAMES_SCALE),
        "variance_explained_by_controls": control_r2,
        "lrt_statistic": lrt,
        "lrt_p_value": 0.5 * float(stats.chi2.sf(lrt, 1)) if lrt > 0 else 0.5,
        "moment_sigma_manager_signed_per_162": float(
            np.sign(moment) * np.sqrt(abs(moment)) * GAMES_SCALE
        ),
        "permutation_within_franchise_p": perm_block["p_value"],
        "contiguity_placebo_p": placebo["p_value"],
        "contiguity_observed": placebo["observed"],
        "contiguity_placebo_mean": placebo["placebo_mean"],
        "out_of_sample": oos,
        "out_of_sample_season_only": oos_no_franchise,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-log", type=Path, default=GAME_LOG_PATH)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--skip-postgres", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    stints, meta = load_stints(args.game_log, 1901, 2100)
    k = fit_pythagorean_exponent(stints)
    report: dict = {"meta": meta}

    print("=== Part 1: differenced bound on manager effects on run totals, 1901-2025 ===")
    transitions = build_transitions(stints, k)
    changed = int(transitions["changed"].sum())
    print(
        f"  {transitions.height} consecutive franchise-season pairs "
        f"({changed} new manager, {transitions.height - changed} same manager)"
    )
    report["part1"] = []
    for column, name in (
        ("d_runs_scored", "runs scored per game"),
        ("d_runs_allowed", "runs allowed per game"),
        ("d_run_diff", "run differential per game"),
    ):
        row = contrast_bound(transitions, column, GAMES_SCALE, args.bootstrap, rng)
        row["name"] = name
        report["part1"].append(row)
        if "signed_sigma_manager" in row:
            lo, hi = row["ci95"]  # type: ignore[misc]
            print(
                f"  {name:<28} sd(changed)={row['sd_changed']:7.2f} "
                f"sd(stayed)={row['sd_stayed']:7.2f} -> sigma_m={row['signed_sigma_manager']:+7.2f} "
                f"[{lo:+.2f}, {hi:+.2f}] runs/162"
            )

    if args.skip_postgres:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.report}")
        return

    print("\n=== Part 2: talent-controlled run models, 2009-2025 ===")
    frames = read_player_seasons()
    crosswalk = crosswalk_franchises(frames["club"], stints)
    full = stints.filter(pl.col("full_season"))
    report["part2"] = []

    for weighting in ("playing_time", "allocation_neutral"):
        talent, talent_meta = team_talent(frames, weighting)
        panel = (
            full.join(crosswalk, on=["season", "franchise_id"], how="inner")
            .join(talent, on=["season", "team_abbrev"], how="inner")
            .filter(pl.col("games") >= MIN_FULL_SEASON_GAMES)
        )
        print(
            f"\n  talent weighting = {weighting}  "
            f"(PA coverage {talent_meta['batter_pa_coverage']:.3f}, "
            f"IP coverage {talent_meta['pitcher_ip_coverage']:.3f}, "
            f"panel {panel.height} stints / {panel['manager_id'].n_unique()} managers)"
        )
        for response, response_label in RESPONSES.items():
            controls = ["bat_talent", "pit_talent"]
            row = fit_run_model(
                panel, response, controls, weighting, args.permutations, rng
            )
            report["part2"].append(row)

            def fmt(block: dict) -> str:
                if "slope" not in block:
                    return "n/a"
                return f"{block['slope']:+.2f} [{block['ci95'][0]:+.2f}, {block['ci95'][1]:+.2f}]"

            print(
                f"    {response_label:<26} sigma_mgr={row['sigma_manager_per_162']:6.2f} "
                f"sigma_fr={row['sigma_franchise_per_162']:6.2f} "
                f"resid_sd={row['residual_sd_per_162']:6.2f} "
                f"R2ctl={row['variance_explained_by_controls']:+.3f}"
            )
            print(
                f"      LRT p={row['lrt_p_value']:.4f}  "
                f"perm p={row['permutation_within_franchise_p']:.4f}  "
                f"CONTIGUITY PLACEBO p={row['contiguity_placebo_p']:.4f}  "
                f"moment={row['moment_sigma_manager_signed_per_162']:+6.2f}"
            )
            print(
                f"      out-of-sample slope: franchise-absorbed "
                f"{fmt(row['out_of_sample'])}  season-only "
                f"{fmt(row['out_of_sample_season_only'])}"
            )

    print("\n=== Part 3: what allocation weighting is worth ===")
    report["part3_allocation_gap"] = []
    for response, response_label in RESPONSES.items():
        pt = next(
            r for r in report["part2"]
            if r["response"] == response and r["label"] == "playing_time"
        )
        an = next(
            r for r in report["part2"]
            if r["response"] == response and r["label"] == "allocation_neutral"
        )
        gap = an["sigma_manager_per_162"] - pt["sigma_manager_per_162"]
        report["part3_allocation_gap"].append(
            {
                "response": response,
                "sigma_playing_time": pt["sigma_manager_per_162"],
                "sigma_allocation_neutral": an["sigma_manager_per_162"],
                "gap_per_162": gap,
            }
        )
        print(
            f"  {response_label:<26} allocation-in={an['sigma_manager_per_162']:6.2f} "
            f"allocation-out={pt['sigma_manager_per_162']:6.2f} "
            f"-> allocation worth {gap:+.2f} runs/162"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
