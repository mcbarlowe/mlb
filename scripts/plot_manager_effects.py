"""Graph one season's manager effects against the noise they have to beat.

Two estimands are drawn, because they answer different questions:

Panel A  Wins above *Pythagorean* expectation — how a manager's club converted
         its own runs scored and allowed into wins. This residual really is
         sampling noise that shrinks like ``1/G`` (verified empirically in
         ``analyze_manager_effects.py``), so each manager gets a genuine 95%
         interval from ``SE = sqrt(G * sigma^2_e)`` with ``sigma^2_e`` taken
         from the 1901-2025 fit rather than from the season being drawn.

Panel B  The same residuals as z-scores against the standard normal. If the
         season's managers differ only by chance, the histogram matches the
         curve and the observed SD of z is 1.

Panel C  Wins above *talent* expectation — the club's runs are no longer given,
         so this includes everything a manager might do to the run totals. Team
         talent is leave-one-season-out player quality (batting linear weights
         and pitcher FIP from other seasons) weighted by this season's playing
         time. Reference bands are the historical spread of the same quantity
         over 2009-2025 team-seasons; they are wide because unmeasured team
         quality lands here, which is exactly why this panel is not a manager
         ranking.

The figure is annotated with the historical bound on true manager variance
(<= 1.05 wins/162, the 1901-2025 differenced estimate) so a reader can see how
much of any single season's spread can possibly be real.

Usage::

    uv run python scripts/build_manager_game_log.py --statsapi-season 2026 \
        --output data/managers/manager_game_log_2026.parquet
    uv run python scripts/plot_manager_effects.py --season 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_manager_effects import (
    fit_pythagorean_exponent,
    load_stints,
    pythagorean,
)
from analyze_manager_win_effects import build_talent, read_postgres

from mlb.analysis.mixed import residualize

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"
OUTPUT_DIR = REPO_ROOT / "output" / "manager_effects"

# Two different bounds on true between-manager SD, in wins per 162. They are
# NOT interchangeable: panel A is the conversion estimand, panel C is total
# wins. Using the looser total-wins number on panel A overstates how much of
# panel A's spread could be real.
CONVERSION_SIGMA_BOUND = 0.56  # 95% profile likelihood, Pythagorean REML, 1901-2025
TOTAL_WINS_SIGMA_BOUND = 1.05  # 97.5% end of the differenced contrast on win pct

COLOR_POS = "#1b6ca8"
COLOR_NEG = "#c0392b"
COLOR_NULL = "#95a5a6"


def season_stints(path: Path, season: int) -> pl.DataFrame:
    log = pl.read_parquet(path).filter((pl.col("season") == season) & ~pl.col("tied"))
    if log.height == 0:
        raise SystemExit(f"no {season} rows in {path}")
    return (
        log.group_by("franchise_id", "manager_id", "manager_name")
        .agg(
            pl.len().alias("games"),
            pl.col("won").sum().alias("wins"),
            pl.col("runs_scored").sum().alias("runs_scored"),
            pl.col("runs_allowed").sum().alias("runs_allowed"),
            pl.col("game_date").min().alias("first_game"),
            pl.col("game_date").max().alias("last_game"),
        )
        .sort("franchise_id", "first_game")
    )


def historical_scale(path: Path) -> tuple[float, float]:
    """Pythagorean exponent and per-game residual variance from all history."""

    stints, _ = load_stints(path, 1901, 2100)
    k = fit_pythagorean_exponent(stints)
    full = stints.filter(pl.col("full_season"))
    games = full["games"].to_numpy().astype(float)
    rs = full["runs_scored"].to_numpy().astype(float)
    ra = full["runs_allowed"].to_numpy().astype(float)
    y = full["win_pct"].to_numpy() - pythagorean(rs, ra, k)
    X = np.ones((full.height, 1))
    resid = residualize(y, X, games)
    sigma2_e = float(np.sum(games * resid**2) / (full.height - 1))
    return k, sigma2_e


def talent_expectation(season: int) -> tuple[pl.DataFrame, float]:
    """Per-club expected win pct for ``season`` plus the historical residual SD.

    The talent -> win-percentage mapping is fitted on 2009-2025 team-seasons
    using talent standardized within season, so applying it to a season whose
    own coefficients are unknown needs no season effect.
    """

    frames = read_postgres()
    talent, _ = build_talent(frames)
    club = frames["club"].join(talent, on=["season", "team_abbrev"], how="inner")

    def standardize_within_season(frame: pl.DataFrame, column: str) -> pl.DataFrame:
        return frame.with_columns(
            (
                (pl.col(column) - pl.col(column).mean().over("season"))
                / pl.col(column).std().over("season")
            ).alias(f"z_{column}")
        )

    club = standardize_within_season(club, "bat_talent")
    club = standardize_within_season(club, "pit_talent")
    train = club.filter(pl.col("season") < season)
    games = train["games"].to_numpy().astype(float)
    y = train["wins"].to_numpy() / games
    X = np.column_stack(
        [
            np.ones(train.height),
            train["z_bat_talent"].to_numpy(),
            train["z_pit_talent"].to_numpy(),
        ]
    )
    sw = np.sqrt(games)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    train_resid_wins = (y - X @ coef) * games
    historical_sd = float(np.std(train_resid_wins / games * 162.0, ddof=1))

    current = club.filter(pl.col("season") == season)
    if current.height == 0:
        raise SystemExit(f"no {season} rows in the talent panel")
    Xc = np.column_stack(
        [
            np.ones(current.height),
            current["z_bat_talent"].to_numpy(),
            current["z_pit_talent"].to_numpy(),
        ]
    )
    return (
        current.select("team_abbrev", "games", "wins").with_columns(
            pl.Series("expected_win_pct", Xc @ coef)
        ),
        historical_sd,
    )


def forest(
    axis: Axes,
    labels: list[str],
    values: np.ndarray,
    errors: np.ndarray | None,
    title: str,
    xlabel: str,
    bands: tuple[float, float] | None = None,
) -> None:
    order = np.argsort(values)
    y = np.arange(len(values))
    values, labels = values[order], [labels[i] for i in order]
    if errors is not None:
        errors = errors[order]
        significant = np.abs(values) > errors
        colors = [
            (COLOR_POS if v > 0 else COLOR_NEG) if s else COLOR_NULL
            for v, s in zip(values, significant)
        ]
        axis.errorbar(
            values, y, xerr=errors, fmt="none", ecolor="#7f8c8d", elinewidth=1.1, capsize=2.5
        )
    else:
        colors = [COLOR_POS if v > 0 else COLOR_NEG for v in values]
    if bands is not None:
        one, two = bands
        axis.axvspan(-two, two, color="#bdc3c7", alpha=0.20, lw=0, zorder=0)
        axis.axvspan(-one, one, color="#95a5a6", alpha=0.25, lw=0, zorder=0)
    axis.scatter(values, y, s=34, c=colors, zorder=3, edgecolor="white", linewidth=0.6)
    axis.axvline(0.0, color="black", lw=1.0, zorder=2)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=7.4)
    axis.set_ylim(-0.8, len(values) - 0.2)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_title(title, fontsize=10.5, loc="left", pad=8)
    axis.grid(axis="x", color="#dfe4e6", lw=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--game-log",
        type=Path,
        default=REPO_ROOT / "data" / "managers" / "manager_game_log_2026.parquet",
    )
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--skip-talent", action="store_true")
    args = parser.parse_args()

    stints = season_stints(args.game_log, args.season)
    k, sigma2_e = historical_scale(args.history)
    print(f"historical scale: k={k:.4f}, sigma^2_e={sigma2_e:.6f} win-pct^2 per game")

    drawn = stints.filter(pl.col("games") >= args.min_games)
    dropped = stints.height - drawn.height
    games = drawn["games"].to_numpy().astype(float)
    rs = drawn["runs_scored"].to_numpy().astype(float)
    ra = drawn["runs_allowed"].to_numpy().astype(float)
    wins = drawn["wins"].to_numpy().astype(float)
    expected = games * pythagorean(rs, ra, k)
    residual_wins = wins - expected
    se_wins = np.sqrt(games * sigma2_e)
    z = residual_wins / se_wins
    labels = [
        f"{row['manager_name']}  ({row['franchise_id']}, {row['games']}g)"
        for row in drawn.iter_rows(named=True)
    ]

    # Two formal reads on "is this season's spread more than chance?".
    # Under the null every z is standard normal, so sum(z^2) ~ chi^2_n and the
    # count of intervals excluding zero is Binomial(n, 0.05).
    n_excluding_zero = int(np.sum(np.abs(z) > 1.96))
    n_managers = drawn.height
    chi2_stat = float(np.sum(z**2))
    chi2_p = float(stats.chi2.sf(chi2_stat, n_managers))
    count_p = float(stats.binom.sf(n_excluding_zero - 1, n_managers, 0.05))
    sd_z = float(z.std(ddof=1))
    sd_z_se = 1.0 / np.sqrt(2.0 * (n_managers - 1))
    print(
        f"{n_managers} managers drawn ({dropped} with <{args.min_games} games omitted)\n"
        f"  sd(z) = {sd_z:.3f} vs 1.000 under pure noise (SE {sd_z_se:.3f})\n"
        f"  sum(z^2) = {chi2_stat:.1f} on {n_managers} df -> p = {chi2_p:.3f}\n"
        f"  {n_excluding_zero} of {n_managers} 95% intervals exclude zero "
        f"({0.05 * n_managers:.1f} expected) -> p = {count_p:.3f}"
    )

    talent_residual = np.zeros(0)
    historical_sd = 0.0
    have_talent = not args.skip_talent
    if have_talent:
        talent, historical_sd = talent_expectation(args.season)
        club_expected = dict(
            zip(talent["team_abbrev"].to_list(), talent["expected_win_pct"].to_list())
        )
        missing = [c for c in drawn["franchise_id"].to_list() if c not in club_expected]
        if missing:
            print(f"no talent expectation for {sorted(set(missing))}; skipping panel C")
            have_talent = False
    if have_talent:
        talent_expected = np.array(
            [club_expected[c] for c in drawn["franchise_id"].to_list()]
        ) * games
        talent_residual = wins - talent_expected
        print(
            f"talent model: historical per-team residual SD "
            f"{historical_sd:.2f} wins/162; {args.season} spread "
            f"{np.std(talent_residual / games * 162.0, ddof=1):.2f} wins/162"
        )

    n_panels = 3 if have_talent else 2
    fig = plt.figure(figsize=(15.5 if have_talent else 11.5, 9.4))
    grid = fig.add_gridspec(
        1, n_panels, width_ratios=[1.25, 0.85, 1.25][:n_panels], wspace=0.42
    )
    ax_a = fig.add_subplot(grid[0, 0])
    forest(
        ax_a,
        labels,
        residual_wins,
        se_wins * 1.96,
        "A. Wins above Pythagorean expectation\n(run-to-win conversion, 95% intervals)",
        "wins above expected",
    )
    handles = [
        Patch(color=COLOR_POS, label="interval excludes 0 (above)"),
        Patch(color=COLOR_NEG, label="interval excludes 0 (below)"),
        Patch(color=COLOR_NULL, label="interval includes 0"),
    ]
    ax_a.legend(
        handles=handles,
        fontsize=7.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.1,
    )

    ax_b = fig.add_subplot(grid[0, 1])
    ax_b.hist(
        z, bins=list(np.arange(-3.25, 3.5, 0.5)), density=True, color="#7fb3d5",
        edgecolor="white", linewidth=0.8,
    )
    xs = np.linspace(-3.4, 3.4, 300)
    ax_b.plot(
        xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), color="#c0392b", lw=1.8,
        label="standard normal\n(pure chance)",
    )
    ax_b.axvline(0, color="black", lw=1.0)
    ax_b.set_title(
        f"B. Same residuals as z-scores\n"
        f"observed SD {sd_z:.2f} vs 1.00 +/- {sd_z_se:.2f} if chance",
        fontsize=10.5, loc="left", pad=8,
    )
    ax_b.set_xlabel("(wins above expected) / standard error", fontsize=9)
    ax_b.set_ylabel("density", fontsize=9)
    ax_b.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax_b.text(
        0.5, -0.075,
        f"sum(z$^2$) = {chi2_stat:.0f} on {n_managers} df, p = {chi2_p:.2f}\n"
        f"{n_excluding_zero} of {n_managers} intervals exclude 0 "
        f"({0.05 * n_managers:.1f} expected), p = {count_p:.2f}",
        transform=ax_b.transAxes, ha="center", va="top", fontsize=7.8, color="#4a4a4a",
    )
    ax_b.grid(axis="y", color="#dfe4e6", lw=0.7)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)

    if have_talent:
        ax_c = fig.add_subplot(grid[0, 2])
        forest(
            ax_c,
            labels,
            talent_residual,
            None,
            "C. Wins above talent expectation\n(roster quality from other seasons)",
            "wins above expected",
            bands=(historical_sd * games.mean() / 162.0, 2 * historical_sd * games.mean() / 162.0),
        )
        ax_c.legend(
            handles=[
                Patch(color="#95a5a6", alpha=0.25, label="±1 SD of 2009-2025 team-seasons"),
                Patch(color="#bdc3c7", alpha=0.20, label="±2 SD of 2009-2025 team-seasons"),
            ],
            fontsize=7.4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.075),
            ncol=2,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.1,
        )

    last = stints["last_game"].max()
    fig.suptitle(
        f"MLB {args.season} manager effects through {last}  —  "
        f"{drawn.height} managers, {int(games.sum())} club-games",
        fontsize=14, x=0.006, ha="left", y=0.982, weight="bold",
    )
    skill_var = float(np.mean((games * CONVERSION_SIGMA_BOUND / 162.0) ** 2))
    noise_var = float(np.mean(se_wins**2))
    skill_share = 100.0 * skill_var / (noise_var + skill_var)
    fig.text(
        0.006, 0.958,
        "Panel A carries the only honest per-manager error bars: its residual is pure sampling "
        "noise, which shrinks like 1/games.\n"
        "Over 1901-2025 the between-manager SD of conversion is bounded at "
        f"{CONVERSION_SIGMA_BOUND:.2f} wins/162 (REML point estimate 0) — even at that ceiling "
        f"only {skill_share:.0f}% of the spread below is skill.",
        fontsize=8.8, ha="left", va="top", color="#4a4a4a", linespacing=1.5,
    )
    fig.text(
        0.006, 0.918,
        "Panel C is NOT a manager ranking and gets no error bars: unmeasured team quality "
        "(defense, health, roster moves) lands in it and does not shrink with games.\n"
        "Re-cutting each club's seasons into arbitrary contiguous blocks explains the same "
        f"variance as real tenures (p = 0.37). Total-wins SD is bounded at "
        f"{TOTAL_WINS_SIGMA_BOUND:.2f} wins/162.",
        fontsize=8.8, ha="left", va="top", color="#4a4a4a", linespacing=1.5,
    )
    fig.subplots_adjust(top=0.845, bottom=0.115, left=0.148, right=0.988)

    output = args.output or OUTPUT_DIR / f"manager_effects_{args.season}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
