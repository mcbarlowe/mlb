"""Can individual manager effects on *wins* be recovered at all?

`analyze_manager_effects.py` answers a narrower question: managers have no
persistent run-to-win conversion skill, so the Pythagorean-residual model
shrinks every manager fully to the league mean and its "individual effects"
are identically zero. That is a property of the data, not of the estimator.

This script attacks the wider question in three parts.

Part 1 — why no individual conversion effect is extractable
    The per-manager career residual with its standard error, the reliability
    ``lambda = sigma^2_m / (sigma^2_m + sigma^2_e / G)`` that a shrunken
    estimate would carry, and the decisive out-of-sample check: fit each
    manager's effect on his odd seasons and see whether it predicts his even
    seasons. A slope of 0 means the per-manager numbers are noise, whatever
    their spread looks like.

Part 2 — an upper bound on manager effects on total wins (1901-2025)
    Uses only the game log, so it spans every season. For consecutive seasons
    of the same franchise, with exactly one manager in each::

        Var(delta | manager changed) - Var(delta | manager stayed) = 2 sigma^2_m

    because the manager term cancels in the "stayed" difference and appears
    twice in the "changed" difference. Both biases in play — clubs that change
    managers churn rosters harder, and a fired manager's club is regressing up
    from a bad year — inflate the changed set, so this is an upper bound.
    Running the same contrast on the Pythagorean residual reproduces Part 1 and
    validates the estimator.

Part 3 — talent-controlled individual effects on wins (2009-2025)
    The only design here that yields per-manager numbers on wins. Team talent
    is measured from player performance in *other* seasons (leave-one-season-out
    batting linear weights and pitcher FIP from ``mlb.batting`` / ``mlb.pitching``),
    weighted by playing time, so the control is not itself this season's
    results. Manager effects are then estimated relative to other managers of
    the same franchise, and validated out of sample the same way as Part 1.

    Caveat that cannot be designed away: any team quality the proxy misses
    (defense, health, coaching staff) lands in the manager term, and playing
    time is itself a managerial decision. Part 3 is therefore also an upper
    bound, not a causal effect.

Usage::

    uv run python scripts/analyze_manager_win_effects.py
    uv run python scripts/analyze_manager_win_effects.py --skip-postgres
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

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
    pythagorean,
)

from mlb.analysis.mixed import (
    REMLProblem,
    between_group_statistic,
    fit_without_block,
    moment_variance_from_permutation,
    permutation_test,
    residualize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_LOG_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"
REPORT_PATH = REPO_ROOT / "output" / "manager_effects" / "manager_win_effects_report.json"

# Seasons shorter than this are excluded from the season-to-season contrast so
# the sampling-noise term is comparable across pairs (drops 1918-19, 1981,
# 1994-95, 2020).
MIN_FULL_SEASON_GAMES = 140

# wOBA-style linear weights, per plate appearance.
WEIGHT_BB, WEIGHT_HBP, WEIGHT_1B, WEIGHT_2B, WEIGHT_3B, WEIGHT_HR = (
    0.69, 0.72, 0.89, 1.27, 1.62, 2.10,
)
# Empirical-Bayes shrinkage of a player's other-season quality toward the
# league mean, in plate appearances / batters faced of prior weight.
BAT_SHRINK_PA = 400.0
PIT_SHRINK_BF = 500.0


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def out_of_sample_slope(
    frame: pl.DataFrame,
    resid: np.ndarray,
    min_seasons: int,
    label: str,
) -> dict[str, object]:
    """Do odd-season manager estimates predict even-season performance?

    Regresses each manager's even-season weighted mean residual on his
    odd-season one. Slope 1 = the estimate is fully real and unattenuated;
    slope 0 = the per-manager numbers carry no out-of-sample signal.
    """

    work = frame.with_columns(pl.Series("resid", resid))
    eligible = (
        work.group_by("manager_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_seasons)
        .select("manager_id")
    )
    work = work.join(eligible, on="manager_id", how="semi").with_columns(
        (pl.col("season") % 2).alias("half")
    )
    halves = work.group_by("manager_id", "half").agg(
        ((pl.col("resid") * pl.col("games")).sum() / pl.col("games").sum()).alias("mean_resid"),
        pl.col("games").sum().alias("games"),
    )
    even = halves.filter(pl.col("half") == 0).select(
        "manager_id", pl.col("mean_resid").alias("x"), pl.col("games").alias("gx")
    )
    odd = halves.filter(pl.col("half") == 1).select(
        "manager_id", pl.col("mean_resid").alias("y"), pl.col("games").alias("gy")
    )
    paired = even.join(odd, on="manager_id", how="inner")
    if paired.height < 15:
        return {"label": label, "n_managers": paired.height}
    x = paired["x"].to_numpy()
    y = paired["y"].to_numpy()
    # Harmonic mean of the two halves' games: precision of the pair.
    w = 1.0 / (1.0 / paired["gx"].to_numpy() + 1.0 / paired["gy"].to_numpy())
    xc = x - np.average(x, weights=w)
    yc = y - np.average(y, weights=w)
    slope = float(np.sum(w * xc * yc) / np.sum(w * xc**2))
    resid_var = float(np.sum(w * (yc - slope * xc) ** 2) / (x.size - 2))
    se = float(np.sqrt(resid_var / np.sum(w * xc**2)))
    return {
        "label": label,
        "n_managers": int(x.size),
        "min_seasons": min_seasons,
        "slope": slope,
        "se": se,
        "ci95": [slope - 1.96 * se, slope + 1.96 * se],
    }


def reliability(sigma2_m: float, resid_variance: float, games: np.ndarray) -> dict[str, float]:
    """Share of a manager's observed mean that is signal, at various careers."""

    out = {}
    for career in (500, 1000, 2000, 4000):
        noise = resid_variance / career
        out[f"games_{career}"] = float(sigma2_m / (sigma2_m + noise)) if sigma2_m > 0 else 0.0
    out["median_career_games"] = float(np.median(games))
    return out


def contiguity_placebo(
    frame: pl.DataFrame,
    resid: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Is the manager grouping better than an arbitrary contiguous era?

    A within-franchise permutation of manager labels rejects for *any* effect
    shared by consecutive seasons of one club — unmeasured defense, a front
    office's acquisition run, a pitching coach — because manager tenures are
    contiguous runs of seasons. This placebo keeps the franchise, the number of
    tenures, the multiset of tenure lengths, and contiguity, and only moves
    where the cut points fall. If the real manager grouping explains no more
    between-group variance than randomly re-cut contiguous blocks of the same
    lengths, the signal is a franchise-era effect and cannot be attributed to
    the individual manager.
    """

    work = (
        frame.with_columns(pl.Series("resid", resid))
        .select("franchise_id", "season", "manager_id", "games", "resid")
        .sort("franchise_id", "season")
    )
    franchises = work["franchise_id"].to_numpy()
    managers = work["manager_id"].to_numpy()
    weights = work["games"].to_numpy().astype(float)
    values = work["resid"].to_numpy()

    groups = np.empty(work.height, dtype=np.int64)
    run_lengths: dict[object, list[int]] = {}
    index: dict[object, np.ndarray] = {}
    next_code = 0
    for franchise in np.unique(franchises):
        idx = np.flatnonzero(franchises == franchise)
        index[franchise] = idx
        lengths: list[int] = []
        start = 0
        for position in range(1, idx.size + 1):
            if position == idx.size or managers[idx[position]] != managers[idx[start]]:
                lengths.append(position - start)
                groups[idx[start:position]] = next_code
                next_code += 1
                start = position
        run_lengths[franchise] = lengths

    n_levels = next_code
    observed = between_group_statistic(values, weights, groups, n_levels)
    null = np.empty(n_draws)
    synthetic = np.empty_like(groups)
    for draw in range(n_draws):
        code = 0
        for franchise, idx in index.items():
            offset = 0
            for length in rng.permutation(np.asarray(run_lengths[franchise])):
                synthetic[idx[offset : offset + int(length)]] = code
                offset += int(length)
                code += 1
        null[draw] = between_group_statistic(values, weights, synthetic, code)
    return {
        "n_tenures": int(n_levels),
        "observed": observed,
        "placebo_mean": float(null.mean()),
        "placebo_p95": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_draws + 1)),
    }


# ---------------------------------------------------------------------------
# Part 2: season-to-season contrast around manager changes
# ---------------------------------------------------------------------------


def build_transitions(stints: pl.DataFrame, k: float) -> pl.DataFrame:
    """Consecutive franchise seasons with exactly one manager in each."""

    sole = (
        stints.group_by("season", "franchise_id")
        .agg(
            pl.len().alias("n_managers"),
            pl.col("games").sum().alias("club_games"),
            pl.col("wins").sum().alias("club_wins"),
            pl.col("runs_scored").sum().alias("club_rs"),
            pl.col("runs_allowed").sum().alias("club_ra"),
            pl.col("manager_id").first().alias("manager_id"),
            pl.col("manager_name").first().alias("manager_name"),
        )
        .filter((pl.col("n_managers") == 1) & (pl.col("club_games") >= MIN_FULL_SEASON_GAMES))
    )
    rs = sole["club_rs"].to_numpy().astype(float)
    ra = sole["club_ra"].to_numpy().astype(float)
    games = sole["club_games"].to_numpy().astype(float)
    win_pct = sole["club_wins"].to_numpy() / games
    sole = sole.with_columns(
        pl.Series("win_pct", win_pct),
        pl.Series("pythag_resid", win_pct - pythagorean(rs, ra, k)),
        pl.Series("run_diff", (rs - ra) / games),
        pl.Series("runs_scored_pg", rs / games),
        pl.Series("runs_allowed_pg", ra / games),
    )
    prior = sole.select(
        pl.col("franchise_id"),
        (pl.col("season") + 1).alias("season"),
        pl.col("manager_id").alias("prior_manager_id"),
        pl.col("win_pct").alias("prior_win_pct"),
        pl.col("pythag_resid").alias("prior_pythag_resid"),
        pl.col("run_diff").alias("prior_run_diff"),
        pl.col("runs_scored_pg").alias("prior_runs_scored_pg"),
        pl.col("runs_allowed_pg").alias("prior_runs_allowed_pg"),
        pl.col("club_games").alias("prior_games"),
    )
    return (
        sole.join(prior, on=["franchise_id", "season"], how="inner")
        .with_columns(
            (pl.col("manager_id") != pl.col("prior_manager_id")).alias("changed"),
            (pl.col("win_pct") - pl.col("prior_win_pct")).alias("d_win_pct"),
            (pl.col("pythag_resid") - pl.col("prior_pythag_resid")).alias("d_pythag_resid"),
            (pl.col("run_diff") - pl.col("prior_run_diff")).alias("d_run_diff"),
            (pl.col("runs_scored_pg") - pl.col("prior_runs_scored_pg")).alias("d_runs_scored"),
            (pl.col("runs_allowed_pg") - pl.col("prior_runs_allowed_pg")).alias("d_runs_allowed"),
        )
        .sort("season", "franchise_id")
    )


def contrast_bound(
    transitions: pl.DataFrame,
    column: str,
    scale: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """2 sigma^2_m = Var(delta | changed) - Var(delta | stayed), bootstrapped.

    Bootstrap resamples franchises, not rows, so the CI respects the fact that
    a franchise contributes many correlated transitions.
    """

    changed = transitions.filter(pl.col("changed"))[column].to_numpy()
    stayed = transitions.filter(~pl.col("changed"))[column].to_numpy()
    if changed.size < 30 or stayed.size < 30:
        return {"column": column, "n_changed": int(changed.size), "n_stayed": int(stayed.size)}

    def sigma_from(c: np.ndarray, s: np.ndarray) -> float:
        diff = np.var(c, ddof=1) - np.var(s, ddof=1)
        return float(np.sign(diff) * np.sqrt(abs(diff) / 2.0) * scale)

    point = sigma_from(changed, stayed)
    franchises = transitions["franchise_id"].to_numpy()
    unique = np.unique(franchises)
    flags = transitions["changed"].to_numpy()
    values = transitions[column].to_numpy()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique, unique.size, replace=True)
        idx = np.concatenate([np.flatnonzero(franchises == f) for f in picked])
        c, s = values[idx][flags[idx]], values[idx][~flags[idx]]
        boot[i] = sigma_from(c, s) if c.size > 5 and s.size > 5 else np.nan
    boot = boot[~np.isnan(boot)]
    return {
        "column": column,
        "n_changed": int(changed.size),
        "n_stayed": int(stayed.size),
        "sd_changed": float(np.std(changed, ddof=1) * scale),
        "sd_stayed": float(np.std(stayed, ddof=1) * scale),
        "signed_sigma_manager": point,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


# ---------------------------------------------------------------------------
# Part 3: talent-controlled manager effects, 2009-2025
# ---------------------------------------------------------------------------


BATTING_SQL = """
    select g.season,
           t.abbreviation as team_abbrev,
           b.player_id,
           sum(b.plateappearances) as pa,
           sum(b.hits) as h,
           sum(b.doubles) as d2,
           sum(b.triples) as d3,
           sum(b.homeruns) as hr,
           sum(b.baseonballs) as bb,
           sum(b.hitbypitch) as hbp
    from mlb.batting b
    join mlb.games g on g.game_pk = b.game_pk
    join lateral (
        values (g.home_team_id, 'home'), (g.away_team_id, 'away')
    ) as s(team_id, side) on s.side = b.team_type
    join mlb.teams t on t.team_id = s.team_id
    where g.game_type = 'R' and g.abstract_game_state = 'Final'
    group by 1, 2, 3
    having sum(b.plateappearances) > 0
"""

PITCHING_SQL = """
    select g.season,
           t.abbreviation as team_abbrev,
           p.player_id,
           sum(p.battersfaced) as bf,
           sum(p.outs) as outs,
           sum(p.homeruns) as hr,
           sum(p.baseonballs) as bb,
           sum(p.hitbatsmen) as hbp,
           sum(p.strikeouts) as k
    from mlb.pitching p
    join mlb.games g on g.game_pk = p.game_pk
    join lateral (
        values (g.home_team_id, 'home'), (g.away_team_id, 'away')
    ) as s(team_id, side) on s.side = p.team_type
    join mlb.teams t on t.team_id = s.team_id
    where g.game_type = 'R' and g.abstract_game_state = 'Final'
    group by 1, 2, 3
    having sum(p.outs) > 0
"""

CLUB_SEASON_SQL = """
    with team_game as (
        select g.season,
               g.game_pk,
               t.abbreviation as team_abbrev,
               sum(case when l.team_type = s.side then l.runs else 0 end) as rs,
               sum(case when l.team_type <> s.side then l.runs else 0 end) as ra
        from mlb.games g
        join lateral (
            values (g.home_team_id, 'home'), (g.away_team_id, 'away')
        ) as s(team_id, side) on true
        join mlb.teams t on t.team_id = s.team_id
        join mlb.linescore l on l.game_pk = g.game_pk
        where g.game_type = 'R' and g.abstract_game_state = 'Final'
        group by 1, 2, 3
    )
    select season,
           team_abbrev,
           count(*) as games,
           sum(case when rs > ra then 1 else 0 end) as wins,
           sum(rs) as runs_scored,
           sum(ra) as runs_allowed
    from team_game
    where rs <> ra
    group by 1, 2
"""


def read_postgres() -> dict[str, pl.DataFrame]:
    import psycopg
    from psycopg.abc import Query

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    frames = {}
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    ) as connection:
        for name, query in (
            ("batting", BATTING_SQL),
            ("pitching", PITCHING_SQL),
            ("club", CLUB_SEASON_SQL),
        ):
            with connection.cursor() as cursor:
                cursor.execute(cast(Query, query))
                columns = [c.name for c in cursor.description or ()]
                rows = cursor.fetchall()
            # Postgres sum() over integers returns numeric, which arrives as
            # Decimal and will not join against the game log's Int64 keys.
            frames[name] = pl.DataFrame(
                {col: [r[i] for r in rows] for i, col in enumerate(columns)}
            ).with_columns((pl.selectors.integer() | pl.selectors.decimal()).cast(pl.Int64))
    return frames


def leave_one_season_out(
    frame: pl.DataFrame,
    value_numerator: pl.Expr,
    weight: str,
    shrink: float,
) -> pl.DataFrame:
    """Player quality for each season, computed from that player's OTHER seasons.

    The numerator and weight are summed over a player's career, the focal
    season is subtracted, and the remainder is shrunk toward the league mean by
    ``shrink`` units of prior weight. Players with no other season fall back
    entirely to the league mean, which is the correct no-information default.
    """

    per = frame.group_by("season", "player_id").agg(
        value_numerator.sum().alias("num"), pl.col(weight).sum().alias("den")
    )
    league = float(per["num"].sum() / per["den"].sum())
    career = per.group_by("player_id").agg(
        pl.col("num").sum().alias("career_num"), pl.col("den").sum().alias("career_den")
    )
    return (
        per.join(career, on="player_id", how="left")
        .with_columns(
            (pl.col("career_num") - pl.col("num")).alias("other_num"),
            (pl.col("career_den") - pl.col("den")).alias("other_den"),
        )
        .with_columns(
            (
                (pl.col("other_num") + shrink * league) / (pl.col("other_den") + shrink)
            ).alias("quality"),
            (pl.col("other_den") > 0).alias("has_other_season"),
        )
        .select("season", "player_id", "quality", "has_other_season")
    )


def build_talent(frames: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict]:
    batting = frames["batting"].with_columns(
        (pl.col("h") - pl.col("d2") - pl.col("d3") - pl.col("hr")).alias("d1")
    )
    bat_numerator = (
        WEIGHT_BB * pl.col("bb")
        + WEIGHT_HBP * pl.col("hbp")
        + WEIGHT_1B * pl.col("d1")
        + WEIGHT_2B * pl.col("d2")
        + WEIGHT_3B * pl.col("d3")
        + WEIGHT_HR * pl.col("hr")
    )
    bat_quality = leave_one_season_out(batting, bat_numerator, "pa", BAT_SHRINK_PA)

    pitching = frames["pitching"].with_columns((pl.col("outs") / 3.0).alias("ip"))
    # FIP numerator; denominator is innings, so it is scaled to per-9 later.
    pit_numerator = 13 * pl.col("hr") + 3 * (pl.col("bb") + pl.col("hbp")) - 2 * pl.col("k")
    pit_quality = leave_one_season_out(pitching, pit_numerator, "ip", PIT_SHRINK_BF / 4.3)

    bat_team = (
        batting.join(bat_quality, on=["season", "player_id"], how="left")
        .group_by("season", "team_abbrev")
        .agg(
            ((pl.col("quality") * pl.col("pa")).sum() / pl.col("pa").sum()).alias("bat_talent"),
            (
                (pl.col("pa") * pl.col("has_other_season")).sum() / pl.col("pa").sum()
            ).alias("bat_coverage"),
        )
    )
    pit_team = (
        pitching.join(pit_quality, on=["season", "player_id"], how="left")
        .group_by("season", "team_abbrev")
        .agg(
            ((pl.col("quality") * pl.col("ip")).sum() / pl.col("ip").sum()).alias("pit_talent"),
            (
                (pl.col("ip") * pl.col("has_other_season")).sum() / pl.col("ip").sum()
            ).alias("pit_coverage"),
        )
    )
    talent = bat_team.join(pit_team, on=["season", "team_abbrev"], how="inner")
    meta = {
        "team_seasons": talent.height,
        "batter_pa_coverage": float(talent["bat_coverage"].mean()),
        "pitcher_ip_coverage": float(talent["pit_coverage"].mean()),
    }
    return talent, meta


def crosswalk_franchises(club: pl.DataFrame, stints: pl.DataFrame) -> pl.DataFrame:
    """Map StatsAPI abbreviations to Retrosheet franchise ids.

    Keyed on (season, games, wins, runs scored, runs allowed), which
    ``build_manager_game_log.py --verify-postgres`` already proved identical
    between the two sources for every overlapping season. Any key that is not
    unique on both sides is dropped rather than guessed.
    """

    gl = (
        stints.group_by("season", "franchise_id")
        .agg(
            pl.col("games").sum().alias("games"),
            pl.col("wins").sum().alias("wins"),
            pl.col("runs_scored").sum().alias("runs_scored"),
            pl.col("runs_allowed").sum().alias("runs_allowed"),
        )
    )
    keys = ["season", "games", "wins", "runs_scored", "runs_allowed"]
    gl_unique = gl.join(gl.group_by(keys).len().filter(pl.col("len") == 1), on=keys, how="semi")
    club_unique = club.join(
        club.group_by(keys).len().filter(pl.col("len") == 1), on=keys, how="semi"
    )
    return gl_unique.join(club_unique, on=keys, how="inner").select(
        "season", "franchise_id", "team_abbrev"
    )


def fit_talent_model(
    panel: pl.DataFrame,
    controls: list[str],
    label: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    games = panel["games"].to_numpy().astype(float)
    y = panel["wins"].to_numpy() / games
    codes = build_codes(panel)
    columns: list[np.ndarray] = [np.ones((panel.height, 1))]
    for name in controls:
        column = panel[name].to_numpy().astype(float)
        columns.append(((column - column.mean()) / column.std())[:, None])
    columns.append(dummy_design(codes["season"]))
    X = np.column_stack(columns)

    blocks = {"manager": codes["manager"], "franchise": codes["franchise"]}
    problem = REMLProblem(y, X, games, blocks)
    fit = problem.fit()
    lrt = max(
        fit_without_block(y, X, games, blocks, "manager").neg2_reml - fit.neg2_reml, 0.0
    )
    resid = residualize(y, X, games)
    talent_r2 = 1.0 - float(np.var(resid) / np.var(residualize(y, np.ones((y.size, 1)), games)))
    # Manager tenure is nested inside a franchise-era, so any team quality the
    # talent proxy misses can masquerade as a manager effect. Shuffling manager
    # labels *within franchise* holds the franchise fixed and is the test that
    # separates the two; the global shuffle does not.
    resid_franchise = residualize(
        y, np.column_stack([X, dummy_design(codes["franchise"])]), games
    )
    perm_global = permutation_test(
        resid_franchise, games, codes["manager"], None, permutations, rng
    )
    perm_block = permutation_test(
        resid_franchise, games, codes["manager"], codes["franchise"], permutations, rng
    )
    # resid_franchise has franchise and season absorbed, so the analytic
    # degrees-of-freedom moment estimator is biased low; take the null
    # expectation from the permutation instead.
    moment = moment_variance_from_permutation(perm_block, games, codes["manager"])

    info = (
        panel.group_by("manager_id")
        .agg(
            pl.col("manager_name").first().alias("manager"),
            pl.len().alias("seasons"),
            pl.col("games").sum().alias("games"),
        )
        .sort("manager_id")
    )
    blup = fit.blups["manager"] * GAMES_SCALE
    blup_sd = fit.blup_sd["manager"] * GAMES_SCALE
    order = np.argsort(-blup)
    extremes = [
        {
            "manager": info["manager"][int(i)],
            "seasons": int(info["seasons"][int(i)]),
            "games": int(info["games"][int(i)]),
            "blup_wins_per_162": float(blup[int(i)]),
            "blup_sd": float(blup_sd[int(i)]),
        }
        for i in list(order[:8]) + list(order[-8:])
    ]
    return {
        "label": label,
        "controls": controls,
        "n_team_seasons": panel.height,
        "n_managers": int(codes["manager"].max() + 1),
        "sigma_manager_wins_per_162": float(np.sqrt(fit.variances["manager"]) * GAMES_SCALE),
        "sigma_franchise_wins_per_162": float(np.sqrt(fit.variances["franchise"]) * GAMES_SCALE),
        "residual_sd_wins_per_162": float(np.sqrt(fit.resid_variance / GAMES_SCALE) * GAMES_SCALE),
        "control_beta": [float(b) for b in fit.beta[1 : 1 + len(controls)] * GAMES_SCALE],
        "variance_explained_by_controls": talent_r2,
        "lrt_statistic": lrt,
        "lrt_p_value": 0.5 * float(stats.chi2.sf(lrt, 1)) if lrt > 0 else 0.5,
        "moment_sigma_manager_signed_per_162": float(
            np.sign(moment) * np.sqrt(abs(moment)) * GAMES_SCALE
        ),
        "permutation_global": perm_global,
        "permutation_within_franchise": perm_block,
        "reliability": reliability(
            fit.variances["manager"], fit.resid_variance, info["games"].to_numpy().astype(float)
        ),
        "manager_blup_extremes": extremes,
        "resid": resid,
        "resid_franchise": resid_franchise,
        "panel": panel,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-log", type=Path, default=GAME_LOG_PATH)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--skip-postgres", action="store_true", help="skip Part 3")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    stints, meta = load_stints(args.game_log, 1901, 2100)
    k = fit_pythagorean_exponent(stints)
    full = stints.filter(pl.col("full_season"))
    report: dict = {"meta": meta, "pythagorean_k": k}

    # --- Part 1 ------------------------------------------------------------
    print("=== Part 1: is the individual Pythagorean-residual effect extractable? ===")
    games = full["games"].to_numpy().astype(float)
    rs = full["runs_scored"].to_numpy().astype(float)
    ra = full["runs_allowed"].to_numpy().astype(float)
    y = full["win_pct"].to_numpy() - pythagorean(rs, ra, k)
    codes = build_codes(full)
    X = np.column_stack(
        [
            np.ones((full.height, 1)),
            dummy_design(codes["franchise"]),
            dummy_design(codes["season"]),
        ]
    )
    resid = residualize(y, X, games)
    resid_variance = float(np.sum(games * resid**2) / (full.height - X.shape[1]))
    career_games = (
        full.group_by("manager_id").agg(pl.col("games").sum())["games"].to_numpy().astype(float)
    )
    report["part1_reliability_if_sigma_m_were"] = {
        f"{s:.2f}_wins_per_162": reliability(
            (s / GAMES_SCALE) ** 2, resid_variance, career_games
        )
        for s in (0.0, 0.56, 1.0, 2.0)
    }
    print(f"  residual variance sigma^2_e = {resid_variance:.6f} (win-pct^2 per game)")
    print("  reliability lambda of a manager's own mean, by career length:")
    for sigma, table in report["part1_reliability_if_sigma_m_were"].items():
        print(
            f"    if sigma_m = {sigma:<20} "
            + "  ".join(f"{key.split('_')[1]}g: {value:.3f}" for key, value in table.items() if key.startswith("games_"))
        )
    report["part1_out_of_sample"] = [
        out_of_sample_slope(full, resid, min_seasons, f"pythag residual, >= {min_seasons} seasons")
        for min_seasons in (4, 6, 8)
    ]
    print("  out-of-sample: odd-season estimate predicting even seasons")
    for row in report["part1_out_of_sample"]:
        if "slope" in row:
            lo, hi = row["ci95"]
            print(
                f"    {row['label']:<38} n={row['n_managers']:>3} "
                f"slope={row['slope']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )

    # --- Part 2 ------------------------------------------------------------
    print("\n=== Part 2: upper bound on manager effects on wins, 1901-2025 ===")
    transitions = build_transitions(stints, k)
    changed = int(transitions["changed"].sum())
    print(
        f"  {transitions.height} consecutive franchise-season pairs "
        f"({changed} with a new manager, {transitions.height - changed} same manager)"
    )
    report["part2"] = []
    for column, scale, name in (
        ("d_win_pct", GAMES_SCALE, "win pct -> wins per 162"),
        ("d_pythag_resid", GAMES_SCALE, "pythagorean residual -> wins per 162"),
        ("d_run_diff", GAMES_SCALE, "run differential -> runs per 162"),
    ):
        row = contrast_bound(transitions, column, scale, args.bootstrap, rng)
        row["name"] = name
        report["part2"].append(row)
        if "signed_sigma_manager" in row:
            lo, hi = row["ci95"]
            print(
                f"  {name:<44} sd(changed)={row['sd_changed']:6.2f} "
                f"sd(stayed)={row['sd_stayed']:6.2f} -> sigma_m={row['signed_sigma_manager']:+6.2f} "
                f"[{lo:+.2f}, {hi:+.2f}]"
            )

    # --- Part 3 ------------------------------------------------------------
    if not args.skip_postgres:
        print("\n=== Part 3: talent-controlled individual effects on wins, 2009-2025 ===")
        frames = read_postgres()
        talent, talent_meta = build_talent(frames)
        crosswalk = crosswalk_franchises(frames["club"], stints)
        print(
            f"  talent proxy: {talent_meta['team_seasons']} team-seasons, "
            f"batter PA coverage {talent_meta['batter_pa_coverage']:.3f}, "
            f"pitcher IP coverage {talent_meta['pitcher_ip_coverage']:.3f}"
        )
        print(f"  franchise crosswalk resolved {crosswalk.height} club-seasons")
        report["part3_talent_meta"] = talent_meta
        report["part3_crosswalk_rows"] = crosswalk.height

        panel = (
            full.join(crosswalk, on=["season", "franchise_id"], how="inner")
            .join(talent, on=["season", "team_abbrev"], how="inner")
            .filter(pl.col("games") >= MIN_FULL_SEASON_GAMES)
        )
        print(f"  panel: {panel.height} full-season stints, {panel['manager_id'].n_unique()} managers")
        report["part3"] = []
        results = []
        for controls, label in (
            ([], "no talent control"),
            (["bat_talent", "pit_talent"], "leave-one-season-out talent control"),
        ):
            result = fit_talent_model(panel, controls, label, args.permutations, rng)
            results.append(result)
            payload = {
                key: value
                for key, value in result.items()
                if key not in ("resid", "resid_franchise", "panel")
            }
            report["part3"].append(payload)
            print(
                f"  {label:<38} sigma_mgr={payload['sigma_manager_wins_per_162']:6.3f} "
                f"sigma_fr={payload['sigma_franchise_wins_per_162']:6.3f} "
                f"resid_sd={payload['residual_sd_wins_per_162']:5.2f} "
                f"R2_controls={payload['variance_explained_by_controls']:+.3f}"
            )
            print(
                f"    LRT p={payload['lrt_p_value']:.4f}  "
                f"moment sigma_m={payload['moment_sigma_manager_signed_per_162']:+.3f}  "
                f"permutation p: global={payload['permutation_global']['p_value']:.4f} "
                f"within-franchise={payload['permutation_within_franchise']['p_value']:.4f}"
            )
        controlled = results[-1]
        report["part3_out_of_sample"] = [
            out_of_sample_slope(
                controlled["panel"], controlled["resid"], min_seasons,
                f"talent-controlled wins, >= {min_seasons} seasons",
            )
            for min_seasons in (4, 6)
        ]
        # Does the manager grouping beat an arbitrary contiguous franchise era?
        placebo = contiguity_placebo(
            controlled["panel"], controlled["resid_franchise"], args.permutations, rng
        )
        report["part3_contiguity_placebo"] = placebo
        print(
            f"  contiguity placebo ({placebo['n_tenures']} tenures re-cut into random "
            f"contiguous blocks of the same lengths):"
        )
        print(
            f"    observed between-tenure SS={placebo['observed']:.5f} vs placebo mean "
            f"{placebo['placebo_mean']:.5f}, p={placebo['p_value']:.4f}"
        )
        print("  out-of-sample: odd-season estimate predicting even seasons")
        for row in report["part3_out_of_sample"]:
            if "slope" in row:
                lo, hi = row["ci95"]
                print(
                    f"    {row['label']:<42} n={row['n_managers']:>3} "
                    f"slope={row['slope']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )
        print("  reliability of a manager's own talent-controlled mean:")
        for key, value in controlled["reliability"].items():
            if key.startswith("games_"):
                print(f"    {key.split('_')[1]:>5} career games -> lambda {value:.3f}")
        print("\n  best / worst shrunken talent-controlled manager estimates:")
        for row in controlled["manager_blup_extremes"]:
            print(
                f"    {row['blup_wins_per_162']:+6.2f} +/- {row['blup_sd']:.2f} wins/162  "
                f"{row['manager']:<20} {row['seasons']:>2} seasons {row['games']:>5} games"
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
