"""Is the runs-allowed manager signal actually pitcher deployment?

`analyze_manager_run_effects.py` found the only real candidate signal in this
whole line of work: with leak-free talent controlled, sigma_manager on runs
allowed is 22.3 runs/162 (LRT p = 0.0018), but it fails the contiguity placebo
(p = 0.13), so it cannot be separated from a persistent club-level pitching
environment that happens to align with manager tenure.

The fix is to stop modelling the outcome and model the DECISION. Runs allowed
is an outcome contaminated by everything about a club. Who pitches to whom, in
what state, is something the manager literally chooses, pitch by pitch, and it
is measurable from `mlb.pitches`.

Three decision indices, all built per club-season from plate-appearance level
state, all holding the club's actual arms fixed so roster quality cannot leak
in:

``leverage_alignment``
    Did the good arms get the important innings? Empirical leverage index per
    base-out-inning-score state, then

        (PA-weighted mean pitcher quality) - (leverage-weighted mean quality)

    with quality = each pitcher's leak-free FIP from his OTHER seasons. Because
    both terms average over the same club's same pitchers, roster quality
    cancels by construction and only the allocation survives. Positive = better
    arms in higher-leverage spots. Units are FIP points; scaled to runs/162.

``third_time_share``
    Share of a starter's plate appearances taken at the third-or-later time
    through the order, where the times-through-order penalty is well
    established. High = slow hook.

``platoon_advantage``
    Share of plate appearances where the pitcher held the handedness
    advantage, relative to what the club's own arms and its opponents' lineups
    made available.

For each index: is it a stable manager trait (REML sigma_manager + contiguity
placebo)? A decision index passing the placebo is far closer to causal than an
outcome, because the manager's hand is the direct cause of the number.

Then the mediation test that ties it together: add the indices to the
runs-allowed model. If the 22.3 runs/162 is deployment, sigma_manager on runs
allowed must shrink when deployment is controlled.

Usage::

    uv run python scripts/analyze_pitcher_usage_effects.py
    uv run python scripts/analyze_pitcher_usage_effects.py --first-season 2015
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
    load_stints,
)
from analyze_manager_win_effects import (
    MIN_FULL_SEASON_GAMES,
    contiguity_placebo,
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
from mlb.analysis.talent import (
    PIT_SHRINK_IP,
    leave_one_season_out,
    read_player_seasons,
    team_talent,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_LOG_PATH = REPO_ROOT / "data" / "managers" / "manager_game_log.parquet"
REPORT_PATH = REPO_ROOT / "output" / "manager_effects" / "pitcher_usage_report.json"
CACHE_DIR = REPO_ROOT / "data" / "managers"

# State discretization for the win-probability and leverage tables. Coarse
# enough that every cell is well estimated (~12M plate appearances over
# ~6k cells) and fine enough to carry real leverage variation.
MAX_INNING = 10
SCORE_CLIP = 6

# One row per plate appearance with the state the pitcher inherited. `outs` and
# the runner flags are at-bat-start values (repaired in place 2026-08-09), and
# the scores are the scores entering the at-bat, so this is the decision state.
PA_STATE_SQL = f"""
    create temp table pa_states as
    with first_pitch as (
        select distinct on (p.game_pk, p.at_bat_index)
               p.game_pk,
               p.season,
               p.at_bat_index,
               p.inning,
               p.half_inning,
               p.outs,
               p.batter_id,
               p.pitcher_id,
               p.bat_side,
               p.throw_side,
               p.home_score,
               p.away_score,
               p.is_runner_on_first,
               p.is_runner_on_second,
               p.is_runner_on_third,
               case when p.half_inning = 'top' then p.home_team_id else p.away_team_id end
                   as pitching_team_id
        from mlb.pitches p
        where p.game_type = 'R'
          and p.season between %(first_season)s and %(last_season)s
          and p.pitch_number > 0
        order by p.game_pk, p.at_bat_index, p.pitch_number
    )
    select f.game_pk,
           f.season,
           f.at_bat_index,
           f.pitching_team_id,
           f.pitcher_id,
           f.batter_id,
           least(f.inning, {MAX_INNING}) as inning_key,
           f.half_inning,
           f.outs,
           (f.is_runner_on_first::int
            + 2 * f.is_runner_on_second::int
            + 4 * f.is_runner_on_third::int) as runners,
           greatest(-{SCORE_CLIP}, least({SCORE_CLIP},
               case when f.half_inning = 'top'
                    then f.away_score - f.home_score
                    else f.home_score - f.away_score end)) as bat_score_diff,
           (f.bat_side <> f.throw_side) as platoon_neutral,
           g.home_won
    from first_pitch f
    join (
        select gg.game_pk,
               (sum(case when l.team_type = 'home' then l.runs else 0 end)
                > sum(case when l.team_type = 'away' then l.runs else 0 end)) as home_won
        from mlb.games gg
        join mlb.linescore l on l.game_pk = gg.game_pk
        where gg.game_type = 'R' and gg.abstract_game_state = 'Final'
        group by gg.game_pk
    ) g on g.game_pk = f.game_pk
"""

WP_STATE_SQL = """
    select inning_key, half_inning, outs, runners, bat_score_diff,
           count(*) as n,
           sum(home_won::int) as home_wins
    from pa_states
    group by 1, 2, 3, 4, 5
"""

# Consecutive plate appearances within a game give the leverage index: how much
# the home win probability actually moves across one plate appearance.
TRANSITION_SQL = """
    with ordered as (
        select inning_key, half_inning, outs, runners, bat_score_diff, home_won,
               lead(inning_key) over w as n_inning,
               lead(half_inning) over w as n_half,
               lead(outs) over w as n_outs,
               lead(runners) over w as n_runners,
               lead(bat_score_diff) over w as n_diff
        from pa_states
        window w as (partition by game_pk order by at_bat_index)
    )
    select inning_key, half_inning, outs, runners, bat_score_diff,
           n_inning, n_half, n_outs, n_runners, n_diff, home_won,
           count(*) as n
    from ordered
    group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
"""

# Per (season, club, pitcher): plate appearances, summed leverage, starter
# workload split by time through the order, and platoon advantage. The leverage
# table is pushed back in as a VALUES join because it is only a few thousand
# rows and this keeps the 12M-row pass server-side.
USAGE_SQL_TEMPLATE = """
    with li(inning_key, half_inning, outs, runners, bat_score_diff, li) as (
        values {li_values}
    ),
    starters as (
        select distinct on (game_pk, pitching_team_id)
               game_pk, pitching_team_id, pitcher_id as starter_id
        from pa_states
        order by game_pk, pitching_team_id, at_bat_index
    ),
    tto as (
        select s.game_pk,
               s.pitching_team_id,
               s.pitcher_id,
               s.platoon_neutral,
               s.inning_key, s.half_inning, s.outs, s.runners, s.bat_score_diff,
               s.season,
               (st.starter_id = s.pitcher_id) as is_starter,
               ceil(row_number() over (
                   partition by s.game_pk, s.pitching_team_id
                   order by s.at_bat_index
               ) / 9.0) as time_through
        from pa_states s
        join starters st
          on st.game_pk = s.game_pk and st.pitching_team_id = s.pitching_team_id
    )
    select t.season,
           t.pitching_team_id,
           t.pitcher_id,
           count(*) as pa,
           sum(coalesce(li.li, 1.0)) as li_sum,
           sum(case when t.is_starter then 1 else 0 end) as starter_pa,
           sum(case when t.is_starter and t.time_through >= 3 then 1 else 0 end)
               as starter_pa_tto3,
           sum(case when t.platoon_neutral then 0 else 1 end) as platoon_adv_pa
    from tto t
    left join li
      on li.inning_key = t.inning_key
     and li.half_inning = t.half_inning
     and li.outs = t.outs
     and li.runners = t.runners
     and li.bat_score_diff = t.bat_score_diff
    group by 1, 2, 3
"""


def fetch_usage(first_season: int, last_season: int) -> dict[str, pl.DataFrame]:
    """Build the PA state temp table once, then run all aggregations on it."""

    import psycopg

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    frames: dict[str, pl.DataFrame] = {}

    def to_frame(cursor) -> pl.DataFrame:
        columns = [c.name for c in cursor.description or ()]
        rows = cursor.fetchall()
        return pl.DataFrame(
            {col: [r[i] for r in rows] for i, col in enumerate(columns)}
        ).with_columns((pl.selectors.integer() | pl.selectors.decimal()).cast(pl.Int64))

    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    ) as connection, connection.cursor() as cursor:
        cursor.execute("set statement_timeout = 0")
        print("  building plate-appearance state table ...", flush=True)
        cursor.execute(
            PA_STATE_SQL,  # type: ignore[arg-type]
            {"first_season": first_season, "last_season": last_season},
        )
        cursor.execute("select count(*) from pa_states")  # type: ignore[arg-type]
        row = cursor.fetchone()
        n_pa = int(row[0]) if row else 0
        print(f"  {n_pa:,} plate appearances", flush=True)

        cursor.execute(WP_STATE_SQL)  # type: ignore[arg-type]
        frames["wp"] = to_frame(cursor)
        print(f"  {frames['wp'].height:,} distinct states", flush=True)

        cursor.execute(TRANSITION_SQL)  # type: ignore[arg-type]
        frames["transitions"] = to_frame(cursor)
        print(f"  {frames['transitions'].height:,} state transitions", flush=True)

        leverage = build_leverage(frames["wp"], frames["transitions"])
        frames["leverage"] = leverage
        values = ", ".join(
            f"({r['inning_key']}, '{r['half_inning']}', {r['outs']}, {r['runners']}, "
            f"{r['bat_score_diff']}, {r['li']:.6f})"
            for r in leverage.iter_rows(named=True)
        )
        print(f"  leverage table: {leverage.height:,} states; aggregating usage ...", flush=True)
        cursor.execute(USAGE_SQL_TEMPLATE.format(li_values=values))  # type: ignore[arg-type]
        frames["usage"] = to_frame(cursor)
        print(f"  {frames['usage'].height:,} club-season-pitcher rows", flush=True)
        frames["n_pa"] = pl.DataFrame({"n_pa": [n_pa]})
    return frames


# Real leverage index tops out around 10-11 for the most extreme late-game
# spots. Anything past this is a sparse-cell artifact, not baseball.
MAX_LEVERAGE = 10.0
# Empirical-Bayes prior weight, in plate appearances, pulling each state's win
# probability toward the smooth logistic fit.
WP_SHRINK_PA = 200.0


def smooth_win_probability(wp: pl.DataFrame) -> pl.DataFrame:
    """State win probability: empirical rate shrunk toward a logistic fit.

    Raw per-cell rates are unusable for leverage. Leverage is a *difference*
    of two win probabilities, so independent noise in two sparse cells adds and
    then gets divided by nothing — rare states like "1st inning, down 5, two
    on" came out at LI 28 when the true value is under 1. Shrinking each cell
    toward a smooth model of score difference, inning, half, outs, and base
    state removes that without flattening the real structure, because dense
    cells (which are most plate appearances) barely move.
    """

    from sklearn.linear_model import LogisticRegression

    keys = ["inning_key", "half_inning", "outs", "runners", "bat_score_diff"]
    diff = wp["bat_score_diff"].to_numpy().astype(float)
    inning = wp["inning_key"].to_numpy().astype(float)
    is_bottom = (wp["half_inning"].to_numpy() == "bottom").astype(float)
    outs = wp["outs"].to_numpy().astype(float)
    runners = wp["runners"].to_numpy().astype(int)
    # Sign the score difference from the home team's perspective so one smooth
    # surface covers both halves of the inning.
    home_diff = np.where(is_bottom > 0, diff, -diff)
    features = np.column_stack(
        [
            home_diff,
            home_diff * inning,
            np.sign(home_diff) * np.sqrt(np.abs(home_diff)),
            inning,
            is_bottom,
            outs,
            is_bottom * inning,
            np.eye(8)[runners],
        ]
    )
    counts = wp["n"].to_numpy().astype(float)
    wins = wp["home_wins"].to_numpy().astype(float)
    # Weighted logistic on aggregated cells: each cell contributes its wins as
    # positives and its losses as negatives.
    X = np.vstack([features, features])
    y = np.concatenate([np.ones(features.shape[0]), np.zeros(features.shape[0])])
    weights = np.concatenate([wins, counts - wins])
    keep = weights > 0
    model = LogisticRegression(max_iter=2000, C=10.0).fit(
        X[keep], y[keep], sample_weight=weights[keep]
    )
    prior = model.predict_proba(features)[:, 1]
    shrunk = (wins + WP_SHRINK_PA * prior) / (counts + WP_SHRINK_PA)
    return wp.with_columns(
        pl.Series("wp", shrunk), pl.Series("wp_model", prior)
    ).select(*keys, "wp", "n")


def build_leverage(wp: pl.DataFrame, transitions: pl.DataFrame) -> pl.DataFrame:
    """Empirical leverage index: mean |change in home win probability| per PA.

    Leverage is the average absolute move in win probability across one plate
    appearance, normalized so the PA-weighted mean is 1.0. Terminal plate
    appearances (no successor) resolve to the realized result, which is the
    correct 0/1 win probability.
    """

    keys = ["inning_key", "half_inning", "outs", "runners", "bat_score_diff"]
    lookup = smooth_win_probability(wp).select(*keys, "wp")

    after = lookup.rename(
        {
            "inning_key": "n_inning",
            "half_inning": "n_half",
            "outs": "n_outs",
            "runners": "n_runners",
            "bat_score_diff": "n_diff",
            "wp": "wp_after",
        }
    )

    joined = (
        transitions.join(lookup, on=keys, how="left")
        .join(after, on=["n_inning", "n_half", "n_outs", "n_runners", "n_diff"], how="left")
        .with_columns(
            # A terminal plate appearance has no successor state; the game is
            # decided, so the post-PA win probability is the realized result.
            pl.when(pl.col("wp_after").is_null())
            .then(pl.col("home_won").cast(pl.Float64))
            .otherwise(pl.col("wp_after"))
            .alias("wp_after")
        )
        .with_columns((pl.col("wp_after") - pl.col("wp")).abs().alias("swing"))
    )
    leverage = joined.group_by(keys).agg(
        ((pl.col("swing") * pl.col("n")).sum() / pl.col("n").sum()).alias("raw_li"),
        pl.col("n").sum().alias("n"),
    )
    # Normalize so the PA-weighted mean is 1.0, clip the tail, then renormalize
    # so the clip does not shift the scale.
    mean_li = float((leverage["raw_li"] * leverage["n"]).sum() / leverage["n"].sum())
    leverage = leverage.with_columns(
        (pl.col("raw_li") / mean_li).clip(0.0, MAX_LEVERAGE).alias("li")
    )
    mean_clipped = float((leverage["li"] * leverage["n"]).sum() / leverage["n"].sum())
    return leverage.with_columns((pl.col("li") / mean_clipped).alias("li")).select(
        *keys, "li", "n"
    )


def build_indices(
    usage: pl.DataFrame, pitching: pl.DataFrame, crosswalk_teams: pl.DataFrame
) -> pl.DataFrame:
    """Per club-season decision indices from the usage aggregates."""

    # Leak-free pitcher quality: FIP numerator over innings from OTHER seasons.
    quality = leave_one_season_out(
        pitching.with_columns((pl.col("outs") / 3.0).alias("ip")),
        13 * pl.col("hr") + 3 * (pl.col("bb") + pl.col("hbp")) - 2 * pl.col("k"),
        "ip",
        PIT_SHRINK_IP,
    ).select("season", "player_id", "quality")

    joined = usage.join(
        quality, left_on=["season", "pitcher_id"], right_on=["season", "player_id"], how="left"
    )
    league = float(
        (joined["quality"] * joined["pa"]).sum() / joined["pa"].sum()
    )
    joined = joined.with_columns(pl.col("quality").fill_null(league))

    return (
        joined.group_by("season", "pitching_team_id")
        .agg(
            pl.col("pa").sum().alias("pa"),
            # Volume-weighted minus leverage-weighted quality. Lower FIP is
            # better, so this is positive when the better arms took the
            # higher-leverage plate appearances. Both terms run over the same
            # club's same pitchers, so roster quality cancels.
            (
                (pl.col("quality") * pl.col("pa")).sum() / pl.col("pa").sum()
                - (pl.col("quality") * pl.col("li_sum")).sum() / pl.col("li_sum").sum()
            ).alias("leverage_alignment"),
            (pl.col("starter_pa_tto3").sum() / pl.col("starter_pa").sum()).alias(
                "third_time_share"
            ),
            (pl.col("platoon_adv_pa").sum() / pl.col("pa").sum()).alias("platoon_advantage"),
        )
        .join(crosswalk_teams, on="pitching_team_id", how="inner")
    )


def fit_index_model(
    panel: pl.DataFrame,
    response: str,
    controls: list[str],
    permutations: int,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> dict:
    games = panel["games"].to_numpy().astype(float)
    y = panel[response].to_numpy().astype(float) * scale
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
    resid_franchise = residualize(
        y, np.column_stack([X, dummy_design(codes["franchise"])]), games
    )
    perm = permutation_test(
        resid_franchise, games, codes["manager"], codes["franchise"], permutations, rng
    )
    moment = moment_variance_from_permutation(perm, games, codes["manager"])
    placebo = contiguity_placebo(panel, resid_franchise, permutations, rng)
    # Do the controls move the response at all? A control that is unrelated to
    # the response cannot mediate anything, so this is what makes a null
    # mediation interpretable rather than merely unsurprising.
    control_betas = {
        name: float(fit.beta[1 + i]) for i, name in enumerate(controls)
    }
    baseline = residualize(
        y, np.column_stack([np.ones((panel.height, 1)), dummy_design(codes["season"])]), games
    )
    control_r2 = (
        1.0 - float(np.var(resid) / np.var(baseline)) if controls else 0.0
    )
    return {
        "response": response,
        "controls": controls,
        "n_team_seasons": panel.height,
        "control_betas_per_sd": control_betas,
        "variance_explained_by_controls": control_r2,
        "n_managers": int(codes["manager"].max() + 1),
        "observed_sd": float(np.std(y, ddof=1)),
        "sigma_manager": float(np.sqrt(fit.variances["manager"])),
        "sigma_franchise": float(np.sqrt(fit.variances["franchise"])),
        "residual_sd_per_season": float(np.sqrt(fit.resid_variance / float(np.median(games)))),
        "share_manager_of_observed": float(
            fit.variances["manager"] / max(float(np.var(y, ddof=1)), 1e-30)
        ),
        "lrt_p_value": 0.5 * float(stats.chi2.sf(lrt, 1)) if lrt > 0 else 0.5,
        "permutation_p": perm["p_value"],
        "contiguity_placebo_p": placebo["p_value"],
        "moment_sigma_signed": float(np.sign(moment) * np.sqrt(abs(moment))),
        "out_of_sample_franchise_absorbed": out_of_sample_slope(
            panel, resid_franchise, 4, f"{response} (franchise-absorbed)"
        ),
        "out_of_sample_season_only": out_of_sample_slope(
            panel, resid, 4, f"{response} (season-only)"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2009)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--refresh", action="store_true", help="ignore the usage cache")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cache = CACHE_DIR / f"pitcher_usage_{args.first_season}_{args.last_season}.parquet"
    lev_cache = CACHE_DIR / f"leverage_{args.first_season}_{args.last_season}.parquet"

    print("=== pitch-level usage extraction ===")
    if cache.exists() and lev_cache.exists() and not args.refresh:
        usage = pl.read_parquet(cache)
        leverage = pl.read_parquet(lev_cache)
        print(f"  cache hit: {usage.height:,} club-season-pitcher rows")
    else:
        frames = fetch_usage(args.first_season, args.last_season)
        usage, leverage = frames["usage"], frames["leverage"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        usage.write_parquet(cache)
        leverage.write_parquet(lev_cache)

    top = leverage.sort("li", descending=True).head(3)
    bottom = leverage.sort("li").head(3)
    print("  highest-leverage states (inning/half/outs/runners/diff -> LI):")
    for r in top.iter_rows(named=True):
        print(
            f"    inn {r['inning_key']} {r['half_inning']:<6} {r['outs']} out "
            f"runners {r['runners']} diff {r['bat_score_diff']:+d} -> LI {r['li']:.2f}"
        )
    for r in bottom.iter_rows(named=True):
        print(
            f"    inn {r['inning_key']} {r['half_inning']:<6} {r['outs']} out "
            f"runners {r['runners']} diff {r['bat_score_diff']:+d} -> LI {r['li']:.2f}"
        )

    player_frames = read_player_seasons()
    stints, meta = load_stints(GAME_LOG_PATH, args.first_season, args.last_season)
    crosswalk = crosswalk_franchises(player_frames["club"], stints)
    team_ids = (
        player_frames["club"]
        .select("team_abbrev")
        .unique()
        .join(
            _team_id_map(),
            on="team_abbrev",
            how="inner",
        )
    )
    indices = build_indices(usage, player_frames["pitching"], team_ids)
    talent, _ = team_talent(player_frames, "playing_time")

    full = stints.filter(pl.col("full_season"))
    panel = (
        full.join(crosswalk, on=["season", "franchise_id"], how="inner")
        .join(indices.drop("pa"), on=["season", "team_abbrev"], how="inner")
        .join(talent, on=["season", "team_abbrev"], how="inner")
        .filter(pl.col("games") >= MIN_FULL_SEASON_GAMES)
    )
    print(
        f"\n=== decision indices, {args.first_season}-{args.last_season} ===\n"
        f"  panel: {panel.height} full-season stints, {panel['manager_id'].n_unique()} managers"
    )
    for column in ("leverage_alignment", "third_time_share", "platoon_advantage"):
        series = panel[column]
        print(
            f"  {column:<20} mean {series.mean():+.4f}  sd {series.std():.4f}  "
            f"range [{series.min():+.4f}, {series.max():+.4f}]"
        )

    report: dict = {"meta": meta, "panel_rows": panel.height, "indices": [], "mediation": []}

    print("\n=== is each decision a stable manager trait? ===")
    for column in ("leverage_alignment", "third_time_share", "platoon_advantage"):
        row = fit_index_model(panel, column, [], args.permutations, rng)
        report["indices"].append(row)
        oos_a = row["out_of_sample_franchise_absorbed"]
        oos_b = row["out_of_sample_season_only"]

        def fmt(block: dict) -> str:
            return (
                f"{block['slope']:+.2f} [{block['ci95'][0]:+.2f}, {block['ci95'][1]:+.2f}]"
                if "slope" in block
                else "n/a"
            )

        print(
            f"  {column:<20} sigma_mgr={row['sigma_manager']:.4f} "
            f"sigma_fr={row['sigma_franchise']:.4f} obs_sd={row['observed_sd']:.4f} "
            f"({100 * row['share_manager_of_observed']:.0f}% of variance)"
        )
        print(
            f"    LRT p={row['lrt_p_value']:.4f}  perm p={row['permutation_p']:.4f}  "
            f"CONTIGUITY PLACEBO p={row['contiguity_placebo_p']:.4f}"
        )
        print(f"    out-of-sample: absorbed {fmt(oos_a)}  season-only {fmt(oos_b)}")

    print("\n=== mediation: does deployment explain the runs-allowed signal? ===")
    panel = panel.with_columns(
        (pl.col("runs_allowed") / pl.col("games")).alias("ra_per_game")
    )
    specs = [
        ("talent only", ["bat_talent", "pit_talent"]),
        (
            "talent + deployment",
            [
                "bat_talent",
                "pit_talent",
                "leverage_alignment",
                "third_time_share",
                "platoon_advantage",
            ],
        ),
    ]
    for label, controls in specs:
        row = fit_index_model(
            panel, "ra_per_game", controls, args.permutations, rng, scale=GAMES_SCALE
        )
        row["label"] = label
        report["mediation"].append(row)
        print(
            f"  {label:<22} sigma_mgr={row['sigma_manager']:6.2f} runs/162  "
            f"sigma_fr={row['sigma_franchise']:6.2f}  LRT p={row['lrt_p_value']:.4f}  "
            f"placebo p={row['contiguity_placebo_p']:.4f}"
        )
        if row["control_betas_per_sd"]:
            betas = "  ".join(
                f"{name}={value:+.2f}" for name, value in row["control_betas_per_sd"].items()
            )
            print(
                f"    runs/162 per 1 SD of each control:  {betas}"
                f"   (controls explain {100 * row['variance_explained_by_controls']:.1f}%)"
            )
    if len(report["mediation"]) == 2:
        before = report["mediation"][0]["sigma_manager"]
        after = report["mediation"][1]["sigma_manager"]
        share = 100.0 * (1.0 - (after / before) ** 2) if before > 0 else float("nan")
        report["mediation_share_explained"] = share
        print(
            f"  -> deployment absorbs {share:.0f}% of the manager VARIANCE in runs allowed "
            f"({before:.2f} -> {after:.2f} runs/162)"
        )

    # Translate manager-to-manager variation in each decision into runs. The
    # manager-attributable share of an index's spread is sigma_manager / sd,
    # and the runs-allowed model prices one SD of that index, so the product is
    # what manager differences in that decision are actually worth.
    priced = report["mediation"][-1]["control_betas_per_sd"]
    report["implied_run_value"] = []
    print("\n=== what manager variation in each decision is worth ===")
    total = 0.0
    for row in report["indices"]:
        name = row["response"]
        beta = priced.get(name)
        if beta is None:
            continue
        manager_share = row["sigma_manager"] / row["observed_sd"]
        runs = abs(beta) * manager_share
        total += runs**2
        report["implied_run_value"].append(
            {
                "index": name,
                "manager_share_of_sd": manager_share,
                "runs_per_162_per_sd": beta,
                "implied_runs_per_162": runs,
                "contiguity_placebo_p": row["contiguity_placebo_p"],
            }
        )
        print(
            f"  {name:<20} manager share of spread {manager_share:.2f} SD x "
            f"{abs(beta):5.2f} runs/162 per SD = {runs:5.2f} runs/162"
        )
    unexplained = report["mediation"][0]["sigma_manager"]
    print(
        f"  all three in quadrature = {total**0.5:.2f} runs/162, against "
        f"{unexplained:.2f} runs/162 of manager variance in runs allowed "
        f"({100 * total / unexplained**2:.0f}% of the variance)"
    )
    report["implied_run_value_total"] = total**0.5

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.report}")


def _team_id_map() -> pl.DataFrame:
    """StatsAPI team id -> abbreviation, for joining usage back to the panel."""

    import psycopg

    from mlb.database import PostgresConfig

    config = PostgresConfig.from_env()
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(  # type: ignore[arg-type]
            "select team_id, abbreviation from mlb.teams where abbreviation is not null"
        )
        rows = cursor.fetchall()
    return pl.DataFrame(
        {
            "pitching_team_id": [int(r[0]) for r in rows],
            "team_abbrev": [str(r[1]) for r in rows],
        }
    )


if __name__ == "__main__":
    main()
