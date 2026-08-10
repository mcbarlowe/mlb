"""Calibrate the game simulator against actual league rates.

Measures the simulated per-pitch result distribution PER COUNT and the
in-play event distribution, per batting side (top = away batting, bottom =
home batting), against actual league rates, and stores per-class
multipliers in ``models/sim/sim_calibration.json``. Matching every count's
result distribution matches the count-machine transition kernel, so PA
outcome rates and run totals follow by construction. Two passes: the
second corrects the residual after applying pass-one multipliers.

    uv run python scripts/calibrate_sim.py --seasons 2023 2024 2025

Home-field advantage falls out of the per-side split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.outcome.dataset import build_training_frame, load_pitches
from src.outcome.inference import PitchOutcomePredictor
from src.sim.calibration import (
    DEFAULT_CALIBRATION_PATH,
    SimCalibration,
    derive_multipliers,
)
from src.sim.count_machine import apply_pitch_result
from src.sim.lineups import lineup_from_feed
from src.sim.matchup import MatchupProviderFactory
from src.sim.pa import EVENT_CLASSES, RESULT_CLASSES
from src.sim.pitch_mix import COUNTS, PitchMixProfiles

LIVEFEED_ROOT = Path("data/raw/livefeeds")
SIDES = {"top": True, "bottom": False}


def count_key(balls: int, strikes: int) -> str:
    return f"{balls}-{strikes}"


def actual_rates_by_side(seasons: list[int]) -> tuple[dict, dict, dict]:
    """Actual league rates: aggregate result, per-count result, event."""
    frame = build_training_frame(load_pitches(seasons))
    result_rates: dict[str, dict[str, float]] = {}
    result_by_count: dict[str, dict[str, dict[str, float]]] = {}
    event_rates: dict[str, dict[str, float]] = {}
    for side, is_top in SIDES.items():
        side_rows = frame.filter(
            (pl.col("is_top_half") == int(is_top))
            & pl.col("label_result").is_not_null()
        )
        counts = side_rows["label_result"].value_counts()
        total = side_rows.height
        result_rates[side] = {row[0]: row[1] / total for row in counts.rows()}

        result_by_count[side] = {}
        for balls, strikes in COUNTS:
            count_rows = side_rows.filter(
                (pl.col("balls_before") == balls)
                & (pl.col("strikes_before") == strikes)
            )
            if count_rows.height == 0:
                continue
            counts = count_rows["label_result"].value_counts()
            result_by_count[side][count_key(balls, strikes)] = {
                row[0]: row[1] / count_rows.height for row in counts.rows()
            }

        event_rows = side_rows.filter(pl.col("label_event").is_not_null())
        counts = event_rows["label_event"].value_counts()
        event_rates[side] = {
            row[0]: row[1] / event_rows.height for row in counts.rows()
        }
    return result_rates, result_by_count, event_rates


def sample_matchups(season: int, n_games: int, seed: int) -> list[tuple]:
    """(pitcher, batter, is_top) triples from recent archived finals."""
    files = sorted((LIVEFEED_ROOT / str(season)).glob("*.json"), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(files)
    matchups: list[tuple] = []
    used = 0
    for path in files:
        if used >= n_games:
            break
        feed = json.loads(path.read_text())
        status = feed.get("gameData", {}).get("status", {}).get("abstractGameState")
        if feed.get("gameData", {}).get("game", {}).get("type") != "R" or status != "Final":
            continue
        try:
            away = lineup_from_feed(feed, "away")
            home = lineup_from_feed(feed, "home")
        except (ValueError, KeyError):
            continue
        used += 1
        for batter in away.batters:
            matchups.append((home.starter, batter, True))
        for batter in home.batters:
            matchups.append((away.starter, batter, False))
    return matchups


def simulate_rates(
    factory: MatchupProviderFactory,
    matchups: list[tuple],
    pas_per_matchup: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    """Simulated aggregate/per-count result and event rates per side.

    PAs run inside a rolling base-out context (transitions sampled from the
    empirical engine, provider stretch selected by current runners) so the
    measured rates reflect the same windup/stretch mixture the game loop
    produces — calibrating windup-only providers understates offense.
    """
    from src.sim.base_out import BaseOutEngine

    rng = random.Random(seed)
    engine = BaseOutEngine.load(seed=seed)
    result_counts = {side: Counter() for side in SIDES}
    count_counts: dict[str, dict[str, Counter]] = {
        side: {count_key(b, s): Counter() for b, s in COUNTS} for side in SIDES
    }
    event_counts = {side: Counter() for side in SIDES}
    for pitcher, batter, is_top in matchups:
        side = "top" if is_top else "bottom"
        runners, outs = 0, 0
        for _ in range(pas_per_matchup):
            provider = factory(pitcher, batter, is_top, runners != 0)
            balls = strikes = 0
            pa_outcome: str | None = None
            while True:
                result = rng.choices(
                    RESULT_CLASSES,
                    weights=[
                        max(provider.result_probabilities(balls, strikes).get(c, 0.0), 0.0)
                        for c in RESULT_CLASSES
                    ],
                )[0]
                result_counts[side][result] += 1
                count_counts[side][count_key(balls, strikes)][result] += 1
                transition = apply_pitch_result(balls, strikes, result)
                if transition.terminal is None:
                    balls, strikes = transition.balls, transition.strikes
                    continue
                if transition.in_play:
                    event = rng.choices(
                        EVENT_CLASSES,
                        weights=[
                            max(provider.event_probabilities(balls, strikes).get(c, 0.0), 0.0)
                            for c in EVENT_CLASSES
                        ],
                    )[0]
                    event_counts[side][event] += 1
                    pa_outcome = event
                else:
                    pa_outcome = transition.terminal
                break
            base_out = engine.sample(pa_outcome, runners, outs)
            runners, outs = base_out.runners_after, base_out.outs_after
            if outs >= 3:
                runners, outs = 0, 0

    def normalize(counts: Counter) -> dict[str, float]:
        total = sum(counts.values())
        return {cls: n / total for cls, n in counts.items()} if total else {}

    return (
        {side: normalize(result_counts[side]) for side in SIDES},
        {
            side: {key: normalize(c) for key, c in count_counts[side].items()}
            for side in SIDES
        },
        {side: normalize(event_counts[side]) for side in SIDES},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate simulator rates.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--matchup-season", type=int, default=2025)
    parser.add_argument("--games", type=int, default=15)
    parser.add_argument("--pas", type=int, default=300)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()

    print(f"Computing actual league rates for {args.seasons}...")
    start = time.perf_counter()
    actual_result, actual_by_count, actual_event = actual_rates_by_side(args.seasons)
    print(f"...done in {time.perf_counter() - start:.0f}s")

    pointer = Path("models/outcome/latest_run.txt").read_text().strip()
    predictor = PitchOutcomePredictor(Path("models/outcome") / pointer)
    mix = PitchMixProfiles.load(seed=args.seed)
    matchups = sample_matchups(args.matchup_season, args.games, args.seed)
    print(f"Sampled {len(matchups)} matchups from {args.games} games")

    result_mults: dict[str, dict[str, float]] = {side: {} for side in SIDES}
    by_count_mults: dict[str, dict[str, dict[str, float]]] = {
        side: {} for side in SIDES
    }
    event_mults: dict[str, dict[str, float]] = {side: {} for side in SIDES}
    for pass_index in range(1, args.passes + 1):
        calibration = SimCalibration(
            result=result_mults, event=event_mults, result_by_count=by_count_mults
        )
        factory = MatchupProviderFactory(
            predictor,
            mix,
            season=args.matchup_season,
            seed=args.seed,
            calibration=calibration,
        )
        sim_result, sim_by_count, sim_event = simulate_rates(
            factory, matchups, args.pas, args.seed
        )
        print(f"\nPass {pass_index} simulated rates vs actual:")
        for side in SIDES:
            strikes_rate = sim_result[side].get("swinging_strike", 0) + sim_result[
                side
            ].get("called_strike", 0)
            print(
                f"  {side:6s} in_play sim {sim_result[side].get('in_play', 0):.3f}"
                f"/act {actual_result[side].get('in_play', 0):.3f}"
                f"  hbp sim {sim_result[side].get('hit_by_pitch', 0):.4f}"
                f"/act {actual_result[side].get('hit_by_pitch', 0):.4f}"
                f"  strikes sim {strikes_rate:.3f}"
            )
        result_mults = {
            side: derive_multipliers(
                actual_result[side], sim_result[side], result_mults[side]
            )
            for side in SIDES
        }
        by_count_mults = {
            side: {
                key: derive_multipliers(
                    actual_by_count[side].get(key, {}),
                    sim_by_count[side].get(key, {}),
                    by_count_mults[side].get(key, {}),
                )
                for key in {*actual_by_count[side]} | {*by_count_mults[side]}
            }
            for side in SIDES
        }
        event_mults = {
            side: derive_multipliers(
                actual_event[side], sim_event[side], event_mults[side]
            )
            for side in SIDES
        }

    calibration = SimCalibration(
        result=result_mults, event=event_mults, result_by_count=by_count_mults
    )
    calibration.save(
        Path(args.output),
        meta={
            "seasons": args.seasons,
            "matchup_season": args.matchup_season,
            "games": args.games,
            "pas": args.pas,
            "passes": args.passes,
            "outcome_run": pointer,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"\nWrote calibration -> {args.output}")
    for side in SIDES:
        print(
            f"  {side} aggregate result multipliers:",
            {k: round(v, 3) for k, v in sorted(result_mults[side].items())},
        )
    for side in SIDES:
        print(
            f"  {side} event multipliers: ",
            {k: round(v, 3) for k, v in sorted(event_mults[side].items())},
        )


if __name__ == "__main__":
    main()
