"""Calibrate the game simulator against actual league rates.

Measures the simulated per-pitch result distribution PER COUNT and the
in-play event distribution, conditioned on batting side (top = away
batting) AND base state (windup = bases empty, stretch = runners on),
against actual league rates, and stores per-class multipliers in
``models/sim/sim_calibration.json``.

Conditioning on stretch matters: the outcome models' runner features
partly encode pitcher-quality selection effects; without pinning the
conditional rates, the simulator turns that correlation into a
runner -> uplift -> runner feedback loop and inflates totals. PAs are
measured inside a rolling base-out context so state frequencies match the
game loop. Two passes: the second corrects the residual.

    uv run python scripts/calibrate_sim.py --seasons 2023 2024 2025
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
    stretch_key,
)
from src.sim.count_machine import apply_pitch_result
from src.sim.lineups import lineup_from_feed
from src.sim.matchup import MatchupProviderFactory
from src.sim.pa import EVENT_CLASSES, RESULT_CLASSES
from src.sim.pitch_mix import COUNTS, PitchMixProfiles

LIVEFEED_ROOT = Path("data/raw/livefeeds")
SIDES = {"top": True, "bottom": False}
STRETCHES = {"windup": False, "stretch": True}


def count_key(balls: int, strikes: int) -> str:
    return f"{balls}-{strikes}"


def actual_rates(seasons: list[int]) -> tuple[dict, dict, dict, dict, dict]:
    """Actual league rates.

    Returns aggregate result rates, per (side, stretch, count) result
    rates, per (side, stretch) event rates, and the stretch-BLIND
    per (side, count) result and per side event rates used by the
    default shared-multiplier derivation.
    """
    frame = build_training_frame(load_pitches(seasons)).with_columns(
        (
            (pl.col("runner_on_first") + pl.col("runner_on_second") + pl.col("runner_on_third")) > 0
        ).alias("stretch_state")
    )
    result_agg: dict[str, dict[str, float]] = {}
    result_cells: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    event_cells: dict[str, dict[str, dict[str, float]]] = {}
    combined_cells: dict[str, dict[str, dict[str, float]]] = {}
    combined_events: dict[str, dict[str, float]] = {}
    for side, is_top in SIDES.items():
        side_rows = frame.filter(
            (pl.col("is_top_half") == int(is_top))
            & pl.col("label_result").is_not_null()
        )
        counts = side_rows["label_result"].value_counts()
        result_agg[side] = {
            row[0]: row[1] / side_rows.height for row in counts.rows()
        }
        combined_cells[side] = {}
        for balls, strikes in COUNTS:
            cell = side_rows.filter(
                (pl.col("balls_before") == balls)
                & (pl.col("strikes_before") == strikes)
            )
            if cell.height == 0:
                continue
            counts = cell["label_result"].value_counts()
            combined_cells[side][count_key(balls, strikes)] = {
                row[0]: row[1] / cell.height for row in counts.rows()
            }
        event_rows = side_rows.filter(pl.col("label_event").is_not_null())
        counts = event_rows["label_event"].value_counts()
        combined_events[side] = {
            row[0]: row[1] / event_rows.height for row in counts.rows()
        }
        result_cells[side] = {}
        event_cells[side] = {}
        for skey, stretch in STRETCHES.items():
            stretch_rows = side_rows.filter(pl.col("stretch_state") == stretch)
            result_cells[side][skey] = {}
            for balls, strikes in COUNTS:
                cell = stretch_rows.filter(
                    (pl.col("balls_before") == balls)
                    & (pl.col("strikes_before") == strikes)
                )
                if cell.height == 0:
                    continue
                counts = cell["label_result"].value_counts()
                result_cells[side][skey][count_key(balls, strikes)] = {
                    row[0]: row[1] / cell.height for row in counts.rows()
                }
            event_rows = stretch_rows.filter(pl.col("label_event").is_not_null())
            counts = event_rows["label_event"].value_counts()
            event_cells[side][skey] = {
                row[0]: row[1] / event_rows.height for row in counts.rows()
            }
    return result_agg, result_cells, event_cells, combined_cells, combined_events


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
) -> tuple[dict, dict, dict, dict, dict]:
    """Simulated aggregate/per-cell result and event rates.

    PAs run inside a rolling base-out context so the windup/stretch state
    frequencies match the game loop's.
    """
    from src.sim.base_out import BaseOutEngine

    rng = random.Random(seed)
    engine = BaseOutEngine.load(seed=seed)
    result_agg = {side: Counter() for side in SIDES}
    result_cells: dict[str, dict[str, dict[str, Counter]]] = {
        side: {
            skey: {count_key(b, s): Counter() for b, s in COUNTS}
            for skey in STRETCHES
        }
        for side in SIDES
    }
    event_cells: dict[str, dict[str, Counter]] = {
        side: {skey: Counter() for skey in STRETCHES} for side in SIDES
    }
    for pitcher, batter, is_top in matchups:
        side = "top" if is_top else "bottom"
        runners, outs = 0, 0
        for _ in range(pas_per_matchup):
            stretch = runners != 0
            skey = stretch_key(stretch)
            provider = factory(pitcher, batter, is_top, stretch)
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
                result_agg[side][result] += 1
                result_cells[side][skey][count_key(balls, strikes)][result] += 1
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
                    event_cells[side][skey][event] += 1
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

    combined_counter: dict[str, dict[str, Counter]] = {
        side: {} for side in SIDES
    }
    combined_event_counter: dict[str, Counter] = {side: Counter() for side in SIDES}
    for side in SIDES:
        for skey in STRETCHES:
            for key, counter in result_cells[side][skey].items():
                combined_counter[side].setdefault(key, Counter()).update(counter)
            combined_event_counter[side].update(event_cells[side][skey])

    return (
        {side: normalize(result_agg[side]) for side in SIDES},
        {
            side: {
                skey: {key: normalize(c) for key, c in cells.items()}
                for skey, cells in result_cells[side].items()
            }
            for side in SIDES
        },
        {
            side: {skey: normalize(c) for skey, c in event_cells[side].items()}
            for side in SIDES
        },
        {
            side: {key: normalize(c) for key, c in combined_counter[side].items()}
            for side in SIDES
        },
        {side: normalize(combined_event_counter[side]) for side in SIDES},
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
    parser.add_argument(
        "--stretch-split",
        action="store_true",
        help=(
            "Derive separate multipliers for windup/stretch cells. Off by "
            "default: the split measurably regressed game-level accuracy "
            "(sparser sim cells -> noisier multipliers; stretch-specific "
            "event boosts compound run value)."
        ),
    )
    args = parser.parse_args()

    print(f"Computing actual league rates for {args.seasons}...")
    start = time.perf_counter()
    (
        actual_agg,
        actual_cells,
        actual_events,
        actual_combined,
        actual_events_combined,
    ) = actual_rates(args.seasons)
    print(f"...done in {time.perf_counter() - start:.0f}s")

    from src.sim.artifacts import ensure_sim_artifacts
    from src.sim.slate import resolve_outcome_model_dirs

    ensure_sim_artifacts()
    run_dir, profiles_dir = resolve_outcome_model_dirs("auto", tracking_uri=None)
    print(f"Loaded outcome models from {run_dir}")
    predictor = PitchOutcomePredictor(run_dir, profiles_dir=profiles_dir)
    mix = PitchMixProfiles.load(seed=args.seed)
    matchups = sample_matchups(args.matchup_season, args.games, args.seed)
    print(f"Sampled {len(matchups)} matchups from {args.games} games")

    result_mults: dict[str, dict[str, float]] = {side: {} for side in SIDES}
    cell_mults: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        side: {skey: {} for skey in STRETCHES} for side in SIDES
    }
    event_mults: dict[str, dict[str, float]] = {side: {} for side in SIDES}
    event_stretch_mults: dict[str, dict[str, dict[str, float]]] = {
        side: {skey: {} for skey in STRETCHES} for side in SIDES
    }
    for pass_index in range(1, args.passes + 1):
        calibration = SimCalibration(
            result=result_mults,
            event=event_mults,
            result_by_count=cell_mults,
            event_by_stretch=event_stretch_mults,
        )
        factory = MatchupProviderFactory(
            predictor,
            mix,
            season=args.matchup_season,
            seed=args.seed,
            calibration=calibration,
        )
        sim_agg, sim_cells, sim_events, sim_combined, sim_events_combined = (
            simulate_rates(factory, matchups, args.pas, args.seed)
        )
        print(f"\nPass {pass_index} simulated rates vs actual (aggregate):")
        for side in SIDES:
            print(
                f"  {side:6s} in_play sim {sim_agg[side].get('in_play', 0):.3f}"
                f"/act {actual_agg[side].get('in_play', 0):.3f}"
                f"  hbp sim {sim_agg[side].get('hit_by_pitch', 0):.4f}"
                f"/act {actual_agg[side].get('hit_by_pitch', 0):.4f}"
            )
        result_mults = {
            side: derive_multipliers(actual_agg[side], sim_agg[side], result_mults[side])
            for side in SIDES
        }
        if args.stretch_split:
            cell_mults = {
                side: {
                    skey: {
                        key: derive_multipliers(
                            actual_cells[side][skey].get(key, {}),
                            sim_cells[side][skey].get(key, {}),
                            cell_mults[side][skey].get(key, {}),
                        )
                        for key in {*actual_cells[side][skey]} | {*cell_mults[side][skey]}
                    }
                    for skey in STRETCHES
                }
                for side in SIDES
            }
            event_stretch_mults = {
                side: {
                    skey: derive_multipliers(
                        actual_events[side][skey],
                        sim_events[side][skey],
                        event_stretch_mults[side][skey],
                    )
                    for skey in STRETCHES
                }
                for side in SIDES
            }
        else:
            # Shared multipliers across stretch states (round-A behavior):
            # derive from the stretch-blind combined rates, write under
            # both stretch keys so provider lookups behave identically.
            shared = {
                side: {
                    key: derive_multipliers(
                        actual_combined[side].get(key, {}),
                        sim_combined[side].get(key, {}),
                        cell_mults[side]["windup"].get(key, {}),
                    )
                    for key in {
                        *actual_combined[side],
                        *cell_mults[side]["windup"],
                    }
                }
                for side in SIDES
            }
            cell_mults = {
                side: {skey: shared[side] for skey in STRETCHES} for side in SIDES
            }
            event_stretch_mults = {
                side: {skey: {} for skey in STRETCHES} for side in SIDES
            }
        event_mults = {
            side: derive_multipliers(
                actual_events_combined[side],
                sim_events_combined[side],
                event_mults[side],
            )
            for side in SIDES
        }

    calibration = SimCalibration(
        result=result_mults,
        event=event_mults,
        result_by_count=cell_mults,
        event_by_stretch=event_stretch_mults,
    )
    calibration.save(
        Path(args.output),
        meta={
            "seasons": args.seasons,
            "matchup_season": args.matchup_season,
            "games": args.games,
            "pas": args.pas,
            "passes": args.passes,
            "outcome_run": run_dir.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"\nWrote calibration -> {args.output}")
    for side in SIDES:
        for skey in STRETCHES:
            print(
                f"  {side}/{skey} event multipliers:",
                {k: round(v, 3) for k, v in sorted(event_stretch_mults[side][skey].items())},
            )


if __name__ == "__main__":
    main()
