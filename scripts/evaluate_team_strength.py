"""Evaluate the leak-free team-strength win model on a held-out season."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sim.team_strength import (
    FEATURE_NAMES,
    TeamStrengthPredictor,
    fit_strength_predictor,
    load_completed_games,
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    brier: float
    log_loss: float
    pick_accuracy: float
    mean_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pregame home-win probabilities on a held-out season."
    )
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument(
        "--train-seasons",
        type=int,
        nargs="*",
        default=None,
        help="Defaults to the four seasons immediately before --test-season.",
    )
    parser.add_argument(
        "--home-rate-baseline",
        type=float,
        default=0.543,
        help="Pregame constant home-win probability baseline.",
    )
    return parser.parse_args()


def score(probabilities: np.ndarray, outcomes: np.ndarray) -> ProbabilityMetrics:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return ProbabilityMetrics(
        brier=float(np.mean((clipped - outcomes) ** 2)),
        log_loss=float(
            np.mean(
                -(
                    outcomes * np.log(clipped)
                    + (1.0 - outcomes) * np.log(1.0 - clipped)
                )
            )
        ),
        pick_accuracy=float(np.mean((clipped >= 0.5) == outcomes.astype(bool))),
        mean_probability=float(np.mean(clipped)),
    )


def predict_frame(predictor: TeamStrengthPredictor, frame) -> np.ndarray:
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    coefficients = np.asarray(predictor.coefficients, dtype=float)
    log_odds = predictor.intercept + values @ coefficients
    return 1.0 / (1.0 + np.exp(-log_odds))


def print_metrics(label: str, metrics: ProbabilityMetrics) -> None:
    print(
        f"{label:<22} "
        f"Brier={metrics.brier:.4f}  "
        f"log_loss={metrics.log_loss:.4f}  "
        f"accuracy={metrics.pick_accuracy:.1%}  "
        f"mean_p(home)={metrics.mean_probability:.3f}"
    )


def main() -> None:
    args = parse_args()
    train_seasons = args.train_seasons or list(
        range(args.test_season - 4, args.test_season)
    )
    if any(season >= args.test_season for season in train_seasons):
        raise ValueError("Training seasons must precede the held-out test season")
    if not 0.0 < args.home_rate_baseline < 1.0:
        raise ValueError("--home-rate-baseline must be between zero and one")

    games = load_completed_games(
        start_season=args.start_season,
        end_season=args.test_season,
    )
    predictor, feature_frame = fit_strength_predictor(
        games,
        prediction_season=args.test_season,
        train_seasons=train_seasons,
    )
    test = feature_frame[feature_frame["season"] == args.test_season]
    if test.empty:
        raise ValueError(f"No held-out games found for {args.test_season}")
    outcomes = test["home_won"].to_numpy(dtype=float)
    model = score(predict_frame(predictor, test), outcomes)
    home_rate = score(
        np.full(len(test), args.home_rate_baseline, dtype=float), outcomes
    )
    coin_flip = score(np.full(len(test), 0.5, dtype=float), outcomes)

    print(f"Held-out season: {args.test_season} ({len(test):,} games)")
    print(f"Training seasons: {', '.join(map(str, train_seasons))}")
    print(f"Actual home-win rate: {float(np.mean(outcomes)):.3f}")
    print_metrics("Team-strength model", model)
    print_metrics("League home rate", home_rate)
    print_metrics("Coin flip", coin_flip)
    print("Coefficients:")
    for name, coefficient in zip(FEATURE_NAMES, predictor.coefficients):
        print(f"  {name:<22} {coefficient:+.6f}")
    print(f"  {'intercept':<22} {predictor.intercept:+.6f}")

    checks = {
        "Brier": model.brier < home_rate.brier,
        "log loss": model.log_loss < home_rate.log_loss,
        "pick accuracy": model.pick_accuracy > home_rate.pick_accuracy,
    }
    print("Gate: " + ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items()))
    if not all(checks.values()):
        raise SystemExit(1)

    brier_gain = home_rate.brier - model.brier
    log_loss_gain = home_rate.log_loss - model.log_loss
    accuracy_gain = model.pick_accuracy - home_rate.pick_accuracy
    if not all(math.isfinite(value) for value in (brier_gain, log_loss_gain, accuracy_gain)):
        raise RuntimeError("Non-finite evaluation improvement")
    print(
        "Promotion gate passed: "
        f"Brier {brier_gain:+.4f}, log loss {log_loss_gain:+.4f}, "
        f"accuracy {accuracy_gain:+.1%} versus league home rate."
    )


if __name__ == "__main__":
    main()
