"""Train, validate, serialize, and register Bayesian pitcher-prop components.

The default chronology is locked to 2015-2023 training and 2024 validation.
Versions register as MLflow challengers; ``--set-champion`` promotes only
components that beat the league baseline on both MAE and count log loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlb.pitcher_props.data import load_pitcher_starts, seasons
from mlb.pitcher_props.evaluation import EvaluationReport, evaluate_walk_forward
from mlb.pitcher_props.mlflow_registry import register_pitcher_prop_models
from mlb.pitcher_props.model import (
    MODEL_CONTRACT_VERSION,
    PitcherPropConfig,
    PitcherPropModel,
    PitcherStart,
)

DEFAULT_MODEL_OUTPUT = Path("models/pitcher_props") / f"{MODEL_CONTRACT_VERSION}.json"


def train_and_evaluate(
    training: list[PitcherStart],
    validation: list[PitcherStart],
    *,
    config: PitcherPropConfig,
    evaluation_draws: int,
    evaluation_seed: int,
) -> tuple[PitcherPropModel, EvaluationReport]:
    model = PitcherPropModel.fit(training, config=config)
    report = evaluate_walk_forward(
        model,
        validation,
        draws_per_start=evaluation_draws,
        seed=evaluation_seed,
    )
    return model, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--train-through", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--recency-half-life-days", type=float, default=365.0)
    parser.add_argument("--workload-prior-starts", type=float, default=25.0)
    parser.add_argument("--rate-prior-batters", type=float, default=250.0)
    parser.add_argument("--opponent-weight", type=float, default=0.5)
    parser.add_argument("--evaluation-draws", type=int, default=1_000)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="mlb-pitcher-prop-models")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--set-champion", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use bounded chronological samples for a local smoke run",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.start_season > args.train_through:
        raise SystemExit("--start-season cannot exceed --train-through")
    if args.validation_season <= args.train_through:
        raise SystemExit("--validation-season must follow --train-through")

    all_starts = load_pitcher_starts(
        start_season=args.start_season,
        end_season=args.validation_season,
    )
    training = seasons(
        all_starts,
        range(args.start_season, args.train_through + 1),
    )
    validation = seasons(all_starts, [args.validation_season])
    if args.quick:
        training = training[-2_000:]
        validation = validation[:200]
    if not training or not validation:
        raise SystemExit("training and validation seasons must both contain starts")

    config = PitcherPropConfig(
        recency_half_life_days=args.recency_half_life_days,
        workload_prior_starts=args.workload_prior_starts,
        rate_prior_batters=args.rate_prior_batters,
        opponent_weight=args.opponent_weight,
        min_draws=1_000 if args.quick else 4_000,
        max_draws=4_000 if args.quick else 40_000,
        draw_batch=1_000 if args.quick else 4_000,
        mc_tolerance=0.01 if args.quick else 0.005,
    )
    model, report = train_and_evaluate(
        training,
        validation,
        config=config,
        evaluation_draws=args.evaluation_draws,
        evaluation_seed=args.evaluation_seed,
    )
    model_path = model.save(args.model_output)
    report_path = model_path.with_suffix(".evaluation.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    training_cutoff = model.trained_through
    if training_cutoff is None:
        raise RuntimeError("fitted pitcher model has no training cutoff")

    print(f"training starts: {model.training_starts:,}")
    print(f"training cutoff: {training_cutoff.isoformat()}")
    print(f"validation starts: {report.starts:,} ({report.validation_season})")
    for market, metrics in sorted(report.components.items()):
        print(
            f"{market}: MAE {metrics.mae:.4f} "
            f"(league {metrics.league_mae:.4f}, delta {metrics.mae_improvement:+.4f}); "
            f"log loss {metrics.count_log_loss:.4f} "
            f"(delta {metrics.log_loss_improvement:+.4f})"
        )
    print(f"model artifact: {model_path}")
    print(f"evaluation artifact: {report_path}")

    if args.skip_register:
        return
    registrations = register_pitcher_prop_models(
        model,
        report,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        set_champion=args.set_champion,
    )
    for registration in registrations:
        action = "existing" if registration.skipped_existing else "registered"
        print(
            f"{action} {registration.registered_model_name} "
            f"v{registration.version} ({registration.market}, "
            f"gate={registration.promotion_gate})"
        )


if __name__ == "__main__":
    main()
