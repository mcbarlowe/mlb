"""Evaluate the production pitch-type model against its operational baseline.

Two questions, per the outcome-model evaluation discipline:

1. Does the LSTM beat the pitcher's empirical count/stretch pitch mix
   (``P(type | pitcher, count, runners-on)`` with league shrinkage)?
   Both are scored on IDENTICAL pitch rows.
2. Is the model marginally calibrated across seasons, including seasons
   after its training window (the off-window cliff check that exposed the
   CatBoost CTR pathology)?

    uv run python scripts/evaluate_pitch_model.py --seasons 2023 2024 2025

Pulls the pitch mix tables through the shared MLflow store when missing.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import numpy as np
import polars as pl
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.features import PITCH_TYPE_CODES
from src.ml.pitch_predictor import PitchPredictor
from src.ml.postgres_data import load_pitches_from_postgres
from src.ml.run_dirs import resolve_pitch_type_run_dir
from src.sim.artifacts import ensure_sim_artifacts
from src.sim.pitch_mix import PitchMixProfiles

DEFAULT_MODEL_DIR = "auto"
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pitch-type model.")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--limit-games", type=int, default=None)
    return parser.parse_args()


def add_pre_pitch_count_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add pre-pitch count columns from historical post-pitch count rows."""
    at_bat_key = ["game_pk", "at_bat_index"]
    df = df.sort(
        at_bat_key + ["pitch_start_time", "pitch_number"],
        nulls_last=True,
    )
    if {"_balls_before", "_strikes_before"}.issubset(df.columns):
        return df.with_columns(
            (
                pl.col("is_runner_on_first").fill_null(False)
                | pl.col("is_runner_on_second").fill_null(False)
                | pl.col("is_runner_on_third").fill_null(False)
            ).alias("_stretch")
        )
    counts = pl.col("count_after_pitch").str.split_exact("-", 1)
    return (
        df.with_columns(
            counts.struct.field("field_0")
            .cast(pl.Int64, strict=False)
            .alias("_balls_after"),
            counts.struct.field("field_1")
            .cast(pl.Int64, strict=False)
            .alias("_strikes_after"),
        )
        .with_columns(
            pl.col("_balls_after")
            .shift(1)
            .over(at_bat_key)
            .fill_null(0)
            .alias("_balls_before"),
            pl.col("_strikes_after")
            .shift(1)
            .over(at_bat_key)
            .fill_null(0)
            .alias("_strikes_before"),
            (
                pl.col("is_runner_on_first").fill_null(False)
                | pl.col("is_runner_on_second").fill_null(False)
                | pl.col("is_runner_on_third").fill_null(False)
            ).alias("_stretch"),
        )
    )


def historical_rows_as_live_inputs(raw: pl.DataFrame) -> pl.DataFrame:
    """Rewrite historical rows so feature engineering sees the pre-pitch count.

    ``mlb.pitches.count_after_pitch`` is the count after an archived pitch.
    Live next-pitch rows use that same raw column for the count before the
    pending pitch. Rewriting keeps the evaluator aligned with deployed inputs.
    """
    frame = add_pre_pitch_count_columns(raw)
    return frame.with_columns(
        pl.concat_str(["_balls_before", "_strikes_before"], separator="-").alias(
            "count_after_pitch"
        )
    )


def iter_sequences(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    max_seq_len: int,
) -> Iterator[dict]:
    """Yield at-bat sequences with aligned context for the baseline."""
    df = add_pre_pitch_count_columns(df)

    for _, group in df.group_by(["game_pk", "at_bat_index"], maintain_order=True):
        features = group.select(feature_columns).cast(pl.Float32).to_numpy()
        targets = group.select(target_columns).cast(pl.Float32).to_numpy()
        if len(features) == 0 or np.isnan(features).any():
            continue
        aux = {
            "pitcher_id": group["pitcher_id"].to_list(),
            "balls": group["_balls_before"].to_list(),
            "strikes": group["_strikes_before"].to_list(),
            "stretch": group["_stretch"].to_list(),
        }
        if len(features) > max_seq_len:
            features = features[:max_seq_len]
            targets = targets[:max_seq_len]
            aux = {k: v[:max_seq_len] for k, v in aux.items()}
        yield {
            "features": features,
            "targets": targets,
            "aux": aux,
            "length": len(features),
        }


def batched(items: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    """Yield bounded batches without retaining a full season of sequences."""
    while batch := list(islice(items, batch_size)):
        yield batch


def evaluate_season(
    predictor: PitchPredictor,
    mix: PitchMixProfiles,
    raw: pl.DataFrame,
    batch_size: int,
    max_seq_len: int,
    device: str,
) -> dict:
    engine = predictor.feature_engine
    assert engine is not None
    raw = historical_rows_as_live_inputs(raw)
    df = engine.transform(raw)
    sequences = iter_sequences(
        df, engine.get_feature_columns(), engine.get_target_columns(), max_seq_len
    )

    model = predictor.lstm_model
    assert model is not None
    torch_device = torch.device(device)
    model.to(torch_device).eval()

    n_classes = len(PITCH_TYPE_CODES)
    lstm_log_loss = 0.0
    lstm_top1 = 0
    base_log_loss = 0.0
    base_top1 = 0
    n_rows = 0
    n_invalid_context = 0
    n_unlabeled = 0
    predicted_mass = np.zeros(n_classes)
    actual_counts = np.zeros(n_classes)
    mix_cache: dict[tuple, dict[str, float]] = {}

    with torch.no_grad():
        for batch in batched(sequences, batch_size):
            lengths = torch.tensor([s["length"] for s in batch])
            max_len = int(lengths.max())
            feature_dim = batch[0]["features"].shape[1]
            padded = torch.zeros(len(batch), max_len, feature_dim)
            for i, s in enumerate(batch):
                padded[i, : s["length"]] = torch.from_numpy(s["features"])
            mask = torch.arange(max_len)[None, :] < lengths[:, None]

            logits, _ = model(
                padded.to(torch_device), lengths.to(torch_device), mask.to(torch_device)
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i, s in enumerate(batch):
                seq_probs = probs[i, : s["length"]]
                target_values = s["targets"][:, 0]
                for j in range(s["length"]):
                    target_value = float(target_values[j])
                    if not math.isfinite(target_value):
                        n_unlabeled += 1
                        continue
                    truth = int(target_value)
                    if truth < 0 or truth >= n_classes:
                        n_unlabeled += 1
                        continue
                    balls = int(s["aux"]["balls"][j])
                    strikes = int(s["aux"]["strikes"][j])
                    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
                        n_invalid_context += 1
                        continue
                    p = seq_probs[j]
                    lstm_log_loss += -math.log(max(p[truth], EPS))
                    lstm_top1 += int(np.argmax(p) == truth)
                    predicted_mass += p
                    actual_counts[truth] += 1

                    key = (
                        int(s["aux"]["pitcher_id"][j]),
                        balls,
                        strikes,
                        bool(s["aux"]["stretch"][j]),
                    )
                    dist = mix_cache.get(key)
                    if dist is None:
                        dist = mix.type_distribution(*key[:3], stretch=key[3])
                        mix_cache[key] = dist
                    truth_code = PITCH_TYPE_CODES[truth]
                    base_log_loss += -math.log(max(dist.get(truth_code, 0.0), EPS))
                    base_top1 += int(
                        max(dist.items(), key=lambda kv: kv[1])[0] == truth_code
                    )
                    n_rows += 1

    return {
        "n_rows": n_rows,
        "n_invalid_context": n_invalid_context,
        "n_unlabeled": n_unlabeled,
        "lstm_log_loss": lstm_log_loss / n_rows,
        "lstm_top1": lstm_top1 / n_rows,
        "baseline_log_loss": base_log_loss / n_rows,
        "baseline_top1": base_top1 / n_rows,
        "predicted_marginal": predicted_mass / n_rows,
        "actual_marginal": actual_counts / n_rows,
    }


def main() -> None:
    args = parse_args()
    ensure_sim_artifacts()
    mix = PitchMixProfiles.load(seed=0)
    model_dir = resolve_pitch_type_run_dir(args.model_dir)
    print(f"Loading model from {model_dir}...")
    predictor = PitchPredictor.load_lstm(model_dir, device=args.device)

    summary = {}
    for season in args.seasons:
        start = time.perf_counter()
        print(f"\nLoading season {season}...")
        raw = load_pitches_from_postgres([str(season)])
        if args.limit_games:
            keep = raw["game_pk"].unique().sort()[: args.limit_games]
            raw = raw.filter(pl.col("game_pk").is_in(keep.implode()))
        print(f"  {raw.height:,} pitches; evaluating...")
        result = evaluate_season(
            predictor, mix, raw, args.batch_size, args.max_seq_len, args.device
        )
        summary[season] = result
        print(
            f"  rows={result['n_rows']:,}"
            f" invalid_context={result['n_invalid_context']:,}"
            f"  LSTM log_loss={result['lstm_log_loss']:.4f} top1={result['lstm_top1']:.3f}"
            f"  | mix baseline log_loss={result['baseline_log_loss']:.4f}"
            f" top1={result['baseline_top1']:.3f}"
            f"  ({time.perf_counter() - start:.0f}s)"
        )

    print("\n=== Marginal calibration by season (predicted vs actual share) ===")
    gaps = defaultdict(dict)
    for season, result in summary.items():
        for i, code in enumerate(PITCH_TYPE_CODES):
            gaps[code][season] = (
                result["predicted_marginal"][i],
                result["actual_marginal"][i],
            )
    header = "  ".join(f"{season:>13}" for season in summary)
    print(f"{'type':6s} {header}   (pred/actual)")
    for code, by_season in gaps.items():
        cells = "  ".join(
            f"{pred:.3f}/{act:.3f}" for pred, act in by_season.values()
        )
        print(f"{code:6s} {cells}")

    print("\n=== Verdict ===")
    for season, result in summary.items():
        edge = result["baseline_log_loss"] - result["lstm_log_loss"]
        max_gap = float(
            np.max(np.abs(result["predicted_marginal"] - result["actual_marginal"]))
        )
        print(
            f"{season}: LSTM edge over mix baseline {edge:+.4f} log loss"
            f" ({'beats' if edge > 0 else 'LOSES to'} baseline);"
            f" max marginal gap {max_gap:.3f}"
        )


if __name__ == "__main__":
    main()
