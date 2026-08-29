"""
Analyze and visualize an entire at-bat with pitch predictions.

This script loads all pitches from a single at-bat, generates predictions
for each pitch, creates a visualization, and calculates evaluation metrics.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier, Pool
from matplotlib.gridspec import GridSpec

from mlb.ml.features import IDX_TO_PITCH_TYPE, PitchFeatureEngine
from mlb.ml.mdn_location_model import BivariateMDN, get_location_density

# Mapping from pitch type codes to full names
PITCH_TYPE_NAMES = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "CH": "Changeup",
    "SL": "Slider",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "ST": "Sweeper",
    "FS": "Splitter",
    "KN": "Knuckleball",
    "OTHER": "Other",
    "NONE": "None",
}


def get_pitch_name(code: str) -> str:
    """Get full pitch name from code."""
    return PITCH_TYPE_NAMES.get(code, code)


@dataclass
class AtBatPrediction:
    """Container for a single pitch prediction within an at-bat."""
    pitch_number: int
    count: str  # "0-0", "1-2", etc.
    predicted_type: str
    predicted_type_prob: float
    actual_type: str
    type_correct: bool
    in_top_3: bool
    predicted_location: tuple[float, float]
    actual_location: tuple[float, float]
    location_error: float  # Euclidean distance
    type_probabilities: np.ndarray
    location_density: np.ndarray
    px_grid: np.ndarray
    pz_grid: np.ndarray


@dataclass
class AtBatEvaluation:
    """Evaluation metrics for an entire at-bat."""
    n_pitches: int
    n_correct: int
    n_in_top_3: int
    accuracy: float
    top_3_accuracy: float
    mean_location_error: float
    predictions: list[AtBatPrediction]

    # Context
    pitcher_name: str
    batter_name: str
    pitcher_hand: str
    batter_hand: str
    game_date: str
    outcome: str  # strikeout, walk, hit, etc.

    # Game situation
    home_team: str
    away_team: str
    inning: int
    inning_half: str
    outs_at_start: int
    runners_on: str  # e.g., "1st & 2nd", "Bases Empty"
    score_home: int
    score_away: int


def load_models(model_dir: Path, device: str = "mps"):
    """Load CatBoost and MDN models."""
    # Load CatBoost
    type_model = CatBoostClassifier()
    type_model.load_model(str(model_dir / "catboost" / "type_model.cbm"))

    with open(model_dir / "catboost" / "feature_info.json") as f:
        catboost_info = json.load(f)

    # Load MDN
    checkpoint = torch.load(
        model_dir / "mdn_location_model.pt",
        map_location=device,
        weights_only=False,
    )

    mdn_model = BivariateMDN(
        n_features=checkpoint["config"]["n_features"],
        hidden_dims=checkpoint["config"]["hidden_dims"],
        n_components=checkpoint["config"]["n_components"],
        dropout=checkpoint["config"]["dropout"],
    )
    mdn_model.load_state_dict(checkpoint["model_state_dict"])
    mdn_model = mdn_model.to(device)
    mdn_model.eval()

    return type_model, mdn_model, catboost_info, checkpoint


def predict_pitch(
    row: pl.DataFrame,
    type_model: CatBoostClassifier,
    mdn_model: BivariateMDN,
    catboost_info: dict,
    mdn_feature_cols: list[str],
    device: str = "mps",
) -> AtBatPrediction:
    """Make prediction for a single pitch."""

    def get_val(col, default=None):
        if col in row.columns:
            val = row[col][0]
            return val if val is not None else default
        return default

    # Prepare CatBoost features
    catboost_features = row.select(catboost_info["feature_columns"]).to_pandas()
    cat_indices = [
        i for i, col in enumerate(catboost_info["feature_columns"])
        if col in catboost_info["categorical_features"]
    ]

    # Prepare MDN features
    mdn_features = row.select(mdn_feature_cols).to_numpy().astype(np.float32)
    mdn_features = torch.tensor(mdn_features, dtype=torch.float32).to(device)

    # CatBoost prediction
    pool = Pool(catboost_features, cat_features=cat_indices)
    type_probs = type_model.predict_proba(pool)[0]
    predicted_idx = int(np.argmax(type_probs))
    predicted_type = IDX_TO_PITCH_TYPE.get(predicted_idx, "UNK")

    # Check if in top 3
    top_3_indices = np.argsort(type_probs)[::-1][:3]
    top_3_types = [IDX_TO_PITCH_TYPE.get(i, "UNK") for i in top_3_indices]

    # MDN prediction
    with torch.no_grad():
        params = mdn_model(mdn_features)
        location_point = mdn_model.get_expected_value(params)[0].cpu().numpy()

    # Get density grid
    px_grid, pz_grid, density = get_location_density(
        mdn_model,
        mdn_features,
        px_range=(-2.5, 2.5),
        pz_range=(0.5, 4.5),
        grid_size=50,
        n_samples=500,
    )

    # Get actual values
    actual_type = get_val("pitch_type_code", "UNK")
    actual_px = get_val("px", 0.0)
    actual_pz = get_val("pz", 2.5)

    # Calculate metrics
    type_correct = predicted_type == actual_type
    in_top_3 = actual_type in top_3_types
    location_error = np.sqrt(
        (location_point[0] - actual_px) ** 2 +
        (location_point[1] - actual_pz) ** 2
    )

    return AtBatPrediction(
        pitch_number=get_val("pitch_number", 1),
        count=f"{get_val('count_balls_before', 0)}-{get_val('count_strikes_before', 0)}",
        predicted_type=predicted_type,
        predicted_type_prob=type_probs[predicted_idx],
        actual_type=actual_type,
        type_correct=type_correct,
        in_top_3=in_top_3,
        predicted_location=(location_point[0], location_point[1]),
        actual_location=(actual_px, actual_pz),
        location_error=location_error,
        type_probabilities=type_probs,
        location_density=density,
        px_grid=px_grid,
        pz_grid=pz_grid,
    )


def evaluate_at_bat(
    at_bat_df: pl.DataFrame,
    type_model: CatBoostClassifier,
    mdn_model: BivariateMDN,
    catboost_info: dict,
    mdn_feature_cols: list[str],
    device: str = "mps",
) -> AtBatEvaluation:
    """Evaluate model on entire at-bat."""

    predictions = []

    # Sort by pitch number
    at_bat_df = at_bat_df.sort("pitch_number")

    for i in range(len(at_bat_df)):
        row = at_bat_df.slice(i, 1)
        pred = predict_pitch(
            row, type_model, mdn_model,
            catboost_info, mdn_feature_cols, device
        )
        predictions.append(pred)

    # Calculate summary metrics
    n_pitches = len(predictions)
    n_correct = sum(1 for p in predictions if p.type_correct)
    n_in_top_3 = sum(1 for p in predictions if p.in_top_3)
    mean_location_error = np.mean([p.location_error for p in predictions])

    # Get context from first row
    first_row = at_bat_df.head(1)

    def get_val(col, default=None):
        if col in first_row.columns:
            val = first_row[col][0]
            return val if val is not None else default
        return default

    # Get outcome from last row
    last_row = at_bat_df.tail(1)
    outcome = last_row["event"][0] if "event" in last_row.columns else "Unknown"

    # Build runners string
    on_first = get_val("is_runner_on_first", False)
    on_second = get_val("is_runner_on_second", False)
    on_third = get_val("is_runner_on_third", False)

    runners = []
    if on_first:
        runners.append("1st")
    if on_second:
        runners.append("2nd")
    if on_third:
        runners.append("3rd")
    runners_str = " & ".join(runners) if runners else "Bases Empty"

    return AtBatEvaluation(
        n_pitches=n_pitches,
        n_correct=n_correct,
        n_in_top_3=n_in_top_3,
        accuracy=n_correct / n_pitches if n_pitches > 0 else 0,
        top_3_accuracy=n_in_top_3 / n_pitches if n_pitches > 0 else 0,
        mean_location_error=mean_location_error,
        predictions=predictions,
        pitcher_name=get_val("pitcher_name", "Unknown"),
        batter_name=get_val("batter_name", "Unknown"),
        pitcher_hand=get_val("throw_side", "R"),
        batter_hand=get_val("bat_side", "R"),
        game_date=str(get_val("game_date", "")),
        outcome=outcome,
        home_team=get_val("home_team_name", "HOME"),
        away_team=get_val("away_team_name", "AWAY"),
        inning=get_val("inning", 1),
        inning_half="Top" if get_val("half_inning", "top") == "top" else "Bot",
        outs_at_start=get_val("outs", 0),
        runners_on=runners_str,
        score_home=get_val("home_score", 0),
        score_away=get_val("away_score", 0),
    )


def create_at_bat_visualization(
    evaluation: AtBatEvaluation,
    save_path: str | None = None,
) -> plt.Figure:
    """Create visualization for entire at-bat."""

    n_pitches = evaluation.n_pitches

    # Determine grid layout
    if n_pitches <= 3:
        n_cols = n_pitches
        n_rows = 1
    elif n_pitches <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = (n_pitches + 3) // 4

    # Create figure
    fig_width = 4 * n_cols + 2
    fig_height = 5 * n_rows + 3.5  # More vertical space for expanded header
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid with space for header and footer
    gs = GridSpec(n_rows + 2, n_cols, figure=fig,
                  height_ratios=[0.5] + [1] * n_rows + [0.25],
                  hspace=0.6, wspace=0.3)  # Increased hspace for label room

    # === HEADER ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    # Title: Pitcher vs Batter with handedness
    title = f"{evaluation.pitcher_name} ({evaluation.pitcher_hand}HP) vs {evaluation.batter_name} ({evaluation.batter_hand}HB)"
    ax_header.text(0.5, 0.82, title, fontsize=18, fontweight="bold",
                   ha="center", va="center", transform=ax_header.transAxes)

    # Game info line: Teams and score
    game_line = f"{evaluation.away_team} {evaluation.score_away} @ {evaluation.home_team} {evaluation.score_home}"
    ax_header.text(0.5, 0.55, game_line, fontsize=13,
                   ha="center", va="center", transform=ax_header.transAxes)

    # Situation line: Date, Inning, Outs, Runners
    situation_line = (
        f"{evaluation.game_date[:10] if len(evaluation.game_date) > 10 else evaluation.game_date} | "
        f"{evaluation.inning_half} {evaluation.inning} | "
        f"{evaluation.outs_at_start} out | "
        f"{evaluation.runners_on}"
    )
    ax_header.text(0.5, 0.28, situation_line, fontsize=11, color="gray",
                   ha="center", va="center", transform=ax_header.transAxes)

    # Outcome line
    outcome_line = f"Outcome: {evaluation.outcome}"
    ax_header.text(0.5, 0.05, outcome_line, fontsize=11, fontweight="bold",
                   ha="center", va="center", transform=ax_header.transAxes,
                   color="darkgreen" if "Hit" in str(evaluation.outcome) or "Double" in str(evaluation.outcome)
                         or "Triple" in str(evaluation.outcome) or "Home" in str(evaluation.outcome)
                   else "darkred" if "Strikeout" in str(evaluation.outcome) else "black")

    # === PITCH PANELS ===
    for i, pred in enumerate(evaluation.predictions):
        row = i // n_cols
        col = i % n_cols

        ax = fig.add_subplot(gs[row + 1, col])

        # Plot density
        extent = [pred.px_grid.min(), pred.px_grid.max(),
                  pred.pz_grid.min(), pred.pz_grid.max()]

        ax.imshow(
            pred.location_density,
            extent=extent,
            origin="lower",
            cmap="YlOrRd",
            aspect="auto",
            alpha=0.7,
        )

        # Draw strike zone
        sz_left, sz_right = -0.7083, 0.7083
        sz_bottom, sz_top = 1.5, 3.5
        ax.plot(
            [sz_left, sz_right, sz_right, sz_left, sz_left],
            [sz_bottom, sz_bottom, sz_top, sz_top, sz_bottom],
            "k-", linewidth=1.5
        )

        # Plot predicted location
        ax.scatter(
            pred.predicted_location[0], pred.predicted_location[1],
            c="blue", s=100, marker="o", edgecolors="white", linewidths=1.5,
            zorder=10
        )

        # Plot actual location
        ax.scatter(
            pred.actual_location[0], pred.actual_location[1],
            c="red", s=100, marker="x", linewidths=2,
            zorder=11
        )

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(0.5, 4.5)
        ax.set_aspect("equal")

        # Title with pitch info
        pred_name = get_pitch_name(pred.predicted_type)
        actual_name = get_pitch_name(pred.actual_type)

        # Color code title based on correctness
        if pred.type_correct:
            title_color = "green"
            symbol = "✓"
        elif pred.in_top_3:
            title_color = "orange"
            symbol = "~"
        else:
            title_color = "red"
            symbol = "✗"

        title_text = f"#{pred.pitch_number} ({pred.count})"
        ax.set_title(title_text, fontsize=11, fontweight="bold")

        # Add prediction/actual labels below
        label_text = f"Pred: {pred_name} ({pred.predicted_type_prob:.0%})\nActual: {actual_name} {symbol}"
        ax.text(0.5, -0.15, label_text, fontsize=9, ha="center", va="top",
                transform=ax.transAxes, color=title_color)

        # Remove tick labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # === FOOTER with metrics ===
    ax_footer = fig.add_subplot(gs[-1, :])
    ax_footer.axis("off")

    metrics_text = (
        f"Pitch Type: {evaluation.n_correct}/{evaluation.n_pitches} correct ({evaluation.accuracy:.0%}) | "
        f"Top-3: {evaluation.n_in_top_3}/{evaluation.n_pitches} ({evaluation.top_3_accuracy:.0%}) | "
        f"Avg Location Error: {evaluation.mean_location_error:.2f} ft"
    )
    ax_footer.text(0.5, 0.6, metrics_text, fontsize=11,
                   ha="center", va="center", transform=ax_footer.transAxes)

    legend_text = "Blue ○ = Predicted | Red ✗ = Actual | Green = Correct | Orange = In Top-3 | Red = Missed"
    ax_footer.text(0.5, 0.1, legend_text, fontsize=9, color="gray",
                   ha="center", va="center", transform=ax_footer.transAxes)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def main():
    import random

    # Configuration
    model_dir = Path("models/combined_20260118_140106")
    data_dir = Path("data/processed/livefeeds/2025")
    output_path = "at_bat_analysis.png"

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load a random game file
    game_files = sorted(data_dir.glob("*.parquet"))
    sample_game = random.choice(game_files)
    print(f"Loading game: {sample_game}")

    df = pl.read_parquet(sample_game)
    print(f"Loaded {len(df)} pitches")

    # Parse count_after_pitch, then compute count_before by shifting
    df = df.with_columns([
        pl.col("count_after_pitch").str.split("-").list.get(0).cast(pl.Int64).alias("balls_after"),
        pl.col("count_after_pitch").str.split("-").list.get(1).cast(pl.Int64).alias("strikes_after"),
    ])

    # Count before = previous pitch's count after (first pitch is always 0-0)
    # Use unique names so feature_engine.transform() doesn't overwrite them
    df = df.with_columns([
        pl.col("balls_after")
        .shift(1)
        .over(["game_pk", "at_bat_index"])
        .fill_null(0)
        .alias("count_balls_before"),
        pl.col("strikes_after")
        .shift(1)
        .over(["game_pk", "at_bat_index"])
        .fill_null(0)
        .alias("count_strikes_before"),
    ])

    # Find an at-bat with multiple pitches
    at_bat_counts = df.group_by(["game_pk", "at_bat_index"]).agg(
        pl.len().alias("n_pitches")
    ).filter(pl.col("n_pitches") >= 4)

    if len(at_bat_counts) == 0:
        print("No at-bats with 4+ pitches found")
        return

    # Pick a random at-bat
    random_idx = random.randint(0, len(at_bat_counts) - 1)
    selected = at_bat_counts.slice(random_idx, 1)
    game_pk = selected["game_pk"][0]
    at_bat_index = selected["at_bat_index"][0]

    at_bat_df = df.filter(
        (pl.col("game_pk") == game_pk) &
        (pl.col("at_bat_index") == at_bat_index)
    )

    print(f"\nSelected at-bat: {at_bat_df['pitcher_name'][0]} vs {at_bat_df['batter_name'][0]}")
    print(f"Pitches in at-bat: {len(at_bat_df)}")

    # Load models
    print("\nLoading models...")
    type_model, mdn_model, catboost_info, checkpoint = load_models(model_dir, device)

    # Load feature engine
    print("Preparing feature engine...")
    feature_engine = PitchFeatureEngine()

    train_dir = Path("data/processed/livefeeds/2023")
    train_files = list(train_dir.glob("*.parquet"))[:50]
    train_dfs = [pl.read_parquet(f) for f in train_files]
    train_sample = pl.concat(train_dfs, how="diagonal")
    print(f"Fitting feature engine on {len(train_sample)} pitches...")
    feature_engine.fit(train_sample)

    # Transform at-bat data
    at_bat_transformed = feature_engine.transform(at_bat_df)

    # Add prev_pitch_type for CatBoost
    at_bat_transformed = at_bat_transformed.with_columns([
        pl.col("pitch_type_code")
        .shift(1)
        .over(["game_pk", "at_bat_index"])
        .fill_null("NONE")
        .alias("prev_pitch_type"),
    ])

    # Evaluate at-bat
    print("\nEvaluating at-bat...")
    evaluation = evaluate_at_bat(
        at_bat_transformed,
        type_model,
        mdn_model,
        catboost_info,
        checkpoint["feature_columns"],
        device,
    )

    # Print results
    print(f"\n{'='*60}")
    print("AT-BAT EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Pitcher: {evaluation.pitcher_name}")
    print(f"Batter: {evaluation.batter_name}")
    print(f"Outcome: {evaluation.outcome}")
    print(f"\nPitches: {evaluation.n_pitches}")
    print(f"Pitch Type Accuracy: {evaluation.accuracy:.1%} ({evaluation.n_correct}/{evaluation.n_pitches})")
    print(f"Top-3 Accuracy: {evaluation.top_3_accuracy:.1%} ({evaluation.n_in_top_3}/{evaluation.n_pitches})")
    print(f"Mean Location Error: {evaluation.mean_location_error:.2f} ft")

    print("\nPitch-by-pitch:")
    for pred in evaluation.predictions:
        status = "✓" if pred.type_correct else ("~" if pred.in_top_3 else "✗")
        pred_name = get_pitch_name(pred.predicted_type)
        actual_name = get_pitch_name(pred.actual_type)
        print(f"  #{pred.pitch_number} ({pred.count}): Pred={pred_name} ({pred.predicted_type_prob:.0%}), "
              f"Actual={actual_name}, Loc Error={pred.location_error:.2f}ft {status}")

    # Create visualization
    print("\nCreating visualization...")
    fig = create_at_bat_visualization(evaluation, save_path=output_path)
    print(f"Saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
