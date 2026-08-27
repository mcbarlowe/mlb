"""
Create a sample pitch card from 2025 data.

This script loads trained models and generates a pitch card visualization
for a sample pitch from the 2025 season.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier, Pool

from src.ml.features import IDX_TO_PITCH_TYPE, PitchFeatureEngine
from src.ml.mdn_location_model import BivariateMDN, get_location_density
from src.ml.pitch_predictor import GameContext, PitchPrediction

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


def main():
    import random

    # Configuration
    model_dir = Path("models/combined_20260118_140106")
    data_dir = Path("data/processed/livefeeds/2025")
    output_path = "sample_pitch_card.png"

    # Pick a random game file
    game_files = sorted(data_dir.glob("*.parquet"))
    if not game_files:
        raise FileNotFoundError("No 2025 game files found")

    # Use random game file
    sample_game = random.choice(game_files)
    print(f"Loading game: {sample_game}")

    # Load game data
    df = pl.read_parquet(sample_game)
    print(f"Loaded {len(df)} pitches")

    # Parse the count_after_pitch column to extract balls and strikes
    # Format is like "1-2" (balls-strikes)
    df = df.with_columns([
        pl.col("count_after_pitch").str.split("-").list.get(0).cast(pl.Int64).alias("balls"),
        pl.col("count_after_pitch").str.split("-").list.get(1).cast(pl.Int64).alias("strikes"),
    ])

    # Select a random pitch from the game
    random_idx = random.randint(0, len(df) - 1)
    row = df.slice(random_idx, 1)
    print(f"Selected pitch: {row['pitcher_name'][0]} vs {row['batter_name'][0]}")
    print(f"Count: {row['balls'][0]}-{row['strikes'][0]}")
    print(f"Actual pitch: {row['pitch_type_code'][0]}")

    # Load CatBoost pitch type model
    print("\nLoading CatBoost model...")
    type_model = CatBoostClassifier()
    type_model.load_model(str(model_dir / "catboost" / "type_model.cbm"))

    # Load CatBoost feature info
    with open(model_dir / "catboost" / "feature_info.json") as f:
        catboost_info = json.load(f)

    # Load MDN model
    print("Loading MDN model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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

    # Initialize feature engine and fit on training data
    print("\nPreparing feature engine...")
    feature_engine = PitchFeatureEngine()

    # Load training data to fit the engine (use 2023 for simplicity)
    train_dir = Path("data/processed/livefeeds/2023")
    train_files = list(train_dir.glob("*.parquet"))[:50]  # Sample for speed
    train_dfs = [pl.read_parquet(f) for f in train_files]
    train_sample = pl.concat(train_dfs, how="diagonal")
    print(f"Fitting feature engine on {len(train_sample)} pitches...")
    feature_engine.fit(train_sample)

    # Transform the single pitch
    transformed = feature_engine.transform(row)

    # Add prev_pitch_type as string (CatBoost uses this as categorical)
    transformed = transformed.with_columns([
        pl.col("pitch_type_code")
        .shift(1)
        .over(["game_pk", "at_bat_index"])
        .fill_null("NONE")
        .alias("prev_pitch_type"),
    ])

    # Prepare CatBoost features
    catboost_features = transformed.select(catboost_info["feature_columns"]).to_pandas()
    cat_indices = [
        i for i, col in enumerate(catboost_info["feature_columns"])
        if col in catboost_info["categorical_features"]
    ]

    # Prepare MDN features
    mdn_feature_cols = checkpoint["feature_columns"]
    mdn_features = transformed.select(mdn_feature_cols).to_numpy().astype(np.float32)
    mdn_features = torch.tensor(mdn_features, dtype=torch.float32).to(device)

    # Make predictions
    print("\nMaking predictions...")

    # CatBoost pitch type prediction
    pool = Pool(catboost_features, cat_features=cat_indices)
    type_probs = type_model.predict_proba(pool)[0]
    predicted_idx = int(np.argmax(type_probs))
    predicted_type = IDX_TO_PITCH_TYPE.get(predicted_idx, "UNK")

    # Top 3 pitch types
    top_3_indices = np.argsort(type_probs)[::-1][:3]
    top_3_types = [
        (IDX_TO_PITCH_TYPE.get(i, "UNK"), type_probs[i])
        for i in top_3_indices
    ]

    print(f"Predicted pitch type: {predicted_type}")
    print(f"Top 3: {top_3_types}")

    # MDN location prediction
    with torch.no_grad():
        params = mdn_model(mdn_features)
        location_point = mdn_model.get_expected_value(params)[0].cpu().numpy()
        location_mode = mdn_model.get_mode(params)[0].cpu().numpy()
        mixture_weights = params["pi"][0].cpu().numpy()
        mixture_means = params["mu"][0].cpu().numpy()
        mixture_stds = params["sigma"][0].cpu().numpy()

    # Get location density
    px_grid, pz_grid, density = get_location_density(
        mdn_model,
        mdn_features,
        px_range=(-2.5, 2.5),
        pz_range=(0.5, 4.5),
        grid_size=100,
        n_samples=1000,
    )

    print(f"Predicted location: ({location_point[0]:.2f}, {location_point[1]:.2f})")

    # Create PitchPrediction object
    prediction = PitchPrediction(
        type_probabilities=type_probs,
        predicted_type_idx=predicted_idx,
        predicted_type=predicted_type,
        top_3_types=top_3_types,
        location_point=location_point,
        location_mode=location_mode,
        px_grid=px_grid,
        pz_grid=pz_grid,
        location_density=density,
        mixture_weights=mixture_weights,
        mixture_means=mixture_means,
        mixture_stds=mixture_stds,
    )

    # Build game context
    def get_val(col, default=None):
        if col in row.columns:
            val = row[col][0]
            return val if val is not None else default
        return default

    # Build runners string using correct column names
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

    context = GameContext(
        pitcher_name=get_val("pitcher_name", "Unknown"),
        batter_name=get_val("batter_name", "Unknown"),
        pitcher_hand=get_val("throw_side", "R"),
        batter_hand=get_val("bat_side", "R"),
        home_team=get_val("home_team_name", "HOME"),
        away_team=get_val("away_team_name", "AWAY"),
        inning=get_val("inning", 1),
        inning_half="Top" if get_val("half_inning", "top") == "top" else "Bot",
        balls=get_val("balls", 0),
        strikes=get_val("strikes", 0),
        outs=get_val("outs", 0),
        date=str(get_val("game_date", "")),
        runners_on=runners_str,
        score_home=get_val("home_score"),
        score_away=get_val("away_score"),
        pitch_number=get_val("pitch_number"),
    )

    # Get actual values
    actual_type = get_val("pitch_type_code")
    actual_px = get_val("px")
    actual_pz = get_val("pz")
    actual_location = (actual_px, actual_pz) if actual_px is not None and actual_pz is not None else None

    print(f"\nActual pitch: {actual_type} at ({actual_px:.2f}, {actual_pz:.2f})")

    # Create the pitch card manually (since we don't have full PitchPredictor.load())
    print("\nCreating pitch card...")
    fig = create_pitch_card_manual(
        prediction=prediction,
        context=context,
        actual_pitch_type=actual_type,
        actual_location=actual_location,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved pitch card to: {output_path}")
    plt.close(fig)


def create_pitch_card_manual(
    prediction: PitchPrediction,
    context: GameContext,
    actual_pitch_type: str | None = None,
    actual_location: tuple | None = None,
) -> plt.Figure:
    """Create a comprehensive pitch prediction card."""

    # Create figure with specific layout
    fig = plt.figure(figsize=(14, 10))

    # Define grid for subplots
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.2, 1.5, 0.8],
        height_ratios=[0.3, 1.5, 0.3],
        hspace=0.3,
        wspace=0.3,
    )

    # Title area (spans top)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")

    # Main location plot (center)
    ax_location = fig.add_subplot(gs[1, 1])

    # Pitch type probabilities (left)
    ax_types = fig.add_subplot(gs[1, 0])

    # Game info (right)
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")

    # Footer area (spans bottom)
    ax_footer = fig.add_subplot(gs[2, :])
    ax_footer.axis("off")

    # === TITLE ===
    title_text = f"{context.pitcher_name} vs {context.batter_name}"
    ax_title.text(0.5, 0.7, title_text, fontsize=20, fontweight="bold",
                  ha="center", va="center", transform=ax_title.transAxes)

    subtitle = f"{context.matchup_str} | {context.inning_half} {context.inning}"
    if context.date:
        subtitle = f"{context.date} | " + subtitle
    ax_title.text(0.5, 0.25, subtitle, fontsize=12, color="gray",
                  ha="center", va="center", transform=ax_title.transAxes)

    # === LOCATION DENSITY PLOT ===
    # Plot KDE density
    extent = [prediction.px_grid.min(), prediction.px_grid.max(),
              prediction.pz_grid.min(), prediction.pz_grid.max()]

    im = ax_location.imshow(
        prediction.location_density,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        aspect="auto",
        alpha=0.8,
    )

    # Draw strike zone
    sz_left, sz_right = -0.7083, 0.7083  # 17 inches / 2
    sz_bottom, sz_top = 1.5, 3.5  # Approximate strike zone
    ax_location.plot(
        [sz_left, sz_right, sz_right, sz_left, sz_left],
        [sz_bottom, sz_bottom, sz_top, sz_top, sz_bottom],
        "k-", linewidth=2
    )

    # Plot predicted location
    ax_location.scatter(
        prediction.location_point[0], prediction.location_point[1],
        c="blue", s=150, marker="o", edgecolors="white", linewidths=2,
        label="Predicted",
        zorder=10
    )

    # Plot actual location if available
    if actual_location:
        ax_location.scatter(
            actual_location[0], actual_location[1],
            c="red", s=150, marker="x", linewidths=3,
            label="Actual",
            zorder=11
        )

    ax_location.set_xlim(-2.5, 2.5)
    ax_location.set_ylim(0.5, 4.5)
    ax_location.set_xlabel("Horizontal Location (ft)", fontsize=11)
    ax_location.set_ylabel("Vertical Location (ft)", fontsize=11)
    ax_location.set_title("Location Prediction", fontsize=14, fontweight="bold")
    # Place legend below the plot to avoid obscuring the strike zone
    ax_location.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                       fontsize=8, ncol=2, frameon=True)
    ax_location.set_aspect("equal")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_location, shrink=0.8)
    cbar.set_label("Probability Density", fontsize=10)

    # === PITCH TYPE PROBABILITIES ===
    # Sort by probability for display
    sorted_indices = np.argsort(prediction.type_probabilities)[::-1]

    # Show top 8 pitch types
    n_show = min(8, len(sorted_indices))
    top_indices = sorted_indices[:n_show]

    type_codes = [IDX_TO_PITCH_TYPE.get(i, f"T{i}") for i in top_indices]
    types = [get_pitch_name(code) for code in type_codes]
    probs = [prediction.type_probabilities[i] for i in top_indices]

    # Colors - highlight predicted type
    colors = []
    for i, idx in enumerate(top_indices):
        if idx == prediction.predicted_type_idx:
            colors.append("#2ecc71")  # Green for predicted
        elif actual_pitch_type and IDX_TO_PITCH_TYPE.get(idx, "") == actual_pitch_type:
            colors.append("#e74c3c")  # Red for actual
        else:
            colors.append("#3498db")  # Blue for others

    bars = ax_types.barh(range(n_show), probs, color=colors, edgecolor="white", linewidth=1)
    ax_types.set_yticks(range(n_show))
    ax_types.set_yticklabels(types, fontsize=11)
    ax_types.set_xlabel("Probability", fontsize=11)
    ax_types.set_title("Pitch Type Prediction", fontsize=14, fontweight="bold")
    ax_types.set_xlim(0, 1)
    ax_types.invert_yaxis()

    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax_types.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                      f"{prob:.1%}", va="center", fontsize=10)

    # === GAME INFO ===
    info_lines = []

    # Game and score
    info_lines.append(("Game", context.game_str))

    # Count
    count_str = f"{context.balls}-{context.strikes}, {context.outs} out"
    if context.pitch_number:
        count_str += f" (Pitch #{context.pitch_number})"
    info_lines.append(("Count", count_str))

    # Runners
    info_lines.append(("Runners", context.runners_on))

    # Prediction summary
    info_lines.append(("", ""))  # Spacer
    info_lines.append(("PREDICTED", ""))
    predicted_name = get_pitch_name(prediction.predicted_type)
    info_lines.append(("Type", f"{predicted_name} ({prediction.top_3_types[0][1]:.1%})"))
    info_lines.append(("Location", f"({prediction.location_point[0]:.2f}, {prediction.location_point[1]:.2f})"))

    # Actual if available
    if actual_pitch_type:
        info_lines.append(("", ""))  # Spacer
        info_lines.append(("ACTUAL", ""))
        actual_name = get_pitch_name(actual_pitch_type)
        info_lines.append(("Type", actual_name))
        if actual_location:
            info_lines.append(("Location", f"({actual_location[0]:.2f}, {actual_location[1]:.2f})"))

    # Render info text
    y_pos = 0.95
    for label, value in info_lines:
        if label in ["PREDICTED", "ACTUAL"]:
            ax_info.text(0.05, y_pos, label, fontsize=11, fontweight="bold",
                        transform=ax_info.transAxes, va="top")
        elif label == "":
            pass  # Spacer
        else:
            ax_info.text(0.05, y_pos, f"{label}:", fontsize=10, fontweight="bold",
                        transform=ax_info.transAxes, va="top")
            ax_info.text(0.4, y_pos, value, fontsize=10,
                        transform=ax_info.transAxes, va="top")
        y_pos -= 0.08

    # === FOOTER ===
    # Legend for colors
    legend_text = "Green = Predicted | Red = Actual | Blue = Other"
    ax_footer.text(0.5, 0.5, legend_text, fontsize=10, color="gray",
                   ha="center", va="center", transform=ax_footer.transAxes)

    # Tighten layout
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    main()
