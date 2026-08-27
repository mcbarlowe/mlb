#!/usr/bin/env python
"""
Generate pitch cards using the combined model pipeline.

Uses:
- Pitch type model: LSTM+Attention from models/attention_full
- Location model: PitchTypeConditionedMDN from models/pitch_type_location

Usage:
    uv run python scripts/generate_combined_pitch_cards.py
    uv run python scripts/generate_combined_pitch_cards.py --n-cards 10
    uv run python scripts/generate_combined_pitch_cards.py --output-dir output/combined_cards
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy import stats
from tqdm import tqdm

from src.ml.features import (
    IDX_TO_PITCH_TYPE,
    PITCH_TYPE_CODES,
    PitchFeatureEngine,
)
from src.ml.model import create_model
from src.ml.pitch_predictor import (
    PITCH_TYPE_FULL_NAMES,
    GameContext,
    PitchPrediction,
    fetch_mlb_headshot,
)
from src.ml.pitch_type_location_model import (
    PitchTypeConditionedMDN,
    PitchTypeThenLocationPredictor,
)


def get_device() -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pitch_type_model(model_dir: Path, device: torch.device):
    """Load the trained pitch type model."""
    checkpoint = torch.load(model_dir / "final_model.pt", map_location=device)

    n_pitchers = checkpoint.get("n_pitchers", 3633)
    n_batters = checkpoint.get("n_batters", 4607)
    n_features = checkpoint.get("n_features", 49)
    n_pitch_types = checkpoint.get("n_pitch_types", len(PITCH_TYPE_CODES))
    feature_indices = checkpoint.get("feature_indices", None)
    config = checkpoint.get("config", {})

    model = create_model(
        n_pitch_types=n_pitch_types,
        n_pitchers=n_pitchers,
        n_batters=n_batters,
        n_features=n_features,
        model_type=config.get("model_type", "lstm_attention"),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        embedding_dim=config.get("embedding_dim", 32),
        n_location_components=config.get("n_location_components", 3),
        n_attention_heads=config.get("n_attention_heads", 8),
        n_attention_layers=config.get("n_attention_layers", 2),
        feature_indices=feature_indices,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


def load_location_model(model_dir: Path, device: torch.device):
    """Load the trained pitch-type-conditioned location model."""
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    n_features = config.get("n_features")
    if n_features is None and "feature_columns" in config:
        n_features = len(config["feature_columns"])

    model = PitchTypeConditionedMDN(
        n_features=n_features,
        n_pitch_types=config.get("n_pitch_types", 11),
        hidden_dims=config["hidden_dims"],
        n_components=config["n_components"],
        dropout=config["dropout"],
    )

    checkpoint = torch.load(model_dir / "pitch_type_location_model.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


def load_sample_at_bats(
    data_path: Path,
    season: str = "2025",
    n_at_bats: int = 10,
    min_pitches: int = 3,
    seed: int | None = None,
) -> list[pl.DataFrame]:
    """Load sample at-bats from parquet files."""
    print(f"Loading data from {season}...")
    df = pl.scan_parquet(str(data_path / season / "*.parquet")).collect()
    print(f"Loaded {len(df):,} pitches")

    df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

    at_bat_groups = df.group_by(["game_pk", "at_bat_index"]).agg(
        pl.count().alias("n_pitches"),
        pl.first("pitcher_name").alias("pitcher_name"),
        pl.first("batter_name").alias("batter_name"),
    ).filter(pl.col("n_pitches") >= min_pitches)

    print(f"Found {len(at_bat_groups):,} at-bats with >= {min_pitches} pitches")

    if seed is not None:
        np.random.seed(seed)
    sampled = at_bat_groups.sample(n=min(n_at_bats, len(at_bat_groups)), seed=seed)

    at_bats = []
    for row in sampled.iter_rows(named=True):
        at_bat_df = df.filter(
            (pl.col("game_pk") == row["game_pk"]) &
            (pl.col("at_bat_index") == row["at_bat_index"])
        ).sort("pitch_number")
        at_bats.append(at_bat_df)

    return at_bats


def get_context_from_row(row: pl.DataFrame) -> GameContext:
    """Extract GameContext from a pitch row."""
    def get_val(col, default=None):
        if col in row.columns:
            val = row[col][0]
            return val if val is not None else default
        return default

    from datetime import datetime

    def format_date(date_val):
        if date_val is None:
            return None
        date_str = str(date_val)
        try:
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str)
            else:
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
            return dt.strftime('%B %d, %Y')
        except Exception:
            return date_str[:10] if len(date_str) >= 10 else date_str

    runner_on_1b = bool(get_val("is_runner_on_first", False))
    runner_on_2b = bool(get_val("is_runner_on_second", False))
    runner_on_3b = bool(get_val("is_runner_on_third", False))

    return GameContext(
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
        date=format_date(get_val("game_date")),
        runner_on_1b=runner_on_1b,
        runner_on_2b=runner_on_2b,
        runner_on_3b=runner_on_3b,
        score_home=get_val("home_score"),
        score_away=get_val("away_score"),
        pitch_number=get_val("pitch_number"),
        pitcher_id=get_val("pitcher_id"),
        batter_id=get_val("batter_id"),
        pitch_result=get_val("pitch_call_description"),
    )


def compute_location_density_from_mdn(
    mdn_params: dict,
    grid_size: int = 100,
    n_samples: int = 2000,
    px_range: tuple = (-2.5, 2.5),
    pz_range: tuple = (0.5, 4.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute location density grid from MDN parameters using sampling and KDE."""
    pi = mdn_params["pi"].cpu().numpy()  # [K]
    mu = mdn_params["mu"].cpu().numpy()  # [K, 2]
    sigma = mdn_params["sigma"].cpu().numpy()  # [K, 2]
    rho = mdn_params["rho"].cpu().numpy()  # [K]

    K = len(pi)

    # Sample from the mixture
    samples = []
    for _ in range(n_samples):
        k = np.random.choice(K, p=pi)
        mean = mu[k]
        std = sigma[k]
        correlation = rho[k]

        cov = np.array([
            [std[0]**2, correlation * std[0] * std[1]],
            [correlation * std[0] * std[1], std[1]**2]
        ])

        try:
            sample = np.random.multivariate_normal(mean, cov)
            samples.append(sample)
        except np.linalg.LinAlgError:
            samples.append(mean)

    samples = np.array(samples)

    px_grid = np.linspace(px_range[0], px_range[1], grid_size)
    pz_grid = np.linspace(pz_range[0], pz_range[1], grid_size)

    try:
        kernel = stats.gaussian_kde(samples.T)
        PX, PZ = np.meshgrid(px_grid, pz_grid)
        positions = np.vstack([PX.ravel(), PZ.ravel()])
        density = kernel(positions).reshape(grid_size, grid_size)
    except (np.linalg.LinAlgError, ValueError):
        density = np.zeros((grid_size, grid_size))

    return px_grid, pz_grid, density


def compute_top2_location_densities(
    location_model,
    loc_features: torch.Tensor,
    top_2_indices: list[int],
    device: torch.device,
    grid_size: int = 100,
) -> dict:
    """
    Compute separate location densities for top 2 pitch types.

    Args:
        location_model: PitchTypeConditionedMDN model
        loc_features: Location features [1, n_features]
        top_2_indices: List of top 2 pitch type indices
        device: Torch device
        grid_size: Resolution for density grid

    Returns:
        Dictionary with {pitch_type_idx: (px_grid, pz_grid, density, expected_loc)}
    """
    loc_features = loc_features.to(device)

    with torch.no_grad():
        # Get MDN params for all pitch types
        all_params = location_model.forward_all_types(loc_features)
        # all_params["pi"]: [batch, n_pitch_types, K]
        # all_params["mu"]: [batch, n_pitch_types, K, 2]
        # all_params["sigma"]: [batch, n_pitch_types, K, 2]
        # all_params["rho"]: [batch, n_pitch_types, K]

    result = {}
    for pt_idx in top_2_indices:
        # Extract params for this pitch type
        pi = all_params["pi"][0, pt_idx]  # [K]
        mu = all_params["mu"][0, pt_idx]  # [K, 2]
        sigma = all_params["sigma"][0, pt_idx]  # [K, 2]
        rho = all_params["rho"][0, pt_idx]  # [K]

        # Compute expected location
        expected_loc = (pi.unsqueeze(-1) * mu).sum(dim=0).cpu().numpy()

        # Compute density grid
        mdn_params = {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}
        px_grid, pz_grid, density = compute_location_density_from_mdn(
            mdn_params, grid_size=grid_size
        )

        result[pt_idx] = {
            "px_grid": px_grid,
            "pz_grid": pz_grid,
            "density": density,
            "expected_loc": expected_loc,
        }

    return result


def predict_combined(
    combined_model: PitchTypeThenLocationPredictor,
    features: torch.Tensor,
    loc_features: torch.Tensor,
    device: torch.device,
    grid_size: int = 100,
    compute_top2_densities: bool = True,
) -> tuple[PitchPrediction, dict | None]:
    """
    Make prediction using the combined pitch type + location model.

    Args:
        combined_model: The combined predictor
        features: Full features for pitch type model [seq_len, n_features]
        loc_features: Location features subset [seq_len, n_loc_features]
        device: Torch device
        grid_size: Resolution for density grid
        compute_top2_densities: Whether to compute separate densities for top 2 types

    Returns:
        Tuple of (PitchPrediction, top2_densities dict or None)
    """
    # Add batch dimension
    if features.dim() == 2:
        features = features.unsqueeze(0)
    if loc_features.dim() == 2:
        loc_features = loc_features.unsqueeze(0)

    features = features.to(device)
    loc_features = loc_features.to(device)

    batch_size, seq_len, _ = features.shape
    lengths = torch.tensor([seq_len], dtype=torch.long)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Flatten location features
    loc_features_flat = loc_features.reshape(-1, loc_features.shape[-1])

    with torch.no_grad():
        output = combined_model(features, lengths, mask, location_features=loc_features_flat)

        # Get predictions for last position
        type_logits_last = output["pitch_type_logits"][0, -1, :]
        type_probs = torch.softmax(type_logits_last, dim=-1).cpu().numpy()

        predicted_idx = int(np.argmax(type_probs))
        predicted_type = IDX_TO_PITCH_TYPE.get(predicted_idx, "UNK")

        top_3_indices = np.argsort(type_probs)[::-1][:3]
        top_3_types = [
            (IDX_TO_PITCH_TYPE.get(i, "UNK"), type_probs[i])
            for i in top_3_indices
        ]

        # Get MDN parameters for last position
        mdn_params = output["mdn_params"]
        # Reshape back to get last position
        mdn_params["pi"].shape[-1]
        last_idx = seq_len - 1

        pi = mdn_params["pi"][last_idx]  # [K]
        mu = mdn_params["mu"][last_idx]  # [K, 2]
        sigma = mdn_params["sigma"][last_idx]  # [K, 2]
        rho = mdn_params["rho"][last_idx]  # [K]

        mixture_weights = pi.cpu().numpy()
        mixture_means = mu.cpu().numpy()
        mixture_stds = sigma.cpu().numpy()

        # Expected value
        location_point = (pi.unsqueeze(-1) * mu).sum(dim=0).cpu().numpy()

    # Compute density grid
    single_mdn_params = {
        "pi": pi,
        "mu": mu,
        "sigma": sigma,
        "rho": rho,
    }
    px_grid, pz_grid, density = compute_location_density_from_mdn(
        single_mdn_params, grid_size=grid_size
    )

    # Mode: peak of density
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    location_mode = np.array([px_grid[max_idx[1]], pz_grid[max_idx[0]]])

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

    # Compute separate densities for top 2 pitch types
    top2_densities = None
    if compute_top2_densities:
        top_2_indices = list(top_3_indices[:2])
        # Get last position features for location model
        last_loc_features = loc_features_flat[-1:, :]  # [1, n_loc_features]
        top2_densities = compute_top2_location_densities(
            combined_model.location_model,
            last_loc_features,
            top_2_indices,
            device,
            grid_size=grid_size,
        )

    return prediction, top2_densities


def create_pitch_card(
    prediction: PitchPrediction,
    context: GameContext,
    actual_pitch_type: str | None = None,
    actual_location: tuple[float, float] | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 10),
    top2_densities: dict | None = None,
) -> plt.Figure:
    """Create a comprehensive pitch prediction card.

    Args:
        prediction: PitchPrediction object with type/location predictions
        context: GameContext with game state info
        actual_pitch_type: Actual pitch type code (optional)
        actual_location: Actual (px, pz) location (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size in inches
        top2_densities: Dict with per-pitch-type location densities for top 2 types
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    fig = plt.figure(figsize=figsize, facecolor='white')

    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.5, 0.8, 3, 0.5],
        width_ratios=[1.2, 0.6, 1.4],
        hspace=0.15,
        wspace=0.2,
    )

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')

    home_team = context.home_team if context.home_team else "HOME"
    away_team = context.away_team if context.away_team else "AWAY"

    if context.score_home is not None and context.score_away is not None:
        header_text = f"{away_team} {context.score_away}  @  {home_team} {context.score_home}"
    else:
        header_text = f"{away_team}  @  {home_team}"

    if context.date:
        header_text += f"   |   {context.date}"

    ax_header.text(
        0.5, 0.6, header_text,
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        transform=ax_header.transAxes,
    )

    inning_text = f"{context.inning_half} {context.inning}"
    ax_header.text(
        0.5, 0.15, inning_text,
        ha='center', va='center',
        fontsize=14, color='#444444',
        transform=ax_header.transAxes,
    )

    # Matchup with headshots
    ax_matchup = fig.add_subplot(gs[1, :])
    ax_matchup.axis('off')
    ax_matchup.set_xlim(0, 1)
    ax_matchup.set_ylim(0, 1)

    pitcher_headshot = fetch_mlb_headshot(context.pitcher_id, size=80)
    batter_headshot = fetch_mlb_headshot(context.batter_id, size=80)

    if pitcher_headshot is not None:
        im_pitcher = OffsetImage(pitcher_headshot, zoom=0.5)
        ab_pitcher = AnnotationBbox(im_pitcher, (0.08, 0.5), frameon=True,
                                    bboxprops={"edgecolor": '#cccccc', "linewidth": 1})
        ax_matchup.add_artist(ab_pitcher)
        ax_matchup.text(0.20, 0.5, f"P: {context.pitcher_name} ({context.pitcher_hand})",
                        ha='left', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)
    else:
        ax_matchup.text(0.12, 0.5, f"P: {context.pitcher_name} ({context.pitcher_hand})",
                        ha='left', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)

    ax_matchup.text(0.50, 0.5, "vs", ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#666666',
                    transform=ax_matchup.transAxes)

    if batter_headshot is not None:
        ax_matchup.text(0.80, 0.5, f"B: {context.batter_name} ({context.batter_hand})",
                        ha='right', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)
        im_batter = OffsetImage(batter_headshot, zoom=0.5)
        ab_batter = AnnotationBbox(im_batter, (0.92, 0.5), frameon=True,
                                   bboxprops={"edgecolor": '#cccccc', "linewidth": 1})
        ax_matchup.add_artist(ab_batter)
    else:
        ax_matchup.text(0.88, 0.5, f"B: {context.batter_name} ({context.batter_hand})",
                        ha='right', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)

    # Pitch type probabilities
    ax_probs = fig.add_subplot(gs[2, 0])
    ax_probs.set_xlim(0, 1)
    ax_probs.set_ylim(0, 1)
    ax_probs.axis('off')

    count_text = f"Count: {context.count_str}"
    if context.pitch_number:
        count_text += f"  (Pitch #{context.pitch_number})"
    ax_probs.set_title(count_text, fontsize=11, fontweight='bold', pad=8)

    probs = prediction.type_probabilities
    sorted_idx = np.argsort(probs)[::-1]
    sorted_codes = [PITCH_TYPE_CODES[i] for i in sorted_idx]
    sorted_probs = probs[sorted_idx]

    mask = sorted_probs > 0.01
    n_show = min(sum(mask), 6)

    table_top = 0.88
    row_height = 0.12

    for i in range(n_show):
        y_pos = table_top - (i * row_height)
        pitch_code = sorted_codes[i]
        pitch_name = PITCH_TYPE_FULL_NAMES.get(pitch_code, pitch_code)
        prob = sorted_probs[i]

        is_actual = actual_pitch_type and pitch_code == actual_pitch_type
        if is_actual:
            text_color = '#27ae60'
            font_weight = 'bold'
        elif i == 0:
            text_color = '#2c3e50'
            font_weight = 'bold'
        else:
            text_color = '#666666'
            font_weight = 'normal'

        row_text = f"{pitch_code:<3} {pitch_name:<20} {prob:>4.0%}"
        ax_probs.text(0.05, y_pos, row_text, fontsize=10,
                      color=text_color, fontweight=font_weight,
                      family='monospace', va='center',
                      transform=ax_probs.transAxes)

        bar_color = '#27ae60' if is_actual else '#3498db'
        bar_width = 0.18 * prob
        ax_probs.plot([0.78, 0.78 + bar_width], [y_pos, y_pos],
                      color=bar_color, linewidth=5, solid_capstyle='round',
                      transform=ax_probs.transAxes)

    if context.pitch_result:
        result_y = table_top - (n_show * row_height) - 0.08
        ax_probs.text(0.05, result_y, f"Result: {context.pitch_result}",
                      fontsize=10, va='top',
                      color='#c0392b', fontweight='bold',
                      transform=ax_probs.transAxes)

    # Baseball diamond
    ax_diamond = fig.add_subplot(gs[2, 1])
    _draw_baseball_diamond(
        ax_diamond,
        runner_on_1b=context.runner_on_1b,
        runner_on_2b=context.runner_on_2b,
        runner_on_3b=context.runner_on_3b,
        outs=context.outs,
    )

    # Strike zone with location density
    ax_zone = fig.add_subplot(gs[2, 2])

    # Draw top 2 pitch type location densities with different colors
    if top2_densities and len(top2_densities) >= 2:
        # Colors for top 2 types: Blue for #1, Orange/Red for #2
        top2_colors = [
            ('#3498db', '#1a5276', 'Blues'),  # Top 1: Blue
            ('#e74c3c', '#922b21', 'Reds'),   # Top 2: Red
        ]

        top2_indices = list(top2_densities.keys())
        legend_handles = []

        for i, pt_idx in enumerate(top2_indices[:2]):
            pt_data = top2_densities[pt_idx]
            pt_code = IDX_TO_PITCH_TYPE.get(pt_idx, "UNK")

            PX, PZ = np.meshgrid(-pt_data["px_grid"], pt_data["pz_grid"])
            density = pt_data["density"]

            # Normalize density for consistent visualization
            density_norm = density / (density.max() + 1e-8)

            marker_color, contour_color, cmap = top2_colors[i]

            # Draw contour fill with transparency
            ax_zone.contourf(
                PX, PZ, density_norm,
                levels=np.linspace(0.1, 1.0, 10),
                cmap=cmap,
                alpha=0.4 if i == 1 else 0.5,
            )
            ax_zone.contour(
                PX, PZ, density_norm,
                levels=[0.3, 0.5, 0.7],
                colors=contour_color,
                alpha=0.6,
                linewidths=1.5,
            )

            # Draw expected location marker
            exp_px_flipped = -pt_data["expected_loc"][0]
            exp_pz = pt_data["expected_loc"][1]
            marker = 'D' if i == 0 else 's'  # Diamond for #1, Square for #2
            ax_zone.scatter(
                exp_px_flipped, exp_pz,
                c=marker_color, s=120, marker=marker,
                edgecolors='white', linewidths=2,
                zorder=10 - i,
            )
            prob = prediction.top_3_types[i][1] if i < len(prediction.top_3_types) else 0
            legend_handles.append(
                plt.Line2D([0], [0], marker=marker, color='w',
                           markerfacecolor=marker_color, markersize=10,
                           label=f'{pt_code} ({prob:.0%})')
            )

        ax_zone.legend(handles=legend_handles, loc='upper right', fontsize=9,
                       title='Top 2 Types', title_fontsize=9)
    else:
        # Single-density visualization with Blues style
        PX, PZ = np.meshgrid(-prediction.px_grid, prediction.pz_grid)

        # Normalize density
        density_norm = prediction.location_density / (prediction.location_density.max() + 1e-8)

        ax_zone.contourf(
            PX, PZ, density_norm,
            levels=np.linspace(0.1, 1.0, 10),
            cmap='Blues',
            alpha=0.5,
        )
        ax_zone.contour(
            PX, PZ, density_norm,
            levels=[0.3, 0.5, 0.7],
            colors='#1a5276',
            alpha=0.6,
            linewidths=1.5,
        )

    zone_width = 17 / 12
    zone_left = -zone_width / 2
    zone_bottom = 1.5
    zone_height = 2.0

    strike_zone = plt.Rectangle(
        (zone_left, zone_bottom), zone_width, zone_height,
        fill=False, edgecolor='black', linewidth=3,
    )
    ax_zone.add_patch(strike_zone)

    plate_y = 0.8
    plate = plt.Polygon(
        [[-0.708, plate_y + 0.3], [0.708, plate_y + 0.3],
         [0.708, plate_y + 0.2], [0, plate_y],
         [-0.708, plate_y + 0.2]],
        fill=True, facecolor='white', edgecolor='black', linewidth=2,
    )
    ax_zone.add_patch(plate)

    # Draw expected location if not using top2 densities (which already draws markers)
    if not top2_densities:
        pred_exp_px_flipped = -prediction.location_point[0]
        ax_zone.scatter(
            pred_exp_px_flipped,
            prediction.location_point[1],
            c='#3498db', s=100, marker='D',
            label='Expected',
            zorder=9, edgecolors='#2471a3', linewidths=1.5,
        )

    # Always draw actual location if available
    if actual_location is not None:
        actual_px_flipped = -actual_location[0]
        ax_zone.scatter(
            actual_px_flipped, actual_location[1],
            c='#2ecc71', s=200, marker='X',
            label='Actual',
            zorder=11, edgecolors='#27ae60', linewidths=2,
        )

    ax_zone.set_xlim(-2.2, 2.2)
    ax_zone.set_ylim(0.5, 4.5)
    ax_zone.set_xlabel("Horizontal (ft) - Pitcher's View", fontsize=10)
    ax_zone.set_ylabel('Vertical (ft)', fontsize=10)

    title = 'Top 2 Pitch Type Locations' if top2_densities else 'Pitch Location (Combined)'
    ax_zone.set_title(title, fontsize=12, fontweight='bold')
    ax_zone.set_aspect('equal')

    # Only add legend if not using top2 (which has its own legend)
    if not top2_densities:
        ax_zone.legend(loc='upper right', fontsize=10)

    ax_zone.text(-1.8, 0.6, 'LHB', fontsize=8, color='#666666')
    ax_zone.text(1.5, 0.6, 'RHB', fontsize=8, color='#666666')

    # Footer
    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.axis('off')

    if actual_pitch_type or actual_location:
        result_parts = []
        if actual_pitch_type:
            actual_full_name = PITCH_TYPE_FULL_NAMES.get(actual_pitch_type, actual_pitch_type)
            result_parts.append(f"Actual: {actual_full_name}")
        if actual_location:
            result_parts.append(f"Location: ({actual_location[0]:.2f}, {actual_location[1]:.2f})")

            exp_error = np.sqrt(
                (prediction.location_point[0] - actual_location[0])**2 +
                (prediction.location_point[1] - actual_location[1])**2
            )
            result_parts.append(f"Error: {exp_error:.2f} ft")

        if actual_pitch_type:
            if actual_pitch_type == prediction.predicted_type:
                result_parts.append("Type Correct")
            else:
                predicted_full_name = PITCH_TYPE_FULL_NAMES.get(prediction.predicted_type, prediction.predicted_type)
                actual_prob = probs[PITCH_TYPE_CODES.index(actual_pitch_type)] if actual_pitch_type in PITCH_TYPE_CODES else 0
                result_parts.append(f"Predicted: {predicted_full_name} (Actual had {actual_prob:.1%})")

        result_text = "  |  ".join(result_parts)

        if actual_pitch_type == prediction.predicted_type:
            text_color = '#27ae60'
        else:
            text_color = '#e74c3c'

        ax_footer.text(
            0.5, 0.5, result_text,
            ha='center', va='center',
            fontsize=11, color=text_color, fontweight='bold',
            transform=ax_footer.transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def _draw_baseball_diamond(
    ax: plt.Axes,
    runner_on_1b: bool = False,
    runner_on_2b: bool = False,
    runner_on_3b: bool = False,
    outs: int = 0,
) -> None:
    """Draw a baseball diamond with runners highlighted."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')

    home = (0, 0)
    first = (1, 0.7)
    second = (0, 1.4)
    third = (-1, 0.7)

    diamond_coords = [home, first, second, third, home]
    xs = [p[0] for p in diamond_coords]
    ys = [p[1] for p in diamond_coords]
    ax.plot(xs, ys, 'k-', linewidth=2)

    base_size = 0.18

    home_plate = plt.Polygon(
        [(-0.1, 0), (0.1, 0), (0.1, 0.1), (0, 0.18), (-0.1, 0.1)],
        fill=True, facecolor='white', edgecolor='black', linewidth=2,
    )
    ax.add_patch(home_plate)

    def draw_base(center, is_occupied):
        x, y = center
        color = '#f1c40f' if is_occupied else 'white'
        edge_color = '#d68910' if is_occupied else 'black'
        base = plt.Polygon(
            [(x, y - base_size), (x + base_size, y),
             (x, y + base_size), (x - base_size, y)],
            fill=True, facecolor=color, edgecolor=edge_color, linewidth=2,
        )
        ax.add_patch(base)

    draw_base(first, runner_on_1b)
    draw_base(second, runner_on_2b)
    draw_base(third, runner_on_3b)

    out_y = -0.35
    for i in range(3):
        out_x = -0.3 + i * 0.3
        is_out = i < outs
        circle = plt.Circle(
            (out_x, out_y), 0.08,
            fill=True,
            facecolor='#e74c3c' if is_out else 'white',
            edgecolor='#c0392b' if is_out else '#666666',
            linewidth=1.5,
        )
        ax.add_patch(circle)

    ax.text(0, out_y - 0.22, 'OUTS', ha='center', va='center',
            fontsize=8, color='#666666', fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description="Generate pitch cards using combined model")
    parser.add_argument("--pitch-type-model", default="models/attention_full/run_20260119_124719",
                        help="Path to pitch type LSTM model")
    parser.add_argument("--location-model", default="models/pitch_type_location_20260121_003206",
                        help="Path to location MDN model")
    parser.add_argument("--data-path", default="data/processed/livefeeds",
                        help="Path to processed pitch data")
    parser.add_argument("--season", default="2025", help="Season to sample from")
    parser.add_argument("--n-cards", type=int, default=5, help="Number of pitch cards")
    parser.add_argument("--output-dir", default="output/combined_pitch_cards",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None for random)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMBINED MODEL PITCH CARD GENERATOR")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading pitch type model from {args.pitch_type_model}...")
    pitch_type_model, pt_config = load_pitch_type_model(Path(args.pitch_type_model), device)
    print(f"  Model type: {pt_config.get('model_type', 'lstm_attention')}")

    print(f"\nLoading location model from {args.location_model}...")
    location_model, loc_config = load_location_model(Path(args.location_model), device)
    print(f"  Hidden dims: {loc_config['hidden_dims']}")

    # Create combined predictor
    print("\nCreating combined predictor...")
    combined_model = PitchTypeThenLocationPredictor(
        pitch_type_model=pitch_type_model,
        location_model=location_model,
        use_soft_conditioning=True,
    )
    combined_model.to(device)
    combined_model.eval()

    # Load feature engine
    feature_engine_path = Path(args.pitch_type_model) / "feature_engine.json"
    if not feature_engine_path.exists():
        feature_engine_path = Path(args.pitch_type_model).parent / "feature_engine.json"

    print(f"\nLoading feature engine from {feature_engine_path}...")
    feature_engine = PitchFeatureEngine.load(feature_engine_path)

    # Get feature column mappings
    full_feature_cols = feature_engine.get_feature_columns()
    loc_feature_cols = loc_config.get("feature_columns", [])

    if loc_feature_cols:
        loc_feature_indices = []
        for col in loc_feature_cols:
            if col in full_feature_cols:
                loc_feature_indices.append(full_feature_cols.index(col))
        loc_feature_indices = torch.tensor(loc_feature_indices, device=device)
        print(f"  Location model uses {len(loc_feature_indices)}/{len(full_feature_cols)} features")
    else:
        loc_feature_indices = None

    # Load sample at-bats
    at_bats = load_sample_at_bats(
        data_path=Path(args.data_path),
        season=args.season,
        n_at_bats=args.n_cards,
        seed=args.seed,
    )

    print(f"\nGenerating {len(at_bats)} pitch cards...")

    correct_predictions = 0
    total_predictions = 0

    for i, at_bat_df in enumerate(tqdm(at_bats, desc="Generating cards")):
        try:
            # Transform the at-bat
            transformed = feature_engine.transform(at_bat_df)
            feature_cols = feature_engine.get_feature_columns()

            # Extract features
            features = transformed.select(feature_cols).to_numpy()
            features = torch.tensor(features, dtype=torch.float32).to(device)

            # Extract location features subset
            if loc_feature_indices is not None:
                loc_features = features[:, loc_feature_indices]
            else:
                loc_features = features

            # Get context from last pitch
            target_row = at_bat_df.slice(-1, 1)
            context = get_context_from_row(target_row)

            # Make prediction
            prediction, _ = predict_combined(
                combined_model, features, loc_features, device,
                compute_top2_densities=False
            )

            # Get actual values
            actual_type = target_row["pitch_type_code"][0]
            actual_px = target_row["px"][0]
            actual_pz = target_row["pz"][0]
            actual_location = (actual_px, actual_pz) if actual_px is not None and actual_pz is not None else None

            # Track accuracy
            total_predictions += 1
            if prediction.predicted_type == actual_type:
                correct_predictions += 1

            # Generate card
            filename = f"pitch_card_{i+1:02d}_{context.pitcher_name.replace(' ', '_')}_{context.batter_name.replace(' ', '_')}.png"
            save_path = output_dir / filename

            fig = create_pitch_card(
                prediction=prediction,
                context=context,
                actual_pitch_type=actual_type,
                actual_location=actual_location,
                save_path=str(save_path),
            )
            plt.close(fig)

        except Exception as e:
            print(f"\nError processing at-bat {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {total_predictions} pitch cards in {output_dir}")
    if total_predictions > 0:
        print(f"Pitch type accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")

    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
