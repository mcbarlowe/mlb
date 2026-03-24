#!/usr/bin/env python
"""
Generate pitch cards for pitchers using the LSTM+Attention model.

Creates visual summaries showing:
- Pitch type distribution
- Location tendencies by pitch type
- Count-based tendencies
- Model predictions for key situations

Usage:
    uv run python scripts/generate_pitch_cards.py --pitcher "Gerrit Cole"
    uv run python scripts/generate_pitch_cards.py --top 10
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import polars as pl
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model import create_model
from src.ml.features import PitchFeatureEngine, PITCH_TYPE_CODES


def load_model(model_dir: str, device: str = "cpu"):
    """Load trained model and feature engine."""
    model_path = Path(model_dir)

    # Find the run directory
    run_dirs = list(model_path.glob("run_*"))
    if run_dirs:
        run_dir = sorted(run_dirs)[-1]  # Get most recent
    else:
        run_dir = model_path

    # Load model checkpoint
    checkpoint_path = run_dir / "final_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    config = checkpoint.get("config", {})

    # Create model
    model = create_model(
        n_pitch_types=checkpoint["n_pitch_types"],
        n_pitchers=checkpoint["n_pitchers"],
        n_batters=checkpoint["n_batters"],
        n_features=checkpoint["n_features"],
        model_type=config.get("model_type", "lstm_attention"),
        feature_indices=checkpoint.get("feature_indices"),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        embedding_dim=config.get("embedding_dim", 32),
        n_location_components=config.get("n_location_components", 3),
        n_attention_heads=config.get("n_attention_heads", 8),
        n_attention_layers=config.get("n_attention_layers", 2),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_pitcher_data(data_path: str, season: str = "2025"):
    """Load pitch data for a season."""
    season_path = Path(data_path) / season

    dfs = []
    for parquet_file in season_path.glob("*.parquet"):
        df = pl.read_parquet(parquet_file)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found in {season_path}")

    return pl.concat(dfs, how="diagonal")


def get_pitcher_stats(df: pl.DataFrame, pitcher_name: str = None, pitcher_id: int = None):
    """Get statistics for a specific pitcher."""
    if pitcher_name:
        pitcher_df = df.filter(pl.col("pitcher_name") == pitcher_name)
    elif pitcher_id:
        pitcher_df = df.filter(pl.col("pitcher_id") == pitcher_id)
    else:
        raise ValueError("Must provide pitcher_name or pitcher_id")

    if len(pitcher_df) == 0:
        return None

    # Get pitcher info
    pitcher_info = {
        "name": pitcher_df.select("pitcher_name").head(1).item(),
        "id": pitcher_df.select("pitcher_id").head(1).item(),
        "throws": pitcher_df.select("pitcher_hand").head(1).item(),
        "total_pitches": len(pitcher_df),
    }

    # Pitch type distribution
    pitch_counts = (
        pitcher_df.group_by("pitch_type")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )

    total = pitch_counts.select("count").sum().item()
    pitch_dist = {}
    for row in pitch_counts.iter_rows(named=True):
        pitch_dist[row["pitch_type"]] = {
            "count": row["count"],
            "pct": row["count"] / total * 100
        }

    pitcher_info["pitch_distribution"] = pitch_dist

    # Location data by pitch type
    locations = {}
    for pitch_type in pitch_dist.keys():
        type_df = pitcher_df.filter(pl.col("pitch_type") == pitch_type)
        px = type_df.select("px").drop_nulls().to_numpy().flatten()
        pz = type_df.select("pz").drop_nulls().to_numpy().flatten()
        if len(px) > 0:
            locations[pitch_type] = {"px": px, "pz": pz}

    pitcher_info["locations"] = locations

    # Count-based tendencies
    count_tendencies = {}
    for balls in range(4):
        for strikes in range(3):
            count_str = f"{balls}-{strikes}"
            count_df = pitcher_df.filter(
                (pl.col("balls") == balls) & (pl.col("strikes") == strikes)
            )
            if len(count_df) > 10:
                count_dist = (
                    count_df.group_by("pitch_type")
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                )
                total_count = count_dist.select("count").sum().item()
                count_tendencies[count_str] = {
                    row["pitch_type"]: row["count"] / total_count * 100
                    for row in count_dist.iter_rows(named=True)
                }

    pitcher_info["count_tendencies"] = count_tendencies

    # Velocity by pitch type
    velocities = {}
    for pitch_type in pitch_dist.keys():
        type_df = pitcher_df.filter(pl.col("pitch_type") == pitch_type)
        speeds = type_df.select("start_speed").drop_nulls().to_numpy().flatten()
        if len(speeds) > 0:
            velocities[pitch_type] = {
                "mean": float(np.mean(speeds)),
                "std": float(np.std(speeds)),
                "min": float(np.min(speeds)),
                "max": float(np.max(speeds)),
            }

    pitcher_info["velocities"] = velocities

    return pitcher_info


def draw_strike_zone(ax, zone_height_bot=1.5, zone_height_top=3.5):
    """Draw strike zone on axes."""
    zone_width = 17 / 12  # 17 inches in feet

    # Strike zone rectangle
    zone = patches.Rectangle(
        (-zone_width/2, zone_height_bot),
        zone_width,
        zone_height_top - zone_height_bot,
        linewidth=2,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(zone)

    # Home plate
    plate_width = 17 / 12
    plate = patches.Polygon(
        [
            (-plate_width/2, 0),
            (plate_width/2, 0),
            (plate_width/2, 0.1),
            (0, 0.3),
            (-plate_width/2, 0.1),
        ],
        linewidth=1,
        edgecolor='black',
        facecolor='lightgray'
    )
    ax.add_patch(plate)


def create_pitch_card(pitcher_info: dict, output_path: str = None):
    """Create a visual pitch card for a pitcher."""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{pitcher_info['name']} ({pitcher_info['throws']}HP)",
        fontsize=20,
        fontweight='bold',
        y=0.98
    )

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Pitch Distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    pitch_dist = pitcher_info["pitch_distribution"]
    labels = list(pitch_dist.keys())
    sizes = [pitch_dist[p]["pct"] for p in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    ax1.set_title("Pitch Mix", fontweight='bold')

    # 2. Velocity table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    velocities = pitcher_info.get("velocities", {})
    if velocities:
        table_data = []
        for pitch_type in sorted(velocities.keys()):
            v = velocities[pitch_type]
            table_data.append([
                pitch_type,
                f"{v['mean']:.1f}",
                f"{v['min']:.0f}-{v['max']:.0f}"
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=["Pitch", "Avg MPH", "Range"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    ax2.set_title("Velocity", fontweight='bold')

    # 3. Count tendencies heatmap
    ax3 = fig.add_subplot(gs[0, 2:])
    count_tendencies = pitcher_info.get("count_tendencies", {})

    # Get top 4 pitch types
    top_pitches = sorted(
        pitch_dist.keys(),
        key=lambda x: pitch_dist[x]["pct"],
        reverse=True
    )[:4]

    # Create heatmap data
    counts = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
    heatmap_data = np.zeros((len(top_pitches), len(counts)))

    for i, pitch in enumerate(top_pitches):
        for j, count in enumerate(counts):
            if count in count_tendencies and pitch in count_tendencies[count]:
                heatmap_data[i, j] = count_tendencies[count][pitch]

    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(counts)))
    ax3.set_xticklabels(counts, rotation=45, ha='right')
    ax3.set_yticks(range(len(top_pitches)))
    ax3.set_yticklabels(top_pitches)
    ax3.set_title("Pitch Usage by Count (%)", fontweight='bold')

    # Add text annotations
    for i in range(len(top_pitches)):
        for j in range(len(counts)):
            val = heatmap_data[i, j]
            if val > 0:
                color = 'white' if val > 40 else 'black'
                ax3.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax3, shrink=0.8)

    # 4-7. Location plots for top 4 pitch types
    locations = pitcher_info.get("locations", {})

    for idx, pitch_type in enumerate(top_pitches[:4]):
        ax = fig.add_subplot(gs[1, idx])

        # Draw strike zone
        draw_strike_zone(ax)

        if pitch_type in locations:
            px = locations[pitch_type]["px"]
            pz = locations[pitch_type]["pz"]

            # Filter to reasonable bounds
            mask = (np.abs(px) < 3) & (pz > 0) & (pz < 5)
            px = px[mask]
            pz = pz[mask]

            if len(px) > 10:
                # 2D histogram heatmap
                heatmap, xedges, yedges = np.histogram2d(
                    px, pz, bins=20, range=[[-2, 2], [0, 5]]
                )
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(
                    heatmap.T, extent=extent, origin='lower',
                    cmap='Reds', alpha=0.7, aspect='auto'
                )

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect('equal')

        pct = pitch_dist[pitch_type]["pct"]
        vel = velocities.get(pitch_type, {}).get("mean", 0)
        ax.set_title(f"{pitch_type}\n{pct:.1f}% | {vel:.1f} mph", fontweight='bold')
        ax.set_xlabel("Horizontal (ft)")
        ax.set_ylabel("Vertical (ft)")

    # 8. Stats summary
    ax8 = fig.add_subplot(gs[2, :2])
    ax8.axis('off')

    stats_text = f"""
    Total Pitches: {pitcher_info['total_pitches']:,}

    Arsenal:
    """
    for pitch_type in sorted(pitch_dist.keys(), key=lambda x: pitch_dist[x]["pct"], reverse=True):
        pct = pitch_dist[pitch_type]["pct"]
        count = pitch_dist[pitch_type]["count"]
        vel = velocities.get(pitch_type, {}).get("mean", 0)
        stats_text += f"\n    {pitch_type}: {pct:.1f}% ({count:,} pitches) - {vel:.1f} mph avg"

    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax8.set_title("Season Stats", fontweight='bold', loc='left')

    # 9. Situational tendencies
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')

    # Calculate some key situations
    situations = []

    # First pitch
    if "0-0" in count_tendencies:
        top_first = max(count_tendencies["0-0"].items(), key=lambda x: x[1])
        situations.append(f"First Pitch: {top_first[0]} ({top_first[1]:.1f}%)")

    # Two strikes
    two_strike_counts = ["0-2", "1-2", "2-2", "3-2"]
    two_strike_pitches = {}
    for count in two_strike_counts:
        if count in count_tendencies:
            for pitch, pct in count_tendencies[count].items():
                two_strike_pitches[pitch] = two_strike_pitches.get(pitch, 0) + pct
    if two_strike_pitches:
        top_two_strike = max(two_strike_pitches.items(), key=lambda x: x[1])
        situations.append(f"Two Strikes: {top_two_strike[0]} ({top_two_strike[1]/len(two_strike_counts):.1f}% avg)")

    # Behind in count
    behind_counts = ["1-0", "2-0", "2-1", "3-0", "3-1"]
    behind_pitches = {}
    for count in behind_counts:
        if count in count_tendencies:
            for pitch, pct in count_tendencies[count].items():
                behind_pitches[pitch] = behind_pitches.get(pitch, 0) + pct
    if behind_pitches:
        top_behind = max(behind_pitches.items(), key=lambda x: x[1])
        situations.append(f"Behind in Count: {top_behind[0]} ({top_behind[1]/len(behind_counts):.1f}% avg)")

    sit_text = "Key Tendencies:\n\n" + "\n".join(f"  • {s}" for s in situations)
    ax9.text(0.1, 0.9, sit_text, transform=ax9.transAxes,
             fontsize=12, verticalalignment='top')
    ax9.set_title("Situational Analysis", fontweight='bold', loc='left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved pitch card to {output_path}")

    return fig


def get_top_pitchers(df: pl.DataFrame, n: int = 10):
    """Get top N pitchers by pitch count."""
    pitcher_counts = (
        df.group_by(["pitcher_id", "pitcher_name"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(n)
    )
    return pitcher_counts


def main():
    parser = argparse.ArgumentParser(description="Generate pitcher pitch cards")
    parser.add_argument("--pitcher", type=str, help="Pitcher name to generate card for")
    parser.add_argument("--top", type=int, default=0, help="Generate cards for top N pitchers")
    parser.add_argument("--model-dir", type=str, default="models/attention_full",
                        help="Model directory")
    parser.add_argument("--data-path", type=str, default="data/processed/livefeeds",
                        help="Path to processed data")
    parser.add_argument("--season", type=str, default="2025", help="Season to analyze")
    parser.add_argument("--output-dir", type=str, default="output/pitch_cards",
                        help="Output directory for cards")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {args.season} season...")
    df = load_pitcher_data(args.data_path, args.season)
    print(f"Loaded {len(df):,} pitches")

    pitchers_to_process = []

    if args.pitcher:
        pitchers_to_process.append(args.pitcher)
    elif args.top > 0:
        top_pitchers = get_top_pitchers(df, args.top)
        for row in top_pitchers.iter_rows(named=True):
            pitchers_to_process.append(row["pitcher_name"])
        print(f"\nTop {args.top} pitchers by pitch count:")
        for i, row in enumerate(top_pitchers.iter_rows(named=True), 1):
            print(f"  {i}. {row['pitcher_name']}: {row['count']:,} pitches")
    else:
        # Default: show top pitchers
        top_pitchers = get_top_pitchers(df, 10)
        print("\nTop 10 pitchers by pitch count:")
        for i, row in enumerate(top_pitchers.iter_rows(named=True), 1):
            print(f"  {i}. {row['pitcher_name']}: {row['count']:,} pitches")
        print("\nUse --pitcher 'Name' or --top N to generate cards")
        return

    print(f"\nGenerating pitch cards...")
    for pitcher_name in pitchers_to_process:
        print(f"\nProcessing {pitcher_name}...")

        pitcher_info = get_pitcher_stats(df, pitcher_name=pitcher_name)

        if pitcher_info is None:
            print(f"  No data found for {pitcher_name}")
            continue

        # Create safe filename
        safe_name = pitcher_name.replace(" ", "_").replace(".", "").lower()
        output_path = output_dir / f"{safe_name}_pitch_card.png"

        create_pitch_card(pitcher_info, str(output_path))

    print(f"\nDone! Cards saved to {output_dir}/")


if __name__ == "__main__":
    main()
