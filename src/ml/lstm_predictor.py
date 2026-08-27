"""
LSTM-based pitch predictor for generating predictions and pitch cards.

This module provides inference capabilities using the trained LSTM+Attention model.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ml.features import IDX_TO_PITCH_TYPE, PITCH_TYPE_CODES, PitchFeatureEngine
from src.ml.model import create_model
from src.ml.pitch_predictor import (
    PITCH_TYPE_FULL_NAMES,
    GameContext,
)


@dataclass
class LSTMPrediction:
    """Container for LSTM model predictions."""

    # Pitch type
    type_probabilities: np.ndarray  # [n_classes]
    predicted_type_idx: int
    predicted_type: str
    top_3_types: list[tuple[str, float]]

    # Location (from MDN)
    location_point: np.ndarray  # [px, pz] expected location
    location_components: dict  # MDN mixture parameters

    # Attention weights (optional)
    attention_weights: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type_probabilities": {
                PITCH_TYPE_CODES[i]: float(p)
                for i, p in enumerate(self.type_probabilities)
            },
            "predicted_type": self.predicted_type,
            "top_3_types": [
                {"type": t, "probability": float(p)}
                for t, p in self.top_3_types
            ],
            "location_point": {
                "px": float(self.location_point[0]),
                "pz": float(self.location_point[1]),
            },
        }


class LSTMPitchPredictor:
    """
    Predictor using LSTM+Attention model for pitch type and location.

    The LSTM model processes at-bat sequences and predicts:
    - Pitch type probabilities
    - Location via MDN (mixture density network)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_engine: PitchFeatureEngine,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.feature_engine = feature_engine
        self.device = device

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "LSTMPitchPredictor":
        """
        Load a trained LSTM predictor from disk.

        Args:
            model_dir: Path to the model directory
            device: Device for inference

        Returns:
            Loaded LSTMPitchPredictor instance
        """
        model_dir = Path(model_dir)

        # Find the run directory
        run_dirs = list(model_dir.glob("run_*"))
        if run_dirs:
            run_dir = max(run_dirs)
        else:
            run_dir = model_dir

        # Load model checkpoint
        checkpoint_path = run_dir / "final_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get config
        config = checkpoint.get("config", {})
        model_type = config.get("model_type", "lstm_attention")

        # Create model
        model = create_model(
            n_pitch_types=checkpoint["n_pitch_types"],
            n_pitchers=checkpoint["n_pitchers"],
            n_batters=checkpoint["n_batters"],
            n_features=checkpoint["n_features"],
            model_type=model_type,
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

        # Create feature engine
        feature_engine = PitchFeatureEngine()

        # Load pitcher/batter mappings if available
        feature_engine_path = run_dir / "feature_engine.json"
        if feature_engine_path.exists():
            feature_engine.load(str(feature_engine_path))
        else:
            # Try parent directory
            parent_engine_path = model_dir / "feature_engine.json"
            if parent_engine_path.exists():
                feature_engine.load(str(parent_engine_path))

        return cls(model=model, feature_engine=feature_engine, device=device)

    def predict_sequence(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> list[LSTMPrediction]:
        """
        Make predictions for a sequence of pitches in an at-bat.

        Args:
            features: [batch, seq_len, n_features]
            lengths: [batch] sequence lengths
            mask: [batch, seq_len] attention mask
            return_attention: Whether to return attention weights

        Returns:
            List of LSTMPrediction for each pitch in sequence
        """
        with torch.no_grad():
            features = features.to(self.device)
            lengths = lengths.to(self.device)
            mask = mask.to(self.device)

            if return_attention:
                type_logits, mdn_params, attn_weights = self.model(
                    features, lengths, mask, return_attention=True
                )
            else:
                type_logits, mdn_params = self.model(features, lengths, mask)
                attn_weights = None

            # Convert logits to probabilities
            type_probs = torch.softmax(type_logits, dim=-1)

            # Get location expected values
            pi = mdn_params["pi"]  # [batch, seq, K]
            mu = mdn_params["mu"]  # [batch, seq, K, 2]

            # Expected value = sum(pi * mu)
            location_points = (pi.unsqueeze(-1) * mu).sum(dim=2)  # [batch, seq, 2]

        predictions = []
        batch_size = features.shape[0]
        features.shape[1]

        for b in range(batch_size):
            for t in range(int(lengths[b].item())):
                probs = type_probs[b, t].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_type = IDX_TO_PITCH_TYPE.get(pred_idx, "UNK")

                # Top 3
                top_3_idx = np.argsort(probs)[::-1][:3]
                top_3 = [(IDX_TO_PITCH_TYPE.get(i, "UNK"), probs[i]) for i in top_3_idx]

                # Location
                loc_point = location_points[b, t].cpu().numpy()

                # MDN components
                components = {
                    "pi": pi[b, t].cpu().numpy(),
                    "mu": mu[b, t].cpu().numpy(),
                    "sigma": mdn_params["sigma"][b, t].cpu().numpy(),
                }

                # Attention
                attn = None
                if attn_weights is not None:
                    attn = attn_weights[b, :, t, :].cpu().numpy()

                predictions.append(LSTMPrediction(
                    type_probabilities=probs,
                    predicted_type_idx=pred_idx,
                    predicted_type=pred_type,
                    top_3_types=top_3,
                    location_point=loc_point,
                    location_components=components,
                    attention_weights=attn,
                ))

        return predictions

    def predict_from_context(
        self,
        pitcher_id: int,
        batter_id: int,
        balls: int,
        strikes: int,
        outs: int,
        inning: int,
        score_diff: int = 0,
        runner_on_1b: bool = False,
        runner_on_2b: bool = False,
        runner_on_3b: bool = False,
        prev_pitches: list[dict] | None = None,
        throw_side: str = "R",
        bat_side: str = "R",
    ) -> LSTMPrediction:
        """
        Make a prediction from game context.

        Args:
            pitcher_id: MLB pitcher ID
            batter_id: MLB batter ID
            balls: Current ball count
            strikes: Current strike count
            outs: Current outs
            inning: Current inning
            score_diff: Score difference (positive = home leading)
            runner_on_1b: Runner on first base
            runner_on_2b: Runner on second base
            runner_on_3b: Runner on third base
            prev_pitches: List of previous pitches in at-bat
            throw_side: Pitcher handedness ("L" or "R")
            bat_side: Batter handedness ("L" or "R")

        Returns:
            LSTMPrediction for the next pitch
        """
        # Build feature tensor for current situation
        # This is a simplified version - full implementation would need
        # all the feature engineering from PitchFeatureEngine

        # Get pitcher/batter indices
        self.feature_engine.pitcher_to_idx.get(pitcher_id, 0)
        self.feature_engine.batter_to_idx.get(batter_id, 0)

        # Build runners bitmap
        int(runner_on_1b) + 2 * int(runner_on_2b) + 4 * int(runner_on_3b)

        # Build basic features (this is simplified - real implementation needs full feature set)
        n_features = len(self.feature_engine.get_feature_columns())

        # Create feature vector
        features = torch.zeros(1, 1, n_features)

        # Fill in known features (indices depend on feature_engine order)
        # This is a simplified implementation

        features = features.to(self.device)
        lengths = torch.tensor([1])
        mask = torch.ones(1, 1, dtype=torch.bool)

        predictions = self.predict_sequence(features, lengths, mask)
        return predictions[0]

    def create_pitch_card(
        self,
        prediction: LSTMPrediction,
        context: GameContext,
        actual_pitch_type: str | None = None,
        actual_location: tuple[float, float] | None = None,
        save_path: str | None = None,
        figsize: tuple[float, float] = (14, 10),
    ) -> plt.Figure:
        """
        Create a pitch card visualization.

        Args:
            prediction: LSTMPrediction from predict
            context: GameContext with game information
            actual_pitch_type: Actual pitch type thrown
            actual_location: Actual (px, pz) location
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
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
            header_text += f"   •   {context.date}"

        ax_header.text(0.5, 0.6, header_text, ha='center', va='center',
                       fontsize=18, fontweight='bold', transform=ax_header.transAxes)

        inning_text = f"{context.inning_half} {context.inning}"
        ax_header.text(0.5, 0.15, inning_text, ha='center', va='center',
                       fontsize=14, color='#444444', transform=ax_header.transAxes)

        # Matchup
        ax_matchup = fig.add_subplot(gs[1, :])
        ax_matchup.axis('off')

        ax_matchup.text(0.25, 0.5, f"P: {context.pitcher_name} ({context.pitcher_hand})",
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)
        ax_matchup.text(0.50, 0.5, "vs", ha='center', va='center',
                        fontsize=14, fontweight='bold', color='#666666',
                        transform=ax_matchup.transAxes)
        ax_matchup.text(0.75, 0.5, f"B: {context.batter_name} ({context.batter_hand})",
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        transform=ax_matchup.transAxes)

        # Pitch type probabilities
        ax_probs = fig.add_subplot(gs[2, 0])

        count_text = f"Count: {context.count_str}"
        if context.pitch_number:
            count_text += f"  (Pitch #{context.pitch_number})"
        ax_probs.set_title(count_text, fontsize=12, fontweight='bold', pad=10)

        probs = prediction.type_probabilities
        sorted_idx = np.argsort(probs)[::-1]
        sorted_codes = [PITCH_TYPE_CODES[i] for i in sorted_idx]
        sorted_probs = probs[sorted_idx]

        mask = sorted_probs > 0.01
        n_show = sum(mask)

        sorted_full_names = [PITCH_TYPE_FULL_NAMES.get(code, code) for code in sorted_codes[:n_show]]

        colors = []
        for i, idx in enumerate(sorted_idx[:n_show]):
            pitch_type = PITCH_TYPE_CODES[idx]
            if actual_pitch_type and pitch_type == actual_pitch_type:
                colors.append('#2ecc71')
            else:
                colors.append('#3498db')

        bars = ax_probs.barh(range(n_show), sorted_probs[:n_show], color=colors,
                             edgecolor='#2c3e50', linewidth=1)

        ax_probs.set_yticks(range(n_show))
        ax_probs.set_yticklabels(sorted_full_names, fontsize=10)
        ax_probs.set_xlabel('Probability', fontsize=10)
        ax_probs.set_xlim(0, 1)
        ax_probs.invert_yaxis()

        for bar, prob in zip(bars, sorted_probs[:n_show]):
            ax_probs.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                          f'{prob:.1%}', va='center', fontsize=10)

        ax_probs.spines['top'].set_visible(False)
        ax_probs.spines['right'].set_visible(False)

        # Diamond
        ax_diamond = fig.add_subplot(gs[2, 1])
        self._draw_baseball_diamond(ax_diamond, context.runner_on_1b,
                                    context.runner_on_2b, context.runner_on_3b, context.outs)

        # Strike zone with location
        ax_zone = fig.add_subplot(gs[2, 2])

        # Draw strike zone
        zone_width = 17 / 12
        zone_left = -zone_width / 2
        zone_bottom = 1.5
        zone_height = 2.0

        strike_zone = plt.Rectangle((zone_left, zone_bottom), zone_width, zone_height,
                                     fill=False, edgecolor='black', linewidth=3)
        ax_zone.add_patch(strike_zone)

        # Draw home plate
        plate_y = 0.8
        plate = plt.Polygon([[-0.708, plate_y + 0.3], [0.708, plate_y + 0.3],
                             [0.708, plate_y + 0.2], [0, plate_y], [-0.708, plate_y + 0.2]],
                            fill=True, facecolor='white', edgecolor='black', linewidth=2)
        ax_zone.add_patch(plate)

        # Plot MDN components
        pi = prediction.location_components["pi"]
        mu = prediction.location_components["mu"]
        sigma = prediction.location_components["sigma"]

        for k in range(len(pi)):
            if pi[k] > 0.1:
                # Draw ellipse for each component
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(
                    xy=(-mu[k, 0], mu[k, 1]),  # Flip x for pitcher's view
                    width=2 * sigma[k, 0],
                    height=2 * sigma[k, 1],
                    alpha=pi[k] * 0.5,
                    facecolor='red',
                    edgecolor='darkred',
                )
                ax_zone.add_patch(ellipse)

        # Plot predicted location
        pred_px = -prediction.location_point[0]  # Flip for pitcher's view
        pred_pz = prediction.location_point[1]
        ax_zone.scatter(pred_px, pred_pz, c='#27ae60', s=150, marker='*',
                        label='Predicted', zorder=10, edgecolors='#1e8449', linewidths=2)

        # Plot actual location
        if actual_location:
            actual_px = -actual_location[0]
            ax_zone.scatter(actual_px, actual_location[1], c='#e74c3c', s=200, marker='X',
                            label='Actual', zorder=11, edgecolors='#c0392b', linewidths=2)

        ax_zone.set_xlim(-2.2, 2.2)
        ax_zone.set_ylim(0.5, 4.5)
        ax_zone.set_xlabel("Horizontal (ft) - Pitcher's View", fontsize=10)
        ax_zone.set_ylabel('Vertical (ft)', fontsize=10)
        ax_zone.set_title('Pitch Location Prediction', fontsize=12, fontweight='bold')
        ax_zone.set_aspect('equal')
        ax_zone.legend(loc='upper right', fontsize=10)

        ax_zone.text(-1.8, 0.6, '← LHB', fontsize=8, color='#666666')
        ax_zone.text(1.4, 0.6, 'RHB →', fontsize=8, color='#666666')

        # Footer
        ax_footer = fig.add_subplot(gs[3, :])
        ax_footer.axis('off')

        if actual_pitch_type:
            actual_full = PITCH_TYPE_FULL_NAMES.get(actual_pitch_type, actual_pitch_type)
            if actual_pitch_type == prediction.predicted_type:
                footer_text = f"✓ Correctly predicted: {actual_full}"
                color = '#27ae60'
            else:
                pred_full = PITCH_TYPE_FULL_NAMES.get(prediction.predicted_type, prediction.predicted_type)
                footer_text = f"Predicted: {pred_full} | Actual: {actual_full}"
                color = '#e74c3c'

            ax_footer.text(0.5, 0.5, footer_text, ha='center', va='center',
                           fontsize=12, fontweight='bold', color=color,
                           transform=ax_footer.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        return fig

    def _draw_baseball_diamond(self, ax, runner_1b, runner_2b, runner_3b, outs):
        """Draw baseball diamond with runners."""
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

        home_plate = plt.Polygon([(-0.1, 0), (0.1, 0), (0.1, 0.1), (0, 0.18), (-0.1, 0.1)],
                                 fill=True, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(home_plate)

        def draw_base(center, is_occupied):
            x, y = center
            color = '#f1c40f' if is_occupied else 'white'
            edge_color = '#d68910' if is_occupied else 'black'
            base = plt.Polygon([(x, y - base_size), (x + base_size, y),
                               (x, y + base_size), (x - base_size, y)],
                              fill=True, facecolor=color, edgecolor=edge_color, linewidth=2)
            ax.add_patch(base)

        draw_base(first, runner_1b)
        draw_base(second, runner_2b)
        draw_base(third, runner_3b)

        out_y = -0.35
        for i in range(3):
            out_x = -0.3 + i * 0.3
            is_out = i < outs
            circle = plt.Circle((out_x, out_y), 0.08, fill=True,
                                facecolor='#e74c3c' if is_out else 'white',
                                edgecolor='#c0392b' if is_out else '#666666', linewidth=1.5)
            ax.add_patch(circle)

        ax.text(0, out_y - 0.22, 'OUTS', ha='center', va='center',
                fontsize=8, color='#666666', fontweight='bold')
