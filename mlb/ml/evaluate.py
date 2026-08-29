"""
Evaluation metrics and visualization for pitch prediction models.

Provides functions for:
- Classification metrics (accuracy, F1, confusion matrix)
- MDN evaluation (NLL, coverage, calibration)
- Sampling from MDN distributions
- Visualization of predictions with uncertainty ellipses
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader

from mlb.ml.features import PITCH_TYPE_CODES


def get_mdn_mean_prediction(mdn_params: dict) -> np.ndarray:
    """
    Get the mean prediction from MDN (weighted average of component means).

    Args:
        mdn_params: Dict with 'pi' [N, K] and 'mu' [N, K, 2]

    Returns:
        Weighted mean locations [N, 2]
    """
    pi = mdn_params["pi"]  # [N, K]
    mu = mdn_params["mu"]  # [N, K, 2]

    # Weighted average: sum_k pi_k * mu_k
    weighted_mean = (pi[..., None] * mu).sum(axis=-2)  # [N, 2]
    return weighted_mean


def compute_mdn_nll(
    mdn_params: dict,
    targets: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-sample NLL for MDN predictions.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho (all numpy arrays)
        targets: Target locations [N, 2]
        eps: Small constant for numerical stability

    Returns:
        NLL values [N]
    """
    pi = mdn_params["pi"]  # [N, K]
    mu = mdn_params["mu"]  # [N, K, 2]
    sigma = mdn_params["sigma"]  # [N, K, 2]
    rho = mdn_params["rho"]  # [N, K]

    # Expand targets: [N, 2] -> [N, 1, 2]
    targets = targets[:, None, :]

    x = targets[..., 0]  # [N, 1]
    y = targets[..., 1]
    mu_x = mu[..., 0]  # [N, K]
    mu_y = mu[..., 1]
    sigma_x = sigma[..., 0]
    sigma_y = sigma[..., 1]

    dx = (x - mu_x) / (sigma_x + eps)
    dy = (y - mu_y) / (sigma_y + eps)

    one_minus_rho_sq = np.clip(1 - rho**2, eps, None)
    z = (dx**2 - 2 * rho * dx * dy + dy**2) / one_minus_rho_sq

    log_component_prob = (
        -math.log(2 * math.pi)
        - np.log(sigma_x + eps)
        - np.log(sigma_y + eps)
        - 0.5 * np.log(one_minus_rho_sq)
        - 0.5 * z
    )

    log_pi = np.log(pi + eps)
    # Log-sum-exp
    max_val = np.max(log_pi + log_component_prob, axis=-1, keepdims=True)
    log_mixture_prob = max_val.squeeze(-1) + np.log(
        np.sum(np.exp(log_pi + log_component_prob - max_val), axis=-1)
    )

    return -log_mixture_prob


def compute_mdn_coverage(
    mdn_params: dict,
    targets: np.ndarray,
    confidence_level: float = 0.95,
    eps: float = 1e-6,
) -> float:
    """
    Compute coverage: what fraction of targets fall within confidence region.

    Uses Mahalanobis distance to the nearest component.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho
        targets: Target locations [N, 2]
        confidence_level: Confidence level (0.90, 0.95, etc.)
        eps: Numerical stability constant

    Returns:
        Coverage fraction (0 to 1)
    """
    from scipy import stats

    # Chi-squared threshold for 2D at given confidence
    chi2_threshold = stats.chi2.ppf(confidence_level, df=2)

    mu = mdn_params["mu"]  # [N, K, 2]
    sigma = mdn_params["sigma"]  # [N, K, 2]
    rho = mdn_params["rho"]  # [N, K]

    _N, _K, _ = mu.shape
    targets_expanded = targets[:, None, :]  # [N, 1, 2]

    # Compute Mahalanobis distance for each component
    dx = targets_expanded[..., 0] - mu[..., 0]  # [N, K]
    dy = targets_expanded[..., 1] - mu[..., 1]
    sigma_x = sigma[..., 0]
    sigma_y = sigma[..., 1]

    one_minus_rho_sq = np.clip(1 - rho**2, eps, None)

    # Mahalanobis distance squared
    mahal_sq = (
        dx**2 / (sigma_x**2 + eps)
        - 2 * rho * dx * dy / ((sigma_x * sigma_y) + eps)
        + dy**2 / (sigma_y**2 + eps)
    ) / one_minus_rho_sq

    # For each sample, check if ANY component contains the target
    min_mahal_sq = np.min(mahal_sq, axis=-1)  # [N]
    covered = min_mahal_sq <= chi2_threshold

    return covered.mean()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
) -> dict:
    """
    Evaluate MDN model on a dataset.

    Args:
        model: Trained PyTorch model with MDN location head.
        data_loader: DataLoader for evaluation data.
        device: Device to use.

    Returns:
        Dictionary with evaluation metrics including MDN-specific metrics.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    all_type_preds = []
    all_type_probs = []
    all_type_targets = []
    all_loc_targets = []
    all_mdn_params = {"pi": [], "mu": [], "sigma": [], "rho": []}

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            pitch_type_logits, mdn_params = model(features, lengths, mask)

            # Get predictions
            type_probs = torch.softmax(pitch_type_logits, dim=-1)
            type_preds = pitch_type_logits.argmax(dim=-1)

            # Extract valid predictions (where mask is True)
            for i in range(features.size(0)):
                seq_len = int(lengths[i].item())
                all_type_preds.extend(type_preds[i, :seq_len].cpu().numpy())
                all_type_probs.extend(type_probs[i, :seq_len].cpu().numpy())
                all_type_targets.extend(targets[i, :seq_len, 0].cpu().numpy())
                all_loc_targets.extend(targets[i, :seq_len, 1:3].cpu().numpy())

                # Collect MDN params
                all_mdn_params["pi"].extend(mdn_params["pi"][i, :seq_len].cpu().numpy())
                all_mdn_params["mu"].extend(mdn_params["mu"][i, :seq_len].cpu().numpy())
                all_mdn_params["sigma"].extend(mdn_params["sigma"][i, :seq_len].cpu().numpy())
                all_mdn_params["rho"].extend(mdn_params["rho"][i, :seq_len].cpu().numpy())

    # Convert to numpy
    all_type_preds = np.array(all_type_preds)
    all_type_probs = np.array(all_type_probs)
    all_type_targets = np.array(all_type_targets).astype(int)
    all_loc_targets = np.array(all_loc_targets)

    mdn_params_np = {
        "pi": np.array(all_mdn_params["pi"]),
        "mu": np.array(all_mdn_params["mu"]),
        "sigma": np.array(all_mdn_params["sigma"]),
        "rho": np.array(all_mdn_params["rho"]),
    }

    # Classification metrics
    accuracy = accuracy_score(all_type_targets, all_type_preds)
    f1_macro = f1_score(all_type_targets, all_type_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_type_targets, all_type_preds, average="weighted", zero_division=0)

    # Top-3 accuracy
    try:
        top3_acc = top_k_accuracy_score(all_type_targets, all_type_probs, k=3)
    except Exception:
        top3_acc = 0.0

    # MDN location metrics
    # Mean prediction (weighted average of components)
    all_loc_preds = get_mdn_mean_prediction(mdn_params_np)

    loc_errors = all_loc_preds - all_loc_targets
    mae_px = np.abs(loc_errors[:, 0]).mean()
    mae_pz = np.abs(loc_errors[:, 1]).mean()
    rmse_px = np.sqrt((loc_errors[:, 0] ** 2).mean())
    rmse_pz = np.sqrt((loc_errors[:, 1] ** 2).mean())
    euclidean_error = np.sqrt((loc_errors**2).sum(axis=1)).mean()

    # MDN-specific metrics
    nll_values = compute_mdn_nll(mdn_params_np, all_loc_targets)
    mean_nll = nll_values.mean()

    coverage_90 = compute_mdn_coverage(mdn_params_np, all_loc_targets, 0.90)
    coverage_95 = compute_mdn_coverage(mdn_params_np, all_loc_targets, 0.95)

    return {
        # Classification metrics
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        # Location metrics (using mean prediction)
        "mae_px": mae_px,
        "mae_pz": mae_pz,
        "rmse_px": rmse_px,
        "rmse_pz": rmse_pz,
        "euclidean_error": euclidean_error,
        # MDN-specific metrics
        "nll": mean_nll,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        # Raw predictions for further analysis
        "type_preds": all_type_preds,
        "type_targets": all_type_targets,
        "type_probs": all_type_probs,
        "loc_preds": all_loc_preds,
        "loc_targets": all_loc_targets,
        "mdn_params": mdn_params_np,
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> str:
    """
    Print detailed classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional class labels.

    Returns:
        Classification report string.
    """
    if labels is None:
        labels = PITCH_TYPE_CODES

    # Filter to only classes that appear in the data
    present_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [labels[i] for i in present_classes if i < len(labels)]

    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=present_labels,
        zero_division=0,
    )
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    normalize: bool = True,
    figsize: tuple = (10, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot confusion matrix for pitch type predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional class labels.
        normalize: Whether to normalize the confusion matrix.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    if labels is None:
        labels = PITCH_TYPE_CODES

    # Filter to only classes that appear in the data
    present_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [labels[i] for i in present_classes if i < len(labels)]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Pitch Type Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_location_predictions(
    loc_preds: np.ndarray,
    loc_targets: np.ndarray,
    type_preds: np.ndarray | None = None,
    n_samples: int = 500,
    figsize: tuple = (12, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize predicted vs actual pitch locations.

    Args:
        loc_preds: Predicted locations [n, 2].
        loc_targets: Actual locations [n, 2].
        type_preds: Optional pitch type predictions for coloring.
        n_samples: Number of samples to plot.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Sample if too many points
    if len(loc_preds) > n_samples:
        idx = np.random.choice(len(loc_preds), n_samples, replace=False)
        loc_preds = loc_preds[idx]
        loc_targets = loc_targets[idx]
        if type_preds is not None:
            type_preds = type_preds[idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Strike zone boundaries (approximate)
    sz_left = -0.83
    sz_right = 0.83
    sz_bottom = 1.5
    sz_top = 3.5

    # Plot 1: Actual locations
    ax = axes[0]
    ax.scatter(loc_targets[:, 0], loc_targets[:, 1], alpha=0.3, s=10)
    ax.add_patch(plt.Rectangle(
        (sz_left, sz_bottom), sz_right - sz_left, sz_top - sz_bottom,
        fill=False, edgecolor="red", linewidth=2
    ))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("px (feet)")
    ax.set_ylabel("pz (feet)")
    ax.set_title("Actual Locations")
    ax.set_aspect("equal")

    # Plot 2: Predicted locations
    ax = axes[1]
    ax.scatter(loc_preds[:, 0], loc_preds[:, 1], alpha=0.3, s=10)
    ax.add_patch(plt.Rectangle(
        (sz_left, sz_bottom), sz_right - sz_left, sz_top - sz_bottom,
        fill=False, edgecolor="red", linewidth=2
    ))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("px (feet)")
    ax.set_ylabel("pz (feet)")
    ax.set_title("Predicted Locations")
    ax.set_aspect("equal")

    # Plot 3: Error vectors
    ax = axes[2]
    errors = loc_preds - loc_targets
    ax.scatter(errors[:, 0], errors[:, 1], alpha=0.3, s=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("px error (feet)")
    ax.set_ylabel("pz error (feet)")
    ax.set_title("Prediction Errors")
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: list[dict],
    figsize: tuple = (12, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: List of epoch metrics dictionaries.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, val_acc, label="Validation Accuracy", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def analyze_predictions_by_count(
    type_preds: np.ndarray,
    type_targets: np.ndarray,
    counts: np.ndarray,
) -> dict:
    """
    Analyze prediction accuracy by ball-strike count.

    Args:
        type_preds: Predicted pitch types.
        type_targets: Actual pitch types.
        counts: Ball-strike counts (as encoded values or strings).

    Returns:
        Dictionary with accuracy by count.
    """
    unique_counts = np.unique(counts)
    results = {}

    for count in unique_counts:
        mask = counts == count
        if mask.sum() > 0:
            acc = accuracy_score(type_targets[mask], type_preds[mask])
            results[str(count)] = {
                "accuracy": acc,
                "n_samples": int(mask.sum()),
            }

    return results


def sample_from_mdn(
    mdn_params: dict,
    n_samples: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample locations from MDN distribution.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho (for a single prediction).
            - pi: [K] mixture weights
            - mu: [K, 2] component means
            - sigma: [K, 2] component standard deviations
            - rho: [K] correlations
        n_samples: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        Sampled locations [n_samples, 2]
    """
    if seed is not None:
        np.random.seed(seed)

    pi = mdn_params["pi"]
    mu = mdn_params["mu"]
    sigma = mdn_params["sigma"]
    rho = mdn_params["rho"]

    K = len(pi)
    samples = []

    for _ in range(n_samples):
        # Sample component from categorical distribution
        k = np.random.choice(K, p=pi)

        # Get component parameters
        mu_k = mu[k]  # [2]
        sigma_k = sigma[k]  # [2]
        rho_k = rho[k]

        # Build covariance matrix
        cov = np.array([
            [sigma_k[0] ** 2, rho_k * sigma_k[0] * sigma_k[1]],
            [rho_k * sigma_k[0] * sigma_k[1], sigma_k[1] ** 2],
        ])

        # Sample from bivariate Gaussian
        sample = np.random.multivariate_normal(mu_k, cov)
        samples.append(sample)

    return np.array(samples)


def plot_mdn_predictions(
    mdn_params: dict,
    targets: np.ndarray | None = None,
    n_samples: int = 20,
    figsize: tuple = (10, 10),
    confidence_level: float = 0.95,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize MDN predictions with uncertainty ellipses.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho for multiple predictions.
            - pi: [N, K] mixture weights
            - mu: [N, K, 2] component means
            - sigma: [N, K, 2] component standard deviations
            - rho: [N, K] correlations
        targets: Optional actual locations [N, 2] to overlay.
        n_samples: Number of predictions to visualize.
        figsize: Figure size.
        confidence_level: Confidence level for ellipses.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    from scipy import stats

    # Chi-squared threshold for ellipse scale
    chi2_val = stats.chi2.ppf(confidence_level, df=2)

    pi = mdn_params["pi"]
    mu = mdn_params["mu"]
    sigma = mdn_params["sigma"]
    rho = mdn_params["rho"]

    N = len(pi)
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        pi = pi[idx]
        mu = mu[idx]
        sigma = sigma[idx]
        rho = rho[idx]
        if targets is not None:
            targets = targets[idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Strike zone
    sz_left, sz_right = -0.83, 0.83
    sz_bottom, sz_top = 1.5, 3.5
    ax.add_patch(plt.Rectangle(
        (sz_left, sz_bottom), sz_right - sz_left, sz_top - sz_bottom,
        fill=False, edgecolor="red", linewidth=2, label="Strike Zone"
    ))

    # Color map for components
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each prediction
    for i in range(len(pi)):
        K = len(pi[i])

        for k in range(K):
            weight = pi[i, k]
            if weight < 0.05:  # Skip low-weight components
                continue

            mu_k = mu[i, k]
            sigma_k = sigma[i, k]
            rho_k = rho[i, k]

            # Compute ellipse parameters
            # Eigenvalue decomposition of covariance matrix
            cov = np.array([
                [sigma_k[0] ** 2, rho_k * sigma_k[0] * sigma_k[1]],
                [rho_k * sigma_k[0] * sigma_k[1], sigma_k[1] ** 2],
            ])

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Ellipse angle
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # Ellipse width and height (scaled by chi-squared value)
            width = 2 * np.sqrt(chi2_val * eigenvalues[0])
            height = 2 * np.sqrt(chi2_val * eigenvalues[1])

            # Draw ellipse with alpha proportional to weight
            ellipse = Ellipse(
                xy=mu_k,
                width=width,
                height=height,
                angle=angle,
                alpha=0.3 * weight,
                facecolor=colors[k % 10],
                edgecolor=colors[k % 10],
                linewidth=1,
            )
            ax.add_patch(ellipse)

            # Mark component center
            ax.scatter(
                mu_k[0], mu_k[1],
                c=[colors[k % 10]],
                s=50 * weight,
                marker="x",
                alpha=0.7,
            )

    # Plot actual targets if provided
    if targets is not None:
        ax.scatter(
            targets[:, 0], targets[:, 1],
            c="black", s=20, marker="o", alpha=0.7,
            label="Actual Locations", zorder=5
        )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("px (feet)")
    ax.set_ylabel("pz (feet)")
    ax.set_title(f"MDN Predictions ({int(confidence_level * 100)}% Confidence Ellipses)")
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def evaluate_pitch_type_location_model(
    model,
    dataloader,
    device: str = "auto",
) -> dict:
    """
    Evaluate a pitch-type-conditioned location model with per-pitch-type metrics.

    Args:
        model: PitchTypeConditionedMDN or similar model with forward(features, pitch_type_idx).
        dataloader: DataLoader yielding (features, pitch_type_idx, location) tuples.
        device: Device to use.

    Returns:
        Dictionary with overall and per-pitch-type metrics.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    all_params = {"pi": [], "mu": [], "sigma": [], "rho": []}
    all_targets = []
    all_pitch_types = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                features, pitch_type_idx, location = batch
            else:
                features = batch["features"]
                pitch_type_idx = batch["pitch_type_idx"]
                location = batch["location"]

            features = features.to(device)
            pitch_type_idx = pitch_type_idx.to(device)
            location = location.to(device)

            params = model(features, pitch_type_idx)

            # Collect results
            all_params["pi"].append(params["pi"].cpu().numpy())
            all_params["mu"].append(params["mu"].cpu().numpy())
            all_params["sigma"].append(params["sigma"].cpu().numpy())
            all_params["rho"].append(params["rho"].cpu().numpy())
            all_targets.append(location.cpu().numpy())
            all_pitch_types.append(pitch_type_idx.cpu().numpy())

    # Concatenate
    mdn_params = {
        "pi": np.concatenate(all_params["pi"]),
        "mu": np.concatenate(all_params["mu"]),
        "sigma": np.concatenate(all_params["sigma"]),
        "rho": np.concatenate(all_params["rho"]),
    }
    targets = np.concatenate(all_targets)
    pitch_types = np.concatenate(all_pitch_types)

    # Compute overall and per-pitch-type metrics
    return compute_per_pitch_type_metrics(mdn_params, targets, pitch_types)


def compute_per_pitch_type_metrics(
    mdn_params: dict,
    targets: np.ndarray,
    pitch_types: np.ndarray,
) -> dict:
    """
    Compute NLL, MAE, and coverage metrics broken down by pitch type.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho (all numpy arrays).
        targets: Target locations [N, 2].
        pitch_types: Pitch type indices [N].

    Returns:
        Dictionary with overall and per-pitch-type metrics.
    """
    # Overall metrics
    nll_values = compute_mdn_nll(mdn_params, targets)
    coverage_90 = compute_mdn_coverage(mdn_params, targets, 0.90)
    coverage_95 = compute_mdn_coverage(mdn_params, targets, 0.95)

    # Point predictions (weighted mean)
    preds = get_mdn_mean_prediction(mdn_params)
    errors = preds - targets
    mae_px = np.abs(errors[:, 0]).mean()
    mae_pz = np.abs(errors[:, 1]).mean()
    euclidean = np.sqrt((errors ** 2).sum(axis=1)).mean()

    metrics = {
        "overall": {
            "nll": float(nll_values.mean()),
            "mae_px": float(mae_px),
            "mae_pz": float(mae_pz),
            "euclidean": float(euclidean),
            "coverage_90": float(coverage_90),
            "coverage_95": float(coverage_95),
            "n_samples": len(targets),
        },
        "per_pitch_type": {},
    }

    # Per-pitch-type breakdown
    unique_pitch_types = np.unique(pitch_types)

    for pt_idx in unique_pitch_types:
        pt_idx = int(pt_idx)
        mask = pitch_types == pt_idx

        if mask.sum() == 0:
            continue

        # Get pitch type name
        if pt_idx < len(PITCH_TYPE_CODES):
            pt_code = PITCH_TYPE_CODES[pt_idx]
        else:
            pt_code = f"UNK{pt_idx}"

        # Filter for this pitch type
        pt_mdn_params = {
            "pi": mdn_params["pi"][mask],
            "mu": mdn_params["mu"][mask],
            "sigma": mdn_params["sigma"][mask],
            "rho": mdn_params["rho"][mask],
        }
        pt_targets = targets[mask]

        # Compute metrics
        pt_nll = compute_mdn_nll(pt_mdn_params, pt_targets)
        pt_coverage_90 = compute_mdn_coverage(pt_mdn_params, pt_targets, 0.90)
        pt_coverage_95 = compute_mdn_coverage(pt_mdn_params, pt_targets, 0.95)

        pt_preds = get_mdn_mean_prediction(pt_mdn_params)
        pt_errors = pt_preds - pt_targets
        pt_mae_px = np.abs(pt_errors[:, 0]).mean()
        pt_mae_pz = np.abs(pt_errors[:, 1]).mean()
        pt_euclidean = np.sqrt((pt_errors ** 2).sum(axis=1)).mean()

        metrics["per_pitch_type"][pt_code] = {
            "nll": float(pt_nll.mean()),
            "mae_px": float(pt_mae_px),
            "mae_pz": float(pt_mae_pz),
            "euclidean": float(pt_euclidean),
            "coverage_90": float(pt_coverage_90),
            "coverage_95": float(pt_coverage_95),
            "count": int(mask.sum()),
        }

    return metrics


def print_per_pitch_type_metrics(metrics: dict) -> None:
    """
    Print per-pitch-type metrics in a formatted table.

    Args:
        metrics: Dictionary from compute_per_pitch_type_metrics.
    """
    print("\nOverall Metrics:")
    overall = metrics["overall"]
    print(f"  NLL:        {overall['nll']:.4f}")
    print(f"  MAE px:     {overall['mae_px']:.4f} ft")
    print(f"  MAE pz:     {overall['mae_pz']:.4f} ft")
    print(f"  Euclidean:  {overall['euclidean']:.4f} ft")
    print(f"  Coverage 90%: {overall['coverage_90']:.1%}")
    print(f"  Coverage 95%: {overall['coverage_95']:.1%}")
    print(f"  N samples:  {overall['n_samples']:,}")

    print("\nPer-Pitch-Type Metrics:")
    print(f"{'Pitch':<8} {'NLL':>8} {'MAE_px':>8} {'MAE_pz':>8} {'Eucl':>8} {'Cov90':>8} {'Cov95':>8} {'Count':>10}")
    print("-" * 78)

    # Sort by count (most common first)
    sorted_types = sorted(
        metrics["per_pitch_type"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )

    for pt_code, pt_metrics in sorted_types:
        print(
            f"{pt_code:<8} "
            f"{pt_metrics['nll']:>8.3f} "
            f"{pt_metrics['mae_px']:>8.3f} "
            f"{pt_metrics['mae_pz']:>8.3f} "
            f"{pt_metrics['euclidean']:>8.3f} "
            f"{pt_metrics['coverage_90']:>7.1%} "
            f"{pt_metrics['coverage_95']:>7.1%} "
            f"{pt_metrics['count']:>10,}"
        )


def plot_location_by_pitch_type(
    mdn_params: dict,
    targets: np.ndarray,
    pitch_types: np.ndarray,
    n_samples_per_type: int = 200,
    figsize: tuple = (15, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot actual vs predicted locations grouped by pitch type.

    Args:
        mdn_params: Dict with pi, mu, sigma, rho.
        targets: Target locations [N, 2].
        pitch_types: Pitch type indices [N].
        n_samples_per_type: Max samples to plot per pitch type.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    unique_types = np.unique(pitch_types)
    n_types = len(unique_types)

    # Determine grid layout
    n_cols = min(4, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_types == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    # Strike zone boundaries
    sz_left, sz_right = -0.83, 0.83
    sz_bottom, sz_top = 1.5, 3.5

    for idx, pt_idx in enumerate(sorted(unique_types)):
        ax = axes[idx]
        mask = pitch_types == pt_idx

        # Get pitch type name
        if pt_idx < len(PITCH_TYPE_CODES):
            pt_code = PITCH_TYPE_CODES[int(pt_idx)]
        else:
            pt_code = f"UNK{pt_idx}"

        # Sample if too many
        pt_indices = np.where(mask)[0]
        if len(pt_indices) > n_samples_per_type:
            pt_indices = np.random.choice(pt_indices, n_samples_per_type, replace=False)

        pt_targets = targets[pt_indices]
        pt_preds = get_mdn_mean_prediction({
            "pi": mdn_params["pi"][pt_indices],
            "mu": mdn_params["mu"][pt_indices],
            "sigma": mdn_params["sigma"][pt_indices],
            "rho": mdn_params["rho"][pt_indices],
        })

        # Plot targets
        ax.scatter(
            pt_targets[:, 0], pt_targets[:, 1],
            alpha=0.3, s=10, c="blue", label="Actual"
        )

        # Plot predictions
        ax.scatter(
            pt_preds[:, 0], pt_preds[:, 1],
            alpha=0.3, s=10, c="red", label="Predicted"
        )

        # Strike zone
        ax.add_patch(plt.Rectangle(
            (sz_left, sz_bottom), sz_right - sz_left, sz_top - sz_bottom,
            fill=False, edgecolor="black", linewidth=2
        ))

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_xlabel("px (feet)")
        ax.set_ylabel("pz (feet)")
        ax.set_title(f"{pt_code} (n={mask.sum():,})")
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend()

    # Hide unused axes
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
