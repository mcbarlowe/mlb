"""
Mixture Density Network for pitch location prediction.

This module provides a standalone MDN model for predicting pitch location
as a bivariate probability density, independent of the LSTM architecture.

The MDN outputs parameters for a mixture of bivariate Gaussians,
allowing for multimodal location predictions (e.g., inside vs outside corner).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats


class BivariateMDN(nn.Module):
    """
    Mixture Density Network for bivariate (px, pz) location prediction.

    Outputs a mixture of K bivariate Gaussians, parameterized by:
    - pi: mixture weights (K)
    - mu: means (K x 2)
    - sigma: standard deviations (K x 2)
    - rho: correlations (K)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int] = [256, 128, 64],
        n_components: int = 5,
        dropout: float = 0.2,
    ):
        """
        Initialize the MDN.

        Args:
            n_features: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            n_components: Number of Gaussian mixture components.
            dropout: Dropout rate.
        """
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features

        # Build MLP layers
        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # MDN output heads
        # 6 parameters per component: pi, mu_x, mu_y, sigma_x, sigma_y, rho
        self.pi_head = nn.Linear(in_dim, n_components)
        self.mu_head = nn.Linear(in_dim, n_components * 2)
        self.sigma_head = nn.Linear(in_dim, n_components * 2)
        self.rho_head = nn.Linear(in_dim, n_components)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: Input features [batch, n_features]

        Returns:
            Dictionary with MDN parameters:
            - pi: [batch, K] mixture weights
            - mu: [batch, K, 2] means
            - sigma: [batch, K, 2] standard deviations
            - rho: [batch, K] correlations
        """
        h = self.mlp(x)

        # Mixture weights (softmax for valid probabilities)
        pi = F.softmax(self.pi_head(h), dim=-1)

        # Means (unconstrained)
        mu = self.mu_head(h).view(-1, self.n_components, 2)

        # Standard deviations (positive, with minimum for stability)
        sigma = F.softplus(self.sigma_head(h)).view(-1, self.n_components, 2)
        sigma = sigma.clamp(min=0.01, max=5.0)

        # Correlation (between -1 and 1, using tanh)
        rho = torch.tanh(self.rho_head(h)) * 0.99  # Avoid exactly +/-1

        return {
            "pi": pi,
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
        }

    def log_prob(self, params: dict, target: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of target under the mixture.

        Args:
            params: MDN parameters from forward()
            target: Target locations [batch, 2]

        Returns:
            Log probability [batch]
        """
        pi = params["pi"]  # [batch, K]
        mu = params["mu"]  # [batch, K, 2]
        sigma = params["sigma"]  # [batch, K, 2]
        rho = params["rho"]  # [batch, K]

        # Expand target for broadcasting
        target = target.unsqueeze(1)  # [batch, 1, 2]

        # Compute bivariate Gaussian log probability for each component
        dx = target[..., 0] - mu[..., 0]  # [batch, K]
        dy = target[..., 1] - mu[..., 1]  # [batch, K]
        sx = sigma[..., 0]
        sy = sigma[..., 1]

        # Bivariate Gaussian log probability
        z = (dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * (dx / sx) * (dy / sy)
        log_exp = -z / (2 * (1 - rho ** 2))
        log_norm = -np.log(2 * np.pi) - torch.log(sx) - torch.log(sy) - 0.5 * torch.log(1 - rho ** 2)

        # Log probability for each component
        log_component = log_norm + log_exp  # [batch, K]

        # Log-sum-exp for mixture
        log_prob = torch.logsumexp(torch.log(pi + 1e-10) + log_component, dim=-1)

        return log_prob

    def sample(self, params: dict, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the mixture distribution.

        Args:
            params: MDN parameters from forward()
            n_samples: Number of samples per input

        Returns:
            Samples [batch, n_samples, 2]
        """
        pi = params["pi"]  # [batch, K]
        mu = params["mu"]  # [batch, K, 2]
        sigma = params["sigma"]  # [batch, K, 2]
        rho = params["rho"]  # [batch, K]

        batch_size = pi.shape[0]
        device = pi.device

        samples = []
        for _ in range(n_samples):
            # Sample component indices
            component_idx = torch.multinomial(pi, 1).squeeze(-1)  # [batch]

            # Get parameters for selected components
            batch_idx = torch.arange(batch_size, device=device)
            mu_selected = mu[batch_idx, component_idx]  # [batch, 2]
            sigma_selected = sigma[batch_idx, component_idx]  # [batch, 2]
            rho_selected = rho[batch_idx, component_idx]  # [batch]

            # Sample from bivariate Gaussian using Cholesky decomposition
            z = torch.randn(batch_size, 2, device=device)

            # Cholesky factor for correlation
            L = torch.zeros(batch_size, 2, 2, device=device)
            L[:, 0, 0] = 1.0
            L[:, 1, 0] = rho_selected
            L[:, 1, 1] = torch.sqrt(1 - rho_selected ** 2)

            # Transform standard normal to correlated
            correlated = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # [batch, 2]

            # Scale and shift
            sample = mu_selected + sigma_selected * correlated
            samples.append(sample)

        return torch.stack(samples, dim=1)  # [batch, n_samples, 2]

    def get_mode(self, params: dict) -> torch.Tensor:
        """
        Get the mode (most likely point) from the mixture.

        Returns the mean of the component with highest weight.

        Args:
            params: MDN parameters

        Returns:
            Mode locations [batch, 2]
        """
        pi = params["pi"]
        mu = params["mu"]

        # Get index of highest weight component
        max_idx = torch.argmax(pi, dim=-1)  # [batch]
        batch_idx = torch.arange(pi.shape[0], device=pi.device)

        return mu[batch_idx, max_idx]  # [batch, 2]

    def get_expected_value(self, params: dict) -> torch.Tensor:
        """
        Get the expected value (weighted mean) from the mixture.

        Args:
            params: MDN parameters

        Returns:
            Expected locations [batch, 2]
        """
        pi = params["pi"]  # [batch, K]
        mu = params["mu"]  # [batch, K, 2]

        # Weighted sum of means
        return torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # [batch, 2]


class MDNLocationTrainer:
    """Trainer for the MDN location model."""

    def __init__(
        self,
        model: BivariateMDN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            params = self.model(X)
            log_prob = self.model.log_prob(params, y)
            loss = -log_prob.mean()  # Negative log likelihood

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_nll = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                params = self.model(X)
                log_prob = self.model.log_prob(params, y)
                total_nll -= log_prob.sum().item()

                # Get point predictions for MAE
                pred = self.model.get_expected_value(params)
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())
                n_batches += 1

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        mae_px = np.mean(np.abs(preds[:, 0] - targets[:, 0]))
        mae_pz = np.mean(np.abs(preds[:, 1] - targets[:, 1]))
        euclidean = np.mean(np.sqrt((preds[:, 0] - targets[:, 0])**2 +
                                     (preds[:, 1] - targets[:, 1])**2))

        return {
            "nll": total_nll / len(torch.cat(all_targets)),
            "mae_px": mae_px,
            "mae_pz": mae_pz,
            "euclidean": euclidean,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> dict:
        """Full training loop."""
        best_val_nll = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_nll": [], "val_mae_px": [], "val_mae_pz": []}

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["nll"])

            history["train_loss"].append(train_loss)
            history["val_nll"].append(val_metrics["nll"])
            history["val_mae_px"].append(val_metrics["mae_px"])
            history["val_mae_pz"].append(val_metrics["mae_pz"])

            if val_metrics["nll"] < best_val_nll:
                best_val_nll = val_metrics["nll"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                      f"val_nll={val_metrics['nll']:.4f}, "
                      f"val_mae={val_metrics['euclidean']:.4f}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        self.model.load_state_dict(best_state)

        return {"history": history, "best_val_nll": best_val_nll}


def plot_density_prediction(
    model: BivariateMDN,
    features: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    n_samples: int = 1000,
    title: str = "Predicted Pitch Location Density",
    save_path: Optional[str] = None,
    ax=None,
):
    """
    Plot the predicted bivariate density for a single pitch.

    Args:
        model: Trained MDN model
        features: Input features [1, n_features] or [n_features]
        target: Optional true location [2]
        n_samples: Number of samples for KDE estimation
        title: Plot title
        save_path: Optional path to save figure
        ax: Optional matplotlib axis
    """
    model.eval()

    if features.dim() == 1:
        features = features.unsqueeze(0)

    # Move features to same device as model
    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        params = model(features)
        samples = model.sample(params, n_samples=n_samples)
        samples = samples[0].cpu().numpy()  # [n_samples, 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    # Create KDE
    px_samples = samples[:, 0]
    pz_samples = samples[:, 1]

    # Grid for density estimation
    px_grid = np.linspace(-2.5, 2.5, 100)
    pz_grid = np.linspace(0, 5, 100)
    PX, PZ = np.meshgrid(px_grid, pz_grid)
    positions = np.vstack([PX.ravel(), PZ.ravel()])

    # Fit KDE
    try:
        kernel = stats.gaussian_kde(samples.T)
        density = kernel(positions).reshape(PX.shape)

        # Plot density contours
        ax.contourf(PX, PZ, density, levels=20, cmap='YlOrRd', alpha=0.8)
        ax.contour(PX, PZ, density, levels=10, colors='darkred', alpha=0.5, linewidths=0.5)
    except:
        # Fallback to scatter if KDE fails
        ax.scatter(px_samples, pz_samples, alpha=0.3, s=5, c='red')

    # Draw strike zone
    strike_zone = plt.Rectangle(
        (-0.83, 1.5), 1.66, 2.0,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(strike_zone)

    # Plot true location if provided
    if target is not None:
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        ax.scatter(target[0], target[1], c='blue', s=100, marker='x',
                   linewidths=3, label='Actual', zorder=10)

    # Plot mixture component means
    pi = params["pi"][0].cpu().numpy()
    mu = params["mu"][0].cpu().numpy()
    for i in range(len(pi)):
        if pi[i] > 0.1:  # Only show significant components
            ax.scatter(mu[i, 0], mu[i, 1], c='green', s=50*pi[i]*10,
                       marker='o', alpha=0.7, edgecolors='darkgreen')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Horizontal Position (ft)')
    ax.set_ylabel('Vertical Position (ft)')
    ax.set_title(title)
    ax.set_aspect('equal')

    if target is not None:
        ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return ax


def get_location_density(
    model: BivariateMDN,
    features: torch.Tensor,
    px_range: tuple = (-2.5, 2.5),
    pz_range: tuple = (0.5, 4.5),
    grid_size: int = 100,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a bivariate kernel density estimate for pitch location.

    This is the main function for getting the location probability density
    that can be used for visualization or downstream analysis.

    Args:
        model: Trained MDN model
        features: Input features [1, n_features] or [n_features]
        px_range: Range for horizontal axis (ft)
        pz_range: Range for vertical axis (ft)
        grid_size: Number of grid points per axis
        n_samples: Number of samples for KDE

    Returns:
        Tuple of (px_grid, pz_grid, density):
        - px_grid: 1D array of horizontal coordinates
        - pz_grid: 1D array of vertical coordinates
        - density: 2D array of probability densities [pz_grid_size, px_grid_size]

    Example:
        >>> px, pz, density = get_location_density(model, features)
        >>> plt.contourf(px, pz, density, levels=20, cmap='YlOrRd')
    """
    model.eval()

    if features.dim() == 1:
        features = features.unsqueeze(0)

    # Move features to same device as model
    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        params = model(features)
        samples = model.sample(params, n_samples=n_samples)
        samples = samples[0].cpu().numpy()  # [n_samples, 2]

    # Create grid
    px_grid = np.linspace(px_range[0], px_range[1], grid_size)
    pz_grid = np.linspace(pz_range[0], pz_range[1], grid_size)
    PX, PZ = np.meshgrid(px_grid, pz_grid)
    positions = np.vstack([PX.ravel(), PZ.ravel()])

    # Fit KDE to samples
    from scipy import stats
    try:
        kernel = stats.gaussian_kde(samples.T, bw_method='scott')
        density = kernel(positions).reshape(PX.shape)
    except Exception:
        # Fallback: use mixture parameters directly
        density = np.zeros_like(PX)
        pi = params["pi"][0].cpu().numpy()
        mu = params["mu"][0].cpu().numpy()
        sigma = params["sigma"][0].cpu().numpy()
        rho = params["rho"][0].cpu().numpy()

        for k in range(len(pi)):
            if pi[k] < 0.01:
                continue
            # Bivariate Gaussian
            dx = PX - mu[k, 0]
            dy = PZ - mu[k, 1]
            sx, sy = sigma[k, 0], sigma[k, 1]
            r = rho[k]

            z = (dx/sx)**2 + (dy/sy)**2 - 2*r*(dx/sx)*(dy/sy)
            coef = 1 / (2 * np.pi * sx * sy * np.sqrt(1 - r**2))
            component_density = coef * np.exp(-z / (2 * (1 - r**2)))
            density += pi[k] * component_density

    return px_grid, pz_grid, density


def get_mixture_parameters(
    model: BivariateMDN,
    features: torch.Tensor,
) -> dict:
    """
    Get the raw mixture parameters for a prediction.

    Useful for understanding the multimodal structure of the prediction.

    Args:
        model: Trained MDN model
        features: Input features

    Returns:
        Dictionary with:
        - weights: Mixture weights [K]
        - means: Component means [K, 2]
        - stds: Component standard deviations [K, 2]
        - correlations: Component correlations [K]
        - expected_location: Weighted mean [2]
        - mode_location: Mean of highest-weight component [2]
    """
    model.eval()

    if features.dim() == 1:
        features = features.unsqueeze(0)

    # Move features to same device as model
    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        params = model(features)

        pi = params["pi"][0].cpu().numpy()
        mu = params["mu"][0].cpu().numpy()
        sigma = params["sigma"][0].cpu().numpy()
        rho = params["rho"][0].cpu().numpy()

        expected = model.get_expected_value(params)[0].cpu().numpy()
        mode = model.get_mode(params)[0].cpu().numpy()

    return {
        "weights": pi,
        "means": mu,
        "stds": sigma,
        "correlations": rho,
        "expected_location": expected,
        "mode_location": mode,
    }


def get_point_estimate(
    model: BivariateMDN,
    features: torch.Tensor,
    method: str = "expected",
) -> np.ndarray:
    """
    Get a point estimate of the pitch location.

    This is a convenience function for when you need a single location prediction
    rather than the full probability density.

    Args:
        model: Trained MDN model
        features: Input features [batch, n_features] or [n_features]
        method: One of:
            - "expected": Weighted mean of all components (default, most stable)
            - "mode": Mean of the highest-weight component (most likely region)

    Returns:
        Point estimate [batch, 2] or [2] if input was unbatched

    Example:
        >>> point = get_point_estimate(model, features)
        >>> print(f"Predicted location: px={point[0]:.2f}, pz={point[1]:.2f}")
    """
    model.eval()
    unbatched = features.dim() == 1

    if unbatched:
        features = features.unsqueeze(0)

    # Move features to same device as model
    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        params = model(features)

        if method == "expected":
            result = model.get_expected_value(params).cpu().numpy()
        elif method == "mode":
            result = model.get_mode(params).cpu().numpy()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'expected' or 'mode'.")

    if unbatched:
        return result[0]
    return result


def predict_location_batch(
    model: BivariateMDN,
    dataloader: DataLoader,
    device: str = "cpu",
    method: str = "expected",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get point estimates for a full dataset.

    Args:
        model: Trained MDN model
        dataloader: DataLoader with (features, targets)
        device: Device to run on
        method: Point estimate method ("expected" or "mode")

    Returns:
        Tuple of (predictions, targets) as numpy arrays, both [N, 2]
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            params = model(X)

            if method == "expected":
                preds = model.get_expected_value(params)
            else:
                preds = model.get_mode(params)

            all_preds.append(preds.cpu())
            all_targets.append(y)

    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def plot_multiple_densities(
    model: BivariateMDN,
    features_list: list[torch.Tensor],
    targets_list: Optional[list[torch.Tensor]] = None,
    titles: Optional[list[str]] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
):
    """
    Plot multiple density predictions in a grid.

    Args:
        model: Trained MDN model
        features_list: List of input features
        targets_list: Optional list of true locations
        titles: Optional list of titles
        n_samples: Number of samples per prediction
        save_path: Optional path to save figure
    """
    n_plots = len(features_list)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, features in enumerate(features_list):
        target = targets_list[i] if targets_list else None
        title = titles[i] if titles else f"Prediction {i+1}"
        plot_density_prediction(
            model, features, target, n_samples, title, ax=axes[i]
        )

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
