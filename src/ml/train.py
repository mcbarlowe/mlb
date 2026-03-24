"""
Training pipeline for pitch prediction models.

Provides training loop, loss functions, and utilities for model checkpointing.
"""

import math
from pathlib import Path
from typing import Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class MDNLoss(nn.Module):
    """
    Negative log-likelihood loss for 2D bivariate Gaussian Mixture.

    Computes the NLL of targets under the mixture distribution:
    NLL = -log(sum_k pi_k * N(x,y | mu_k, Sigma_k))

    where Sigma_k = [[sigma_x^2, rho*sigma_x*sigma_y],
                     [rho*sigma_x*sigma_y, sigma_y^2]]
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize MDN loss.

        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mdn_params: dict,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MDN negative log-likelihood loss.

        Args:
            mdn_params: Dictionary with keys:
                - pi: Mixture weights [batch, seq, K]
                - mu: Means [batch, seq, K, 2]
                - sigma: Standard deviations [batch, seq, K, 2]
                - rho: Correlations [batch, seq, K]
            targets: Target locations [batch, seq, 2] (px, pz)
            mask: Valid position mask [batch, seq]

        Returns:
            Scalar loss value.
        """
        pi = mdn_params["pi"]  # [batch, seq, K]
        mu = mdn_params["mu"]  # [batch, seq, K, 2]
        sigma = mdn_params["sigma"]  # [batch, seq, K, 2]
        rho = mdn_params["rho"]  # [batch, seq, K]

        # Expand targets for broadcasting with K components
        # [batch, seq, 2] -> [batch, seq, 1, 2]
        targets = targets.unsqueeze(-2)

        # Extract x and y components
        x = targets[..., 0]  # [batch, seq, 1]
        y = targets[..., 1]  # [batch, seq, 1]
        mu_x = mu[..., 0]  # [batch, seq, K]
        mu_y = mu[..., 1]  # [batch, seq, K]
        sigma_x = sigma[..., 0]  # [batch, seq, K]
        sigma_y = sigma[..., 1]  # [batch, seq, K]

        # Standardize the differences
        dx = (x - mu_x) / (sigma_x + self.eps)
        dy = (y - mu_y) / (sigma_y + self.eps)

        # Compute z = (dx^2 - 2*rho*dx*dy + dy^2) / (1 - rho^2)
        one_minus_rho_sq = (1 - rho**2).clamp(min=self.eps)
        z = (dx**2 - 2 * rho * dx * dy + dy**2) / one_minus_rho_sq

        # Log probability of bivariate Gaussian
        # log N = -log(2*pi) - log(sigma_x) - log(sigma_y) - 0.5*log(1-rho^2) - 0.5*z
        log_component_prob = (
            -math.log(2 * math.pi)
            - torch.log(sigma_x + self.eps)
            - torch.log(sigma_y + self.eps)
            - 0.5 * torch.log(one_minus_rho_sq)
            - 0.5 * z
        )

        # Log mixture probability: log(sum_k pi_k * exp(log_prob_k))
        # Use log-sum-exp trick for numerical stability
        log_pi = torch.log(pi + self.eps)
        log_mixture_prob = torch.logsumexp(log_pi + log_component_prob, dim=-1)

        # Negative log-likelihood
        nll = -log_mixture_prob

        # Masked mean
        masked_nll = nll * mask.float()
        loss = masked_nll.sum() / mask.float().sum().clamp(min=1)

        return loss


class PitchPredictionLoss(nn.Module):
    """
    Combined loss for pitch prediction.

    Combines:
    - CrossEntropyLoss for pitch type classification
    - MDN NLL loss for location (multi-modal distribution)
    """

    def __init__(
        self,
        type_weight: float = 1.0,
        location_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the loss function.

        Args:
            type_weight: Weight for pitch type classification loss.
            location_weight: Weight for MDN location loss.
            class_weights: Optional class weights for imbalanced pitch types.
        """
        super().__init__()
        self.type_weight = type_weight
        self.location_weight = location_weight

        self.type_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.location_loss = MDNLoss()

    def forward(
        self,
        pitch_type_logits: torch.Tensor,
        mdn_params: dict,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            pitch_type_logits: [batch, seq_len, n_classes]
            mdn_params: Dict with MDN parameters (pi, mu, sigma, rho)
            targets: [batch, seq_len, 3] (pitch_type_idx, px, pz)
            mask: [batch, seq_len] attention mask

        Returns:
            Tuple of (total_loss, loss_dict with individual losses)
        """
        # Extract targets
        pitch_type_target = targets[:, :, 0].long()
        location_target = targets[:, :, 1:3]

        # Reshape for cross-entropy
        batch_size, seq_len, n_classes = pitch_type_logits.shape
        logits_flat = pitch_type_logits.reshape(-1, n_classes)
        type_target_flat = pitch_type_target.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Mask invalid targets
        type_target_flat = torch.where(
            mask_flat, type_target_flat, torch.tensor(-1, device=type_target_flat.device)
        )

        # Classification loss
        type_loss = self.type_loss(logits_flat, type_target_flat)

        # MDN location loss (NLL)
        location_loss = self.location_loss(mdn_params, location_target, mask)

        # Combined loss
        total_loss = self.type_weight * type_loss + self.location_weight * location_loss

        return total_loss, {
            "type_loss": type_loss.item(),
            "location_loss": location_loss.item(),
            "total_loss": total_loss.item(),
        }


class PitchPredictionTrainer:
    """
    Trainer class for pitch prediction models.

    Handles training loop, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        type_weight: float = 1.0,
        location_weight: float = 0.5,
        checkpoint_dir: Optional[Path] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to use ("auto", "cuda", "mps", "cpu").
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight.
            type_weight: Weight for pitch type loss.
            location_weight: Weight for location loss.
            checkpoint_dir: Directory for saving checkpoints.
            class_weights: Optional tensor of class weights for imbalanced pitch types.
        """
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Move class weights to device if provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"Using class weights: {class_weights.tolist()}")

        # Loss function
        self.loss_fn = PitchPredictionLoss(
            type_weight=type_weight,
            location_weight=location_weight,
            class_weights=class_weights,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = []

    def train_epoch(self, show_batch_progress: bool = True) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_type_loss = 0
        total_location_loss = 0
        n_batches = 0

        loader = (
            tqdm(self.train_loader, desc="  Train", leave=False)
            if show_batch_progress
            else self.train_loader
        )
        for batch in loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pitch_type_logits, mdn_params = self.model(features, lengths, mask)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                pitch_type_logits, mdn_params, targets, mask
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_dict["total_loss"]
            total_type_loss += loss_dict["type_loss"]
            total_location_loss += loss_dict["location_loss"]
            n_batches += 1

            if show_batch_progress:
                loader.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "type": f"{loss_dict['type_loss']:.4f}",
                    "loc": f"{loss_dict['location_loss']:.4f}",
                })

        return {
            "train_loss": total_loss / n_batches,
            "train_type_loss": total_type_loss / n_batches,
            "train_location_loss": total_location_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, show_batch_progress: bool = True) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_type_loss = 0
        total_location_loss = 0
        n_correct = 0
        n_total = 0
        n_batches = 0

        loader = (
            tqdm(self.val_loader, desc="  Val", leave=False)
            if show_batch_progress
            else self.val_loader
        )
        for batch in loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            pitch_type_logits, mdn_params = self.model(features, lengths, mask)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                pitch_type_logits, mdn_params, targets, mask
            )

            # Accuracy
            predictions = pitch_type_logits.argmax(dim=-1)
            pitch_type_target = targets[:, :, 0].long()
            correct = (predictions == pitch_type_target) & mask
            n_correct += correct.sum().item()
            n_total += mask.sum().item()

            # Accumulate
            total_loss += loss_dict["total_loss"]
            total_type_loss += loss_dict["type_loss"]
            total_location_loss += loss_dict["location_loss"]
            n_batches += 1

        accuracy = n_correct / n_total if n_total > 0 else 0

        return {
            "val_loss": total_loss / n_batches,
            "val_type_loss": total_type_loss / n_batches,
            "val_location_loss": total_location_loss / n_batches,
            "val_accuracy": accuracy,
        }

    def train(
        self,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
        show_batch_progress: bool = True,
    ) -> dict:
        """
        Full training loop.

        Args:
            n_epochs: Maximum number of epochs.
            early_stopping_patience: Stop after this many epochs without improvement.
            show_batch_progress: Show progress bars for each batch (default True).
                Set to False for cleaner output with only epoch-level progress.

        Returns:
            Training history dictionary.
        """
        print(f"Starting training for up to {n_epochs} epochs")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Early stopping patience: {early_stopping_patience}")

        start_time = time.time()
        epoch_times = []

        # Epoch-level progress bar
        epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(show_batch_progress=show_batch_progress)

            # Validate
            val_metrics = self.validate(show_batch_progress=show_batch_progress)

            # Track epoch time
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Calculate ETA based on average epoch time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = n_epochs - epoch - 1
            eta_seconds = avg_epoch_time * remaining_epochs

            # Update learning rate
            self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self.training_history.append(epoch_metrics)

            # Check for improvement
            improved = ""
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
                improved = " *"
            else:
                self.epochs_without_improvement += 1

            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                "loss": f"{val_metrics['val_loss']:.3f}{improved}",
                "acc": f"{val_metrics['val_accuracy']:.1%}",
                "best": f"{self.best_val_loss:.3f}",
                "eta": f"{eta_seconds/60:.1f}m",
            })

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                epoch_pbar.set_description(f"Early stop @ epoch {epoch + 1}")
                break

        epoch_pbar.close()
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.training_history),
        }

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }, path)
        return path

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]


def train_model(
    data_path: Optional[str] = None,
    seasons: Optional[list[str]] = None,
    batch_size: int = 64,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.3,
    sample_frac: Optional[float] = None,
    device: str = "auto",
) -> dict:
    """
    Train a pitch prediction model from scratch.

    Args:
        data_path: Path to parquet data files.
        seasons: List of seasons to train on.
        batch_size: Training batch size.
        n_epochs: Maximum training epochs.
        learning_rate: Initial learning rate.
        hidden_dim: LSTM hidden dimension.
        n_layers: Number of LSTM layers.
        dropout: Dropout rate.
        sample_frac: Optional fraction to sample for faster iteration.
        device: Device to use.

    Returns:
        Training results dictionary.
    """
    from src.ml.dataset import PitchDataModule
    from src.ml.model import create_model

    # Load data
    print("Loading and preparing data...")
    data_module = PitchDataModule(
        data_path=data_path,
        seasons=seasons,
        batch_size=batch_size,
        sample_frac=sample_frac,
    )
    data_module.setup()

    # Create model
    print("Creating model...")
    model = create_model(
        n_pitch_types=data_module.n_pitch_types,
        n_pitchers=data_module.n_pitchers,
        n_batters=data_module.n_batters,
        n_features=data_module.n_features,
        model_type="lstm",
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = PitchPredictionTrainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        device=device,
        learning_rate=learning_rate,
    )

    # Train
    results = trainer.train(n_epochs=n_epochs)

    return {
        **results,
        "model": model,
        "trainer": trainer,
        "data_module": data_module,
    }
