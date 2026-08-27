"""
Neural network architecture for pitch prediction.

Dual-head LSTM model that predicts:
- Pitch type (classification)
- Pitch location (Mixture Density Network for multi-modal distributions)

Includes attention-based variant for improved sequence modeling.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Allows the model to focus on different parts of the pitch sequence
    when making predictions.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1, causal: bool = True):
        """
        Initialize multi-head attention.

        Args:
            hidden_dim: Dimension of input/output features.
            n_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
            causal: If True, apply causal masking to prevent attending to future positions.
        """
        super().__init__()

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.causal = causal

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim].
            mask: Attention mask [batch, seq_len]. True = valid, False = padding.
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights).
            - output: [batch, seq_len, hidden_dim]
            - attention_weights: [batch, n_heads, seq_len, seq_len] or None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq, hidden]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, n_heads, seq, head_dim] @ [batch, n_heads, head_dim, seq] -> [batch, n_heads, seq, seq]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask if needed (prevent attending to future positions)
        if self.causal:
            # Create lower triangular mask: position i can only attend to positions <= i
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            # Expand for batch and heads: [seq, seq] -> [1, 1, seq, seq]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            # Expand mask for broadcasting: [batch, seq] -> [batch, 1, 1, seq]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, n_heads, seq, seq] @ [batch, n_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.out_proj(attn_output)

        if return_weights:
            return output, attn_weights
        return output, None


class AttentionBlock(nn.Module):
    """
    Transformer-style attention block with residual connection and layer norm.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim].
            mask: Attention mask [batch, seq_len].
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights).
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, mask, return_weights)
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class PitchPredictor(nn.Module):
    """
    LSTM-based pitch prediction model with dual heads.

    Architecture:
    - Embeddings for categorical features (pitcher, batter, pitch type)
    - LSTM encoder for sequence modeling
    - Classification head for pitch type
    - MDN head for location (bivariate Gaussian mixture for multi-modal predictions)
    """

    def __init__(
        self,
        n_pitch_types: int,
        n_pitchers: int,
        n_batters: int,
        n_continuous_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_location_components: int = 3,
        feature_indices: dict | None = None,
    ):
        """
        Initialize the model.

        Args:
            n_pitch_types: Number of pitch type classes.
            n_pitchers: Number of unique pitchers (for embedding).
            n_batters: Number of unique batters (for embedding).
            n_continuous_features: Number of continuous input features.
            embedding_dim: Dimension for embeddings.
            hidden_dim: LSTM hidden dimension.
            n_layers: Number of LSTM layers.
            dropout: Dropout rate.
            n_location_components: Number of mixture components for MDN (default 3).
            feature_indices: Dict with 'embedding_indices' and 'continuous_indices'
                from PitchFeatureEngine.get_feature_indices(). If None, uses defaults.
        """
        super().__init__()

        self.n_pitch_types = n_pitch_types
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_location_components = n_location_components

        # Store feature indices for dynamic extraction
        if feature_indices is not None:
            self.pitcher_idx_pos = feature_indices["embedding_indices"]["pitcher_idx"]
            self.batter_idx_pos = feature_indices["embedding_indices"]["batter_idx"]
            self.prev_pitch_idx_pos = feature_indices["embedding_indices"]["prev_pitch_type_idx"]
            self.continuous_indices = feature_indices["continuous_indices"]
        else:
            # Fallback to current 31-feature layout
            self.pitcher_idx_pos = 10
            self.batter_idx_pos = 11
            self.prev_pitch_idx_pos = 19
            self.continuous_indices = list(range(10)) + list(range(12, 19)) + list(range(20, 31))

        n_continuous = len(self.continuous_indices)

        # Embeddings for categorical features
        self.pitcher_embedding = nn.Embedding(n_pitchers, embedding_dim)
        self.batter_embedding = nn.Embedding(n_batters, embedding_dim)
        self.prev_pitch_embedding = nn.Embedding(
            n_pitch_types + 1, embedding_dim, padding_idx=n_pitch_types
        )  # +1 for "no previous pitch"

        # Calculate total input size
        # Embeddings: pitcher + batter + prev_pitch = 3 * embedding_dim
        # Continuous: n_continuous (already excludes embedding indices)
        self.input_size = 3 * embedding_dim + n_continuous

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.pitch_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_pitch_types),
        )

        # MDN location head outputs 6*K parameters per prediction:
        # K mixture weights (pi), 2K means (mu_x, mu_y), 2K stds (sigma_x, sigma_y), K correlations (rho)
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6 * n_location_components),
        )

    def _parse_mdn_params(self, raw_output: torch.Tensor) -> dict:
        """
        Parse raw MDN output into distribution parameters.

        Args:
            raw_output: Raw output from location head [batch, seq, 6*K].

        Returns:
            Dictionary with keys:
            - pi: Mixture weights [batch, seq, K]
            - mu: Means [batch, seq, K, 2]
            - sigma: Standard deviations [batch, seq, K, 2]
            - rho: Correlations [batch, seq, K]
        """
        K = self.n_location_components
        batch, seq, _ = raw_output.shape

        # Split output into components
        pi_logits = raw_output[..., :K]
        mu = raw_output[..., K : 3 * K].reshape(batch, seq, K, 2)
        log_sigma = raw_output[..., 3 * K : 5 * K].reshape(batch, seq, K, 2)
        rho_raw = raw_output[..., 5 * K : 6 * K]

        # Apply activations
        pi = F.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        rho = torch.tanh(rho_raw) * 0.99  # Avoid exactly +/-1 for numerical stability

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            features: Input features [batch, seq_len, n_features].
            lengths: Sequence lengths [batch].
            mask: Attention mask [batch, seq_len].

        Returns:
            Tuple of (pitch_type_logits, mdn_params).
            - pitch_type_logits: [batch, seq_len, n_pitch_types]
            - mdn_params: dict with keys 'pi', 'mu', 'sigma', 'rho'
        """
        _batch_size, seq_len, _n_features = features.shape

        # Extract embedding indices using stored positions
        pitcher_idx = features[:, :, self.pitcher_idx_pos].long().clamp(
            0, self.pitcher_embedding.num_embeddings - 1
        )
        batter_idx = features[:, :, self.batter_idx_pos].long().clamp(
            0, self.batter_embedding.num_embeddings - 1
        )
        prev_pitch_idx = features[:, :, self.prev_pitch_idx_pos].long().clamp(
            -1, self.n_pitch_types - 1
        )
        # Map -1 (no previous pitch) to the padding index
        prev_pitch_idx = torch.where(
            prev_pitch_idx < 0,
            torch.tensor(self.n_pitch_types, device=prev_pitch_idx.device),
            prev_pitch_idx,
        )

        # Get embeddings
        pitcher_emb = self.pitcher_embedding(pitcher_idx)
        batter_emb = self.batter_embedding(batter_idx)
        prev_pitch_emb = self.prev_pitch_embedding(prev_pitch_idx)

        # Extract continuous features using stored indices
        continuous_features = features[:, :, self.continuous_indices]

        # Concatenate all features
        x = torch.cat([
            pitcher_emb,
            batter_emb,
            prev_pitch_emb,
            continuous_features,
        ], dim=-1)

        # Pack sequence for LSTM
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        lstm_out = self.dropout(lstm_out)

        # Output heads
        pitch_type_logits = self.pitch_type_head(lstm_out)
        raw_location_out = self.location_head(lstm_out)

        # Parse MDN parameters
        mdn_params = self._parse_mdn_params(raw_location_out)

        return pitch_type_logits, mdn_params


class PitchPredictorWithAttention(nn.Module):
    """
    LSTM + Attention model for pitch prediction.

    Architecture:
    - Embeddings for categorical features (pitcher, batter, pitch type)
    - LSTM encoder for sequence modeling
    - Multi-head self-attention layer to focus on relevant previous pitches
    - Classification head for pitch type
    - MDN head for location

    Based on Yu et al. 2022: "Attention-Based LSTM for Pitch Prediction"
    which achieved 76.7% accuracy with per-pitcher models.
    """

    def __init__(
        self,
        n_pitch_types: int,
        n_pitchers: int,
        n_batters: int,
        n_continuous_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_location_components: int = 3,
        n_attention_heads: int = 4,
        n_attention_layers: int = 1,
        feature_indices: dict | None = None,
    ):
        """
        Initialize the attention-based model.

        Args:
            n_pitch_types: Number of pitch type classes.
            n_pitchers: Number of unique pitchers (for embedding).
            n_batters: Number of unique batters (for embedding).
            n_continuous_features: Number of continuous input features.
            embedding_dim: Dimension for embeddings.
            hidden_dim: LSTM hidden dimension.
            n_layers: Number of LSTM layers.
            dropout: Dropout rate.
            n_location_components: Number of mixture components for MDN.
            n_attention_heads: Number of attention heads.
            n_attention_layers: Number of stacked attention blocks.
            feature_indices: Dict with 'embedding_indices' and 'continuous_indices'.
        """
        super().__init__()

        self.n_pitch_types = n_pitch_types
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_location_components = n_location_components

        # Store feature indices for dynamic extraction
        if feature_indices is not None:
            self.pitcher_idx_pos = feature_indices["embedding_indices"]["pitcher_idx"]
            self.batter_idx_pos = feature_indices["embedding_indices"]["batter_idx"]
            self.prev_pitch_idx_pos = feature_indices["embedding_indices"]["prev_pitch_type_idx"]
            self.continuous_indices = feature_indices["continuous_indices"]
        else:
            # Fallback to current layout
            self.pitcher_idx_pos = 10
            self.batter_idx_pos = 11
            self.prev_pitch_idx_pos = 19
            self.continuous_indices = list(range(10)) + list(range(12, 19)) + list(range(20, 31))

        n_continuous = len(self.continuous_indices)

        # Embeddings for categorical features
        self.pitcher_embedding = nn.Embedding(n_pitchers, embedding_dim)
        self.batter_embedding = nn.Embedding(n_batters, embedding_dim)
        self.prev_pitch_embedding = nn.Embedding(
            n_pitch_types + 1, embedding_dim, padding_idx=n_pitch_types
        )

        # Calculate total input size
        self.input_size = 3 * embedding_dim + n_continuous

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention layers (stacked)
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, n_attention_heads, dropout)
            for _ in range(n_attention_layers)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.pitch_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_pitch_types),
        )

        # MDN location head
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6 * n_location_components),
        )

    def _parse_mdn_params(self, raw_output: torch.Tensor) -> dict:
        """Parse raw MDN output into distribution parameters."""
        K = self.n_location_components
        batch, seq, _ = raw_output.shape

        pi_logits = raw_output[..., :K]
        mu = raw_output[..., K : 3 * K].reshape(batch, seq, K, 2)
        log_sigma = raw_output[..., 3 * K : 5 * K].reshape(batch, seq, K, 2)
        rho_raw = raw_output[..., 5 * K : 6 * K]

        pi = F.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        rho = torch.tanh(rho_raw) * 0.99

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
        """
        Forward pass with attention.

        Args:
            features: Input features [batch, seq_len, n_features].
            lengths: Sequence lengths [batch].
            mask: Attention mask [batch, seq_len].
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (pitch_type_logits, mdn_params, attention_weights).
        """
        _batch_size, seq_len, _n_features = features.shape

        # Extract embedding indices
        pitcher_idx = features[:, :, self.pitcher_idx_pos].long().clamp(
            0, self.pitcher_embedding.num_embeddings - 1
        )
        batter_idx = features[:, :, self.batter_idx_pos].long().clamp(
            0, self.batter_embedding.num_embeddings - 1
        )
        prev_pitch_idx = features[:, :, self.prev_pitch_idx_pos].long().clamp(
            -1, self.n_pitch_types - 1
        )
        prev_pitch_idx = torch.where(
            prev_pitch_idx < 0,
            torch.tensor(self.n_pitch_types, device=prev_pitch_idx.device),
            prev_pitch_idx,
        )

        # Get embeddings
        pitcher_emb = self.pitcher_embedding(pitcher_idx)
        batter_emb = self.batter_embedding(batter_idx)
        prev_pitch_emb = self.prev_pitch_embedding(prev_pitch_idx)

        # Extract continuous features
        continuous_features = features[:, :, self.continuous_indices]

        # Concatenate all features
        x = torch.cat([
            pitcher_emb,
            batter_emb,
            prev_pitch_emb,
            continuous_features,
        ], dim=-1)

        # Pack sequence for LSTM
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        # Apply attention layers
        attn_weights = None
        for attn_layer in self.attention_layers:
            lstm_out, attn_weights = attn_layer(
                lstm_out, mask, return_weights=return_attention
            )

        lstm_out = self.dropout(lstm_out)

        # Output heads
        pitch_type_logits = self.pitch_type_head(lstm_out)
        raw_location_out = self.location_head(lstm_out)

        # Parse MDN parameters
        mdn_params = self._parse_mdn_params(raw_location_out)

        if return_attention:
            return pitch_type_logits, mdn_params, attn_weights
        return pitch_type_logits, mdn_params


class HierarchicalMDN(nn.Module):
    """
    Hierarchical Mixture Density Network for pitch location prediction.

    Uses pitch-family-specific mixture components:
    - Fastball family (FF, SI, FC): Higher in zone
    - Offspeed family (CH, FS): Mid-low
    - Breaking family (SL, CU, KC, ST): Low and glove-side

    Each family has its own set of mixture components, and the final
    distribution is a weighted combination based on the predicted pitch type.
    """

    # Pitch family assignments (indices into PITCH_TYPE_CODES)
    FASTBALL_INDICES = (0, 1, 2)  # FF, SI, FC
    OFFSPEED_INDICES = (3, 8)  # CH, FS
    BREAKING_INDICES = (4, 5, 6, 7)  # SL, CU, KC, ST
    # KN (9) and OTHER (10) use a generic head

    def __init__(
        self,
        hidden_dim: int,
        n_pitch_types: int,
        embedding_dim: int = 32,
        n_components_per_family: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize hierarchical MDN.

        Args:
            hidden_dim: Dimension of input hidden state.
            n_pitch_types: Number of pitch type classes.
            embedding_dim: Dimension for pitch type embedding.
            n_components_per_family: Number of mixture components per pitch family.
            dropout: Dropout rate.
        """
        super().__init__()

        self.n_pitch_types = n_pitch_types
        self.n_components_per_family = n_components_per_family
        self.n_families = 4  # fastball, offspeed, breaking, other
        self.total_components = self.n_families * n_components_per_family

        # Pitch type embedding for conditioning
        self.pitch_type_embedding = nn.Embedding(n_pitch_types, embedding_dim)

        # Shared feature processing (takes hidden state + pitch type embedding)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Family-specific MDN heads
        # Each outputs: K*(1 + 2 + 2 + 1) = 6K parameters per family
        # (K weights, 2K means, 2K stds, K correlations)
        self.fastball_head = nn.Linear(hidden_dim, 6 * n_components_per_family)
        self.offspeed_head = nn.Linear(hidden_dim, 6 * n_components_per_family)
        self.breaking_head = nn.Linear(hidden_dim, 6 * n_components_per_family)
        self.other_head = nn.Linear(hidden_dim, 6 * n_components_per_family)

        # Mapping from pitch type index to family index
        self.register_buffer(
            'pitch_to_family',
            self._create_pitch_family_mapping(n_pitch_types)
        )

    def _create_pitch_family_mapping(self, n_pitch_types: int) -> torch.Tensor:
        """Create mapping from pitch type index to family index."""
        mapping = torch.zeros(n_pitch_types, dtype=torch.long)
        for idx in self.FASTBALL_INDICES:
            if idx < n_pitch_types:
                mapping[idx] = 0
        for idx in self.OFFSPEED_INDICES:
            if idx < n_pitch_types:
                mapping[idx] = 1
        for idx in self.BREAKING_INDICES:
            if idx < n_pitch_types:
                mapping[idx] = 2
        # Default (KN, OTHER) to family 3
        for idx in range(n_pitch_types):
            if idx not in self.FASTBALL_INDICES + self.OFFSPEED_INDICES + self.BREAKING_INDICES:
                mapping[idx] = 3
        return mapping

    def _parse_family_params(self, raw_output: torch.Tensor) -> dict:
        """Parse raw output from one family head into MDN parameters."""
        K = self.n_components_per_family

        # Handle both 2D and 3D input
        if raw_output.dim() == 2:
            raw_output = raw_output.unsqueeze(1)

        batch, seq, _ = raw_output.shape

        pi_logits = raw_output[..., :K]
        mu = raw_output[..., K:3*K].reshape(batch, seq, K, 2)
        log_sigma = raw_output[..., 3*K:5*K].reshape(batch, seq, K, 2)
        rho_raw = raw_output[..., 5*K:6*K]

        pi = F.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        rho = torch.tanh(rho_raw) * 0.99

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward(
        self,
        hidden_state: torch.Tensor,
        pitch_type_probs: torch.Tensor,
    ) -> dict:
        """
        Forward pass with pitch-type-conditioned location prediction.

        Args:
            hidden_state: LSTM output [batch, seq, hidden_dim].
            pitch_type_probs: Predicted pitch type probabilities [batch, seq, n_pitch_types].

        Returns:
            Dictionary with combined MDN parameters.
        """
        batch, seq, _ = hidden_state.shape
        device = hidden_state.device

        # Get expected pitch type embedding (weighted by probabilities)
        # [batch, seq, n_pitch_types] @ [n_pitch_types, embedding_dim] -> [batch, seq, embedding_dim]
        pitch_type_emb = torch.matmul(pitch_type_probs, self.pitch_type_embedding.weight)

        # Concatenate hidden state with pitch type embedding
        conditioned = torch.cat([hidden_state, pitch_type_emb], dim=-1)

        # Shared processing
        shared_features = self.shared_layer(conditioned)

        # Get outputs from each family head
        fastball_raw = self.fastball_head(shared_features)
        offspeed_raw = self.offspeed_head(shared_features)
        breaking_raw = self.breaking_head(shared_features)
        other_raw = self.other_head(shared_features)

        # Parse each family's parameters
        fastball_params = self._parse_family_params(fastball_raw)
        offspeed_params = self._parse_family_params(offspeed_raw)
        breaking_params = self._parse_family_params(breaking_raw)
        other_params = self._parse_family_params(other_raw)

        # Compute family weights from pitch type probabilities
        # Sum probabilities for each family
        family_weights = torch.zeros(batch, seq, self.n_families, device=device)

        for idx in self.FASTBALL_INDICES:
            if idx < self.n_pitch_types:
                family_weights[..., 0] += pitch_type_probs[..., idx]
        for idx in self.OFFSPEED_INDICES:
            if idx < self.n_pitch_types:
                family_weights[..., 1] += pitch_type_probs[..., idx]
        for idx in self.BREAKING_INDICES:
            if idx < self.n_pitch_types:
                family_weights[..., 2] += pitch_type_probs[..., idx]
        # Remaining probability goes to 'other'
        family_weights[..., 3] = 1.0 - family_weights[..., :3].sum(dim=-1)
        family_weights = family_weights.clamp(min=1e-6)

        # Combine all components into single MDN
        # Stack parameters from all families

        # Combine mixture weights (scale by family weight)
        # Each family's pi gets multiplied by its family weight
        combined_pi = torch.cat([
            fastball_params["pi"] * family_weights[..., 0:1],
            offspeed_params["pi"] * family_weights[..., 1:2],
            breaking_params["pi"] * family_weights[..., 2:3],
            other_params["pi"] * family_weights[..., 3:4],
        ], dim=-1)  # [batch, seq, total_K]

        # Renormalize mixture weights
        combined_pi = combined_pi / combined_pi.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Stack means, sigmas, rhos
        combined_mu = torch.cat([
            fastball_params["mu"],
            offspeed_params["mu"],
            breaking_params["mu"],
            other_params["mu"],
        ], dim=2)  # [batch, seq, total_K, 2]

        combined_sigma = torch.cat([
            fastball_params["sigma"],
            offspeed_params["sigma"],
            breaking_params["sigma"],
            other_params["sigma"],
        ], dim=2)  # [batch, seq, total_K, 2]

        combined_rho = torch.cat([
            fastball_params["rho"],
            offspeed_params["rho"],
            breaking_params["rho"],
            other_params["rho"],
        ], dim=-1)  # [batch, seq, total_K]

        return {
            "pi": combined_pi,
            "mu": combined_mu,
            "sigma": combined_sigma,
            "rho": combined_rho,
        }


class PitchPredictorEnhanced(nn.Module):
    """
    Enhanced LSTM model with pitch-type-conditioned location prediction.

    Improvements over base PitchPredictor:
    1. Location prediction is conditioned on predicted pitch type
    2. Uses hierarchical MDN with pitch-family-specific components
    3. Better captures the interaction between pitch type and location

    Architecture:
    - Embeddings for categorical features
    - LSTM encoder
    - Pitch type classification head
    - Hierarchical MDN location head (conditioned on pitch type)
    """

    def __init__(
        self,
        n_pitch_types: int,
        n_pitchers: int,
        n_batters: int,
        n_continuous_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_location_components: int = 2,  # Components per family
        feature_indices: dict | None = None,
        use_attention: bool = False,
        n_attention_heads: int = 4,
        n_attention_layers: int = 1,
    ):
        """
        Initialize the enhanced model.

        Args:
            n_pitch_types: Number of pitch type classes.
            n_pitchers: Number of unique pitchers.
            n_batters: Number of unique batters.
            n_continuous_features: Number of continuous input features.
            embedding_dim: Dimension for embeddings.
            hidden_dim: LSTM hidden dimension.
            n_layers: Number of LSTM layers.
            dropout: Dropout rate.
            n_location_components: Components per pitch family (total = 4 * this).
            feature_indices: Dict with embedding and continuous indices.
            use_attention: Whether to use attention layers after LSTM.
            n_attention_heads: Number of attention heads (if using attention).
            n_attention_layers: Number of attention layers (if using attention).
        """
        super().__init__()

        self.n_pitch_types = n_pitch_types
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_location_components = n_location_components * 4  # Total across families
        self.use_attention = use_attention

        # Store feature indices
        if feature_indices is not None:
            self.pitcher_idx_pos = feature_indices["embedding_indices"]["pitcher_idx"]
            self.batter_idx_pos = feature_indices["embedding_indices"]["batter_idx"]
            self.prev_pitch_idx_pos = feature_indices["embedding_indices"]["prev_pitch_type_idx"]
            self.continuous_indices = feature_indices["continuous_indices"]
        else:
            self.pitcher_idx_pos = 10
            self.batter_idx_pos = 11
            self.prev_pitch_idx_pos = 19
            self.continuous_indices = list(range(10)) + list(range(12, 19)) + list(range(20, 31))

        n_continuous = len(self.continuous_indices)

        # Embeddings
        self.pitcher_embedding = nn.Embedding(n_pitchers, embedding_dim)
        self.batter_embedding = nn.Embedding(n_batters, embedding_dim)
        self.prev_pitch_embedding = nn.Embedding(
            n_pitch_types + 1, embedding_dim, padding_idx=n_pitch_types
        )

        # Input size
        self.input_size = 3 * embedding_dim + n_continuous

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        # Optional attention layers
        if use_attention:
            self.attention_layers = nn.ModuleList([
                AttentionBlock(hidden_dim, n_attention_heads, dropout)
                for _ in range(n_attention_layers)
            ])

        self.dropout = nn.Dropout(dropout)

        # Pitch type classification head
        self.pitch_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_pitch_types),
        )

        # Hierarchical MDN for location (pitch-type conditioned)
        self.location_head = HierarchicalMDN(
            hidden_dim=hidden_dim,
            n_pitch_types=n_pitch_types,
            embedding_dim=embedding_dim,
            n_components_per_family=n_location_components // 4 if n_location_components >= 4 else 2,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
        """
        Forward pass with pitch-type-conditioned location prediction.

        Args:
            features: Input features [batch, seq_len, n_features].
            lengths: Sequence lengths [batch].
            mask: Attention mask [batch, seq_len].
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (pitch_type_logits, mdn_params, attention_weights).
        """
        _batch_size, seq_len, _n_features = features.shape

        # Extract embedding indices
        pitcher_idx = features[:, :, self.pitcher_idx_pos].long().clamp(
            0, self.pitcher_embedding.num_embeddings - 1
        )
        batter_idx = features[:, :, self.batter_idx_pos].long().clamp(
            0, self.batter_embedding.num_embeddings - 1
        )
        prev_pitch_idx = features[:, :, self.prev_pitch_idx_pos].long().clamp(
            -1, self.n_pitch_types - 1
        )
        prev_pitch_idx = torch.where(
            prev_pitch_idx < 0,
            torch.tensor(self.n_pitch_types, device=prev_pitch_idx.device),
            prev_pitch_idx,
        )

        # Get embeddings
        pitcher_emb = self.pitcher_embedding(pitcher_idx)
        batter_emb = self.batter_embedding(batter_idx)
        prev_pitch_emb = self.prev_pitch_embedding(prev_pitch_idx)

        # Extract continuous features
        continuous_features = features[:, :, self.continuous_indices]

        # Concatenate
        x = torch.cat([
            pitcher_emb,
            batter_emb,
            prev_pitch_emb,
            continuous_features,
        ], dim=-1)

        # LSTM
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        # Optional attention
        attn_weights = None
        if self.use_attention:
            for attn_layer in self.attention_layers:
                lstm_out, attn_weights = attn_layer(
                    lstm_out, mask, return_weights=return_attention
                )

        lstm_out = self.dropout(lstm_out)

        # Predict pitch type first
        pitch_type_logits = self.pitch_type_head(lstm_out)
        pitch_type_probs = F.softmax(pitch_type_logits, dim=-1)

        # Predict location conditioned on pitch type
        mdn_params = self.location_head(lstm_out, pitch_type_probs)

        if return_attention:
            return pitch_type_logits, mdn_params, attn_weights
        return pitch_type_logits, mdn_params


class PitchPredictorSimple(nn.Module):
    """
    Simpler feedforward model for baseline comparison.

    Uses only the current pitch context (no sequence modeling).
    Also uses MDN for location prediction.
    """

    def __init__(
        self,
        n_pitch_types: int,
        n_pitchers: int,
        n_batters: int,
        n_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        n_location_components: int = 3,
        feature_indices: dict | None = None,
    ):
        super().__init__()

        self.n_pitch_types = n_pitch_types
        self.n_location_components = n_location_components

        # Store feature indices for dynamic extraction
        # Simple model only embeds pitcher and batter (not prev_pitch_type)
        if feature_indices is not None:
            self.pitcher_idx_pos = feature_indices["embedding_indices"]["pitcher_idx"]
            self.batter_idx_pos = feature_indices["embedding_indices"]["batter_idx"]
            # Continuous includes everything except pitcher_idx and batter_idx
            self.continuous_indices = [
                i for i in range(n_features)
                if i not in [self.pitcher_idx_pos, self.batter_idx_pos]
            ]
        else:
            # Fallback to current 31-feature layout
            self.pitcher_idx_pos = 10
            self.batter_idx_pos = 11
            self.continuous_indices = list(range(10)) + list(range(12, n_features))

        n_continuous = len(self.continuous_indices)

        # Embeddings
        self.pitcher_embedding = nn.Embedding(n_pitchers, embedding_dim)
        self.batter_embedding = nn.Embedding(n_batters, embedding_dim)

        # Input size: embeddings + continuous features
        input_size = 2 * embedding_dim + n_continuous

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.pitch_type_head = nn.Linear(hidden_dim, n_pitch_types)
        # MDN location head: 6*K parameters
        self.location_head = nn.Linear(hidden_dim, 6 * n_location_components)

    def _parse_mdn_params(self, raw_output: torch.Tensor) -> dict:
        """Parse raw MDN output into distribution parameters."""
        K = self.n_location_components
        batch, seq, _ = raw_output.shape

        pi_logits = raw_output[..., :K]
        mu = raw_output[..., K : 3 * K].reshape(batch, seq, K, 2)
        log_sigma = raw_output[..., 3 * K : 5 * K].reshape(batch, seq, K, 2)
        rho_raw = raw_output[..., 5 * K : 6 * K]

        pi = F.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        rho = torch.tanh(rho_raw) * 0.99

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass for simple model."""
        # Get embeddings using stored positions
        pitcher_idx = features[:, :, self.pitcher_idx_pos].long().clamp(
            0, self.pitcher_embedding.num_embeddings - 1
        )
        batter_idx = features[:, :, self.batter_idx_pos].long().clamp(
            0, self.batter_embedding.num_embeddings - 1
        )

        pitcher_emb = self.pitcher_embedding(pitcher_idx)
        batter_emb = self.batter_embedding(batter_idx)

        # Extract continuous features using stored indices
        continuous = features[:, :, self.continuous_indices]

        # Concatenate
        x = torch.cat([pitcher_emb, batter_emb, continuous], dim=-1)

        # MLP
        hidden = self.mlp(x)

        # Outputs
        pitch_type_logits = self.pitch_type_head(hidden)
        raw_location_out = self.location_head(hidden)

        # Parse MDN parameters
        mdn_params = self._parse_mdn_params(raw_location_out)

        return pitch_type_logits, mdn_params


def create_model(
    n_pitch_types: int,
    n_pitchers: int,
    n_batters: int,
    n_features: int,
    model_type: str = "lstm",
    feature_indices: dict | None = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a pitch prediction model.

    Args:
        n_pitch_types: Number of pitch type classes.
        n_pitchers: Number of unique pitchers.
        n_batters: Number of unique batters.
        n_features: Number of input features.
        model_type: Type of model:
            - "lstm": Basic LSTM model
            - "lstm_attention": LSTM with attention layers
            - "enhanced": Enhanced model with pitch-type-conditioned location
            - "enhanced_attention": Enhanced model with attention
            - "simple": Simple feedforward baseline
        feature_indices: Dict from PitchFeatureEngine.get_feature_indices().
            If provided, the model uses dynamic feature extraction instead
            of hardcoded indices.
        **kwargs: Additional model arguments.

    Returns:
        PyTorch model.
    """
    if model_type == "lstm":
        return PitchPredictor(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_features,
            feature_indices=feature_indices,
            **kwargs,
        )
    elif model_type == "lstm_attention":
        return PitchPredictorWithAttention(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_features,
            feature_indices=feature_indices,
            **kwargs,
        )
    elif model_type == "enhanced":
        return PitchPredictorEnhanced(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_features,
            feature_indices=feature_indices,
            use_attention=False,
            **kwargs,
        )
    elif model_type == "enhanced_attention":
        return PitchPredictorEnhanced(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_continuous_features=n_features,
            feature_indices=feature_indices,
            use_attention=True,
            **kwargs,
        )
    elif model_type == "simple":
        return PitchPredictorSimple(
            n_pitch_types=n_pitch_types,
            n_pitchers=n_pitchers,
            n_batters=n_batters,
            n_features=n_features,
            feature_indices=feature_indices,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
