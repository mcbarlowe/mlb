# Pitch Prediction Models Documentation

Technical documentation for the MLB pitch prediction pipeline, covering the pitch type prediction model (LSTM+Attention) and the pitch location model (PitchTypeConditionedMDN).

## 1. Executive Summary

This system predicts both **pitch type** and **pitch location** for MLB at-bats using a two-stage approach:

1. **Pitch Type Model**: LSTM+Attention architecture predicting which of 11 pitch types will be thrown
2. **Location Model**: Pitch-type-conditioned MDN predicting where the pitch will be located

### Key Performance Metrics (2025 Test Set - 801,978 pitches)

| Metric | Pitch Type Model | Location Model (Conditioned MDN) |
|--------|------------------|----------------------------------|
| Accuracy | 72.34% | — |
| Top-3 Accuracy | 94.74% | — |
| Macro F1 | 0.6651 | — |
| NLL | 2.083 | **2.048** |
| Euclidean Error | 0.980 ft | **0.961 ft** |
| Coverage @90% | 94.50% | **95.08%** |
| Coverage @95% | 97.45% | **97.62%** |

### Model Comparison

| Aspect | LSTM Built-in MDN | Standalone Conditioned MDN |
|--------|-------------------|---------------------------|
| Location NLL | 2.083 | 2.048 |
| Euclidean Error | 0.980 ft | 0.961 ft |
| Parameters | Shared with type model | ~120K dedicated |
| Pitch-type specificity | None | 11 separate heads |
| Best Use Case | Unified inference | Maximum location accuracy |

---

## 2. Pitch Type Model (LSTM+Attention)

### 2.1 Architecture Overview

- **Model Class**: `PitchPredictorWithAttention` (`src/ml/model.py:386`)
- **Reference**: Yu et al. 2022 "Attention-Based LSTM for Pitch Prediction"
- **Parameters**: 2.84M total

```
Input Features [batch, seq, 49]
        │
        ▼
┌─────────────────────────────────┐
│  Embedding Layers               │
│  • Pitcher (32-dim)             │
│  • Batter (32-dim)              │
│  • Previous Pitch (32-dim)      │
└─────────────────────────────────┘
        │
        ▼  [batch, seq, 142]
┌─────────────────────────────────┐
│  LSTM Encoder                   │
│  • 2 layers                     │
│  • 256 hidden units             │
│  • Causal (unidirectional)      │
└─────────────────────────────────┘
        │
        ▼  [batch, seq, 256]
┌─────────────────────────────────┐
│  Multi-Head Attention           │
│  • 8 heads                      │
│  • 2 stacked layers             │
│  • Causal masking               │
│  • Residual + LayerNorm         │
└─────────────────────────────────┘
        │
        ▼  [batch, seq, 256]
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────────┐
│ Type │  │ Location MDN │
│ Head │  │    Head      │
└──────┘  └──────────────┘
   │           │
   ▼           ▼
[11 classes] [18 params]
```

### 2.2 Model Components

#### Embedding Layer
| Component | Vocabulary Size | Dimension |
|-----------|-----------------|-----------|
| Pitcher | ~3,500 | 32 |
| Batter | ~3,500 | 32 |
| Previous Pitch | 12 (11 + padding) | 32 |

#### LSTM Encoder
- **Layers**: 2
- **Hidden Units**: 256
- **Direction**: Unidirectional (causal)
- **Dropout**: 0.3 (between layers)

#### Multi-Head Attention
- **Heads**: 8
- **Stacked Layers**: 2
- **Head Dimension**: 32 (256 / 8)
- **Masking**: Causal (prevents attending to future positions)
- **Feed-Forward**: 4x expansion (256 → 1024 → 256)
- **Activation**: GELU
- **Normalization**: Pre-LayerNorm

#### Output Heads
| Head | Architecture | Output |
|------|--------------|--------|
| Pitch Type | Linear(256→128) + ReLU + Dropout + Linear(128→11) | 11 class logits |
| Location MDN | Linear(256→128) + ReLU + Dropout + Linear(128→18) | 3 components × 6 params |

### 2.3 Input Features (49 Total)

#### Count Features (6)
| Feature | Description | Range |
|---------|-------------|-------|
| `balls` | Current ball count | 0-3 |
| `strikes` | Current strike count | 0-2 |
| `two_strike_count` | Binary: strikes == 2 | 0/1 |
| `hitters_count` | Binary: balls >= 2 and strikes <= 1 | 0/1 |
| `first_pitch` | Binary: 0-0 count | 0/1 |
| `pitcher_ahead` | Binary: strikes > balls | 0/1 |

#### Game State (4)
| Feature | Description | Range |
|---------|-------------|-------|
| `inning` | Current inning | 1-15+ |
| `outs` | Current out count | 0-2 |
| `runners_bitmap` | 3-bit runner encoding (1B=1, 2B=2, 3B=4) | 0-7 |
| `score_diff` | Batting team's run differential | -20 to 20 |

#### Player IDs (3) - Embedded
| Feature | Description |
|---------|-------------|
| `pitcher_idx` | Pitcher lookup index |
| `batter_idx` | Batter lookup index |
| `prev_pitch_type_idx` | Previous pitch type index (-1 if first pitch) |

#### Handedness (3)
| Feature | Description | Encoding |
|---------|-------------|----------|
| `throw_side_enc` | Pitcher handedness | L=0, R=1 |
| `bat_side_enc` | Batter handedness | L=0, R=1 |
| `platoon_same_side` | Same-side matchup | 0/1 |

#### Pitcher Tendencies (2)
| Feature | Description | Range |
|---------|-------------|-------|
| `pitcher_ff_pct` | Historical fastball percentage | 0-1 |
| `pitcher_repertoire` | Number of pitch types / 7 | 0-1.4 |

#### Batter Zone (2)
| Feature | Description |
|---------|-------------|
| `batter_zone_height` | Strike zone height / 2.0 |
| `batter_zone_mid` | Zone vertical midpoint / 2.5 |

#### Previous Pitch (8)
| Feature | Description |
|---------|-------------|
| `prev_px` | Previous horizontal location (ft) |
| `prev_pz` | Previous vertical location (ft) |
| `prev_speed` | Previous pitch velocity (mph) |
| `prev_is_strike` | Previous pitch was a strike |
| `velocity_delta` | (current - previous) / 10 |
| `prev_swing` | Batter swung at previous pitch |
| `prev_result_type` | 0=ball, 1=called, 2=swing, 3=foul, 4=in_play |

#### Sequence Features (4)
| Feature | Description |
|---------|-------------|
| `n_fastballs_in_ab` | Cumulative fastballs in at-bat |
| `n_breaking_in_ab` | Cumulative breaking balls in at-bat |
| `same_pitch_streak` | Consecutive same pitch types |
| `pitch_number` | Pitch number in at-bat |

#### Weather (4)
| Feature | Description |
|---------|-------------|
| `temp_normalized` | (temp - 70) / 30 |
| `wind_speed` | Wind speed / 20 |
| `wind_direction` | -1 (in), 0 (neutral), 1 (out) |
| `is_night_game` | Night game indicator |

#### Situation (4)
| Feature | Description |
|---------|-------------|
| `season_progress` | (month - 4) / 6, clipped to [0,1] |
| `runners_in_scoring_position` | Runner on 2B or 3B |
| `leverage_approx` | Situational importance score |
| `pitcher_pitch_count` | Pitches thrown in game / 100 |

#### Pitch Family Indicators (3)
| Feature | Description |
|---------|-------------|
| `prev_is_fastball` | Previous pitch was FF/SI/FC/FA/FT |
| `prev_is_offspeed` | Previous pitch was CH/FS |
| `prev_is_breaking` | Previous pitch was SL/CU/KC/ST/SV |

#### Interaction Terms (7)
| Feature | Description |
|---------|-------------|
| `prev_fb_x_rhb` | Fastball × RHB interaction |
| `prev_off_x_rhb` | Offspeed × RHB interaction |
| `prev_brk_x_rhb` | Breaking × RHB interaction |
| `ahead_x_rhb` | Pitcher ahead × RHB |
| `two_strike_x_rhb` | Two-strike × RHB |
| `hitters_x_rhb` | Hitter's count × RHB |
| `platoon_x_breaking` | Platoon × breaking ball |

### 2.4 Attention Mechanism Details

The attention mechanism (`MultiHeadAttention` class, `src/ml/model.py:19`) enables the model to selectively focus on relevant previous pitches in the sequence.

#### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q, K, V are linear projections of the input
- d_k = 32 (head dimension)
- Causal mask prevents attending to future positions

#### Architecture per Block (`AttentionBlock`, `src/ml/model.py:127`)
```
Input
  │
  ├──► Multi-Head Self-Attention
  │         │
  │         ▼
  │    Add & LayerNorm ◄──┘
  │         │
  │         ▼
  ├──► Feed-Forward (256→1024→256)
  │         │
  │         ▼
       Add & LayerNorm ◄──┘
         │
         ▼
      Output
```

### 2.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 50 |
| Early Stopping | 7 epochs patience |
| Loss (Type) | CrossEntropyLoss with class weights |
| Loss (Location) | Negative Log-Likelihood (MDN) |
| Loss Weights | 1.0 × type + 0.5 × location |
| Class Weight Smoothing | 0.5 |
| Gradient Clipping | 1.0 |

#### Data Split
- **Train**: 2021-2023 seasons
- **Validation**: 2024 season
- **Test**: 2025 season
- **Excluded**: 2020 (COVID shortened season)

### 2.6 Performance Metrics

#### Classification (2025 Test Set - 801,978 pitches)

| Metric | Value |
|--------|-------|
| Accuracy | 72.34% |
| Top-3 Accuracy | 94.74% |
| Macro F1 | 0.6651 |
| Weighted F1 | 0.7223 |

#### Per-Pitch-Type Performance

| Pitch Type | Count | Accuracy | Notes |
|------------|-------|----------|-------|
| FF (Four-Seam) | 256,437 | ~75% | Most common pitch |
| SI (Sinker) | 124,955 | ~72% | Good accuracy |
| SL (Slider) | 119,663 | ~70% | Breaking ball |
| CH (Changeup) | 82,416 | ~68% | Offspeed |
| FC (Cutter) | 61,418 | ~65% | Hybrid pitch |
| ST (Sweeper) | 56,039 | ~60% | Modern breaking ball |
| CU (Curveball) | 55,405 | ~62% | Traditional breaking |
| FS (Splitter) | 25,939 | ~55% | Rare offspeed |
| KC (Knuckle Curve) | 12,867 | ~50% | Specialty pitch |
| OTHER | 6,664 | ~40% | Catch-all category |
| KN (Knuckleball) | 175 | ~30% | Very rare |

#### Location (Built-in MDN)

| Metric | Value |
|--------|-------|
| NLL | 2.0829 |
| MAE px | 0.607 ft |
| MAE pz | 0.648 ft |
| RMSE px | 0.779 ft |
| RMSE pz | 0.847 ft |
| Euclidean Error | 0.980 ft |
| Coverage @90% | 94.50% |
| Coverage @95% | 97.45% |

---

## 3. Location Model (PitchTypeConditionedMDN)

### 3.1 Architecture Overview

The `PitchTypeConditionedMDN` (`src/ml/pitch_type_location_model.py:30`) treats pitch type as a **random effect**, providing 11 separate MDN output heads for pitch-type-specific location distributions.

```
Input Features [batch, 43]
        │
        ▼
┌─────────────────────────────────┐
│  Shared Backbone                │
│  Linear(43→256) + BN + ReLU     │
│  Linear(256→128) + BN + ReLU    │
│  Dropout(0.2)                   │
└─────────────────────────────────┘
        │
        ▼  [batch, 128]
   ┌────┴────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
   │    │    │    │    │    │    │    │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
│ FF ││ SI ││ FC ││ CH ││ SL ││ CU ││ KC ││ ST ││ FS ││ KN ││OTH │
│Head││Head││Head││Head││Head││Head││Head││Head││Head││Head││Head│
└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘
   │    │    │    │    │    │    │    │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
[18]  [18]  [18]  [18]  [18]  [18]  [18]  [18]  [18]  [18]  [18]
         MDN Parameters (3 components × 6 params each)
```

### 3.2 Model Components

#### Shared Backbone
```python
Linear(43, 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.2)
```

#### Pitch-Type Heads
- **11 separate heads** (one per pitch type)
- **Each head**: `Linear(128, 18)` producing 3 MDN components
- **Total output**: 18 parameters per pitch type

### 3.3 MDN Parameters (per component)

| Parameter | Symbol | Activation | Description |
|-----------|--------|------------|-------------|
| Weight | π | Softmax | Mixture component weight |
| Mean X | μ_x | None | Horizontal location mean |
| Mean Z | μ_z | None | Vertical location mean |
| Std X | σ_x | Exp (clamped 0.01-5.0) | Horizontal std dev |
| Std Z | σ_z | Exp (clamped 0.01-5.0) | Vertical std dev |
| Correlation | ρ | Tanh × 0.99 | X-Z correlation |

### 3.4 Conditioning Mechanisms

#### Hard Conditioning (Training)
During training, the ground-truth pitch type index directly selects which head to use:
```python
mask = pitch_type_idx == pt_idx
output = self.pitch_type_heads[pt_idx](hidden[mask])
```

#### Soft Conditioning (Inference)
During inference with predicted pitch types, outputs from all heads are combined weighted by pitch type probabilities:

```
p(location | x) = Σ_t P(type=t | x) × p(location | x, type=t)
```

Implementation (`forward_soft` method):
1. Compute MDN parameters from all 11 heads
2. Weight parameters by pitch type probabilities
3. Renormalize mixture weights

### 3.5 Bivariate Gaussian Mixture

The location distribution is modeled as a mixture of K=3 bivariate Gaussians:

```
p(px, pz | params) = Σ_{k=1}^K π_k × N((px, pz) | μ_k, Σ_k)
```

Where the covariance matrix is parameterized via correlation:
```
Σ_k = [σ_x²           ρ·σ_x·σ_z]
      [ρ·σ_x·σ_z      σ_z²     ]
```

#### Log-Likelihood Computation
```python
z = (dx/σx)² + (dy/σy)² - 2ρ(dx/σx)(dy/σy)
log_prob = -z / (2(1-ρ²)) - log(2π) - log(σx) - log(σy) - 0.5·log(1-ρ²)
total_log_prob = logsumexp(log(π) + log_prob)
```

### 3.6 Training Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Dims | [256, 128] |
| Components | 3 per pitch type |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 1e-5 |
| Batch Size | 256 |
| Max Epochs | 100 |
| Early Stopping | 10 epochs patience |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

#### Data Split
- **Train**: 2021-2023 seasons
- **Validation**: 2024 season
- **Test**: 2025 season

### 3.7 Performance Metrics

#### Overall (2025 Test Set - 801,978 pitches)

| Metric | Value |
|--------|-------|
| NLL | 2.0484 |
| MAE px | 0.5879 ft |
| MAE pz | 0.6441 ft |
| Euclidean Error | 0.9606 ft |
| Coverage @90% | 95.08% |
| Coverage @95% | 97.62% |

#### Per-Pitch-Type Performance

| Pitch Type | NLL | Euclidean (ft) | Coverage @90% | Count |
|------------|-----|----------------|---------------|-------|
| FF | 1.983 | 0.933 | 95.40% | 256,437 |
| SI | 1.964 | 0.932 | 95.05% | 124,955 |
| FC | 2.031 | 0.966 | 94.64% | 61,418 |
| CH | 2.059 | 0.965 | 95.93% | 82,416 |
| SL | 2.082 | 0.959 | 94.42% | 119,663 |
| CU | 2.207 | 1.063 | 94.96% | 55,405 |
| KC | 2.163 | 1.031 | 95.35% | 12,867 |
| ST | 2.208 | 0.985 | 94.31% | 56,039 |
| FS | 2.129 | 0.999 | 95.98% | 25,939 |
| KN | 2.898 | 1.336 | 80.57% | 175 |
| OTHER | 2.370 | 1.106 | 92.66% | 6,664 |

**Key Observations**:
- **Best**: Fastballs (FF, SI) have lowest NLL (~1.96-1.98) - most predictable locations
- **Worst**: Knuckleball (KN) has highest NLL (2.90) - inherently unpredictable, limited data
- **Breaking balls** (CU, KC, ST) have higher error - wider location distributions

---

## 4. Combined Pipeline

### 4.1 Integration Architecture

The `PitchTypeThenLocationPredictor` class (`src/ml/pitch_type_location_model.py:400`) combines both models:

```python
class PitchTypeThenLocationPredictor(nn.Module):
    def __init__(self, pitch_type_model, location_model, use_soft_conditioning=True):
        self.pitch_type_model = pitch_type_model  # LSTM+Attention
        self.location_model = location_model       # PitchTypeConditionedMDN
        self.use_soft_conditioning = use_soft_conditioning
```

### 4.2 Inference Flow

```
Input Features [batch, seq, 49]
        │
        ▼
┌─────────────────────────────────┐
│  LSTM+Attention Model           │
│  (PitchPredictorWithAttention)  │
└─────────────────────────────────┘
        │
        ▼
Pitch Type Probabilities [batch, seq, 11]
        │
        ▼
┌─────────────────────────────────┐
│  Extract Location Features      │
│  (49 → 43 features)             │
└─────────────────────────────────┘
        │
        ▼
Location Features [batch×seq, 43]
        │
        ▼
┌─────────────────────────────────┐
│  PitchTypeConditionedMDN        │
│  (soft conditioning mode)       │
└─────────────────────────────────┘
        │
        ▼
Location Distribution (MDN params)
```

### 4.3 Feature Subset

The location model uses **43 features** (excluding 6 features that are either embedded or redundant):

**Excluded from location model**:
- `wind_direction` (minimal impact)
- `is_night_game` (minimal impact)
- `prev_is_fastball`, `prev_is_offspeed`, `prev_is_breaking` (redundant with pitch type conditioning)

### 4.4 Combined Performance

| Metric | LSTM Only | Combined Pipeline |
|--------|-----------|-------------------|
| Pitch Type Accuracy | 72.34% | 72.34% (unchanged) |
| Location NLL | 2.083 | 2.048 (1.7% better) |
| Euclidean Error | 0.980 ft | 0.961 ft (2% better) |

---

## 5. Key Design Decisions

### 5.1 Why Attention?

The attention mechanism provides several advantages over pure LSTM:

1. **Variable-length sequences**: At-bats range from 1 to 20+ pitches
2. **Selective memory**: Model learns which previous pitches are most relevant
3. **Causal masking**: Prevents information leakage from future pitches
4. **Interpretability**: Attention weights show which pitches influenced prediction

### 5.2 Why Random Effects for Location?

Different pitch types have fundamentally different location distributions:

| Pitch Family | Typical Location Pattern |
|--------------|--------------------------|
| Fastballs (FF, SI) | Up in zone, concentrated |
| Breaking (SL, CU) | Low and glove-side, dispersed |
| Offspeed (CH, FS) | Low and arm-side |
| Sweeper (ST) | Wide horizontal movement |

A single MDN head cannot capture these multimodal patterns effectively. The pitch-type-conditioned approach allows each pitch type to have its own specialized distribution.

### 5.3 Why MDN for Location?

Location prediction is inherently probabilistic:

1. **Multiple valid targets**: Pitcher may target inside corner OR outside corner
2. **Command uncertainty**: Even intended location has variance
3. **Proper uncertainty quantification**: MDN provides calibrated confidence intervals
4. **Multimodal distributions**: Mixture model captures multiple likely locations

---

## 6. File References

```
models/
├── attention_full/run_20260119_124719/
│   ├── final_model.pt              # LSTM+Attention weights (2.84M params)
│   ├── feature_engine.json         # Feature preprocessing mappings
│   └── results.json                # Training config & metrics
│
└── pitch_type_location_20260121_003206/
    ├── pitch_type_location_model.pt  # Location MDN weights
    ├── config.json                   # Training configuration
    ├── test_metrics.json             # Detailed per-type metrics
    └── training_history.json         # Training curves

src/ml/
├── model.py                    # LSTM+Attention architecture
│   ├── MultiHeadAttention      # Line 19
│   ├── AttentionBlock          # Line 127
│   ├── PitchPredictor          # Line 175 (base LSTM)
│   └── PitchPredictorWithAttention  # Line 386
│
├── pitch_type_location_model.py  # PitchTypeConditionedMDN
│   ├── PitchTypeConditionedMDN       # Line 30
│   ├── PitchTypeThenLocationPredictor # Line 400
│   └── PitchTypeLocationTrainer      # Line 664
│
├── features.py                 # PitchFeatureEngine
│   ├── PITCH_TYPE_CODES        # Line 16 (pitch type definitions)
│   └── PitchFeatureEngine      # Line 42
│
└── pitch_predictor.py          # Unified inference interface
    ├── PitchPredictor          # Line 202
    ├── PitchPrediction         # Line 155 (dataclass)
    └── GameContext             # Line 115 (dataclass)
```

---

## 7. Usage Examples

### Loading and Running Inference

```python
from src.ml.pitch_predictor import PitchPredictor
import torch

# Load the LSTM+Attention model
predictor = PitchPredictor.load_lstm("models/attention_full/run_20260119_124719")

# Prepare features (from PitchFeatureEngine)
# features shape: [seq_len, 49] or [batch, seq_len, 49]
features = torch.randn(10, 49)  # 10 pitches in at-bat

# Make prediction for last pitch in sequence
prediction = predictor.predict(lstm_features=features)

# Access results
print(f"Predicted pitch type: {prediction.predicted_type}")
print(f"Top 3: {prediction.top_3_types}")
print(f"Location: ({prediction.location_point[0]:.2f}, {prediction.location_point[1]:.2f})")
print(f"Strike zone probability: {predictor.get_strike_zone_probability(prediction):.1%}")
```

### Generating Pitch Cards

```python
from src.ml.pitch_predictor import PitchPredictor, GameContext, create_pitch_card_from_row

# Load model
predictor = PitchPredictor.load_lstm("models/attention_full/run_20260119_124719")

# Create context
context = GameContext(
    pitcher_name="Gerrit Cole",
    batter_name="Aaron Judge",
    pitcher_hand="R",
    batter_hand="R",
    home_team="NYY",
    away_team="BOS",
    inning=7,
    inning_half="Bot",
    balls=1,
    strikes=2,
    outs=1,
)

# Generate visualization
prediction = predictor.predict(lstm_features=features)
fig = predictor.create_pitch_card(
    prediction=prediction,
    context=context,
    save_path="pitch_card.png"
)
```

### Command Line

```bash
# Generate sample pitch cards
uv run python scripts/generate_combined_pitch_cards.py --n-cards 5

# Train location model
uv run python scripts/train_pitch_type_location.py \
    --train-seasons 2021 2022 2023 \
    --val-season 2024 \
    --test-season 2025
```

---

## 8. Appendix: Pitch Type Definitions

| Code | Full Name | Family | Typical Velocity |
|------|-----------|--------|------------------|
| FF | Four-Seam Fastball | Fastball | 93-97 mph |
| SI | Sinker | Fastball | 91-95 mph |
| FC | Cutter | Fastball | 86-92 mph |
| CH | Changeup | Offspeed | 82-88 mph |
| SL | Slider | Breaking | 83-89 mph |
| CU | Curveball | Breaking | 75-82 mph |
| KC | Knuckle Curve | Breaking | 77-84 mph |
| ST | Sweeper | Breaking | 78-85 mph |
| FS | Splitter | Offspeed | 84-89 mph |
| KN | Knuckleball | Specialty | 70-80 mph |
| OTHER | Catch-all | Various | Various |
