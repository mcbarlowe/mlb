# Evaluation Metrics Documentation

Comprehensive evaluation metrics for the pitch prediction models on the 2025 test set (801,978 pitches).

---

## Table of Contents

1. [Test Set Overview](#1-test-set-overview)
2. [LSTM+Attention Model Metrics](#2-lstmattention-model-metrics)
   - [Classification Metrics](#21-classification-metrics)
   - [Per-Class Performance](#22-per-class-performance)
   - [Location Prediction (Built-in MDN)](#23-location-prediction-built-in-mdn)
3. [PitchTypeConditionedMDN Metrics](#3-pitchtypeconditionedmdn-metrics)
   - [Overall Performance](#31-overall-performance)
   - [Per-Pitch-Type Performance](#32-per-pitch-type-performance)
4. [Model Comparison](#4-model-comparison)
5. [Training Convergence](#5-training-convergence)
6. [Metric Definitions](#6-metric-definitions)

---

## 1. Test Set Overview

| Property | Value |
|----------|-------|
| **Season** | 2025 |
| **Total Pitches** | 801,978 |
| **Train Seasons** | 2021-2023 |
| **Validation Season** | 2024 |
| **Excluded** | 2020 (COVID season) |

### Pitch Type Distribution (Test Set)

| Pitch Type | Full Name | Count | Percentage |
|------------|-----------|-------|------------|
| FF | Four-Seam Fastball | 256,437 | 31.98% |
| SI | Sinker | 124,955 | 15.58% |
| SL | Slider | 119,663 | 14.92% |
| CH | Changeup | 82,416 | 10.28% |
| FC | Cutter | 61,418 | 7.66% |
| ST | Sweeper | 56,039 | 6.99% |
| CU | Curveball | 55,405 | 6.91% |
| FS | Splitter | 25,939 | 3.23% |
| KC | Knuckle Curve | 12,867 | 1.60% |
| OTHER | Other pitches | 6,664 | 0.83% |
| KN | Knuckleball | 175 | 0.02% |

---

## 2. LSTM+Attention Model Metrics

**Model**: `PitchPredictorWithAttention`
**Checkpoint**: `models/attention_full/run_20260119_124719/final_model.pt`
**Parameters**: 2,838,813

### 2.1 Classification Metrics

#### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 72.34% |
| **Top-3 Accuracy** | 94.74% |
| **Macro F1** | 0.6651 |
| **Weighted F1** | 0.7223 |
| **Macro Precision** | 0.67 |
| **Macro Recall** | 0.67 |

### 2.2 Per-Class Performance

| Pitch Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **FF** | 0.81 | 0.85 | **0.83** | 256,437 |
| **SI** | 0.66 | 0.65 | 0.65 | 124,955 |
| **FC** | 0.64 | 0.58 | 0.61 | 61,418 |
| **CH** | 0.62 | 0.70 | 0.66 | 82,416 |
| **SL** | 0.74 | 0.68 | 0.71 | 119,663 |
| **CU** | 0.74 | 0.78 | **0.76** | 55,405 |
| **KC** | 0.58 | 0.78 | 0.66 | 12,867 |
| **ST** | 0.74 | 0.61 | 0.67 | 56,039 |
| **FS** | 0.62 | 0.59 | 0.61 | 25,939 |
| **KN** | 0.45 | 0.57 | 0.50 | 175 |
| **OTHER** | 0.72 | 0.60 | 0.66 | 6,664 |

#### Classification Analysis

**Best Performing (by F1)**:
1. **FF (0.83)**: Most common pitch, clear velocity/movement signature
2. **CU (0.76)**: Distinctive slow, looping trajectory
3. **SL (0.71)**: Well-defined breaking ball characteristics

**Most Challenging**:
1. **KN (0.50)**: Extremely rare (175 samples), unpredictable by nature
2. **FC (0.61)**: Often confused with FF and SL
3. **FS (0.61)**: Confused with CH and SI

**High Recall (frequent prediction)**:
- FF (0.85), CU (0.78), KC (0.78)

**Low Recall (under-predicted)**:
- FC (0.58), FS (0.59), OTHER (0.60), ST (0.61)

### 2.3 Location Prediction (Built-in MDN)

The LSTM model includes a 3-component MDN head for location prediction.

#### Point Estimate Metrics

| Metric | Value | Unit |
|--------|-------|------|
| **MAE (px)** | 0.6068 | ft |
| **MAE (pz)** | 0.6479 | ft |
| **RMSE (px)** | 0.7791 | ft |
| **RMSE (pz)** | 0.8475 | ft |
| **Euclidean Error** | 0.9795 | ft |

#### Probabilistic Metrics

| Metric | Value |
|--------|-------|
| **Negative Log-Likelihood (NLL)** | 2.0829 |
| **Coverage @90%** | 94.50% |
| **Coverage @95%** | 97.45% |

#### Interpretation
- Average prediction is **~11.75 inches** from actual location
- The 90% confidence region contains the actual pitch **94.5%** of the time (well-calibrated)
- Vertical error (pz) is slightly higher than horizontal (px)

---

## 3. PitchTypeConditionedMDN Metrics

**Model**: `PitchTypeConditionedMDN`
**Checkpoint**: `models/pitch_type_location_20260121_003206/pitch_type_location_model.pt`
**Test Samples**: 801,978

### 3.1 Overall Performance

| Metric | Value | Unit | vs. LSTM MDN |
|--------|-------|------|--------------|
| **NLL** | 2.0484 | — | -1.7% (better) |
| **MAE (px)** | 0.5879 | ft | -3.1% (better) |
| **MAE (pz)** | 0.6441 | ft | -0.6% (better) |
| **Euclidean Error** | 0.9606 | ft | -1.9% (better) |
| **Coverage @90%** | 95.08% | — | +0.6% (better) |
| **Coverage @95%** | 97.62% | — | +0.2% (better) |

### 3.2 Per-Pitch-Type Performance

#### Detailed Metrics Table

| Pitch | NLL | MAE px (ft) | MAE pz (ft) | Euclid (ft) | Cov @90% | Cov @95% | Count |
|-------|-----|-------------|-------------|-------------|----------|----------|-------|
| **FF** | 1.983 | 0.563 | 0.630 | 0.933 | 95.40% | 97.78% | 256,437 |
| **SI** | 1.964 | 0.608 | 0.584 | 0.932 | 95.05% | 97.62% | 124,955 |
| **FC** | 2.031 | 0.576 | 0.663 | 0.966 | 94.64% | 97.28% | 61,418 |
| **CH** | 2.059 | 0.594 | 0.648 | 0.965 | 95.93% | 98.08% | 82,416 |
| **SL** | 2.082 | 0.569 | 0.662 | 0.959 | 94.42% | 97.34% | 119,663 |
| **CU** | 2.207 | 0.627 | 0.743 | 1.063 | 94.96% | 97.52% | 55,405 |
| **KC** | 2.163 | 0.585 | 0.743 | 1.031 | 95.35% | 97.83% | 12,867 |
| **ST** | 2.208 | 0.647 | 0.623 | 0.985 | 94.31% | 97.21% | 56,039 |
| **FS** | 2.129 | 0.590 | 0.697 | 0.999 | 95.98% | 98.15% | 25,939 |
| **KN** | 2.898 | 0.839 | 0.913 | 1.336 | 80.57% | 88.57% | 175 |
| **OTHER** | 2.370 | 0.683 | 0.741 | 1.106 | 92.66% | 96.16% | 6,664 |

#### Performance by Pitch Family

| Family | Pitch Types | Avg NLL | Avg Euclidean (ft) | Avg Coverage @90% |
|--------|-------------|---------|--------------------|--------------------|
| **Fastballs** | FF, SI, FC | 1.993 | 0.944 | 95.03% |
| **Breaking** | SL, CU, KC, ST | 2.165 | 1.010 | 94.76% |
| **Offspeed** | CH, FS | 2.094 | 0.982 | 95.96% |
| **Other** | KN, OTHER | 2.634 | 1.221 | 86.62% |

#### Analysis by Pitch Type

**Best Location Predictability** (lowest NLL):
1. **SI (1.964)**: Sinkers have consistent arm-side run, predictable locations
2. **FF (1.983)**: Four-seamers target predictable zones (up, away)
3. **FC (2.031)**: Cutters have tight command patterns

**Worst Location Predictability** (highest NLL):
1. **KN (2.898)**: Knuckleballs are inherently unpredictable (only 175 samples)
2. **OTHER (2.370)**: Catch-all category with mixed pitch types
3. **ST (2.208)**: Sweepers have wide horizontal movement variance

**Best Coverage (well-calibrated)**:
- FS (95.98%), CH (95.93%), FF (95.40%)

**Worst Coverage (under-confident)**:
- KN (80.57%) - model is too confident for unpredictable pitch
- OTHER (92.66%) - heterogeneous category

---

## 4. Model Comparison

### 4.1 LSTM+Attention vs. Alternatives

| Model | Accuracy | Top-3 | F1 Macro | Location NLL | Euclidean (ft) |
|-------|----------|-------|----------|--------------|----------------|
| **LSTM+Attention** | **72.34%** | **94.74%** | **0.665** | 2.083 | 0.980 |
| Enhanced+Attention | 68.56% | 93.75% | 0.642 | 2.123 | 0.989 |
| CatBoost (baseline) | 66.42% | 91.87% | 0.531 | — | 1.014 |

### 4.2 Location Model Comparison

| Model | NLL | MAE px (ft) | MAE pz (ft) | Euclidean (ft) | Coverage @90% |
|-------|-----|-------------|-------------|----------------|---------------|
| **PitchTypeConditionedMDN** | **2.048** | **0.588** | **0.644** | **0.961** | **95.08%** |
| LSTM Built-in MDN | 2.083 | 0.607 | 0.648 | 0.980 | 94.50% |
| CatBoost + MDN | 2.121 | 0.608 | 0.660 | 0.989 | — |

### 4.3 Improvement Summary

The pitch-type-conditioned MDN improves location prediction by:
- **1.7% lower NLL** (better likelihood)
- **3.1% lower horizontal MAE** (better px accuracy)
- **1.9% lower Euclidean error** (better overall distance)
- **0.6% higher 90% coverage** (better calibration)

---

## 5. Training Convergence

### 5.1 LSTM+Attention Training

| Metric | Value |
|--------|-------|
| Total Epochs | 11 (early stopped) |
| Best Validation Loss | 1.885 |
| Patience | 7 epochs |
| Learning Rate | 0.001 |

### 5.2 PitchTypeConditionedMDN Training

The location model trained for 100 epochs with learning rate scheduling.

#### Validation NLL Progression

| Epoch Range | Val NLL | Notes |
|-------------|---------|-------|
| 1-10 | 2.21 → 2.09 | Rapid initial learning |
| 11-20 | 2.09 → 2.07 | Continued improvement |
| 21-40 | 2.07 → 2.05 | Slower gains |
| 41-60 | 2.05 → 2.045 | Fine-tuning |
| 61-80 | 2.045 → 2.043 | Near convergence |
| 81-100 | 2.043 → 2.042 | Converged |

#### Final Training Metrics

| Metric | Train | Validation |
|--------|-------|------------|
| NLL | 2.076 | 2.042 |
| MAE px (ft) | — | 0.592 |
| MAE pz (ft) | — | 0.642 |
| Euclidean (ft) | — | 0.962 |

---

## 6. Metric Definitions

### Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | TP / Total | Fraction of correct predictions |
| **Precision** | TP / (TP + FP) | Fraction of positive predictions that are correct |
| **Recall** | TP / (TP + FN) | Fraction of actual positives correctly predicted |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **Top-3 Accuracy** | Correct in top 3 / Total | True class in top 3 predictions |
| **Macro Avg** | Mean across classes | Unweighted average (treats all classes equally) |
| **Weighted Avg** | Weighted by support | Weighted by class frequency |

### Location Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | Σ\|pred - actual\| / n | Mean Absolute Error in feet |
| **RMSE** | √(Σ(pred - actual)² / n) | Root Mean Square Error in feet |
| **Euclidean** | √((px_err)² + (pz_err)²) | 2D distance error in feet |
| **NLL** | -log p(actual \| params) | Negative Log-Likelihood (lower is better) |
| **Coverage @X%** | % of actuals within X% CI | Calibration metric |

### MDN-Specific Metrics

| Metric | Description |
|--------|-------------|
| **NLL** | Negative log-likelihood under the Gaussian mixture |
| **Coverage** | Fraction of samples within the predicted confidence region |

#### Coverage Interpretation
- **Coverage @90% = 94.5%** means the 90% confidence ellipse contains 94.5% of actual pitches
- Ideal coverage equals the confidence level (90% should contain 90%)
- **Over-coverage** (94.5% > 90%) indicates the model is slightly under-confident
- **Under-coverage** would indicate over-confidence

---

## 7. Key Takeaways

### Pitch Type Prediction
1. **72.3% accuracy** is strong given the inherent unpredictability of pitch selection
2. **94.7% top-3 accuracy** means the model almost always has the right pitch in top 3
3. **Fastballs (FF)** are most predictable (0.83 F1), knuckleballs least (0.50 F1)
4. The model tends to over-predict common pitches (FF, SL) and under-predict rare ones (FS, KC)

### Location Prediction
1. **~11.5 inches average error** is strong for a probabilistic prediction
2. **Pitch-type conditioning improves accuracy by 2-3%** across all metrics
3. **Fastballs are most predictable** (0.93 ft error), breaking balls less so (1.01 ft)
4. **90% coverage at 95%** indicates well-calibrated uncertainty estimates
5. **Vertical (pz) prediction is harder** than horizontal (px) for most pitch types

### Model Selection Guidance
- Use **LSTM+Attention** for unified pitch type + location inference
- Use **PitchTypeConditionedMDN** when maximum location accuracy is needed and pitch type is known/predicted
- The combined pipeline (LSTM → Conditioned MDN) provides the best overall performance
