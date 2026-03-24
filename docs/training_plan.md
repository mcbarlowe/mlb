# Pitch Prediction Model Training Plan

## Overview

This document outlines the training strategy for the pitch prediction model with MDN location head. The plan includes data splitting, cross-validation, hyperparameter tuning, and final evaluation.

## Data Inventory

| Season | Games (approx) | Notes |
|--------|---------------|-------|
| 2018 | ~2,965 | Full season |
| 2019 | ~2,953 | Full season |
| 2020 | ~1,280 | COVID-shortened (60 games) |
| 2021 | ~2,884 | Full season |
| 2022 | ~2,741 | Full season |
| 2023 | ~2,957 | Full season |
| 2024 | ~2,940 | Full season |
| 2025 | ~2,903 | Current/partial season |

**Total**: ~21,000+ games, estimated 700,000+ pitches per full season

## Data Splitting Strategy

### Why Time-Series Aware Splitting?

Standard k-fold cross-validation is inappropriate because:
1. Baseball data is temporal - games happen in sequence
2. Player performance evolves over seasons
3. Random splits would leak future information into training

### Recommended Split: Season-Based Time-Series CV

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SPLIT STRATEGY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CROSS-VALIDATION (2018-2024)              │  TEST SET (2025)      │
│  ────────────────────────────────────────  │  ──────────────────── │
│                                            │                       │
│  Fold 1: Train [2018-2021] → Val [2022]    │                       │
│  Fold 2: Train [2018-2022] → Val [2023]    │   HELD OUT            │
│  Fold 3: Train [2018-2023] → Val [2024]    │   Final Evaluation    │
│                                            │                       │
│  (Expanding window cross-validation)       │                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Rationale

1. **Expanding Window CV**: More realistic - we always train on past data to predict future
2. **Season Boundaries**: Natural temporal boundaries in baseball
3. **2025 Holdout**: True out-of-sample test (never seen during any training/validation)
4. **Skip 2020**: COVID season was anomalous; optionally exclude from training

## Cross-Validation Details

### Fold Structure

| Fold | Training Seasons | Validation Season | Training Size (approx) |
|------|-----------------|-------------------|------------------------|
| 1 | 2018-2021 | 2022 | ~10,000 games |
| 2 | 2018-2022 | 2023 | ~13,000 games |
| 3 | 2018-2023 | 2024 | ~16,000 games |

### Metrics to Track Per Fold

**Classification (Pitch Type)**:
- Accuracy
- Top-3 Accuracy
- Macro F1
- Per-class F1 (especially for rare pitches)

**Location (MDN)**:
- Negative Log-Likelihood (NLL) - primary metric
- MAE for px and pz (using mean prediction)
- RMSE for px and pz
- Coverage @ 90% (calibration)
- Coverage @ 95% (calibration)

## Hyperparameter Search

### Parameters to Tune

| Parameter | Search Range | Default |
|-----------|-------------|---------|
| `hidden_dim` | [64, 128, 256] | 128 |
| `n_layers` | [1, 2, 3] | 2 |
| `dropout` | [0.1, 0.3, 0.5] | 0.3 |
| `n_location_components` | [1, 3, 5] | 3 |
| `learning_rate` | [1e-4, 5e-4, 1e-3] | 1e-3 |
| `type_weight` | [0.5, 1.0, 2.0] | 1.0 |
| `location_weight` | [0.25, 0.5, 1.0] | 0.5 |
| `embedding_dim` | [16, 32, 64] | 32 |

### Search Strategy

**Recommended**: Random search with 20-30 configurations
- Faster than grid search
- Better coverage of hyperparameter space
- Each configuration evaluated across all 3 CV folds

### Selection Criterion

Primary: Mean validation NLL across folds (lower is better)
Secondary: Mean validation accuracy for pitch type

## Training Procedure

### Phase 1: Hyperparameter Search

```python
# Pseudocode
for config in random_search_configs:
    fold_scores = []
    for fold in [1, 2, 3]:
        model = create_model(**config)
        trainer = PitchPredictionTrainer(model, train_data[fold], val_data[fold])
        results = trainer.train(n_epochs=30, early_stopping_patience=5)
        fold_scores.append(results['best_val_loss'])

    mean_score = np.mean(fold_scores)
    log_results(config, mean_score)

best_config = select_best_config()
```

### Phase 2: Final Model Training

After selecting best hyperparameters:

1. **Train on all CV data** (2018-2024)
2. Use 2024 as validation for early stopping
3. Train until convergence or patience exceeded

```python
final_model = create_model(**best_config)
trainer = PitchPredictionTrainer(
    model=final_model,
    train_loader=all_training_data,  # 2018-2023
    val_loader=validation_data,       # 2024
)
trainer.train(n_epochs=50, early_stopping_patience=10)
```

### Phase 3: Test Set Evaluation

Evaluate on held-out 2025 data:

```python
test_results = evaluate_model(final_model, test_loader_2025)

print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test Top-3 Accuracy: {test_results['top3_accuracy']:.4f}")
print(f"Test NLL: {test_results['nll']:.4f}")
print(f"Test Coverage @95%: {test_results['coverage_95']:.4f}")
```

## Implementation Checklist

### 1. Data Preparation
- [ ] Implement season-based data loading
- [ ] Create CV fold generator
- [ ] Verify no data leakage between folds

### 2. Cross-Validation Infrastructure
- [ ] Create `TimeSeriesCrossValidator` class
- [ ] Implement fold iteration with proper train/val splits
- [ ] Add logging for per-fold metrics

### 3. Hyperparameter Search
- [ ] Implement random search over config space
- [ ] Add results logging (CSV or JSON)
- [ ] Create visualization of search results

### 4. Training Script
- [ ] Command-line interface for training
- [ ] Support for resuming from checkpoint
- [ ] Proper random seed handling

### 5. Evaluation
- [ ] Generate comprehensive test set report
- [ ] Create visualization of MDN predictions
- [ ] Analyze performance by pitch type, count, etc.

## Expected Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Data prep & CV infrastructure | - |
| 2 | Hyperparameter search (20 configs × 3 folds) | - |
| 3 | Final model training | - |
| 4 | Test evaluation & analysis | - |

## Success Criteria

### Minimum Viable Performance
- Pitch type accuracy > 40% (baseline: ~30% random for 11 classes)
- Top-3 accuracy > 70%
- Coverage @95% between 90-98% (well-calibrated)

### Target Performance
- Pitch type accuracy > 45%
- Top-3 accuracy > 75%
- NLL improvement over single-Gaussian baseline
- Coverage @95% between 93-97%

## Files to Create

1. `src/ml/cross_validation.py` - CV infrastructure
2. `src/ml/hyperparameter_search.py` - HP search utilities
3. `scripts/train_model.py` - Main training script
4. `scripts/evaluate_model.py` - Evaluation script
5. `notebooks/training_analysis.ipynb` - Results visualization

## Notes

### On the 2020 Season
The COVID-shortened 2020 season may have different characteristics:
- Only 60 games instead of 162
- No fans (different pressure?)
- Different schedule density

**Recommendation**: Include in training but monitor if excluding improves validation performance.

### On New Players
Players who appear in 2025 but not in training data will get a default embedding. Consider:
- Using a shared "unknown player" embedding
- Fine-tuning on early 2025 data before final evaluation

### On MDN Components
Start with K=3 components. If coverage metrics are poor:
- K=1: Reduces to standard regression
- K=5+: More expressive but harder to train
