# Experiment 3 Plan: Threshold-Optimized Random Forest with Interaction Features

## Experiment Overview

**Primary Objective**: Implement threshold-optimized Random Forest with advanced feature engineering to significantly improve recall (target: ≥65%) while achieving PR-AUC target of 0.672.

**Key Innovation**: Replace ensemble approach with optimized single model + threshold tuning strategy for better recall-precision balance.

**Expected Impact**: 14.2% PR-AUC improvement (0.650 vs 0.569) and >100% recall improvement (0.86+ vs 0.414).

## Detailed Implementation Plan

### 1. Data Preprocessing Steps

#### 1.1 Feature Selection
- **Remove columns**: `['id', 'f_27']`
- **Rationale**: `id` is non-predictive identifier; `f_27` has 300 unique categorical values (overfitting risk)
- **Resulting features**: 30 numerical features (continuous and discrete)

#### 1.2 Outlier Handling Strategy
- **Approach**: NO aggressive outlier capping (change from iteration 2)
- **Rationale**: Exploration showed that preserving extreme values improves performance
- **Previous issue**: q95/q05 capping on f_20, f_06, f_00 removed critical failure signals
- **New approach**: Keep original feature distributions to preserve failure pattern information

#### 1.3 Data Scaling
- **Not required**: Random Forest handles mixed scales naturally
- **No standardization**: Preserve interpretability and feature importance rankings

### 2. Feature Engineering Steps

#### 2.1 Interaction Feature Creation
Create interaction features using top 3 predictive features: `f_20`, `f_14`, `f_28`

**Multiplicative Interactions**:
```python
# Cross-multiplication terms
f_20_x_f_14 = f_20 * f_14
f_20_x_f_28 = f_20 * f_28  
f_14_x_f_28 = f_14 * f_28
```

**Ratio Interactions**:
```python
# Division ratios (add small epsilon to avoid division by zero)
f_20_div_f_14 = f_20 / (f_14 + 1e-8)
f_20_div_f_28 = f_20 / (f_28 + 1e-8)
f_14_div_f_28 = f_14 / (f_28 + 1e-8)
```

**Polynomial Features**:
```python
# Higher-order terms
f_20_squared = f_20 ** 2
f_14_squared = f_14 ** 2  
f_28_squared = f_28 ** 2
f_20_cubed = f_20 ** 3
f_14_cubed = f_14 ** 3
f_28_cubed = f_28 ** 3
```

**Total Features**: 42 (30 original + 12 engineered)

#### 2.2 Feature Validation
- Verify no infinite or NaN values in engineered features
- Apply clipping if necessary for extreme ratio values
- Log feature creation summary for reproducibility

### 3. Model Selection Steps

#### 3.1 Algorithm Choice
- **Model**: `RandomForestClassifier`
- **Rationale**: Best performing algorithm from exploration; handles interactions naturally; provides feature importance

#### 3.2 Model Configuration
```python
RandomForestClassifier(
    n_estimators=200,           # Increased from 100 for stability
    random_state=42,            # Reproducibility
    class_weight='balanced',    # Address class imbalance (key change)
    max_depth=12,              # Prevent overfitting while allowing complexity
    min_samples_split=5,       # Conservative splitting
    min_samples_leaf=2,        # Prevent overfitting
    bootstrap=True,            # Standard RF bootstrap
    n_jobs=-1                  # Parallel processing
)
```

#### 3.3 Cross-Validation Strategy
- **Method**: 5-fold Stratified Cross-Validation
- **Purpose**: Reliable performance estimation with balanced folds
- **Metrics**: PR-AUC, Recall, Precision, F1-Score

### 4. Threshold Optimization Strategy

#### 4.1 Threshold Selection Methodology
1. **Train model** on full training set with cross-validation
2. **Generate probability predictions** on validation set  
3. **Analyze precision-recall curve** across threshold range [0.1, 0.9]
4. **Identify optimal thresholds** for different business objectives:
   - **Recall-focused**: Threshold achieving recall ≥ 0.65
   - **F1-optimized**: Threshold maximizing F1-score
   - **Balanced**: Threshold balancing precision and recall

#### 4.2 Threshold Evaluation Criteria
- **Primary**: Recall ≥ 0.65 (detect majority of failures)
- **Secondary**: Maintain precision ≥ 0.50 (limit false alarms)  
- **Target**: F1-score ≥ 0.650 (overall balanced performance)

#### 4.3 Deployment Threshold Selection
- **Recommended**: Use recall-focused threshold (≥0.65 recall)
- **Business justification**: In predictive maintenance, missing failures (false negatives) is more costly than false alarms
- **Expected threshold**: ~0.34 based on exploration results

### 5. Evaluation Strategy

#### 5.1 Primary Evaluation Metrics
- **PR-AUC** (primary): Target ≥ 0.672 (15.3% improvement over iteration 2)
- **Recall**: Target ≥ 0.65 (>50% improvement over iteration 2)
- **F1-Score**: Target ≥ 0.650 (minimum threshold for acceptance)

#### 5.2 Secondary Evaluation Metrics
- **Precision**: Monitor to ensure reasonable false alarm rate
- **ROC-AUC**: Overall discrimination ability
- **Specificity**: True negative rate for non-failure detection

#### 5.3 Diagnostic Analysis Requirements

**A. Threshold Analysis**:
- Generate precision-recall curves
- Create threshold performance plots (recall, precision, F1 vs threshold)
- Document optimal thresholds for different objectives
- Analyze trade-offs between recall and precision

**B. Feature Importance Analysis**:
- Rank all 42 features by Random Forest importance scores
- Identify top 10 most predictive features
- Analyze contribution of engineered vs original features
- Validate interaction feature effectiveness

**C. Prediction Pattern Analysis**:
- Confusion matrix analysis at optimal threshold
- Probability distribution analysis by class
- Identification of hard-to-classify cases
- Business impact assessment (cost of false positives vs false negatives)

**D. Model Calibration Assessment**:
- Calibration curve analysis
- Reliability diagram for probability predictions
- Assessment of prediction confidence intervals

### 6. Expected Outputs

#### 6.1 Model Artifacts
- `model.pkl`: Trained RandomForest model
- `feature_names.json`: List of all 42 feature names
- `threshold_analysis.json`: Optimal thresholds for different objectives

#### 6.2 Evaluation Reports
- `experiment_summary.json`: Comprehensive metrics and results
- `experiment_summary.md`: Human-readable analysis and insights
- Performance visualization plots (HTML format):
  - `precision_recall_curve.html`
  - `threshold_analysis.html` 
  - `feature_importance.html`
  - `confusion_matrix.html`
  - `calibration_curve.html`

#### 6.3 Deployment Recommendations
- Optimal threshold recommendation with business justification
- Feature importance insights for model interpretability
- Performance comparison with previous iterations
- Next iteration suggestions if targets not met

## Success Criteria

### Must-Have (Minimum Viable)
- ✅ PR-AUC ≥ 0.600 (improvement over iteration 2: 0.569)
- ✅ Recall ≥ 0.60 (major improvement over iteration 2: 0.414)
- ✅ F1-Score ≥ 0.600 (improvement over iteration 2: 0.558)

### Target Performance
- 🎯 PR-AUC ≥ 0.672 (project target)  
- 🎯 Recall ≥ 0.65 (business requirement for failure detection)
- 🎯 F1-Score ≥ 0.650 (balanced performance threshold)

### Stretch Goals
- 🚀 PR-AUC ≥ 0.700 (exceptional performance)
- 🚀 Recall ≥ 0.75 (detect 3 out of 4 failures)
- 🚀 Maintain Precision ≥ 0.55 (reasonable false alarm rate)

## Risk Mitigation

### Technical Risks
1. **Overfitting with interaction features**: Mitigated by conservative RF hyperparameters and cross-validation
2. **Threshold instability**: Addressed through systematic threshold analysis and validation set evaluation  
3. **Feature scaling issues**: Avoided by using tree-based model (RF) that handles mixed scales

### Business Risks  
1. **False negative rate**: Directly addressed through recall optimization and threshold tuning
2. **False alarm fatigue**: Balanced through precision monitoring and threshold selection
3. **Model interpretability**: Maintained through feature importance analysis and single-model approach

This comprehensive plan builds on successful exploration experiments to deliver a significant improvement in machine failure detection capability while maintaining model reliability and interpretability.