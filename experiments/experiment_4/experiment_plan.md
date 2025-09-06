# Experiment 4: Gradient Boosting with Conservative Feature Engineering

## Overview
**Objective**: Reverse the performance decline trend from previous iterations by switching to Gradient Boosting with optimized feature engineering.

**Key Change**: Move from Random Forest to Gradient Boosting algorithm, which showed superior performance in exploration experiments.

## Background
Previous iterations showed declining performance:
- Experiment 1 (KNN): PR-AUC 0.593
- Experiment 2 (Ensemble): PR-AUC 0.569  
- Experiment 3 (RF + Features): PR-AUC 0.525

Exploration experiments revealed that Gradient Boosting with conservative feature engineering achieved PR-AUC 0.629, indicating a promising path forward.

## Data Preprocessing Steps

1. **Data Loading**
   - Load `train_set.csv` and `test_set.csv` from data/ directory
   - Verify datasets contain 300 training and 80 test samples

2. **Feature Cleaning** 
   - Remove `id` column (identifier, not predictive)
   - **CRITICAL**: Remove `f_27` (categorical feature with 300/80 unique values respectively - no overlap between train/test, causes data leakage)
   - Retain 30 numerical features: `f_00` through `f_30` (excluding `f_27`)

3. **Data Quality Validation**
   - Confirm no missing values in remaining features
   - Verify target distribution: ~48% failure rate in train, ~39% in test

4. **No Scaling Required**
   - Gradient Boosting handles mixed feature scales naturally
   - Use raw numerical values for all features

## Feature Engineering Steps

1. **Ratio Features** (from top correlated features in EDA)
   ```python
   f_20_div_f_14 = f_20 / (f_14 + 1e-8)  # Prevent division by zero
   f_20_div_f_28 = f_20 / (f_28 + 1e-8) 
   f_14_div_f_28 = f_14 / (f_28 + 1e-8)
   ```

2. **Log Transformations** (for highly skewed features from EDA)
   ```python
   f_13_log = log1p(abs(f_13))  # Handle negative values
   f_17_log = log1p(abs(f_17))  # Most skewed features from EDA
   ```

3. **Final Feature Set**
   - 30 original numerical features
   - 5 engineered features  
   - **Total**: 35 features for modeling

## Model Selection and Training

1. **Algorithm**: GradientBoostingClassifier (sklearn.ensemble)

2. **Optimized Hyperparameters** (from exploration results)
   ```python
   n_estimators = 200          # Sufficient for convergence
   learning_rate = 0.05        # Conservative to prevent overfitting  
   max_depth = 4               # Moderate complexity
   min_samples_split = 15      # Regularization for small dataset
   random_state = 42           # Reproducibility
   ```

3. **Training Strategy**
   - No class weighting (dataset is well-balanced)
   - Single model (no ensemble based on exploration results)
   - Train on 300 samples with 35 features

## Evaluation Strategy

### Primary Metrics
- **PR-AUC**: Primary metric for imbalanced failure prediction
- **ROC-AUC**: Secondary discrimination metric

### Business Metrics Analysis
- Calculate precision at 65% recall (business requirement)
- Threshold analysis for optimal operating points
- Confusion matrix for deployment readiness assessment

### Model Diagnostics
1. **Feature Importance Analysis**
   - Validate that engineered features rank in top 15
   - Compare with EDA correlation findings

2. **Model Calibration**
   - Calibration plots for probability reliability
   - Essential for business decision-making

3. **Performance Comparison**
   - Against Experiment 3 baseline (target: > 0.525 PR-AUC)
   - Progress toward business target (0.672 PR-AUC)

### Visualization Outputs
- `precision_recall_curve.html`: PR curve with threshold analysis
- `feature_importance.html`: Feature ranking with engineering validation
- `confusion_matrix.html`: Classification performance breakdown  
- `calibration_plot.html`: Probability reliability assessment
- `threshold_analysis.html`: Business metrics optimization

## Success Criteria

1. **Primary**: PR-AUC > 0.60 (significant improvement over 0.525)
2. **Business**: Achieve 65% recall with precision > 45%
3. **Technical**: Engineered features appear in top 15 by importance
4. **Deployment**: Model calibration suitable for probability-based decisions

## Expected Outcomes

- **PR-AUC Range**: 0.62 - 0.64 (based on exploration results)
- **Performance Gap**: Reduce gap to business target from 21.9% to ~6.4%
- **Algorithm Validation**: Confirm Gradient Boosting > Random Forest for this dataset
- **Engineering Validation**: Conservative feature engineering > complex interactions

## Key Insights to Validate

1. **Model Choice Impact**: GB's sequential learning better suited than RF's bagging
2. **Feature Quality**: Simple ratio/log features more effective than complex interactions  
3. **Regularization**: Conservative hyperparameters prevent overfitting on 300 samples
4. **Data Quality**: Removing f_27 crucial for generalization

This experiment represents a strategic pivot based on thorough exploration, designed to reverse the performance decline and move significantly closer to business deployment requirements.