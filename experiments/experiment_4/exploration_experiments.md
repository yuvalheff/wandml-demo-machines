# Experiment 4 Exploration Summary

## Context Analysis

Based on the previous iterations, there was a concerning performance decline trend:
- **Experiment 1 (KNN)**: PR-AUC 0.593
- **Experiment 2 (Ensemble)**: PR-AUC 0.569  
- **Experiment 3 (RF + Features)**: PR-AUC 0.525

The key insight was that increasing model complexity was actually hurting performance, suggesting overfitting or inappropriate model choice for the dataset characteristics.

## Key Discovery: Categorical Feature Issue

During exploration, I discovered that the categorical feature `f_27` has **completely unique values** in both train and test sets:
- Train: 300 unique values in 300 samples
- Test: 80 unique values in 80 samples
- **Zero overlap** between train and test categorical values

This indicates `f_27` is likely an ID-like feature that should be excluded from modeling, as it provides no generalizable signal.

## Exploration Experiments Conducted

### 1. Baseline Model Comparison (without f_27)
- **KNN**: PR-AUC 0.524, ROC-AUC 0.700
- **Logistic Regression**: PR-AUC 0.458, ROC-AUC 0.614  
- **Random Forest**: PR-AUC 0.558, ROC-AUC 0.649

### 2. Advanced Model Testing
- **Gradient Boosting**: PR-AUC **0.579**, ROC-AUC 0.656
  - Emerged as best performing baseline
  - Feature importance aligned with EDA findings (f_20, f_26, f_05 top features)

### 3. Feature Selection Experiments
- **Top 15 features** (F-test selection): Poor performance with KNN/LR
- **Top 5 correlated features only**: Moderate performance
- **Statistical feature selection** didn't improve over using all features

### 4. Feature Engineering with Gradient Boosting
Applied conservative feature engineering:
- **Ratio features**: f_20_div_f_14, f_20_div_f_28, f_14_div_f_28
- **Log transforms**: f_13_log, f_17_log (for highly skewed features from EDA)
- **Result**: PR-AUC improved to **0.618**

### 5. Hyperparameter Optimization
Tested three configurations for Gradient Boosting:
- Config 1: 100 est, lr=0.1, depth=3 → PR-AUC 0.626
- **Config 2**: 200 est, lr=0.05, depth=4 → PR-AUC **0.629** ⭐
- Config 3: 150 est, lr=0.08, depth=5 → PR-AUC 0.615

## Final Validation Results

**Best Configuration**: 
- Model: Gradient Boosting (200 estimators, lr=0.05, max_depth=4)
- Features: 30 base features + 5 engineered features
- **PR-AUC**: 0.6293 (19.9% improvement over Experiment 3)
- **ROC-AUC**: 0.704
- **Business metrics**: Can achieve 65% recall with 50% precision

## Key Insights

1. **Simpler is often better**: Gradient Boosting with conservative engineering outperformed complex RF approaches
2. **Feature quality > quantity**: 5 well-designed features were more valuable than 12 complex interactions  
3. **Model choice matters**: GB handles this dataset better than RF, likely due to sequential learning
4. **Categorical feature was harmful**: Excluding f_27 was crucial for generalization

## Recommendation for Experiment 4

Switch from Random Forest to **Gradient Boosting with conservative feature engineering**, focusing on:
- Ratio features from top predictive variables
- Log transforms for skewed features
- Optimized hyperparameters for the dataset size
- Comprehensive evaluation including business metrics analysis

This approach closes the gap to target performance from 21.9% (Exp 3) to just 6.4% below target.