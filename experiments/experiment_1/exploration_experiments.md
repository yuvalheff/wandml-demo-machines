# Exploration Experiments Summary - Machine Failure Prediction

## Overview
This document summarizes the exploration experiments conducted to determine the optimal approaches for the machine failure prediction binary classification task. All experiments used 5-fold cross-validation with PR-AUC as the evaluation metric.

## Dataset Characteristics
- **Training samples**: 300 (52% no-failure, 48% failure)  
- **Test samples**: 80 (61% no-failure, 39% failure)
- **Features**: 30 numerical + 1 categorical (f_27)
- **Key finding**: Categorical feature f_27 has 300 unique values (essentially an ID), should be excluded

## Experiment Results

### 1. Preprocessing Experiments
**Objective**: Test different scaling approaches to handle outliers identified in EDA.

| Approach | PR-AUC (CV) | Std Dev |
|----------|-------------|---------|
| No preprocessing | 0.5971 | ±0.0695 |
| Standard scaling | 0.6044 | ±0.0652 |
| **Robust scaling** | 0.5984 | ±0.0695 |

**Key Insight**: Standard scaling shows slight improvement, but differences are minimal. Random Forest is relatively robust to scaling.

### 2. Feature Engineering Experiments
**Objective**: Test transformations based on EDA findings about skewed features and outliers.

| Approach | PR-AUC (CV) | Std Dev |
|----------|-------------|---------|
| Original features | 0.5971 | ±0.0695 |
| Log-transform skewed | 0.5971 | ±0.0695 |
| Feature interactions | 0.5897 | ±0.0281 |
| **Outlier capping (IQR)** | 0.6005 | ±0.0634 |
| Combined approach | 0.5902 | ±0.0217 |

**Key Insights**:
- Outlier capping shows marginal improvement
- Feature interactions did not help (possibly due to Random Forest already capturing non-linear relationships)
- Skewness was not as problematic as initially expected

### 3. Algorithm Comparison
**Objective**: Test various ML algorithms to find the best performer.

| Algorithm | PR-AUC (CV) | Std Dev |
|-----------|-------------|---------|
| Random Forest | 0.6005 | ±0.0634 |
| Extra Trees | 0.6151 | ±0.0404 |
| Gradient Boosting | 0.6173 | ±0.0727 |
| XGBoost | 0.5986 | ±0.0467 |
| Logistic Regression | 0.5560 | ±0.0191 |
| SVM (RBF) | 0.5857 | ±0.0367 |
| **KNN** | 0.6176 | ±0.0698 |

**Key Insights**:
- KNN performed surprisingly well, suggesting local pattern importance
- Tree-based methods showed consistent good performance
- Linear methods underperformed, indicating non-linear relationships

### 4. Advanced Model Experiments
**Objective**: Hyperparameter tuning and ensemble methods for top performers.

| Approach | PR-AUC (CV) | Best Parameters |
|----------|-------------|-----------------|
| Tuned Gradient Boosting | 0.6018 | learning_rate=0.1, max_depth=5, n_estimators=200 |
| **Tuned KNN** | 0.6628 | n_neighbors=11, weights=distance, metric=manhattan |
| Voting Ensemble | 0.6301 | RF + GB + KNN with soft voting |

**Key Insights**:
- Tuned KNN achieved the best single-model performance (0.6628)
- Ensemble provided good balance between performance and stability
- Manhattan distance worked better than Euclidean for KNN

## Top Feature Importance (Random Forest)
Based on the Random Forest analysis:

1. **f_28**: 0.0686 (highest importance, matches EDA correlation findings)
2. **f_20**: 0.0604 (second highest, confirms EDA analysis)
3. **f_19**: 0.0491
4. **f_05**: 0.0483
5. **f_23**: 0.0482

These align well with the EDA correlation analysis, validating our feature understanding.

## Final Recommendations
Based on the exploration experiments:

1. **Best Model**: Tuned KNN (PR-AUC: 0.6628)
2. **Best Preprocessing**: Outlier capping using IQR method + Standard scaling for KNN
3. **Feature Selection**: Use all numerical features, exclude categorical f_27
4. **Ensemble Alternative**: Voting ensemble provides good stability if single-model risk is a concern

## Next Steps for Implementation
The experiment plan should focus on:
1. Implementing the optimized KNN approach as primary model
2. Including ensemble method as secondary approach
3. Comprehensive evaluation including calibration and error analysis
4. Feature importance analysis for model interpretability