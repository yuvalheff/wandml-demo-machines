# Iteration 2: Exploration Experiments Summary

## Overview
This document summarizes the exploration experiments conducted to design Iteration 2 of the machine failure prediction experiment. The goal was to identify improvements over the baseline KNN approach (PR-AUC: 0.593) from Iteration 1.

## Exploration Results

### 1. Individual Model Performance (5-fold CV)
- **Random Forest**: 0.6006 ± 0.0528 PR-AUC
- **Gradient Boosting**: 0.6158 ± 0.0659 PR-AUC  
- **KNN**: 0.6481 ± 0.0342 PR-AUC

**Key Finding**: All individual models outperform the Iteration 1 baseline, with KNN showing the most consistent performance across folds.

### 2. Ensemble Methods Comparison
- **RF + GB**: 0.6094 ± 0.0620 PR-AUC
- **RF + KNN**: **0.6622 ± 0.0478 PR-AUC** ⭐ Best Performance
- **GB + KNN**: 0.6531 ± 0.0614 PR-AUC
- **RF + GB + KNN**: 0.6449 ± 0.0621 PR-AUC

**Key Finding**: The RF+KNN ensemble significantly outperforms all other combinations, achieving 11.7% improvement over the Iteration 1 baseline.

### 3. Feature Engineering Experiments
- **Polynomial Features (degree 2)**: 0.6129 ± 0.0321 PR-AUC
- **Domain-Specific Interactions**: 0.6243 ± 0.0365 PR-AUC

**Key Finding**: Feature engineering approaches showed marginal improvements but didn't exceed the baseline RF+KNN ensemble performance.

### 4. Outlier Handling Strategy Optimization
- **q99/q01 (current)**: 0.6622 ± 0.0478 PR-AUC
- **q95/q05**: **0.6717 ± 0.0538 PR-AUC** ⭐ Best Performance
- **q99.5/q0.5**: 0.6622 ± 0.0478 PR-AUC

**Key Finding**: Using q95/q05 quantiles for outlier capping provides additional 1.4% performance boost.

### 5. Threshold Optimization Analysis
Based on cross-validation predictions from the best RF+KNN ensemble:

| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|---------|----------|
| 0.3       | 0.513     | 0.986   | 0.675    |
| **0.4**   | **0.556** | **0.868**| **0.678** |
| 0.5       | 0.603     | 0.507   | 0.551    |
| 0.6       | 0.750     | 0.229   | 0.351    |
| 0.7       | 0.818     | 0.062   | 0.116    |

**Key Finding**: Threshold 0.4 maximizes F1-score, while threshold 0.7 maximizes precision. For maintenance applications, a balanced threshold of 0.4 is recommended.

## Final Recommendations
1. **Best Configuration**: RF+KNN ensemble with q95/q05 outlier capping
2. **Expected PR-AUC**: 0.6717 (13.2% improvement over Iteration 1)
3. **Optimal Threshold**: 0.4 for balanced precision-recall trade-off
4. **Feature Engineering**: Skip complex feature engineering - the ensemble approach provides sufficient performance gains

## Implementation Strategy
The optimal approach focuses on a single key change: implementing the RF+KNN ensemble with improved outlier handling. This provides substantial performance improvement while maintaining interpretability and avoiding overfitting risks from excessive feature engineering.