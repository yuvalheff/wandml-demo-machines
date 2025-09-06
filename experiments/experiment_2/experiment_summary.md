# Experiment 2: RF+KNN Ensemble with Optimized Outlier Handling - Results Analysis

## Executive Summary

The RF+KNN ensemble experiment with optimized outlier handling **did not meet its performance targets**, achieving a PR-AUC of **0.569** against the target of **0.672** (15.3% below target). While the ensemble approach showed promise with improved discrimination ability (ROC-AUC: 0.714), the model exhibits a conservative prediction pattern with high precision (85.71%) but low recall (41.38%).

## Experiment Configuration

### Model Architecture
- **Algorithm**: Random Forest + K-Nearest Neighbors ensemble with soft voting
- **RF Component**: 100 estimators, random_state=42, no feature scaling
- **KNN Component**: 11 neighbors, distance weights, Manhattan metric, standardized features
- **Voting Strategy**: Soft voting with equal weights

### Key Changes from Iteration 1
- Replaced single KNN model with RF+KNN ensemble
- Optimized outlier capping from q99/q01 to q95/q05 quantiles
- Implemented soft voting ensemble with equal component weights

### Data Preprocessing
- **Feature Exclusion**: Removed `id` and `f_27` (300 unique categorical values)
- **Outlier Handling**: Quantile capping at 95th/5th percentiles for features f_20, f_06, f_00
- **Feature Scaling**: StandardScaler applied only for KNN component

## Performance Results

### Primary Metrics
| Metric | Target | Actual | Gap |
|--------|---------|---------|-----|
| **PR-AUC** | 0.672 | **0.569** | -15.3% |
| **F1-Score** | ≥0.650 | **0.558** | -14.2% |

### Comprehensive Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| PR-AUC | **0.569** | Primary evaluation metric (precision-recall area under curve) |
| ROC-AUC | **0.714** | Discrimination ability between classes |
| Accuracy | **0.548** | Overall classification accuracy |
| Precision | **0.857** | Precision when predicting failures (high confidence) |
| Recall | **0.414** | Sensitivity to actual failures (conservative) |
| F1-Score | **0.558** | Harmonic mean of precision and recall |
| Specificity | **0.846** | True negative rate (good at avoiding false alarms) |

### Confusion Matrix Analysis
- **Test Set Size**: 42 samples
- **True Positives**: 12 (correctly identified failures)
- **False Negatives**: 17 (missed failures - critical for maintenance)
- **False Positives**: 2 (false alarms - low business impact)
- **True Negatives**: 11 (correctly identified non-failures)

## Key Findings

### Strengths
1. **High Precision (85.7%)**: Model is highly accurate when predicting failures, minimizing false maintenance alerts
2. **Strong Specificity (84.6%)**: Excellent at avoiding false positives, reducing unnecessary maintenance costs
3. **Good ROC-AUC (0.714)**: Demonstrates reasonable discrimination between failure and non-failure classes
4. **Ensemble Stability**: Soft voting approach provides robust predictions

### Critical Weaknesses
1. **Low Recall (41.4%)**: **Major concern** - model misses 59% of actual failures, creating significant business risk
2. **Below-Target Performance**: Failed to achieve the 13.2% improvement goal over baseline
3. **Conservative Prediction Pattern**: Model is overly cautious, prioritizing precision over failure detection
4. **Limited Improvement**: Ensemble approach did not deliver expected performance gains

### Ensemble Component Analysis
- The combination of RF (scale-invariant) and KNN (distance-based) should theoretically complement each other
- However, the equal weighting strategy may not be optimal given the different algorithmic strengths
- Outlier capping optimization (q95/q05) may have been too aggressive, potentially removing important signal

## Business Impact Assessment

### Risk Analysis
- **High Business Risk**: With 59% of failures undetected, the model poses significant operational risks
- **Maintenance Efficiency**: Low false positive rate (4.8%) is beneficial for maintenance scheduling
- **Cost Implications**: Missing failures could lead to unplanned downtime exceeding false alarm costs

### Deployment Readiness
- **Not Ready for Production**: Current recall performance is insufficient for critical failure prediction
- **Threshold Optimization Needed**: Current decision threshold may need adjustment to balance precision/recall
- **Model Calibration**: Probability calibration analysis required for deployment confidence

## Technical Insights

### Feature Engineering Impact
- Minimal feature engineering approach may have limited model's learning capacity
- Original 30 features (after exclusions) may not capture complex failure patterns
- Outlier capping strategy needs reevaluation - may have removed critical extreme values

### Model Architecture Assessment
- Ensemble approach is sound but implementation may need refinement
- Equal voting weights assumption needs validation through component analysis
- Feature scaling strategy (only for KNN) creates information asymmetry between components

## Future Suggestions

### Immediate Priority (Next Iteration)
1. **Recall Optimization**: Implement recall-focused threshold optimization or cost-sensitive learning
2. **Ensemble Weight Optimization**: Use validation data to determine optimal component weights rather than equal weighting
3. **Feature Engineering Enhancement**: Explore interaction features, temporal patterns, and domain-specific transformations

### Medium-Term Improvements
1. **Advanced Ensemble Methods**: Consider stacking or blending approaches for better component integration
2. **Outlier Strategy Refinement**: Test multiple outlier handling approaches or feature-specific strategies
3. **Class Imbalance Techniques**: Implement SMOTE, cost-sensitive learning, or focal loss approaches

### Model Architecture Alternatives
1. **Gradient Boosting Integration**: Add XGBoost or LightGBM to the ensemble
2. **Neural Network Component**: Consider deep learning for complex pattern recognition
3. **Time-Series Features**: If temporal information is available, incorporate time-based features

## Context for Next Iteration

### Lessons Learned
- Simple ensemble approaches may not automatically improve performance
- Outlier handling optimization requires careful validation to avoid signal loss
- High precision/low recall pattern suggests need for recall-focused optimization strategies

### Continuity Notes
- Model artifacts saved in MLflow with signature and input example
- All evaluation plots available for detailed analysis (5 visualization files)
- Experiment successfully completed without technical errors
- Preprocessing pipeline validated and reusable

### Research Direction
The focus should shift toward recall enhancement while maintaining reasonable precision, with particular attention to ensemble component optimization and advanced feature engineering techniques.