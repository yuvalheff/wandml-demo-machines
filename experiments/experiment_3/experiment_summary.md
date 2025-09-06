# Experiment 3: Threshold-Optimized Random Forest with Interaction Features

## Executive Summary

**Critical Finding: Experiment 3 significantly underperformed expectations**, achieving a PR-AUC of **0.525** versus the target of **0.672** (21.9% below target). This represents a **declining performance trend** across all three iterations, with performance actually regressing from the baseline KNN approach (0.593 → 0.525).

## Performance Results

### Primary Metrics
- **PR-AUC: 0.525** (Target: 0.672) - **FAILED**
- **ROC-AUC: 0.623** - Reasonable discrimination ability
- **Performance vs Random Baseline: +35.3%** - Model learned patterns but insufficient

### Performance Trend Analysis
| Iteration | Algorithm | PR-AUC | Change |
|-----------|-----------|---------|---------|
| 1 | KNN + Outlier Capping | 0.593 | Baseline |
| 2 | RF+KNN Ensemble | 0.569 | -4.0% |
| 3 | RF + Interaction Features | 0.525 | -11.5% |

**Key Insight: Increasing model complexity degraded performance**, suggesting overfitting or inappropriate feature engineering.

## Model Configuration

### Algorithm & Hyperparameters
- **Random Forest Classifier** with 200 estimators
- **Balanced class weights** to address imbalance
- **Max depth: 12** with conservative splitting parameters
- **Threshold optimization** targeting recall ≥ 65%

### Feature Engineering Strategy
- **Base Features:** Top 3 predictive features (f_20, f_14, f_28)
- **12 Engineered Features:** 3 multiplicative, 3 ratio, 6 polynomial
- **Final Feature Count:** 42 (30 original + 12 engineered)

## Feature Importance Analysis

### Top Performing Features
| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | f_26 | 0.044 | Original |
| 2 | f_05 | 0.041 | Original |
| 3 | f_28 | 0.039 | Original |
| 4 | f_22 | 0.039 | Original |
| 5 | **f_28_cubed** | **0.038** | **Engineered** |

### Feature Engineering Impact
- **7 of top 20 features (35%) are engineered** - feature engineering showed value
- **Top engineered features:** f_28_cubed, f_20_div_f_14, f_20_div_f_28
- **Balanced importance distribution** - no single dominant feature

## Critical Analysis: Why the Experiment Failed

### 1. Feature Engineering Paradox
- **✅ Feature Engineering Worked:** 35% of top features are engineered
- **❌ Overall Performance Declined:** Model couldn't effectively leverage them
- **Root Cause:** Likely overfitting from 30→42 features with complex interactions

### 2. Model Complexity Mismatch
- **Random Forest complexity** inappropriate for dataset characteristics
- **Interaction features added noise** rather than meaningful signal
- **Class balancing ineffective** with this feature combination

### 3. Threshold Optimization Limitations
- **Threshold optimization implemented** but couldn't overcome fundamental model limitations
- **No threshold achieved target performance** - indicates deeper model issues
- **Expected recall target (≥65%) unachievable** with current approach

## Comparison with Previous Iterations

### What Worked Previously
- **Experiment 1 (KNN):** Achieved highest PR-AUC (0.593)
- **Simple outlier handling** more effective than complex feature engineering
- **Simpler models** better suited to dataset characteristics

### What Failed in This Iteration
- **Advanced feature engineering** counterproductive
- **Ensemble complexity** (from Iteration 2) and Random Forest sophistication
- **Threshold optimization** insufficient without fundamental model improvements

## Business Impact Assessment

### Deployment Status: **NOT READY**
- **Missing business requirements** by significant margin (22% below target)
- **Predictive maintenance effectiveness compromised** with low recall
- **Performance regression** from simpler, proven approaches

### Cost-Benefit Analysis
- **High development cost** for advanced feature engineering
- **Negative ROI** due to performance decline
- **Technical debt** from complex feature pipeline

## Future Recommendations

### Immediate Actions (Next Iteration)
1. **Return to Baseline:** Start from Experiment 1's KNN approach (PR-AUC: 0.593)
2. **Selective Feature Engineering:** Test individual engineered features rather than comprehensive approach
3. **Gradient Boosting Trial:** XGBoost/LightGBM may handle feature interactions better

### Strategic Recommendations
1. **Simplicity First:** Prefer simple, interpretable models that perform well
2. **Incremental Engineering:** Add complexity only when proven beneficial
3. **Alternative Algorithms:** Neural networks, support vector machines, or advanced ensembles
4. **Data Quality Investigation:** Assess potential data leakage or quality issues

### Success Criteria for Next Iteration
- **Minimum Target:** PR-AUC ≥ 0.600 (vs current 0.525)
- **Stretch Goal:** PR-AUC ≥ 0.650 (approaching business target)
- **No Performance Regression:** Must exceed Experiment 1 baseline (0.593)

## Lessons Learned

### Technical Insights
1. **Feature engineering value varies:** Individual features worked, but ensemble degraded performance
2. **Model complexity isn't always better:** Simple KNN outperformed sophisticated RF
3. **Class imbalance handling:** Balanced weights insufficient with poor feature combinations

### Methodological Insights
1. **Threshold optimization requires strong base model:** Can't compensate for fundamental limitations
2. **Comprehensive feature engineering risks overfitting:** Selective approach needed
3. **Performance monitoring critical:** Early detection of declining trends needed

## Conclusion

Experiment 3 demonstrates that **sophisticated approaches can be counterproductive** when they don't align with dataset characteristics. The significant performance decline (0.593 → 0.525) across iterations suggests the need to **return to fundamentals** and build incrementally from proven baselines rather than pursuing complex feature engineering without validation.

**Next iteration should focus on recovering baseline performance** before attempting additional complexity.