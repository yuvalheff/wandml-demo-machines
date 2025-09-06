# Machine Failure Prediction - Exploratory Data Analysis Report

## Dataset Overview
- **Dataset Size**: 300 samples × 33 columns (31 features + ID + target)
- **Task Type**: Binary Classification
- **Target Variable**: `target` (0 = No Failure, 1 = Failure)
- **Data Quality**: Excellent - No missing values
- **Feature Composition**: 30 numerical features + 1 categorical feature

## Target Variable Analysis
- **Class Distribution**: Well-balanced
  - No Failure (0): 156 samples (52%)
  - Failure (1): 144 samples (48%)
- **Class Imbalance**: Minimal (4% difference)
- **Recommendation**: No special class balancing techniques required

## Feature Analysis

### Feature Types Breakdown
- **Continuous Features** (f_00-f_06, f_19-f_26, f_28): 15 features
- **Discrete Integer Features** (f_07-f_18, f_29-f_30): 15 features  
- **Categorical Features** (f_27): 1 feature with 300 unique values

### Key Predictive Features
1. **f_20** (correlation: -0.161)
   - Strongest predictor
   - High outlier count
   - Clear class separation (Cohen's d = 0.327)

2. **f_14** (correlation: -0.131)
   - Second strongest predictor
   - Discrete integer with outliers
   - Good discriminative power (Cohen's d = 0.264)

3. **f_28** (correlation: 0.112)
   - Third strongest predictor
   - Continuous feature with large scale values
   - Moderate effect size (Cohen's d = 0.224)

### Data Quality Issues

#### Skewness
- **Highly Skewed**: f_17, f_13, f_14, f_07, f_10, f_08
- **Impact**: May require transformation for linear models
- **Recommendation**: Consider log transformation or robust scaling

#### Outliers
- **Most Problematic**: f_20, f_06, f_00, f_14, f_02, f_17
- **Detection Method**: IQR-based (1.5 × IQR rule)
- **Recommendation**: Apply outlier handling (clipping, robust scaling, or removal)

#### Categorical Feature Challenge
- **f_27**: 300 unique string values (one per sample)
- **Issue**: High cardinality may cause overfitting
- **Recommendation**: Consider feature hashing, embedding, or removal

## Feature Relationships

### Correlation Analysis
- **Overall Correlation Strength**: Moderate
- **Top Correlations with Target**:
  - f_20: -0.161
  - f_14: -0.131
  - f_28: +0.112
  - f_22: +0.104
  - f_26: +0.103

### Class Separation
- **Good Separability**: Features show distinct patterns between failure/non-failure
- **Multi-dimensional**: Combination of features provides better discrimination
- **Visualization**: Scatter plot matrix reveals clear clustering patterns

## Preprocessing Recommendations

### Immediate Actions
1. **Scaling**: Apply robust scaling due to outliers
2. **Categorical Encoding**: Handle f_27's high cardinality
3. **Feature Selection**: Focus on top correlated features

### Optional Enhancements
1. **Outlier Treatment**: Robust preprocessing for f_20, f_06, f_00
2. **Transformation**: Log/sqrt transformation for highly skewed features
3. **Feature Engineering**: Create interaction terms from top features

## Model Recommendations

### Suitable Algorithms
- **Tree-based Models**: Random Forest, XGBoost (handle outliers well)
- **Linear Models**: Logistic Regression (with preprocessing)
- **Ensemble Methods**: Combine multiple algorithms

### Evaluation Strategy
- **Primary Metric**: PR-AUC (as specified)
- **Cross-validation**: Stratified K-fold
- **Feature Importance**: Track top contributing features

## Summary
The dataset shows excellent quality with balanced classes and clear predictive signals. Key challenges include outliers, skewed distributions, and high-cardinality categorical feature. The moderate correlation strengths suggest ensemble methods may perform well. Focus preprocessing efforts on outlier handling and proper scaling for optimal model performance.