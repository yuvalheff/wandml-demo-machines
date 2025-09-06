# Experiment 2: RF+KNN Ensemble with Optimized Outlier Handling

## Experiment Overview
**Objective**: Implement Random Forest and K-Nearest Neighbors ensemble with improved outlier capping to enhance machine failure prediction performance beyond the Iteration 1 baseline (PR-AUC: 0.593).

**Key Innovation**: Replace single KNN approach with RF+KNN ensemble and optimize outlier handling strategy based on exploration experiments.

**Expected Performance**: PR-AUC ≥ 0.672 (13.2% improvement over baseline)

## Data Preprocessing Steps

### 1. Feature Exclusion
- **Exclude columns**: `id`, `f_27`
- **Reasoning**: 
  - `id`: Identifier column with no predictive value
  - `f_27`: Categorical feature with 300 unique values requiring complex encoding (poor signal-to-noise ratio)

### 2. Outlier Handling (KEY IMPROVEMENT)
- **Method**: Quantile-based capping
- **Target features**: `f_20`, `f_06`, `f_00` (highest outlier counts from EDA)
- **Quantiles**: 
  - Upper bound: 95th percentile (q95)
  - Lower bound: 5th percentile (q05)
- **Implementation**: `X[col] = X[col].clip(q05, q95)`
- **Reasoning**: Exploration experiments showed q95/q05 outperforms current q99/q01 by 1.4% PR-AUC

### 3. Feature Standardization
- **Method**: StandardScaler from sklearn
- **Target**: All 30 remaining numerical features
- **Purpose**: Required for KNN distance calculations (RF is scale-invariant)
- **Implementation**: Fit on training set, transform both train and test

## Feature Engineering Steps
- **Approach**: Minimal feature engineering
- **Reasoning**: Exploration showed complex approaches (polynomial features, interactions) provided marginal gains compared to ensemble benefits
- **Final feature set**: 30 original features after exclusion and preprocessing

## Model Selection and Architecture

### Ensemble Configuration
**Method**: Soft Voting Ensemble with two complementary algorithms

#### Component 1: Random Forest
- **Algorithm**: `RandomForestClassifier`
- **Parameters**: 
  - `n_estimators=100`
  - `random_state=42`
- **Input data**: Original features (unscaled)
- **Reasoning**: RF handles feature interactions internally and is robust to feature scales

#### Component 2: K-Nearest Neighbors  
- **Algorithm**: `KNeighborsClassifier`
- **Parameters**: 
  - `n_neighbors=11`
  - `weights='distance'`
  - `metric='manhattan'`
- **Input data**: Standardized features
- **Reasoning**: Distance-based algorithm requires scaled features; Manhattan distance performed well in Iteration 1

#### Ensemble Strategy
- **Voting method**: Soft (probability averaging)
- **Weights**: Equal (1:1 ratio)
- **Final prediction**: `(RF_proba + KNN_proba) / 2`

## Evaluation Strategy

### Primary Evaluation
- **Metric**: PR-AUC (Area under Precision-Recall curve)
- **Target**: ≥ 0.650 (minimum 9.6% improvement)
- **Cross-validation**: 5-fold StratifiedKFold with `random_state=42`

### Secondary Metrics
- ROC-AUC (discrimination ability)
- Precision, Recall, F1-Score
- Accuracy, Balanced Accuracy, Specificity

### Threshold Optimization
- **Recommended threshold**: 0.4
- **Rationale**: Maximizes F1-score (0.678) with balanced precision (0.556) and recall (0.868)
- **Business context**: Balances false alarm costs with failure detection

### Diagnostic Analyses
1. **Precision-Recall Curve**: Visualize threshold trade-offs
2. **ROC Curve**: Compare discrimination with baseline
3. **Calibration Curve**: Assess probability reliability 
4. **Confusion Matrix**: Analyze error patterns
5. **Probability Distribution**: Compare class prediction confidence
6. **Feature Importance**: Identify key predictive features from RF
7. **Ensemble Analysis**: Compare individual RF vs KNN contributions

## Implementation Guidelines

### Code Structure
```python
# 1. Load and preprocess data
X = train_df.drop(['target', 'id', 'f_27'], axis=1)
y = train_df['target']

# 2. Apply optimized outlier capping
for col in ['f_20', 'f_06', 'f_00']:
    q95 = X[col].quantile(0.95)
    q05 = X[col].quantile(0.05)
    X[col] = X[col].clip(q05, q95)

# 3. Scale features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Create ensemble
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan')

# 5. Train on appropriate data
rf.fit(X_train, y_train)  # Unscaled features
knn.fit(X_train_scaled, y_train)  # Scaled features

# 6. Ensemble prediction
rf_proba = rf.predict_proba(X_test)[:, 1]
knn_proba = knn.predict_proba(X_test_scaled)[:, 1]
ensemble_proba = (rf_proba + knn_proba) / 2
```

## Success Criteria
- **Primary**: PR-AUC ≥ 0.650 (9.6% minimum improvement over baseline)
- **Secondary**: F1-Score ≥ 0.650 with balanced precision/recall
- **Deployment**: Well-calibrated probabilities for business decision-making

## Risk Mitigation
- **Overfitting**: Use cross-validation and avoid excessive feature engineering
- **Scalability**: Ensemble adds minimal computational overhead
- **Interpretability**: Feature importance from RF component provides business insights
- **Robustness**: Ensemble reduces variance compared to single model approach

## Expected Deliverables
- Trained ensemble model with preprocessing pipeline
- Comprehensive evaluation plots and metrics
- Performance comparison with Iteration 1 baseline
- Business-ready probability threshold recommendations