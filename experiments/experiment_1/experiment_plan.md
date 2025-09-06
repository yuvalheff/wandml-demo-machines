# Experiment Plan: Optimized KNN with Outlier Capping

## Experiment Overview
**Experiment Name**: Optimized KNN with Outlier Capping  
**Task Type**: Binary Classification  
**Target Variable**: `target` (0 = no failure, 1 = failure)  
**Primary Metric**: PR-AUC  
**Based on**: Comprehensive exploration experiments showing KNN as top performer

## Data Preprocessing Steps

### 1. Feature Selection
- **Remove columns**: `id`, `f_27`
- **Rationale**: 
  - `id` is just an identifier
  - `f_27` has 300 unique values (one per training sample), essentially acting as an ID

### 2. Outlier Handling
- **Method**: IQR-based capping
- **Target features**: `f_20`, `f_06`, `f_00` (identified as high-outlier features in EDA)
- **Implementation**:
  - Calculate Q1 (25th percentile) and Q3 (75th percentile) for each feature
  - Compute IQR = Q3 - Q1
  - Cap values at: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Expected impact**: Marginal improvement (~0.003 PR-AUC based on exploration)

### 3. Feature Scaling
- **Method**: StandardScaler
- **Apply to**: All numerical features after outlier capping
- **Rationale**: KNN is distance-based and requires scaled features for optimal performance
- **Critical requirement**: Fit scaler on training data only, transform both train and test

## Feature Engineering Steps

### What NOT to do (based on exploration results):
- **No feature interactions**: Exploration showed interactions decreased performance
- **No log transformations**: No benefit observed for skewed features
- **No polynomial features**: Would increase dimensionality without proven benefit

### Feature Set:
- Use all 30 numerical features after preprocessing
- Expected final feature count: 30 features

## Model Selection Strategy

### Primary Model: Optimized KNN
- **Algorithm**: KNeighborsClassifier
- **Hyperparameters**:
  - `n_neighbors=11` (found optimal in exploration)
  - `weights='distance'` (distance weighting outperformed uniform)
  - `metric='manhattan'` (outperformed euclidean in exploration)
- **Expected performance**: PR-AUC ≈ 0.6628

### Secondary Model: Ensemble Voting Classifier
- **Purpose**: Higher stability, competitive performance
- **Components**:
  1. Random Forest (n_estimators=100, random_state=42)
  2. Gradient Boosting (n_estimators=200, learning_rate=0.1, max_depth=5)
  3. KNN (same parameters as primary model)
- **Voting method**: Soft voting (uses predict_proba)
- **Expected performance**: PR-AUC ≈ 0.6301

## Comprehensive Evaluation Strategy

### Cross-Validation Setup
- **Method**: 5-fold Stratified Cross-Validation
- **Random state**: 42 for reproducibility
- **Primary metric**: PR-AUC
- **Secondary metrics**: Precision, Recall, F1-Score, ROC-AUC, Accuracy

### Diagnostic Analyses

#### 1. Feature Importance Analysis
- **Method**: Permutation importance for KNN model
- **Purpose**: Understand which features drive predictions despite KNN's black-box nature
- **Expected insights**: Validation of EDA findings (f_28, f_20, f_19 as top features)

#### 2. Model Calibration Analysis  
- **Methods**: 
  - Reliability diagram (calibration curve)
  - Brier score calculation
- **Purpose**: Assess if predicted probabilities reflect true likelihood of failure
- **Critical for**: Business decision-making based on failure probability

#### 3. Error Analysis
- **Components**:
  - False positive analysis (healthy machines predicted as failing)
  - False negative analysis (failing machines predicted as healthy)
  - Error distribution across feature ranges
- **Purpose**: Understand model weaknesses and potential business impact

#### 4. Threshold Optimization
- **Method**: Precision-Recall curve analysis
- **Purpose**: Find optimal classification threshold for business requirements
- **Considerations**: Balance between catching failures (recall) vs false alarms (precision)

#### 5. Prediction Confidence Analysis
- **Method**: Analyze prediction probability distributions
- **Segments**: High confidence (>0.8 or <0.2) vs uncertain (0.3-0.7) predictions
- **Purpose**: Identify cases where model is uncertain for human review

### Performance Slicing

#### 1. Feature Range Analysis
- **Slice by**: Different ranges of top predictive features (f_28, f_20, f_19)
- **Purpose**: Identify if model performs differently across feature value ranges
- **Implementation**: Divide each feature into quartiles, evaluate performance per quartile

#### 2. Prediction Confidence Slicing
- **Segments**:
  - High confidence positive (prob > 0.8)
  - High confidence negative (prob < 0.2) 
  - Uncertain predictions (0.2 ≤ prob ≤ 0.8)
- **Purpose**: Validate model reliability across confidence levels

## Implementation Pipeline

### Step-by-Step Process:
1. **Data Loading**: Load train_set.csv and test_set.csv
2. **Feature Removal**: Drop `id` and `f_27` columns
3. **Outlier Capping**: Apply IQR method to `f_20`, `f_06`, `f_00`
4. **Scaling**: Fit StandardScaler on training data, transform both sets
5. **Model Training**: Train KNN with optimized hyperparameters
6. **Cross-Validation**: 5-fold stratified CV evaluation
7. **Test Evaluation**: Final model evaluation on held-out test set
8. **Diagnostic Analysis**: Run all evaluation analyses
9. **Results Documentation**: Generate reports and visualizations

### Expected Outputs

#### Model Files:
- `knn_model_optimized.pkl`: Trained KNN model
- `ensemble_model.pkl`: Trained ensemble model
- `scaler_pipeline.pkl`: Fitted preprocessing pipeline

#### Evaluation Reports:
- `model_performance_report.json`: Comprehensive performance metrics
- `feature_importance_analysis.json`: Permutation importance results
- `calibration_analysis.json`: Calibration assessment results
- `error_analysis_report.json`: Error pattern analysis

#### Visualizations:
- `precision_recall_curve.png`: PR curve with optimal threshold
- `calibration_plot.png`: Model calibration assessment
- `feature_importance_plot.png`: Top feature importance visualization  
- `error_distribution_plot.png`: Error analysis visualization

## Validation Requirements

### Technical Validation:
- Confirm preprocessing steps match exploration results
- Verify no data leakage (scaler fitted on training data only)
- Ensure hyperparameters match exploration optimal values
- Validate evaluation metrics align with exploration benchmarks

### Business Validation:
- PR-AUC performance should be ≥ 0.65 (above exploration benchmark)
- Model should show stable performance across CV folds (std < 0.1)
- Error analysis should identify actionable patterns for maintenance teams
- Calibration should show reasonable probability estimates for business use

## Success Criteria
- **Primary**: Achieve PR-AUC ≥ 0.65 on test set
- **Secondary**: Ensemble model provides competitive performance with higher stability  
- **Diagnostic**: Comprehensive evaluation reveals clear model strengths/weaknesses
- **Business**: Model probabilities are well-calibrated for maintenance decision-making