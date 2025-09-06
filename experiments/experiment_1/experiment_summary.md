# Experiment 1 Summary: Optimized KNN with Outlier Capping

## Executive Summary

The first iteration focused on implementing an optimized K-Nearest Neighbors classifier with targeted preprocessing improvements based on prior exploratory data analysis. The experiment achieved a **PR-AUC of 0.593** on the test set, falling short of the expected performance target of ~0.663 from exploration.

## Experiment Configuration

- **Model**: KNeighborsClassifier (n_neighbors=11, weights='distance', metric='manhattan')
- **Preprocessing**: 
  - Feature exclusion: id, f_27 (high cardinality)
  - Outlier capping: IQR method on f_20, f_06, f_00
  - Scaling: StandardScaler
- **Evaluation**: 5-fold stratified cross-validation
- **Test Set Size**: 80 samples (31 positive, 49 negative)

## Key Results

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **PR-AUC** | **0.593** | Primary evaluation metric |
| **ROC-AUC** | **0.751** | Area under ROC curve |
| **Accuracy** | **0.662** | Overall classification accuracy |
| **Precision** | **0.553** | Positive class precision |
| **Recall** | **0.677** | Positive class sensitivity |
| **F1-Score** | **0.609** | Harmonic mean of precision/recall |
| **Specificity** | **0.653** | True negative rate |

### Confusion Matrix
- True Positives: 21, True Negatives: 32
- False Positives: 17, False Negatives: 10

## Analysis: Planning vs. Results

**Expected vs. Actual Performance:**
- Expected PR-AUC: ~0.663 (from exploration)  
- Actual PR-AUC: 0.593
- **Gap**: -0.070 (10.5% underperformance)

This significant gap suggests potential issues with:
1. **Data drift** between exploration and final datasets
2. **Overfitting** during exploration phase 
3. **Implementation differences** in preprocessing steps
4. **Cross-validation variance** not captured in exploration

## Key Weaknesses Identified

### 1. **Underperforming Relative to Expectations**
- The model achieved 10.5% lower PR-AUC than anticipated from exploration
- This indicates potential instability in the KNN approach or data/preprocessing inconsistencies

### 2. **High False Positive Rate (34.7%)**
- 17 false positives out of 49 negative samples
- Critical for predictive maintenance where false alarms are costly
- Precision of only 55.3% means nearly half of positive predictions are incorrect

### 3. **Model Complexity vs. Performance Trade-off**
- Simple KNN achieved moderate performance but may have reached its ceiling
- The manhattan distance metric and distance weighting didn't provide the expected boost

### 4. **Limited Feature Engineering**
- Deliberately excluded feature interactions and transformations based on exploration
- This conservative approach may have missed opportunities for improvement

## Recommendations for Next Iteration

### Primary Recommendation: **Ensemble Approach with Feature Engineering**
1. **Implement the planned Voting Classifier** with Random Forest, Gradient Boosting, and KNN
   - Expected to provide more stable performance (~0.630 PR-AUC)
   - Better handling of different data patterns through diverse algorithms

### Secondary Improvements:
2. **Advanced Feature Engineering**
   - Revisit feature interactions despite exploration results
   - Consider polynomial features for continuous variables
   - Implement domain-specific feature combinations

3. **Hyperparameter Optimization**
   - Systematic grid/random search for KNN parameters
   - Optimize ensemble voting weights
   - Cross-validate preprocessing parameters

4. **Data Quality Analysis**
   - Investigate data drift between exploration and current datasets
   - Validate preprocessing implementation consistency
   - Analyze outlier capping effectiveness

## Context for Future Iterations

- **Model artifacts preserved**: All preprocessors and trained model saved for reproducibility
- **Baseline established**: PR-AUC 0.593 represents current KNN ceiling performance
- **Infrastructure validated**: MLflow integration and evaluation pipeline working correctly
- **Next target**: Achieve PR-AUC > 0.630 through ensemble methods

## Technical Details

- **Runtime**: ~1 minute training time
- **Model size**: 93KB (lightweight deployment)
- **Cross-validation strategy**: 5-fold stratified
- **Seed**: 42 (reproducible results)
- **Framework**: scikit-learn 1.7.1

The experiment successfully established a baseline and identified clear areas for improvement, setting the foundation for more sophisticated modeling approaches in the next iteration.