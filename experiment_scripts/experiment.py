from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import sklearn
from typing import Dict, Any, List

from machine_failure_prediction.pipeline.feature_preprocessing import FeatureProcessor
from machine_failure_prediction.pipeline.data_preprocessing import DataProcessor
from machine_failure_prediction.pipeline.model import ModelWrapper
from machine_failure_prediction.config import Config
from experiment_scripts.evaluation import ModelEvaluator
from machine_failure_prediction.model_pipeline import ModelPipeline

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path: str, test_dataset_path: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
        """
        Run the complete ML experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training dataset
        test_dataset_path: Path to test dataset  
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
        
        Returns:
        Dictionary with experiment results and MLflow model information
        """
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        print("🚀 Starting Machine Failure Prediction Experiment")
        
        # Create output directories - harness expects output subdirectory
        output_subdir = os.path.join(output_dir, "output")
        model_artifacts_dir = os.path.join(output_subdir, "model_artifacts")
        general_artifacts_dir = os.path.join(output_subdir, "general_artifacts")
        plots_dir = os.path.join(output_subdir, "plots")
        
        for dir_path in [model_artifacts_dir, general_artifacts_dir, plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. Load datasets
        print("📊 Loading datasets...")
        train_df = pd.read_csv(train_dataset_path)
        test_df = pd.read_csv(test_dataset_path)
        
        # Extract features and target
        target_col = self._config.data_prep.target_column
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # 2. Initialize and fit components
        print("🔧 Fitting preprocessing components...")
        
        # Data processor (outlier capping, column removal)
        data_processor = DataProcessor(self._config.data_prep)
        data_processor.fit(X_train)
        
        # Apply data preprocessing to get clean training data for feature processor
        X_train_processed = data_processor.transform(X_train)
        
        # Feature processor (scaling)
        feature_processor = FeatureProcessor(self._config.feature_prep)
        feature_processor.fit(X_train_processed)
        
        # Apply feature preprocessing to get final training features
        X_train_features = feature_processor.transform(X_train_processed)
        
        print(f"Final feature set: {X_train_features.shape[1]} features")
        
        # 3. Train model
        print("🤖 Training model...")
        model = ModelWrapper(self._config.model)
        model.fit(X_train_features, y_train)
        
        # 4. Evaluate model
        print("📈 Evaluating model...")
        evaluator = ModelEvaluator(self._config.model_evaluation)
        
        # Apply preprocessing to test data
        X_test_processed = data_processor.transform(X_test)
        X_test_features = feature_processor.transform(X_test_processed)
        
        # Evaluate model and generate plots
        metrics = evaluator.evaluate_model(model, X_test_features, y_test, plots_dir)
        
        print(f"Primary metric (PR-AUC): {metrics['pr_auc']:.4f}")
        print(f"Additional metrics - ROC-AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # 5. Save individual model artifacts
        print("💾 Saving model artifacts...")
        data_processor.save(os.path.join(model_artifacts_dir, "data_processor.pkl"))
        feature_processor.save(os.path.join(model_artifacts_dir, "feature_processor.pkl"))
        model.save(os.path.join(model_artifacts_dir, "trained_model.pkl"))
        
        # 6. Create and test ModelPipeline
        print("🔗 Creating MLflow-compatible pipeline...")
        pipeline = ModelPipeline(
            data_processor=data_processor,
            feature_processor=feature_processor,
            model=model
        )
        
        # Set feature names for array input compatibility
        original_feature_names = X_train.columns.tolist()
        pipeline.set_feature_names(original_feature_names)
        
        # Test pipeline end-to-end with sample data
        print("🧪 Testing pipeline...")
        sample_input = X_test.head(5)
        sample_predictions = pipeline.predict(sample_input)
        sample_probabilities = pipeline.predict_proba(sample_input)
        
        print(f"Pipeline test successful - predictions shape: {sample_predictions.shape}")
        print(f"Pipeline test successful - probabilities shape: {sample_probabilities.shape}")
        
        # 7. Save and log MLflow model
        print("📦 Saving MLflow model...")
        
        # Define paths
        mlflow_model_path = os.path.join(model_artifacts_dir, "mlflow_model")
        relative_model_path = "output/model_artifacts/mlflow_model/"
        
        # Always save the model locally for harness validation
        print(f"💾 Saving model to local disk for harness: {mlflow_model_path}")
        
        # Create model signature
        signature = mlflow.models.infer_signature(sample_input, pipeline.predict(sample_input))
        
        # Save model locally
        mlflow.sklearn.save_model(
            pipeline,
            path=mlflow_model_path,
            code_paths=["machine_failure_prediction"],  # Bundle custom code
            signature=signature
        )
        
        # Initialize logged_model_uri
        logged_model_uri = None
        
        # If MLflow run ID is provided, reconnect and log the model as an artifact
        active_run_id = "36640fb31dd54f06b56007fa0093e918"
        
        if active_run_id and active_run_id != 'None' and active_run_id.strip():
            print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
            try:
                with mlflow.start_run(run_id=active_run_id):
                    logged_model_info = mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path="model",  # Standard artifact path
                        code_paths=["machine_failure_prediction"],  # Bundle custom code
                        signature=signature
                    )
                    logged_model_uri = logged_model_info.model_uri
                print(f"📤 Model logged to MLflow with URI: {logged_model_uri}")
            except Exception as e:
                print(f"⚠️ Failed to log model to MLflow: {e}")
                logged_model_uri = None
        else:
            print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
        
        # 8. Prepare results
        model_artifacts = [
            "data_processor.pkl",
            "feature_processor.pkl", 
            "trained_model.pkl",
            "mlflow_model/"
        ]
        
        # Create input example for signature
        input_example = sample_input.to_dict('records')[0]
        
        mlflow_model_info = {
            "model_path": relative_model_path,
            "logged_model_uri": logged_model_uri,
            "model_type": "sklearn",
            "task_type": "classification", 
            "signature": signature.to_dict() if signature else None,
            "input_example": input_example,
            "framework_version": sklearn.__version__
        }
        
        print("✅ Experiment completed successfully!")
        
        return {
            "metric_name": "pr_auc",
            "metric_value": float(metrics['pr_auc']),
            "model_artifacts": model_artifacts,
            "mlflow_model_info": mlflow_model_info
        }