"""
Machine Failure Prediction Pipeline

Complete ML pipeline class that combines data preprocessing, feature engineering,
and model prediction for machine failure prediction.
"""

import pandas as pd
import numpy as np
from typing import Union

from machine_failure_prediction.pipeline.data_preprocessing import DataProcessor
from machine_failure_prediction.pipeline.feature_preprocessing import FeatureProcessor
from machine_failure_prediction.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete ML pipeline for machine failure prediction.
    
    This class combines data preprocessing, feature engineering, and model prediction
    into a single self-contained pipeline that can be saved as an MLflow model.
    """
    
    def __init__(self, data_processor: DataProcessor = None, 
                 feature_processor: FeatureProcessor = None, 
                 model: ModelWrapper = None):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: DataProcessor instance for data preprocessing
        feature_processor: FeatureProcessor instance for feature engineering
        model: ModelWrapper instance for model predictions
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
        self.feature_names_ = None
        
    def _preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the complete preprocessing pipeline to input data.
        
        Parameters:
        X: Raw input features
        
        Returns:
        Preprocessed features ready for model prediction
        """
        # Apply data preprocessing (outlier capping, column removal)
        X_processed = self.data_processor.transform(X)
        
        # Apply feature preprocessing (scaling)
        X_features = self.feature_processor.transform(X_processed)
        
        return X_features
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for input features.
        
        Parameters:
        X: Input features (DataFrame or array)
        
        Returns:
        Predicted class labels (0 = no failure, 1 = failure)
        """
        # Ensure input is DataFrame
        if isinstance(X, np.ndarray):
            if self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                raise ValueError("Cannot convert array to DataFrame without feature names")
        
        # Apply preprocessing pipeline
        X_processed = self._preprocess_input(X)
        
        # Generate predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input features.
        
        Parameters:
        X: Input features (DataFrame or array)
        
        Returns:
        Predicted class probabilities [prob_no_failure, prob_failure]
        """
        # Ensure input is DataFrame
        if isinstance(X, np.ndarray):
            if self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                raise ValueError("Cannot convert array to DataFrame without feature names")
        
        # Apply preprocessing pipeline
        X_processed = self._preprocess_input(X)
        
        # Generate probability predictions
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def set_feature_names(self, feature_names: list):
        """
        Set feature names for array input conversion.
        
        Parameters:
        feature_names: List of feature names
        """
        self.feature_names_ = feature_names
