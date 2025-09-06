import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from machine_failure_prediction.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Create the model based on configuration"""
        if self.config.model_type == 'random_forest':
            return RandomForestClassifier(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores from the trained model.
        
        Returns:
        np.ndarray: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")

    def get_feature_importance_df(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame with feature names.
        
        Parameters:
        feature_names (list): List of feature names
        
        Returns:
        pd.DataFrame: DataFrame with features and their importance scores
        """
        importance = self.get_feature_importance()
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Save the model as an artifact

        Parameters:
        path (str): The file path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)