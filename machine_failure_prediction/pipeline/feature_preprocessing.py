from typing import Optional
import pandas as pd
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from machine_failure_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Store feature names for consistency
        self.feature_names_ = X.columns.tolist()
        
        # Fit the scaler on all numerical features
        if self.config.apply_scaling:
            self.scaler.fit(X)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Make a copy to avoid modifying original data
        X_transformed = X.copy()
        
        # Apply scaling if configured
        if self.config.apply_scaling:
            # Scale the features
            X_scaled = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_scaled,
                columns=X_transformed.columns,
                index=X_transformed.index
            )
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
