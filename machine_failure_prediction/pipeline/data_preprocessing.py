from typing import Optional
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin

from machine_failure_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.outlier_bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        # Remove columns first to get clean data for outlier bound calculation
        X_clean = self._remove_columns(X)
        
        # Calculate outlier bounds for specified features
        for feature in self.config.outlier_features:
            if feature in X_clean.columns:
                if self.config.outlier_method == "quantile":
                    # Use q05/q95 quantile-based capping as per experiment plan
                    lower_bound = X_clean[feature].quantile(0.05)
                    upper_bound = X_clean[feature].quantile(0.95)
                else:
                    # Use IQR method (default/fallback)
                    Q1 = X_clean[feature].quantile(0.25)
                    Q3 = X_clean[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                self.outlier_bounds_[feature] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Make a copy to avoid modifying original data
        X_transformed = X.copy()
        
        # Remove specified columns
        X_transformed = self._remove_columns(X_transformed)
        
        # Apply outlier capping
        X_transformed = self._apply_outlier_capping(X_transformed)
        
        return X_transformed

    def _remove_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove specified columns from DataFrame"""
        cols_to_remove = [col for col in self.config.columns_to_remove if col in X.columns]
        if cols_to_remove:
            return X.drop(columns=cols_to_remove)
        return X

    def _apply_outlier_capping(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply IQR-based outlier capping to specified features"""
        X_capped = X.copy()
        
        for feature, bounds in self.outlier_bounds_.items():
            if feature in X_capped.columns:
                X_capped[feature] = np.clip(
                    X_capped[feature],
                    bounds['lower'],
                    bounds['upper']
                )
        
        return X_capped

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
