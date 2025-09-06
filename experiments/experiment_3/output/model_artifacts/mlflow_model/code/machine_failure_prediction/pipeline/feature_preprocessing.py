from typing import Optional
import pandas as pd
import pickle
import os
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from machine_failure_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.top_predictive_features = ['f_20', 'f_14', 'f_28']  # Top predictive features from EDA

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Create interaction features first to get final feature set
        X_with_interactions = self._create_interaction_features(X)
        
        # Store feature names for consistency
        self.feature_names_ = X_with_interactions.columns.tolist()
        
        # Fit the scaler on all features if scaling is enabled
        if self.config.apply_scaling:
            self.scaler.fit(X_with_interactions)
        
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
        
        # Create interaction features
        X_transformed = self._create_interaction_features(X_transformed)
        
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
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create 12 interaction features from top predictive features (f_20, f_14, f_28):
        - 3 multiplicative interactions
        - 3 ratio features  
        - 6 polynomial features (squared and cubed for each)
        
        Parameters:
        X (pd.DataFrame): Input features
        
        Returns:
        pd.DataFrame: Features with interactions added
        """
        X_interactions = X.copy()
        
        # Verify top predictive features exist
        available_features = [f for f in self.top_predictive_features if f in X.columns]
        if len(available_features) < 3:
            print(f"Warning: Only {len(available_features)} of {len(self.top_predictive_features)} top features found")
        
        # 1. Multiplicative interactions (3 features)
        if 'f_20' in X.columns and 'f_14' in X.columns:
            X_interactions['f_20_x_f_14'] = X['f_20'] * X['f_14']
        if 'f_20' in X.columns and 'f_28' in X.columns:
            X_interactions['f_20_x_f_28'] = X['f_20'] * X['f_28']
        if 'f_14' in X.columns and 'f_28' in X.columns:
            X_interactions['f_14_x_f_28'] = X['f_14'] * X['f_28']
        
        # 2. Ratio features with epsilon for division safety (3 features)
        epsilon = 1e-8
        if 'f_20' in X.columns and 'f_14' in X.columns:
            X_interactions['f_20_div_f_14'] = X['f_20'] / (X['f_14'] + epsilon)
        if 'f_20' in X.columns and 'f_28' in X.columns:
            X_interactions['f_20_div_f_28'] = X['f_20'] / (X['f_28'] + epsilon)
        if 'f_14' in X.columns and 'f_28' in X.columns:
            X_interactions['f_14_div_f_28'] = X['f_14'] / (X['f_28'] + epsilon)
        
        # 3. Polynomial features - squared and cubed terms (6 features)
        for feature in self.top_predictive_features:
            if feature in X.columns:
                X_interactions[f'{feature}_squared'] = X[feature] ** 2
                X_interactions[f'{feature}_cubed'] = X[feature] ** 3
        
        # Validate for infinite/NaN values and apply clipping if needed
        X_interactions = self._validate_and_clip_features(X_interactions)
        
        return X_interactions
    
    def _validate_and_clip_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate interaction features for infinite/NaN values and apply clipping
        
        Parameters:
        X (pd.DataFrame): Features to validate
        
        Returns:
        pd.DataFrame: Validated and clipped features
        """
        X_validated = X.copy()
        
        # Check for infinite values
        inf_mask = np.isinf(X_validated.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            print("Warning: Infinite values detected in interaction features. Clipping to large finite values.")
            # Replace inf with large finite values
            X_validated = X_validated.replace([np.inf, -np.inf], [1e10, -1e10])
        
        # Check for NaN values
        nan_mask = X_validated.isnull()
        if nan_mask.any().any():
            print("Warning: NaN values detected in interaction features. Filling with 0.")
            X_validated = X_validated.fillna(0)
        
        return X_validated

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
