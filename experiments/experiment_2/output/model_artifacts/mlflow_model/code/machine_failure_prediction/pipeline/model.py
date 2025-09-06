import pandas as pd
import numpy as np
import pickle
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

from machine_failure_prediction.config import ModelConfig


class RFKNNEnsemble:
    """
    Custom RF+KNN ensemble where RF uses unscaled features and KNN uses scaled features
    """
    def __init__(self, rf_params, knn_params, weights=[1, 1]):
        self.rf = RandomForestClassifier(**rf_params)
        self.knn = KNeighborsClassifier(**knn_params)
        self.scaler = StandardScaler()
        self.weights = weights
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit both RF on unscaled data and KNN on scaled data"""
        # Fit RF on unscaled features
        self.rf.fit(X, y)
        
        # Fit scaler and KNN on scaled features
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using soft voting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get RF probabilities on unscaled data
        rf_probs = self.rf.predict_proba(X)
        
        # Get KNN probabilities on scaled data
        X_scaled = self.scaler.transform(X)
        knn_probs = self.knn.predict_proba(X_scaled)
        
        # Soft voting with weights
        weighted_probs = (self.weights[0] * rf_probs + self.weights[1] * knn_probs) / sum(self.weights)
        return weighted_probs
    
    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Create the model based on configuration"""
        if self.config.model_type == 'knn':
            return KNeighborsClassifier(**self.config.model_params)
        elif self.config.model_type == 'rf_knn_ensemble':
            # Create custom RF+KNN ensemble with dual data handling
            return RFKNNEnsemble(
                rf_params=self.config.model_params.get('rf_params', {}),
                knn_params=self.config.model_params.get('knn_params', {}),
                weights=self.config.model_params.get('weights', [1, 1])
            )
        elif self.config.model_type == 'voting_classifier':
            # Create ensemble components
            knn = KNeighborsClassifier(**self.config.model_params.get('knn_params', {}))
            rf = RandomForestClassifier(**self.config.model_params.get('rf_params', {}))
            gb = GradientBoostingClassifier(**self.config.model_params.get('gb_params', {}))
            
            estimators = [
                ('knn', knn),
                ('rf', rf),
                ('gb', gb)
            ]
            
            return VotingClassifier(
                estimators=estimators,
                voting=self.config.model_params.get('voting', 'soft')
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

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