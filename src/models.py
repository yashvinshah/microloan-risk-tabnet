"""
Model implementations: baseline RandomForest and TabNet deep learning model.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional, Callable
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib

logger = logging.getLogger(__name__)


class BaselineRandomForest:
    """
    Baseline Random Forest model for microloan default risk prediction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest classifier.
        
        Args:
            **kwargs: Arguments to pass to RandomForestClassifier
        """
        self.model = RandomForestClassifier(**kwargs)
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional arguments (unused, for API consistency)
        """
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Random Forest training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted binary labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.
        
        Returns:
            Array of feature importances
        """
        return self.model.feature_importances_
    
    def save(self, filepath: Path):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class AUCCallback(Callback):
    """
    Custom callback to track validation ROC-AUC during TabNet training.
    """
    
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Initialize AUC callback.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        self.X_val = X_val
        self.y_val = y_val
        self.auc_list = []
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Called at end of each epoch."""
        logs = logs or {}
        # This is called after each epoch in TabNet
        return


class TabNetModel:
    """
    TabNet deep learning model for tabular data with custom loss function support.
    """
    
    def __init__(
        self,
        n_features: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        mask_type: str = "sparsemax",
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize TabNet model.
        
        Args:
            n_features: Number of input features
            n_d: Width of decision-making layer
            n_a: Width of feature attention layer
            n_steps: Number of decision steps
            gamma: Relaxation parameter
            lambda_sparse: Sparsity regularization strength
            mask_type: Type of feature mask ('softmax' or 'sparsemax')
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        self.n_features = n_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        
        logger.info(f"Initializing TabNet on device: {self.device}")
        
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            mask_type=mask_type,
            device_name='cuda' if self.device.type == 'cuda' else 'cpu',
            verbose=0,
            **kwargs
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 2e-2,
        early_stopping_patience: int = 15,
        **kwargs
    ):
        """
        Train the TabNet model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            **kwargs: Additional arguments
        """
        logger.info("Training TabNet model...")
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train the model using pytorch-tabnet's fit method
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['auc'],
            max_epochs=epochs,
            batch_size=batch_size,
            patience=early_stopping_patience,
            virtual_batch_size=128,
        )
        
        self.is_trained = True
        logger.info("TabNet training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted binary labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probability predictions [batch_size, 2]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        predictions = self.model.predict_proba(X)
        return predictions
    
    def get_feature_importance(self, normalize: bool = True) -> np.ndarray:
        """
        Get feature importances from TabNet.
        
        Args:
            normalize: Whether to normalize importances
        
        Returns:
            Array of feature importances
        """
        importance = self.model.feature_importances_
        if normalize:
            importance = importance / importance.sum()
        return importance
    
    def get_mask_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get feature mask values (interpretability).
        
        Args:
            X: Input features
        
        Returns:
            Feature mask values
        """
        return self.model.explain(X)[1]
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save_model(str(filepath))
        logger.info(f"TabNet model saved to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        self.model.load_model(str(filepath))
        self.is_trained = True
        logger.info(f"TabNet model loaded from {filepath}")


class EnsembleModel:
    """
    Ensemble combining RandomForest and TabNet predictions.
    """
    
    def __init__(
        self,
        rf_model: BaselineRandomForest,
        tabnet_model: TabNetModel,
        rf_weight: float = 0.3,
        tabnet_weight: float = 0.7
    ):
        """
        Initialize ensemble model.
        
        Args:
            rf_model: Trained RandomForest model
            tabnet_model: Trained TabNet model
            rf_weight: Weight for RandomForest predictions
            tabnet_weight: Weight for TabNet predictions
        """
        self.rf_model = rf_model
        self.tabnet_model = tabnet_model
        
        # Normalize weights
        total_weight = rf_weight + tabnet_weight
        self.rf_weight = rf_weight / total_weight
        self.tabnet_weight = tabnet_weight / total_weight
        
        logger.info(f"Ensemble weights - RF: {self.rf_weight:.3f}, TabNet: {self.tabnet_weight:.3f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble predictions.
        
        Args:
            X: Features
        
        Returns:
            Ensemble probability predictions
        """
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        tabnet_proba = self.tabnet_model.predict_proba(X)[:, 1]
        
        ensemble_proba = self.rf_weight * rf_proba + self.tabnet_weight * tabnet_proba
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble predictions (binary).
        
        Args:
            X: Features
            threshold: Classification threshold
        
        Returns:
            Ensemble binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
