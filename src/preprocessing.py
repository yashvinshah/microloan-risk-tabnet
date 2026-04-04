"""
Preprocessing module for handling data loading, cleaning, and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from pathlib import Path
from sklearn.impute import SimpleImputer
from src.utils import (
    load_data, get_missing_stats, encode_categorical_features,
    scale_features, stratified_split, compute_class_weights
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing operations for the micro-loan default risk prediction.
    """
    
    def __init__(self, data_dir: Path, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the raw data files
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.categorical_features = []
        self.numeric_features = []
        
    def load_and_prepare_application_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare the main application dataset (with target variable).
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Loading application training data...")
        df = load_data(self.data_dir, "application_train.csv")
        
        # Separate target and features
        y = df['TARGET']
        X = df.drop('TARGET', axis=1)
        
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Class imbalance ratio: 1:{(y==0).sum() / (y==1).sum():.2f}")
        
        return X, y
    
    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy_numeric: str = "median",
        strategy_categorical: str = "most_frequent"
    ) -> pd.DataFrame:
        """
        Handle missing values via imputation.
        
        Args:
            X: Feature DataFrame
            strategy_numeric: Imputation strategy for numeric columns
            strategy_categorical: Imputation strategy for categorical columns
        
        Returns:
            DataFrame with imputed values
        """
        logger.info("Handling missing values...")
        
        # Get missing stats
        missing_stats = get_missing_stats(X)
        if len(missing_stats) > 0:
            logger.info(f"\nMissing value statistics:\n{missing_stats}")
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        
        X_imputed = X.copy()
        
        # Impute numeric columns
        if numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy=strategy_numeric)
            X_imputed[numeric_cols] = self.numeric_imputer.fit_transform(X[numeric_cols])
            logger.info(f"Imputed {len(numeric_cols)} numeric columns with '{strategy_numeric}'")
        
        # Impute categorical columns
        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy=strategy_categorical)
            X_imputed[categorical_cols] = self.categorical_imputer.fit_transform(X[categorical_cols])
            logger.info(f"Imputed {len(categorical_cols)} categorical columns with '{strategy_categorical}'")
        
        return X_imputed
    
    def encode_and_scale(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        encoding_strategy: str = "label",
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode categorical features and scale all features.
        
        Args:
            X_train: Training features
            X_test: Test features
            encoding_strategy: Strategy for encoding categorical features
            scale: Whether to scale features
        
        Returns:
            Tuple of (processed X_train, processed X_test)
        """
        logger.info("Encoding categorical features...")
        
        # Encode categorical features
        if self.categorical_features:
            X_train, X_test, self.encoders = encode_categorical_features(
                X_train.copy(),
                X_test.copy(),
                self.categorical_features,
                strategy=encoding_strategy
            )
        
        # Store feature names before scaling
        self.feature_names = X_train.columns.tolist()
        
        # Scale features if requested
        if scale:
            logger.info("Scaling features...")
            X_train_processed, X_test_processed, self.scaler = scale_features(X_train, X_test)
        else:
            X_train_processed = X_train.values
            X_test_processed = X_test.values
        
        return X_train_processed, X_test_processed
    
    def preprocess_pipeline(
        self,
        handle_missing: bool = True,
        encode_categorical: bool = True,
        scale: bool = True,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            scale: Whether to scale features
            test_size: Proportion for train-test split
        
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load main data
        X, y = self.load_and_prepare_application_data()
        
        # Handle missing values
        if handle_missing:
            X = self.handle_missing_values(X)
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_size, random_state=self.random_state)
        
        # Encode and scale
        X_train_processed, X_test_processed = self.encode_and_scale(
            X_train, X_test,
            encoding_strategy="label",
            scale=scale
        )
        
        # Compute class weights
        class_weights = compute_class_weights(y_train)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'class_weights': class_weights,
            'scaler': self.scaler,
            'encoders': self.encoders,
        }


def preprocess_data(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete preprocessing pipeline.
    
    Args:
        data_dir: Directory containing raw data
        test_size: Proportion for train-test split
        random_state: Random seed
    
    Returns:
        Dictionary with preprocessed data and metadata
    """
    preprocessor = DataPreprocessor(data_dir, random_state=random_state)
    return preprocessor.preprocess_pipeline(test_size=test_size)
