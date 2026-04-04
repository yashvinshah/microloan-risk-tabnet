"""
Utility functions for the micro-loan default risk prediction project.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, log_name: str = "pipeline.log") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_name: Name of the log file
    
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / log_name
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(data_dir: Path, filename: str) -> pd.DataFrame:
    """
    Load CSV data from file.
    
    Args:
        data_dir: Directory containing the CSV file
        filename: Name of the CSV file
    
    Returns:
        Loaded DataFrame
    """
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading data from {filename}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded shape: {df.shape}")
    return df


def get_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics about missing values in the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values,
        'Data_Type': df.dtypes.values
    }).sort_values('Missing_Percent', ascending=False)
    
    return missing_stats[missing_stats['Missing_Count'] > 0]


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
    strategy: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using specified strategy.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_cols: List of categorical column names
        strategy: Encoding strategy ('label' or 'onehot')
    
    Returns:
        Tuple of (encoded X_train, encoded X_test, encoders dict)
    """
    encoders = {}
    
    if strategy == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le
            logger.info(f"Label encoded {col}")
    
    elif strategy == "onehot":
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        # Align columns
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        logger.info(f"One-hot encoded {len(categorical_cols)} features")
    
    return X_train, X_test, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (scaled X_train, scaled X_test, scaler object)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split to maintain class distribution.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"Train set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution in test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Target variable
    
    Returns:
        Dictionary mapping class labels to weights
    """
    class_counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(class_counts)
    
    class_weights = {}
    for class_label, count in class_counts.items():
        weight = n_samples / (n_classes * count)
        class_weights[class_label] = weight
    
    logger.info(f"Computed class weights: {class_weights}")
    return class_weights
