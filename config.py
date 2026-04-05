"""
Configuration file for micro-loan default risk prediction project.
"""

import os
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"

# Generate timestamped run directory
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_ROOT / "outputs" / f"run_{RUN_TIMESTAMP}"

# Create run-specific directories
OUTPUT_DIR = RUN_DIR
MODELS_DIR = RUN_DIR / "models"
LOGS_DIR = RUN_DIR / "logs"
EVALUATION_DIR = RUN_DIR / "evaluation"
HPO_REPORTS_DIR = RUN_DIR / "hpo_reports"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
HPO_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "previous_application": "previous_application.csv",
}

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "n_jobs": -1,
    "test_size": 0.2,
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "tabnet": {
        "n_d": 64,  # Width of decision-making layer
        "n_a": 64,  # Width of feature attention layer
        "n_steps": 3,  # Number of decision steps
        "gamma": 1.5,  # Relaxation parameter for feature reuse
        "lambda_sparse": 1e-4,  # Sparsity regularization strength
        "optimizer_params": {
            "lr": 2e-2,
        },
        "scheduler_params": {
            "step_size": 20,
            "gamma": 0.5,
        },
        "mask_type": "sparsemax",
    },
}

# Export new directory variables for easy access
__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'OUTPUT_DIR', 'MODELS_DIR', 'LOGS_DIR',
    'EVALUATION_DIR', 'HPO_REPORTS_DIR', 'RUN_TIMESTAMP', 'RUN_DIR',
    'DATA_CONFIG', 'MODEL_CONFIG', 'FOCAL_LOSS_CONFIG', 'OPTUNA_CONFIG', 'TRAINING_CONFIG', 'EVALUATION_CONFIG'
]

# Focal Loss configuration
FOCAL_LOSS_CONFIG = {
    "alpha": 0.25,  # Weight for positive class
    "gamma": 2.0,  # Focusing parameter
}

# Optuna HPO configuration
OPTUNA_CONFIG = {
    "name": "microloan_tabnet_hpo",
    "n_trials": 12,
    "timeout": 3600,  # 1 hour timeout
    "n_jobs": 1,
    "sampler": "tpe",  # Tree-structured Parzen Estimator
    "pruner": "median",  # Median stopping rule
    "direction": "maximize",  # Maximize ROC-AUC
    "objective_metric": "val_roc_auc",
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 60,
    "batch_size": 256,
    "validation_split": 0.2,
    "early_stopping_patience": 7,
    "early_stopping_metric": "val_roc_auc",
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "threshold_search_steps": 100,
}
