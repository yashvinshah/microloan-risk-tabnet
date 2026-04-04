"""
Optuna-based hyperparameter optimization for TabNet model.
"""

import logging
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import joblib
from src.models import TabNetModel
from src.loss_functions import FocalLoss

logger = logging.getLogger(__name__)


class TabNetOptimizer:
    """
    Optuna-based hyperparameter optimizer for TabNet.
    """
    
    def __init__(
        self,
        n_features: int,
        study_name: str = "tabnet_optimization",
        direction: str = "maximize",
        n_jobs: int = 1,
        sampler_name: str = "tpe",
        pruner_name: str = "median",
    ):
        """
        Initialize the optimizer.
        
        Args:
            n_features: Number of input features
            study_name: Name for the Optuna study
            direction: Optimization direction ('maximize' or 'minimize')
            n_jobs: Number of parallel jobs
            sampler_name: Sampler type ('tpe', 'random', 'grid')
            pruner_name: Pruner type ('median', 'successive_halving')
        """
        self.n_features = n_features
        self.study_name = study_name
        self.direction = direction
        self.n_jobs = n_jobs
        
        # Initialize sampler
        if sampler_name == "tpe":
            sampler = TPESampler(seed=42, n_startup_trials=10)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # Initialize pruner
        if pruner_name == "median":
            pruner = MedianPruner(n_warmup_steps=5)
        elif pruner_name == "successive_halving":
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_name}")
        
        # Create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        logger.info(f"Initialized Optuna study: {study_name}")
    
    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int = 3,
        metric: str = "roc_auc",
        **kwargs
    ) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            cv_folds: Number of cross-validation folds
            metric: Metric to optimize ('roc_auc', 'f1', 'pr_auc')
            **kwargs: Additional arguments
        
        Returns:
            Metric value to optimize
        """
        # Suggest hyperparameters
        params = {
            'n_d': trial.suggest_int('n_d', 32, 128, step=16),
            'n_a': trial.suggest_int('n_a', 32, 128, step=16),
            'n_steps': trial.suggest_int('n_steps', 2, 5),
            'gamma': trial.suggest_float('gamma', 1.0, 2.5, step=0.1),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_int('batch_size', 128, 512, step=64),
        }
        
        # Focal Loss parameters
        focal_params = {
            'alpha': trial.suggest_float('focal_alpha', 0.1, 0.9, step=0.1),
            'gamma': trial.suggest_float('focal_gamma', 1.0, 5.0, step=0.5),
        }
        
        logger.info(f"\nTrial {trial.number}: Testing parameters:")
        logger.info(f"  TabNet: {params}")
        logger.info(f"  Focal Loss: {focal_params}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            try:
                # Initialize and train model
                model = TabNetModel(
                    n_features=self.n_features,
                    n_d=params['n_d'],
                    n_a=params['n_a'],
                    n_steps=params['n_steps'],
                    gamma=params['gamma'],
                    lambda_sparse=params['lambda_sparse'],
                )
                
                model.train(
                    X_fold_train, y_fold_train,
                    X_val=X_fold_val, y_val=y_fold_val,
                    epochs=kwargs.get('epochs', 50),
                    batch_size=params['batch_size'],
                    early_stopping_patience=kwargs.get('early_stopping_patience', 10),
                )
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                
                if metric == 'roc_auc':
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                elif metric == 'f1':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_fold_val, y_pred, zero_division=0)
                elif metric == 'pr_auc':
                    precision_vals, recall_vals, _ = precision_recall_curve(y_fold_val, y_pred_proba)
                    score = auc(recall_vals, precision_vals)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                cv_scores.append(score)
                logger.info(f"  Fold {fold}: {metric}={score:.4f}")
                
                # Prune if score is too low
                trial.report(score, step=fold)
                if trial.should_prune():
                    logger.info(f"  Trial {trial.number} pruned")
                    raise optuna.TrialPruned()
            
            except Exception as e:
                logger.warning(f"Error in fold {fold}: {str(e)}")
                return 0.0
        
        mean_score = np.mean(cv_scores)
        logger.info(f"Trial {trial.number} mean {metric}: {mean_score:.4f}")
        return mean_score
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        metric: str = "roc_auc",
        cv_folds: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of trials
            timeout: Timeout in seconds
            metric: Metric to optimize
            cv_folds: Number of cross-validation folds
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with best parameters and trial information
        """
        logger.info(f"Starting optimization for {n_trials} trials...")
        logger.info(f"Optimization metric: {metric}")
        
        # If no validation set provided, use stratified split
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
        
        # Run optimization
        self.study.optimize(
            lambda trial: self.objective(
                trial, X_train, y_train, X_val, y_val,
                cv_folds=cv_folds, metric=metric, **kwargs
            ),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_trial = self.study.best_trial
        best_params = best_trial.params
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best {metric}: {best_trial.value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        return {
            'best_params': best_params,
            'best_value': best_trial.value,
            'best_trial': best_trial,
            'study': self.study,
            'n_trials': len(self.study.trials),
        }
    
    def report(self, output_dir: Path = None) -> pd.DataFrame:
        """
        Generate optimization report.
        
        Args:
            output_dir: Directory to save report
        
        Returns:
            DataFrame with trial information
        """
        trials_df = self.study.trials_dataframe()
        
        # Sort by value
        trials_df = trials_df.sort_values('value', ascending=(self.direction == "minimize"))
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "optimization_report.csv"
            trials_df.to_csv(filepath, index=False)
            logger.info(f"Optimization report saved to {filepath}")
        
        return trials_df
    
    def save(self, filepath: Path):
        """Save study to disk."""
        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to {filepath}")
    
    def load(self, filepath: Path):
        """Load study from disk."""
        self.study = joblib.load(filepath)
        logger.info(f"Study loaded from {filepath}")


def run_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_trials: int = 30,
    n_features: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of trials
        n_features: Number of features (inferred if not provided)
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with optimization results
    """
    if n_features is None:
        n_features = X_train.shape[1]
    
    optimizer = TabNetOptimizer(n_features=n_features)
    results = optimizer.optimize(
        X_train, y_train, X_val, y_val,
        n_trials=n_trials,
        **kwargs
    )
    
    return results
