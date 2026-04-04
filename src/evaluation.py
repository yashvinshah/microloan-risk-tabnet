"""
Evaluation metrics and utilities for model performance assessment.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve, 
    f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Comprehensive metrics evaluation for binary classification models.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation plots
        """
        self.output_dir = output_dir or Path("outputs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray = None,
        threshold: float = 0.5,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred: Predicted binary labels (computed from threshold if not provided)
            threshold: Classification threshold
            model_name: Name of the model for logging
        
        Returns:
            Dictionary of computed metrics
        """
        # Compute binary predictions if not provided
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Compute PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        # Compute classification metrics
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'Threshold': threshold
        }
        
        logger.info(f"\n{model_name} Evaluation Metrics:")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = "f1",
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'roc_auc', 'pr_auc')
            n_thresholds: Number of thresholds to evaluate
        
        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        thresholds = np.linspace(0, 1, n_thresholds)
        best_threshold = 0.5
        best_value = -np.inf
        metric_values = []
        
        if metric == 'f1':
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                metric_values.append(f1)
                if f1 > best_value:
                    best_value = f1
                    best_threshold = threshold
        
        elif metric == 'roc_auc':
            best_value = roc_auc_score(y_true, y_pred_proba)
            best_threshold = 0.5  # ROC-AUC doesn't depend on threshold
        
        elif metric == 'pr_auc':
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            best_value = auc(recall_vals, precision_vals)
            best_threshold = 0.5  # PR-AUC doesn't depend on threshold directly
        
        logger.info(f"Optimal threshold ({metric}): {best_threshold:.4f} with value {best_value:.4f}")
        return best_threshold, best_value
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> Dict[str, float]:
        """
        Plot ROC curve and compute ROC-AUC.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        
        Returns:
            Dictionary with ROC metrics
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
        
        return {'ROC_AUC': roc_auc, 'FPR': fpr, 'TPR': tpr}
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> Dict[str, float]:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        
        Returns:
            Dictionary with PR metrics
        """
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='darkgreen', lw=2, label=f'{model_name} (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        plt.close()
        
        return {'PR_AUC': pr_auc, 'Precision': precision_vals, 'Recall': recall_vals}
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray = None,
        threshold: float = 0.5,
        model_name: str = "Model"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred: Predicted binary labels
            threshold: Classification threshold
            model_name: Name of the model
        
        Returns:
            String containing the classification report
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        report = classification_report(y_true, y_pred, target_names=['No Default', 'Default'])
        
        logger.info(f"\n{model_name} Classification Report:\n{report}")
        return report


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Evaluate multiple models and create comparison DataFrame.
    
    Args:
        models: Dictionary of model names to model objects
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save results
    
    Returns:
        DataFrame with evaluation results for all models
    """
    evaluator = MetricsEvaluator(output_dir)
    results = []
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test).squeeze()
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Compute metrics
        metrics = evaluator.compute_metrics(y_test, y_pred_proba, y_pred, model_name=model_name)
        metrics['Model'] = model_name
        results.append(metrics)
        
        # Plot evaluation curves
        evaluator.plot_roc_curve(
            y_test, y_pred_proba, model_name,
            save_path=str(output_dir / f"roc_curve_{model_name}.png") if output_dir else None
        )
        
        evaluator.plot_precision_recall_curve(
            y_test, y_pred_proba, model_name,
            save_path=str(output_dir / f"pr_curve_{model_name}.png") if output_dir else None
        )
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name,
            save_path=str(output_dir / f"confusion_matrix_{model_name}.png") if output_dir else None
        )
    
    results_df = pd.DataFrame(results)
    logger.info(f"\nModel Comparison:\n{results_df.to_string()}")
    
    return results_df
