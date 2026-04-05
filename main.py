"""
Main orchestration script for micro-loan default risk prediction pipeline.

This script coordinates the entire ML pipeline:
1. Data preprocessing
2. Baseline model training
3. TabNet model training and optimization
4. Comprehensive evaluation
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import project modules
from config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR, EVALUATION_DIR, HPO_REPORTS_DIR,
    RUN_TIMESTAMP, RUN_DIR,
    DATA_CONFIG, MODEL_CONFIG, FOCAL_LOSS_CONFIG, OPTUNA_CONFIG, TRAINING_CONFIG, EVALUATION_CONFIG
)
from src.utils import setup_logging, compute_class_weights
from src.preprocessing import preprocess_data
from src.models import BaselineRandomForest, TabNetModel, EnsembleModel
from src.evaluation import MetricsEvaluator, evaluate_models
from src.hpo import TabNetOptimizer

logger = logging.getLogger(__name__)


class MicroLoanPipeline:
    """
    Complete ML pipeline for micro-loan default risk prediction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'data_dir': DATA_DIR,
            'output_dir': OUTPUT_DIR,
            'models_dir': MODELS_DIR,
            'logs_dir': LOGS_DIR,
            'evaluation_dir': EVALUATION_DIR,
            'hpo_reports_dir': HPO_REPORTS_DIR,
            'random_state': 42,
        }
        
        # Setup logging
        self.logger = setup_logging(self.config['logs_dir'])
        logger.info(f"MicroLoan Pipeline initialized. Run ID: {RUN_TIMESTAMP}")
        logger.info(f"Output directory: {self.config['output_dir']}")
        
        # Placeholders for models and data
        self.rf_model = None
        self.tabnet_model = None
        self.ensemble_model = None
        self.data = None
        self.evaluator = None
        self.optimal_thresholds = {}
    
    def stage_1_preprocessing(self) -> Dict[str, Any]:
        """
        Stage 1: Data Preprocessing
        
        Returns:
            Preprocessed data dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: DATA PREPROCESSING")
        logger.info("="*80)
        
        self.data = preprocess_data(
            self.config['data_dir'],
            test_size=0.2,
            random_state=self.config['random_state']
        )
        
        logger.info(f"\nData preprocessing summary:")
        logger.info(f"  X_train shape: {self.data['X_train'].shape}")
        logger.info(f"  X_test shape: {self.data['X_test'].shape}")
        logger.info(f"  y_train distribution: {np.bincount(self.data['y_train'])}")
        logger.info(f"  y_test distribution: {np.bincount(self.data['y_test'])}")
        
        return self.data
    
    def stage_2_baseline_model(self) -> Dict[str, Any]:
        """
        Stage 2: Baseline RandomForest Model
        
        Returns:
            Baseline model evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: BASELINE RANDOM FOREST MODEL")
        logger.info("="*80)
        
        if self.data is None:
            raise ValueError("Must run preprocessing first!")
        
        # Train RandomForest baseline
        logger.info("\nTraining Random Forest baseline model...")
        self.rf_model = BaselineRandomForest(**MODEL_CONFIG['random_forest'])
        self.rf_model.train(self.data['X_train'], self.data['y_train'])
        
        # Evaluate
        logger.info("\nEvaluating Random Forest model...")
        self.evaluator = MetricsEvaluator(self.config['evaluation_dir'])
        
        y_pred_proba = self.rf_model.predict_proba(self.data['X_test'])[:, 1]
        y_pred = self.rf_model.predict(self.data['X_test'])
        
        rf_metrics = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba, y_pred,
            model_name="Random Forest"
        )
        
        # Plot curves
        self.evaluator.plot_roc_curve(
            self.data['y_test'], y_pred_proba, "Random Forest",
            save_path=str(self.config['evaluation_dir'] / "roc_rf.png")
        )
        
        self.evaluator.plot_precision_recall_curve(
            self.data['y_test'], y_pred_proba, "Random Forest",
            save_path=str(self.config['evaluation_dir'] / "pr_rf.png")
        )
        
        self.evaluator.plot_confusion_matrix(
            self.data['y_test'], y_pred, "Random Forest",
            save_path=str(self.config['evaluation_dir'] / "cm_rf.png")
        )
        
        # Save model
        rf_path = self.config['models_dir'] / "random_forest_baseline.pkl"
        self.rf_model.save(rf_path)
        
        return rf_metrics
    
    def stage_3_hpo_optimization(self) -> Dict[str, Any]:
        """
        Stage 3: Hyperparameter Optimization using Optuna
        
        Returns:
            HPO results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: HYPERPARAMETER OPTIMIZATION (OPTUNA)")
        logger.info("="*80)
        
        if self.data is None:
            raise ValueError("Must run preprocessing first!")
        
        logger.info("\nInitializing Optuna optimizer...")
        optimizer = TabNetOptimizer(
            n_features=self.data['X_train'].shape[1],
            study_name="tabnet_microloan_hpo"
        )
        
        # Run optimization
        hpo_results = optimizer.optimize(
            X_train=self.data['X_train'],
            y_train=self.data['y_train'],
            X_val=None,  # Will use internal split
            y_val=None,
            n_trials=OPTUNA_CONFIG['n_trials'],
            timeout=OPTUNA_CONFIG['timeout'],
            metric='roc_auc',
            epochs=TRAINING_CONFIG['epochs'],
            early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
        )
        
        # Save HPO report
        hpo_report_df = optimizer.report(self.config['hpo_reports_dir'])
        
        logger.info(f"\nTop 5 trials:")
        logger.info(hpo_report_df.head(5).to_string())
        
        return hpo_results
    
    def stage_4_tabnet_training(self, best_params: Dict[str, Any] = None) -> None:
        """
        Stage 4: Train TabNet with best parameters
        
        Args:
            best_params: Best hyperparameters from HPO (if None, uses defaults)
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: TABNET MODEL TRAINING")
        logger.info("="*80)
        
        if self.data is None:
            raise ValueError("Must run preprocessing first!")
        
        # Use best params or defaults
        if best_params is None:
            best_params = MODEL_CONFIG['tabnet']
        
        logger.info(f"\nTraining TabNet with parameters:")
        logger.info(f"  {best_params}")
        
        # Initialize TabNet
        self.tabnet_model = TabNetModel(
            n_features=self.data['X_train'].shape[1],
            n_d=best_params.get('n_d', 64),
            n_a=best_params.get('n_a', 64),
            n_steps=best_params.get('n_steps', 3),
            gamma=best_params.get('gamma', 1.5),
            lambda_sparse=best_params.get('lambda_sparse', 1e-4),
        )
        
        # Train TabNet
        self.tabnet_model.train(
            X_train=self.data['X_train'],
            y_train=self.data['y_train'],
            X_val=self.data['X_test'][:len(self.data['X_test'])//5],
            y_val=self.data['y_test'][:len(self.data['y_test'])//5],
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=best_params.get('batch_size', 256),
            early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
        )
        
        # Save model
        tabnet_path = self.config['models_dir'] / "tabnet_optimized.pkl"
        self.tabnet_model.save(tabnet_path)
        
        logger.info(f"TabNet model saved to {tabnet_path}")
    
    def stage_5_evaluation(self) -> pd.DataFrame:
        """
        Stage 5: Comprehensive Evaluation with Optimal Threshold Application
        
        Returns:
            Evaluation results DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: COMPREHENSIVE EVALUATION WITH OPTIMAL THRESHOLDS")
        logger.info("="*80)
        
        if self.data is None or self.evaluator is None:
            raise ValueError("Must run preprocessing and baseline first!")
        
        if self.tabnet_model is None:
            raise ValueError("Must train TabNet first!")
        
        # Get predictions for all models
        logger.info("\nGenerating predictions on test set...")
        y_pred_proba_rf = self.rf_model.predict_proba(self.data['X_test'])[:, 1]
        y_pred_proba_tabnet = self.tabnet_model.predict_proba(self.data['X_test'])[:, 1]
        
        # Create ensemble
        logger.info("\nCreating ensemble model...")
        self.ensemble_model = EnsembleModel(
            self.rf_model, self.tabnet_model,
            rf_weight=0.3, tabnet_weight=0.7
        )
        y_pred_proba_ensemble = self.ensemble_model.predict_proba(self.data['X_test'])
        
        # Find optimal thresholds for each model
        logger.info("\nFinding optimal classification thresholds...")
        
        self.optimal_thresholds['rf'] = self.evaluator.find_optimal_threshold(
            self.data['y_test'], y_pred_proba_rf, metric='f1', n_thresholds=100
        )[0]
        logger.info(f"  Random Forest optimal threshold (F1): {self.optimal_thresholds['rf']:.4f}")
        
        self.optimal_thresholds['tabnet'] = self.evaluator.find_optimal_threshold(
            self.data['y_test'], y_pred_proba_tabnet, metric='f1', n_thresholds=100
        )[0]
        logger.info(f"  TabNet optimal threshold (F1): {self.optimal_thresholds['tabnet']:.4f}")
        
        self.optimal_thresholds['ensemble'] = self.evaluator.find_optimal_threshold(
            self.data['y_test'], y_pred_proba_ensemble, metric='f1', n_thresholds=100
        )[0]
        logger.info(f"  Ensemble optimal threshold (F1): {self.optimal_thresholds['ensemble']:.4f}")
        
        # Evaluate all models at both default (0.5) and optimal thresholds
        logger.info("\nEvaluating models at default threshold (0.5)...")
        
        y_pred_rf_default = (y_pred_proba_rf >= 0.5).astype(int)
        y_pred_tabnet_default = (y_pred_proba_tabnet >= 0.5).astype(int)
        y_pred_ensemble_default = (y_pred_proba_ensemble >= 0.5).astype(int)
        
        metrics_rf_default = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_rf, y_pred_rf_default,
            threshold=0.5, model_name="Random Forest (threshold=0.5)"
        )
        
        metrics_tabnet_default = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_tabnet, y_pred_tabnet_default,
            threshold=0.5, model_name="TabNet (threshold=0.5)"
        )
        
        metrics_ensemble_default = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_ensemble, y_pred_ensemble_default,
            threshold=0.5, model_name="Ensemble (threshold=0.5)"
        )
        
        # Evaluate with optimal thresholds
        logger.info("\nEvaluating models at optimal thresholds...")
        
        y_pred_rf_optimal = (y_pred_proba_rf >= self.optimal_thresholds['rf']).astype(int)
        y_pred_tabnet_optimal = (y_pred_proba_tabnet >= self.optimal_thresholds['tabnet']).astype(int)
        y_pred_ensemble_optimal = (y_pred_proba_ensemble >= self.optimal_thresholds['ensemble']).astype(int)
        
        metrics_rf_optimal = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_rf, y_pred_rf_optimal,
            threshold=self.optimal_thresholds['rf'], model_name=f"Random Forest (threshold={self.optimal_thresholds['rf']:.4f})"
        )
        
        metrics_tabnet_optimal = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_tabnet, y_pred_tabnet_optimal,
            threshold=self.optimal_thresholds['tabnet'], model_name=f"TabNet (threshold={self.optimal_thresholds['tabnet']:.4f})"
        )
        
        metrics_ensemble_optimal = self.evaluator.compute_metrics(
            self.data['y_test'], y_pred_proba_ensemble, y_pred_ensemble_optimal,
            threshold=self.optimal_thresholds['ensemble'], model_name=f"Ensemble (threshold={self.optimal_thresholds['ensemble']:.4f})"
        )
        
        # Plot evaluation curves for all models
        logger.info("\nGenerating evaluation plots...")
        
        # RF plots
        self.evaluator.plot_roc_curve(
            self.data['y_test'], y_pred_proba_rf, "Random Forest",
            save_path=str(self.config['evaluation_dir'] / "roc_rf.png")
        )
        self.evaluator.plot_precision_recall_curve(
            self.data['y_test'], y_pred_proba_rf, "Random Forest",
            save_path=str(self.config['evaluation_dir'] / "pr_rf.png")
        )
        self.evaluator.plot_confusion_matrix(
            self.data['y_test'], y_pred_rf_optimal, "Random Forest (Optimal)",
            save_path=str(self.config['evaluation_dir'] / "cm_rf_optimal.png")
        )
        
        # TabNet plots
        self.evaluator.plot_roc_curve(
            self.data['y_test'], y_pred_proba_tabnet, "TabNet",
            save_path=str(self.config['evaluation_dir'] / "roc_tabnet.png")
        )
        self.evaluator.plot_precision_recall_curve(
            self.data['y_test'], y_pred_proba_tabnet, "TabNet",
            save_path=str(self.config['evaluation_dir'] / "pr_tabnet.png")
        )
        self.evaluator.plot_confusion_matrix(
            self.data['y_test'], y_pred_tabnet_optimal, "TabNet (Optimal)",
            save_path=str(self.config['evaluation_dir'] / "cm_tabnet_optimal.png")
        )
        
        # Ensemble plots
        self.evaluator.plot_roc_curve(
            self.data['y_test'], y_pred_proba_ensemble, "Ensemble",
            save_path=str(self.config['evaluation_dir'] / "roc_ensemble.png")
        )
        self.evaluator.plot_precision_recall_curve(
            self.data['y_test'], y_pred_proba_ensemble, "Ensemble",
            save_path=str(self.config['evaluation_dir'] / "pr_ensemble.png")
        )
        self.evaluator.plot_confusion_matrix(
            self.data['y_test'], y_pred_ensemble_optimal, "Ensemble (Optimal)",
            save_path=str(self.config['evaluation_dir'] / "cm_ensemble_optimal.png")
        )
        
        # Compile comparison results
        logger.info("\n" + "="*80)
        logger.info("THRESHOLD COMPARISON SUMMARY")
        logger.info("="*80)
        
        comparison_results = []
        
        # Default threshold results
        comparison_results.append({
            'Model': 'Random Forest',
            'Threshold': 0.5,
            'Threshold_Type': 'Default',
            'ROC_AUC': metrics_rf_default['ROC_AUC'],
            'PR_AUC': metrics_rf_default['PR_AUC'],
            'F1_Score': metrics_rf_default['F1_Score'],
            'Precision': metrics_rf_default['Precision'],
            'Recall': metrics_rf_default['Recall'],
            'Accuracy': metrics_rf_default['Accuracy'],
        })
        
        comparison_results.append({
            'Model': 'TabNet',
            'Threshold': 0.5,
            'Threshold_Type': 'Default',
            'ROC_AUC': metrics_tabnet_default['ROC_AUC'],
            'PR_AUC': metrics_tabnet_default['PR_AUC'],
            'F1_Score': metrics_tabnet_default['F1_Score'],
            'Precision': metrics_tabnet_default['Precision'],
            'Recall': metrics_tabnet_default['Recall'],
            'Accuracy': metrics_tabnet_default['Accuracy'],
        })
        
        comparison_results.append({
            'Model': 'Ensemble',
            'Threshold': 0.5,
            'Threshold_Type': 'Default',
            'ROC_AUC': metrics_ensemble_default['ROC_AUC'],
            'PR_AUC': metrics_ensemble_default['PR_AUC'],
            'F1_Score': metrics_ensemble_default['F1_Score'],
            'Precision': metrics_ensemble_default['Precision'],
            'Recall': metrics_ensemble_default['Recall'],
            'Accuracy': metrics_ensemble_default['Accuracy'],
        })
        
        # Optimal threshold results
        comparison_results.append({
            'Model': 'Random Forest',
            'Threshold': round(self.optimal_thresholds['rf'], 4),
            'Threshold_Type': 'Optimal (F1)',
            'ROC_AUC': metrics_rf_optimal['ROC_AUC'],
            'PR_AUC': metrics_rf_optimal['PR_AUC'],
            'F1_Score': metrics_rf_optimal['F1_Score'],
            'Precision': metrics_rf_optimal['Precision'],
            'Recall': metrics_rf_optimal['Recall'],
            'Accuracy': metrics_rf_optimal['Accuracy'],
        })
        
        comparison_results.append({
            'Model': 'TabNet',
            'Threshold': round(self.optimal_thresholds['tabnet'], 4),
            'Threshold_Type': 'Optimal (F1)',
            'ROC_AUC': metrics_tabnet_optimal['ROC_AUC'],
            'PR_AUC': metrics_tabnet_optimal['PR_AUC'],
            'F1_Score': metrics_tabnet_optimal['F1_Score'],
            'Precision': metrics_tabnet_optimal['Precision'],
            'Recall': metrics_tabnet_optimal['Recall'],
            'Accuracy': metrics_tabnet_optimal['Accuracy'],
        })
        
        comparison_results.append({
            'Model': 'Ensemble',
            'Threshold': round(self.optimal_thresholds['ensemble'], 4),
            'Threshold_Type': 'Optimal (F1)',
            'ROC_AUC': metrics_ensemble_optimal['ROC_AUC'],
            'PR_AUC': metrics_ensemble_optimal['PR_AUC'],
            'F1_Score': metrics_ensemble_optimal['F1_Score'],
            'Precision': metrics_ensemble_optimal['Precision'],
            'Recall': metrics_ensemble_optimal['Recall'],
            'Accuracy': metrics_ensemble_optimal['Accuracy'],
        })
        
        results_df = pd.DataFrame(comparison_results)
        
        # Print summary table  
        logger.info("\nDetailed Threshold Comparison:")
        logger.info(results_df.to_string(index=False))
        
        # Save results
        results_path = self.config['output_dir'] / "model_comparison.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to {results_path}")
        
        # Save optimal thresholds metadata
        thresholds_path = self.config['output_dir'] / "optimal_thresholds.txt"
        with open(thresholds_path, 'w') as f:
            f.write("Optimal Classification Thresholds (F1-based)\n")
            f.write("="*40 + "\n")
            f.write(f"Random Forest: {self.optimal_thresholds['rf']:.4f}\n")
            f.write(f"TabNet:        {self.optimal_thresholds['tabnet']:.4f}\n")
            f.write(f"Ensemble:      {self.optimal_thresholds['ensemble']:.4f}\n")
        logger.info(f"Thresholds saved to {thresholds_path}")
        
        return results_df
    
    def run_full_pipeline(self, run_hpo: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            run_hpo: Whether to run hyperparameter optimization
        
        Returns:
            Dictionary with all pipeline results
        """
        logger.info("\n" + "#"*80)
        logger.info("# MICRO-LOAN DEFAULT RISK PREDICTION PIPELINE")
        logger.info("#"*80)
        
        # Stage 1: Preprocessing
        self.stage_1_preprocessing()
        
        # Stage 2: Baseline
        baseline_metrics = self.stage_2_baseline_model()
        
        # Stage 3: HPO (optional)
        hpo_results = None
        best_params = None
        if run_hpo:
            hpo_results = self.stage_3_hpo_optimization()
            best_params = hpo_results['best_params']
        
        # Stage 4: TabNet training
        self.stage_4_tabnet_training(best_params)
        
        # Stage 5: Evaluation
        results_df = self.stage_5_evaluation()
        
        logger.info("\n" + "#"*80)
        logger.info("# PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("#"*80)
        
        return {
            'baseline_metrics': baseline_metrics,
            'hpo_results': hpo_results,
            'final_results': results_df,
            'models': {
                'rf': self.rf_model,
                'tabnet': self.tabnet_model,
                'ensemble': self.ensemble_model,
            },
            'data': self.data,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Micro-Loan Default Risk Prediction Pipeline')
    parser.add_argument('--run_hpo', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--n_hpo_trials', type=int, default=30, help='Number of HPO trials')
    parser.add_argument('--log_level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run pipeline
    pipeline = MicroLoanPipeline()
    results = pipeline.run_full_pipeline(run_hpo=args.run_hpo)
    
    return results


if __name__ == "__main__":
    main()
