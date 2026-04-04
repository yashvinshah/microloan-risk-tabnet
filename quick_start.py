"""
Quick start examples and usage guide for the micro-loan default risk prediction pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR
from src.preprocessing import preprocess_data
from src.models import BaselineRandomForest, TabNetModel, EnsembleModel
from src.evaluation import MetricsEvaluator
from src.hpo import run_hpo


# Example 1: Data Preprocessing Only
def example_preprocessing():
    """Example: Load and preprocess data."""
    print("\\n" + "="*80)
    print("Example 1: Data Preprocessing")
    print("="*80)
    
    data = preprocess_data(DATA_DIR)
    
    print(f"\\nTrain set shape: {data['X_train'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
    print(f"Number of features: {len(data['feature_names'])}")
    print(f"Class weights: {data['class_weights']}")
    
    return data


# Example 2: Train Baseline Model
def example_baseline(data):
    """Example: Train and evaluate baseline RandomForest model."""
    print("\\n" + "="*80)
    print("Example 2: Baseline RandomForest Model")
    print("="*80)
    
    # Train model
    rf_model = BaselineRandomForest(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.train(data['X_train'], data['y_train'])
    
    # Make predictions
    y_pred_proba = rf_model.predict_proba(data['X_test'])[:, 1]
    y_pred = rf_model.predict(data['X_test'])
    
    # Evaluate
    evaluator = MetricsEvaluator(OUTPUT_DIR / "evaluation")
    metrics = evaluator.compute_metrics(
        data['y_test'], y_pred_proba, y_pred,
        model_name="Random Forest"
    )
    
    print(f"\\nRandom Forest Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return rf_model, metrics


# Example 3: Train TabNet Model
def example_tabnet(data):
    """Example: Train and evaluate TabNet model."""
    print("\\n" + "="*80)
    print("Example 3: TabNet Deep Learning Model")
    print("="*80)
    
    # Initialize model
    tabnet_model = TabNetModel(
        n_features=data['X_train'].shape[1],
        n_d=64,
        n_a=64,
        n_steps=3,
        gamma=1.5,
        lambda_sparse=1e-4,
    )
    
    # Train model
    print("\\nTraining TabNet (this may take a few minutes)...")
    tabnet_model.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_test'][:100],  # Small validation set for demo
        y_val=data['y_test'][:100],
        epochs=20,  # Reduced for quick demo
        batch_size=256,
        learning_rate=2e-2,
        early_stopping_patience=10,
    )
    
    # Make predictions
    y_pred_proba = tabnet_model.predict_proba(data['X_test'])[:, 1]
    y_pred = tabnet_model.predict(data['X_test'])
    
    # Evaluate
    evaluator = MetricsEvaluator(OUTPUT_DIR / "evaluation")
    metrics = evaluator.compute_metrics(
        data['y_test'], y_pred_proba, y_pred,
        model_name="TabNet"
    )
    
    print(f"\\nTabNet Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get feature importance
    feature_importance = tabnet_model.get_feature_importance(normalize=True)
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"\\nTop 10 Most Important Features:")
    for idx, feat_idx in enumerate(top_features, 1):
        feat_name = data['feature_names'][feat_idx]
        importance = feature_importance[feat_idx]
        print(f"  {idx}. {feat_name}: {importance:.4f}")
    
    return tabnet_model, metrics


# Example 4: Hyperparameter Optimization
def example_hpo(data):
    """Example: Run Optuna hyperparameter optimization."""
    print("\\n" + "="*80)
    print("Example 4: Hyperparameter Optimization (HPO)")
    print("="*80)
    
    print("\\nStarting optimization (this may take 30+ minutes for full pipeline)...")
    print("Running 5 trials for demonstration...\n")
    
    results = run_hpo(
        X_train=data['X_train'],
        y_train=data['y_train'],
        n_trials=5,  # Reduced for quick demo
        n_features=data['X_train'].shape[1],
        metric='roc_auc',
        cv_folds=2,  # Reduced for quick demo
        epochs=10,  # Reduced for quick demo
    )
    
    print(f"\\nBest ROC-AUC: {results['best_value']:.4f}")
    print(f"\\nBest Hyperparameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    
    return results


# Example 5: Ensemble Model
def example_ensemble(rf_model, tabnet_model, data):
    """Example: Create and evaluate ensemble model."""
    print("\\n" + "="*80)
    print("Example 5: Ensemble Model")
    print("="*80)
    
    # Create ensemble
    ensemble_model = EnsembleModel(
        rf_model, tabnet_model,
        rf_weight=0.3,
        tabnet_weight=0.7
    )
    
    # Make predictions
    y_pred_proba = ensemble_model.predict_proba(data['X_test'])
    y_pred = ensemble_model.predict(data['X_test'], threshold=0.5)
    
    # Evaluate
    evaluator = MetricsEvaluator(OUTPUT_DIR / "evaluation")
    metrics = evaluator.compute_metrics(
        data['y_test'], y_pred_proba, y_pred,
        model_name="Ensemble"
    )
    
    print(f"\\nEnsemble Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return ensemble_model, metrics


# Example 6: Threshold Optimization
def example_threshold_optimization(data, model):
    """Example: Find optimal classification threshold."""
    print("\\n" + "="*80)
    print("Example 6: Threshold Optimization")
    print("="*80)
    
    evaluator = MetricsEvaluator(OUTPUT_DIR / "evaluation")
    
    y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
    
    # Find optimal thresholds for different metrics
    metrics_to_optimize = ['f1', 'roc_auc', 'pr_auc']
    
    print("\\nOptimal Thresholds:")
    for metric in metrics_to_optimize:
        threshold, value = evaluator.find_optimal_threshold(
            data['y_test'], y_pred_proba, metric=metric
        )
        print(f"  {metric}: threshold={threshold:.3f}, value={value:.4f}")
    
    return evaluator


# Master Example: Run All
def run_all_examples():
    """Run all examples in sequence."""
    print("\\n" + "#"*80)
    print("# MICRO-LOAN DEFAULT RISK PREDICTION - QUICK START GUIDE")
    print("#"*80)
    
    # 1. Preprocessing
    data = example_preprocessing()
    
    # 2. Baseline
    rf_model, rf_metrics = example_baseline(data)
    
    # 3. TabNet
    tabnet_model, tabnet_metrics = example_tabnet(data)
    
    # 4. Threshold Optimization
    evaluator = example_threshold_optimization(data, tabnet_model)
    
    # 5. Ensemble
    ensemble_model, ensemble_metrics = example_ensemble(rf_model, tabnet_model, data)
    
    # Summary
    print("\\n" + "#"*80)
    print("# MODEL COMPARISON SUMMARY")
    print("#"*80)
    
    summary_data = {
        'Model': ['Random Forest', 'TabNet', 'Ensemble'],
        'ROC-AUC': [rf_metrics['ROC_AUC'], tabnet_metrics['ROC_AUC'], ensemble_metrics['ROC_AUC']],
        'PR-AUC': [rf_metrics['PR_AUC'], tabnet_metrics['PR_AUC'], ensemble_metrics['PR_AUC']],
        'F1-Score': [rf_metrics['F1_Score'], tabnet_metrics['F1_Score'], ensemble_metrics['F1_Score']],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_path = OUTPUT_DIR / "quick_start_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\\nSummary saved to: {summary_path}")
    
    return {
        'data': data,
        'models': {
            'rf': rf_model,
            'tabnet': tabnet_model,
            'ensemble': ensemble_model,
        },
        'metrics': {
            'rf': rf_metrics,
            'tabnet': tabnet_metrics,
            'ensemble': ensemble_metrics,
        }
    }


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Single examples
    # data = example_preprocessing()
    # run_all_examples()
    
    # Run full pipeline
    print("To run examples, uncomment desired function in this script")
    print("or import examples and run individually:")
    print("\\nfrom quick_start import example_preprocessing, example_baseline")
    print("data = example_preprocessing()")
    print("rf_model, metrics = example_baseline(data)")
