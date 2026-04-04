# Micro-Loan Default Risk Prediction using TabNet

A comprehensive deep learning pipeline for predicting short-term default risk in micro-loan and BNPL (Buy Now Pay Later) platforms using the **Home Credit Default Risk** dataset from Kaggle.

## Overview

This project implements a complete machine learning pipeline with the following components:

- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Baseline Model**: Random Forest classifier for performance baseline
- **Deep Learning**: TabNet (Attentive Interpretable Tabular Learning) implementation
- **Custom Loss Function**: Focal Loss to handle extreme class imbalance
- **Hyperparameter Optimization**: Optuna-based automatic tuning of model parameters
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, F1-Score, and threshold optimization

## Architecture

### Project Structure

```
microloan-risk-tabnet/
├── config.py                 # Configuration and hyperparameters
├── requirements.txt          # Python dependencies
├── main.py                   # Main orchestration script
│
├── src/
│   ├── __init__.py
│   ├── utils.py             # Utility functions (loading, encoding, scaling)
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── models.py            # Baseline RandomForest and TabNet implementations
│   ├── loss_functions.py    # Custom Focal Loss for class imbalance
│   ├── evaluation.py        # Evaluation metrics and visualization
│   └── hpo.py               # Optuna hyperparameter optimization
│
├── home-credit-default-risk/  # Data directory
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   └── ... (other datasets)
│
└── outputs/
    ├── models/              # Saved model files
    ├── evaluation/          # Evaluation plots (ROC, PR curves, etc.)
    ├── hpo_reports/        # HPO trial reports
    └── logs/               # Pipeline execution logs
```

## Key Features

### 1. **Data Preprocessing** (`src/preprocessing.py`)
- Intelligent missing value imputation (median for numeric, mode for categorical)
- Categorical feature encoding (label encoding for TabNet)
- Feature scaling using StandardScaler
- Stratified train-test split for class imbalance preservation
- Class weight computation for imbalanced datasets

### 2. **Custom Focal Loss** (`src/loss_functions.py`)
Focal Loss implementation to handle class imbalance:
```
FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
```
- **alpha**: Weighting factor for positive class (0.25 default)
- **gamma**: Focusing parameter (2.0 default) - higher gamma emphasizes hard negatives

### 3. **TabNet Model** (`src/models.py`)
Implements the TabNet architecture from pytorch-tabnet library:
- Multiple decision steps with feature attention masks
- Interpretable feature importance
- Automatic feature selection
- Efficient sparse feature learning

### 4. **Baseline Model** (`src/models.py`)
Random Forest classifier with:
- Balanced class weighting
- Feature importance extraction
- Simple yet competitive baseline

### 5. **Hyperparameter Optimization** (`src/hpo.py`)
Optuna-based HPO pipeline:
- **TabNet Parameters**: n_d, n_a, n_steps, gamma, lambda_sparse, learning_rate, batch_size
- **Focal Loss Parameters**: alpha, gamma
- **Strategy**: Tree-structured Parzen Estimator (TPE) with cross-validation
- **Pruning**: Median stopping rule for early trial termination
- **Parallelization**: Multi-job support for faster optimization

### 6. **Evaluation Metrics** (`src/evaluation.py`)
- **ROC-AUC**: Area under ROC curve (threshold-independent)
- **PR-AUC**: Precision-Recall area (important for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Threshold Optimization**: Find optimal classification threshold
- **Visualization**: ROC curves, PR curves, confusion matrices

### 7. **Ensemble Model** (`src/models.py`)
Combines RandomForest and TabNet predictions:
- Configurable weights for each model
- Leverages strengths of both models
- Typically better performance than individual models

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
cd microloan-risk-tabnet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Home Credit Default Risk dataset from Kaggle:
```bash
# Place CSV files in home-credit-default-risk/ directory
```

## Usage

### Quick Start (Without HPO)

Run the complete pipeline with default hyperparameters:

```bash
python main.py
```

### Full Pipeline (With Hyperparameter Optimization)

Run the pipeline with Optuna HPO:

```bash
python main.py --run_hpo --n_hpo_trials 30
```

### Programmatic Usage

```python
from main import MicroLoanPipeline

# Initialize pipeline
pipeline = MicroLoanPipeline()

# Run preprocessing
data = pipeline.stage_1_preprocessing()

# Train baseline model
baseline_metrics = pipeline.stage_2_baseline_model()

# Run HPO (optional)
hpo_results = pipeline.stage_3_hpo_optimization()

# Train TabNet with best parameters
best_params = hpo_results['best_params']
pipeline.stage_4_tabnet_training(best_params)

# Evaluate all models
results = pipeline.stage_5_evaluation()

# Print results
print(results['final_results'])
```

### Custom Configuration

Edit `config.py` to customize:

```python
# Model hyperparameters
MODEL_CONFIG = {
    "tabnet": {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 3,
        "gamma": 1.5,
        "lambda_sparse": 1e-4,
    }
}

# Focal Loss parameters
FOCAL_LOSS_CONFIG = {
    "alpha": 0.25,
    "gamma": 2.0,
}

# Optuna HPO settings
OPTUNA_CONFIG = {
    "n_trials": 30,
    "timeout": 3600,
}
```

## Output Files

The pipeline generates the following outputs:

### Models
- `outputs/models/random_forest_baseline.pkl` - Trained RandomForest
- `outputs/models/tabnet_optimized.pkl` - Trained TabNet model

### Evaluation
- `outputs/evaluation/roc_curve_*.png` - ROC curves
- `outputs/evaluation/pr_curve_*.png` - PR curves
- `outputs/evaluation/confusion_matrix_*.png` - Confusion matrices
- `outputs/model_comparison.csv` - Final metrics comparison

### Optimization
- `outputs/hpo_reports/optimization_report.csv` - Detailed HPO trial results
- `outputs/logs/pipeline.log` - Complete execution log

## Model Performance

### Expected Results (Baseline)

When trained on the Home Credit Default Risk dataset:

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| Random Forest | ~0.75 | ~0.45 | ~0.55 |
| TabNet (optimized) | ~0.78 | ~0.50 | ~0.58 |
| Ensemble | ~0.79 | ~0.52 | ~0.60 |

*Results vary based on data preprocessing and HPO iterations*

## Key Techniques & Rationale

### 1. **Focal Loss for Class Imbalance**
The dataset has severe class imbalance (only ~8% defaults). Focal Loss:
- Downweights easy negatives
- Focuses training on hard examples
- Better handles minority class

### 2. **TabNet Over Standard Neural Networks**
TabNet advantages:
- Designed for tabular data (better than MLPs)
- Interpretable feature selection masks
- Efficient training on CPU/GPU
- Automatic feature selection

### 3. **Ensemble Approach**
Combining RandomForest + TabNet:
- Captures different pattern types
- Reduces overfitting
- Improves generalization
- Better threshold tolerance

### 4. **Hyperparameter Optimization**
Optuna HPO:
- Automically tunes 8+ hyperparameters
- Cross-validation for robustness
- Median pruning prevents wasteful trials
- Bayesian optimization (TPE) efficient exploration

## Requirements

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities, metrics, preprocessing
- `torch` - PyTorch for computation
- `pytorch-tabnet` - TabNet implementation
- `optuna` - Hyperparameter optimization
- `matplotlib`, `seaborn` - Visualization
- `joblib` - Model serialization
- `tqdm` - Progress bars

## Advanced Usage

### Custom Loss Function

```python
from src.loss_functions import FocalLoss

# Initialize Focal Loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Use in training
loss = focal_loss(logits, targets)
```

### Manual Evaluation

```python
from src.evaluation import MetricsEvaluator

evaluator = MetricsEvaluator(output_dir="outputs/evaluation")

# Compute metrics
metrics = evaluator.compute_metrics(y_true, y_pred_proba)

# Find optimal threshold
optimal_threshold, f1 = evaluator.find_optimal_threshold(y_true, y_pred_proba, metric='f1')

# Plot curves
evaluator.plot_roc_curve(y_true, y_pred_proba, model_name="MyModel")
```

### Custom HPO

```python
from src.hpo import TabNetOptimizer

optimizer = TabNetOptimizer(n_features=X_train.shape[1])

results = optimizer.optimize(
    X_train, y_train,
    n_trials=50,
    metric='roc_auc',
    cv_folds=5
)

# Generate report
report_df = optimizer.report(output_dir="outputs/hpo_reports")
```

## Performance Optimization

### GPU Acceleration
```python
# Automatically uses GPU if available
tabnet_model = TabNetModel(...)  # Detects CUDA
```

### Parallel HPO
```python
optimizer = TabNetOptimizer(..., n_jobs=-1)  # Use all cores
```

### Batch Processing
Adjust `batch_size` in config for memory/speed tradeoff:
```python
TRAINING_CONFIG = {
    "batch_size": 512,  # Increase for faster training
}
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Reduce `n_hpo_trials` for HPO
- Use `n_jobs=1` for HPO

### Slow Training
- Use GPU if available
- Increase `batch_size`
- Reduce number of training epochs
- Use fewer HPO trials

### Poor Model Performance
- Check data quality and preprocessing
- Increase HPO trials and CV folds
- Adjust Focal Loss parameters (alpha, gamma)
- Try different ensemble weights

## Citation

If you use this project, please consider citing:

```bibtex
@software{microloan_tabnet_2024,
  title={{Micro-Loan Default Risk Prediction using TabNet}},
  author={Author Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## References

- **TabNet**: Arik & Pfister (2021). "TabNet: Attentive Interpretable Tabular Learning"
- **Focal Loss**: Lin et al. (2017). "Focal Loss for Dense Object Detection"
- **Optuna**: Akiba et al. (2019). "Optuna: A Next-Generation Hyperparameter Optimization Framework"
- **Home Credit Dataset**: https://www.kaggle.com/competitions/home-credit-default-risk

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Author**: Your Name  
**Last Updated**: April 2024  
**Status**: Production Ready ✓
