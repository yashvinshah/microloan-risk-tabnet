# QUICK REFERENCE GUIDE

## File Location Reference

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| **Configuration** | `config.py` | All global settings |
| **Preprocessing** | `src/preprocessing.py` | `DataPreprocessor` class |
| **Data Utilities** | `src/utils.py` | `load_data()`, `stratified_split()`, etc. |
| **Models** | `src/models.py` | `BaselineRandomForest`, `TabNetModel`, `EnsembleModel` |
| **Loss Functions** | `src/loss_functions.py` | `FocalLoss`, `WeightedBCELoss`, `CombinedLoss` |
| **Evaluation** | `src/evaluation.py` | `MetricsEvaluator` class |
| **HPO** | `src/hpo.py` | `TabNetOptimizer` class |
| **Main Pipeline** | `main.py` | `MicroLoanPipeline` class |
| **Examples** | `quick_start.py` | Example functions |
| **Tests** | `test_pipeline.py` | Component tests |

---

## Common Commands

### Installation & Verification
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python test_pipeline.py

# Clean old outputs
rm -rf outputs/ (Unix/Mac)
rmdir /s outputs (Windows)
```

### Run Pipeline
```bash
# Quick start (demo)
python quick_start.py

# Full pipeline (no optimization)
python main.py

# Full pipeline with HPO
python main.py --run_hpo --n_hpo_trials 30

# Custom: Fewer trials (faster)
python main.py --run_hpo --n_hpo_trials 10
```

### Helper Scripts
```bash
# Unix/Mac
./run.sh install
./run.sh test
./run.sh quick-start
./run.sh run
./run.sh run-hpo
./run.sh clean
./run.sh docs

# Windows
run.bat install
run.bat test
run.bat quick-start
run.bat run
run.bat run-hpo
run.bat clean
run.bat docs
```

---

## Python API Reference

### Quick Start: Preprocessing
```python
from src.preprocessing import preprocess_data
from config import DATA_DIR

# Load and preprocess all data
data = preprocess_data(DATA_DIR, test_size=0.2, random_state=42)

# Access preprocessed data
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
class_weights = data['class_weights']
```

### Quick Start: RandomForest Baseline
```python
from src.models import BaselineRandomForest

# Initialize and train
rf_model = BaselineRandomForest(n_estimators=100, max_depth=15)
rf_model.train(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Save/Load model
rf_model.save(Path('my_model.pkl'))
rf_model.load(Path('my_model.pkl'))
```

### Quick Start: TabNet
```python
from src.models import TabNetModel

# Initialize and train
tabnet = TabNetModel(n_features=200, n_d=64, n_a=64, n_steps=3)
tabnet.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=100,
    batch_size=256,
    learning_rate=2e-2,
    early_stopping_patience=15
)

# Make predictions  
y_pred = tabnet.predict(X_test)
y_pred_proba = tabnet.predict_proba(X_test)

# Get feature importance
importance = tabnet.get_feature_importance(normalize=True)
```

### Quick Start: Focal Loss
```python
import torch
from src.loss_functions import FocalLoss

# Initialize loss function
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Compute loss
logits = model(X)  # Your model output
targets = y
loss = focal_loss(logits, targets)
```

### Quick Start: Evaluation
```python
from src.evaluation import MetricsEvaluator

# Initialize evaluator
evaluator = MetricsEvaluator(output_dir='outputs/evaluation')

# Compute metrics
metrics = evaluator.compute_metrics(y_test, y_pred_proba, y_pred)

# Find optimal threshold
threshold, f1 = evaluator.find_optimal_threshold(y_test, y_pred_proba, metric='f1')

# Plot curves
evaluator.plot_roc_curve(y_test, y_pred_proba, model_name='MyModel')
evaluator.plot_precision_recall_curve(y_test, y_pred_proba, model_name='MyModel')
evaluator.plot_confusion_matrix(y_test, y_pred, model_name='MyModel')
```

### Quick Start: HPO
```python
from src.hpo import run_hpo

# Run hyperparameter optimization
results = run_hpo(
    X_train, y_train,
    n_trials=30,
    n_features=X_train.shape[1],
    metric='roc_auc',
    cv_folds=5
)

# Get best parameters
best_params = results['best_params']
best_value = results['best_value']

# Use best parameters for training
tabnet = TabNetModel(
    n_features=X_train.shape[1],
    **{k: v for k, v in best_params.items() 
       if k in ['n_d', 'n_a', 'n_steps', 'gamma', 'lambda_sparse']}
)
```

### Quick Start: Ensemble
```python
from src.models import EnsembleModel

# Create ensemble from trained models
ensemble = EnsembleModel(
    rf_model=trained_rf,
    tabnet_model=trained_tabnet,
    rf_weight=0.3,
    tabnet_weight=0.7
)

# Make predictions
y_pred_ensemble = ensemble.predict(X_test)
y_pred_proba_ensemble = ensemble.predict_proba(X_test)
```

---

## Configuration Parameters

### Model Hyperparameters
```python
# TabNet parameters
n_d = 64                # Decision layer width (32-128)
n_a = 64                # Attention layer width (32-128)
n_steps = 3             # Decision steps (2-5)
gamma = 1.5             # Relaxation parameter (1.0-2.5)
lambda_sparse = 1e-4    # Sparsity regularization (1e-6 to 1e-3)

# Training parameters
batch_size = 256        # Batch size (128-512)
learning_rate = 2e-2    # Learning rate (1e-3 to 1e-1)
epochs = 100            # Training epochs
early_stopping_patience = 15  # Patience for early stopping
```

### Loss Function Parameters
```python
# Focal Loss
alpha = 0.25            # Positive class weight (0.1-0.9)
gamma = 2.0             # Focusing parameter (1.0-5.0)
```

### HPO Parameters
```python
# Optuna settings
n_trials = 30           # Number of trials
timeout = 3600          # Timeout in seconds (1 hour)
n_jobs = 1              # Parallel jobs (1 or -1 for all cores)
sampler = 'tpe'         # Sampler type
pruner = 'median'       # Pruner type
cv_folds = 3            # Cross-validation folds
```

---

## Metric Definitions

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **ROC-AUC** | Area under ROC curve | 0-1 | Threshold-independent ranking |
| **PR-AUC** | Area under Precision-Recall curve | 0-1 | Imbalanced classes (rare events) |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | 0-1 | Balanced precision/recall |
| **Precision** | TP / (TP + FP) | 0-1 | Minimize false positives |
| **Recall** | TP / (TP + FN) | 0-1 | Minimize false negatives |
| **Accuracy** | (TP + TN) / Total | 0-1 | ⚠️ Misleading for imbalanced data |

---

## Data Pipeline Steps

```
1. Load application_train.csv
   ↓
2. Separate TARGET (y) from features (X)
   ↓
3. Handle Missing Values
   • Numeric: Median imputation
   • Categorical: Mode imputation
   ↓
4. Encode Categorical Features
   • Label encoding (0, 1, 2, ...)
   ↓
5. Scale Features
   • StandardScaler: mean=0, std=1
   ↓
6. Stratified Train-Test Split
   • Maintains class distribution
   • 80% train, 20% test
   ↓
7. Output: X_train, X_test, y_train, y_test
```

---

## Hyperparameter Tuning Guide

### For Better Performance
```python
# More model capacity
n_d = 128
n_a = 128
n_steps = 5              # More decision steps

# More exploration
OPTUNA_CONFIG['n_trials'] = 100

# Focus on hard examples
focal_alpha = 0.3       # Increase from 0.25
focal_gamma = 3.0       # Increase from 2.0
```

### For Faster Training
```python
# Reduce model complexity
n_d = 32
n_a = 32
n_steps = 2

# Faster HPO
n_trials = 10           # Reduce from 30

# Larger batches
batch_size = 512        # Increase from 256
```

### For Memory Efficiency
```python
# Reduce batch size
batch_size = 128        # Reduce from 256

# Reduce model size
n_d = 32
n_a = 32

# Fewer HPO trials
n_trials = 10
```

---

## Diagnostic Output

### Expected ROC-AUC Ranges
- **Random Model**: 0.50 (baseline)
- **Weak Model**: 0.60-0.65 (chance)
- **OK Model**: 0.70-0.75 (acceptable)
- **Good Model**: 0.75-0.80 (good)
- **Excellent Model**: 0.80+ (very good)

### Class Imbalance Ratio
```
Home Credit Dataset:
- Total loans: ~307,000
- Defaults: ~24,000 (8%)
- Non-defaults: ~283,000 (92%)
- Imbalance ratio: 1:12

This extreme imbalance is why:
- Focal Loss is necessary
- ROC-AUC & PR-AUC matter more than accuracy
- Stratified splits are critical
```

---

## File I/O Operations

### Save Models
```python
from pathlib import Path

# Save RandomForest
rf_model.save(Path('outputs/models/rf_model.pkl'))

# Save TabNet
tabnet_model.save(Path('outputs/models/tabnet_model.pkl'))
```

### Load Models
```python
from src.models import BaselineRandomForest, TabNetModel

# Load RandomForest
rf = BaselineRandomForest()
rf.load(Path('outputs/models/rf_model.pkl'))

# Load TabNet
tabnet = TabNetModel(n_features=200)
tabnet.load(Path('outputs/models/tabnet_model.pkl'))
```

### Save Results
```python
import pandas as pd

# Save metrics to CSV
results_df.to_csv('outputs/results.csv', index=False)

# Save plots (automatic in evaluator)
evaluator.plot_roc_curve(..., save_path='outputs/roc.png')
```

---

## Environment Setup

### Python Version
```bash
# Check Python version
python --version
# Should be 3.8 or higher
```

### Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Unix/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

### GPU Support
```python
import torch
print(torch.cuda.is_available())  # True if GPU available
print(torch.cuda.get_device_name(0))  # GPU name
```

---

## Version Info

| Component | Version |
|-----------|---------|
| PyTorch | 2.0.1 |
| PyTorch-TabNet | 4.1.0 |
| Optuna | 3.12.0 |
| Scikit-learn | 1.3.0 |
| Pandas | 2.0.3 |
| NumPy | 1.24.3 |

---

## Support & Resources

- **Documentation**: See `README.md`, `DOCUMENTATION.py`, `GETTING_STARTED.md`
- **Examples**: Run `python quick_start.py`
- **Tests**: Run `python test_pipeline.py`
- **Logs**: Check `outputs/logs/pipeline.log`
- **Results**: See `outputs/model_comparison.csv`

---

**This quick reference covers 90% of common use cases. For detailed info, see full documentation files!**
