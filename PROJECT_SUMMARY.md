# PROJECT SUMMARY: Micro-Loan Default Risk Prediction Pipeline

## Overview

A complete, production-ready machine learning pipeline for predicting short-term default risk in micro-loan and BNPL platforms using the Home Credit Default Risk dataset. The pipeline implements all requested architectural and evaluation requirements.

---

## ✅ Deliverables Completed

### 1. **Data Preprocessing** ✓
- **File**: `src/preprocessing.py`
- **Features**:
  - Intelligent missing value imputation (median for numeric, mode for categorical)
  - Categorical feature encoding (label encoding optimized for deep learning)
  - Feature scaling using StandardScaler
  - Stratified train-test splitting (maintains class distribution)
  - Class weight computation for imbalanced datasets
  - Comprehensive logging and validation

### 2. **Baseline Model** ✓
- **File**: `src/models.py` (BaselineRandomForest class)
- **Features**:
  - RandomForestClassifier with balanced class weights
  - Feature importance extraction
  - Save/load functionality
  - Consistent API with deep learning models

### 3. **Deep Learning Architecture (TabNet)** ✓
- **File**: `src/models.py` (TabNetModel class)
- **Features**:
  - Attentive Interpretable Tabular Learning (TabNet) implementation
  - Multi-step decision-making with feature attention masks
  - Interpretable feature selection
  - Automatic GPU acceleration detection
  - Training with early stopping
  - Batch training support

### 4. **Custom Loss Function (Focal Loss)** ✓
- **File**: `src/loss_functions.py`
- **Features**:
  - Focal Loss implementation: `FL(pt) = -alpha * (1-pt)^gamma * log(pt)`
  - Handles extreme class imbalance (8% defaults in dataset)
  - Parametric design (tunable alpha, gamma)
  - Alternative WeightedBCELoss and CombinedLoss with L2 regularization
  - PyTorch compatible, GPU-optimized

### 5. **Hyperparameter Optimization (Optuna)** ✓
- **File**: `src/hpo.py`
- **Features**:
  - Bayesian optimization using Tree-structured Parzen Estimator (TPE)
  - Automatic hyperparameter tuning:
    - TabNet params: n_d, n_a, n_steps, gamma, lambda_sparse, learning_rate, batch_size
    - Focal Loss params: alpha, gamma
  - Cross-validation integration (3-5 folds)
  - Median pruning for early trial termination
  - Parallel job support
  - Trial reporting and analysis

### 6. **Evaluation Metrics** ✓
- **File**: `src/evaluation.py`
- **Features**:
  - ROC-AUC (threshold-independent, standard metric)
  - PR-AUC (Precision-Recall area, important for imbalanced data)
  - F1-Score (balance precision and recall)
  - Confusion matrices and classification reports
  - Optimal threshold finding (customizable for any metric)
  - Visualization (ROC curves, PR curves, confusion matrix heatmaps)

### 7. **Code Style & Modularity** ✓
- **Well-organized structure**:
  - Separation of concerns (preprocessing, models, evaluation, HPO)
  - Object-oriented design with clear interfaces
  - Functional utilities for flexibility
  - Comprehensive docstrings and type hints
  - Logging throughout for debugging
  
- **Scikit-learn integration**:
  - Preprocessing: StandardScaler, SimpleImputer, LabelEncoder
  - Metrics: roc_auc_score, f1_score, confusion_matrix, etc.
  - Model selection: train_test_split, StratifiedKFold, cross_val_score
  
- **PyTorch-TabNet integration**:
  - Clean wrapper around pytorch-tabnet library
  - Consistent API with sklearn models
  
- **Optuna integration**:
  - Full HPO pipeline with custom objective function
  - Trial history and study management
  - Visualization support

---

## 📁 Project Structure

```
microloan-risk-tabnet/
├── config.py                      # Central configuration file
├── requirements.txt               # All dependencies
├── main.py                        # Main orchestration script
├── quick_start.py                 # Quick examples for each component
├── test_pipeline.py               # Component test suite
├── DOCUMENTATION.py               # Detailed architecture & design docs
├── README.md                      # Comprehensive README
├── run.sh                         # Unix/Linux/Mac helper script
├── run.bat                        # Windows helper script
├── .gitignore                     # Git ignore rules
│
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Low-level utilities (loading, encoding, scaling)
│   ├── preprocessing.py           # Data preprocessing pipeline (DataPreprocessor class)
│   ├── models.py                  # RandomForest, TabNet, Ensemble models
│   ├── loss_functions.py          # Focal Loss, WeightedBCE, Combined Loss
│   ├── evaluation.py              # MetricsEvaluator class & visualization
│   └── hpo.py                     # TabNetOptimizer & Optuna pipeline
│
├── home-credit-default-risk/      # Data directory (7+ CSV files)
│   ├── application_train.csv
│   ├── application_test.csv
│   └── ... (other datasets)
│
└── outputs/                       # Generated during pipeline execution
    ├── models/                    # Saved model files
    ├── evaluation/                # Plots (ROC, PR, confusion matrices)
    ├── hpo_reports/              # HPO optimization results
    └── logs/                      # Execution logs
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Tests
```bash
python test_pipeline.py
```

### Step 3: Run Full Pipeline
```bash
# Without HPO (fast)
python main.py

# With HPO (slow, ~1-2 hours)
python main.py --run_hpo --n_hpo_trials 30
```

### Step 4: View Results
- Model comparison: `outputs/model_comparison.csv`
- ROC curves: `outputs/evaluation/roc_curve_*.png`
- PR curves: `outputs/evaluation/pr_curve_*.png`
- Confusion matrices: `outputs/evaluation/confusion_matrix_*.png`
- Logs: `outputs/logs/pipeline.log`

### Helper Commands (Unix/Linux/Mac)
```bash
./run.sh install         # Install dependencies
./run.sh test            # Run tests
./run.sh quick-start     # Run quick examples
./run.sh run             # Full pipeline
./run.sh run-hpo         # Full pipeline with HPO
./run.sh clean           # Clean outputs
./run.sh docs            # Display documentation
```

### Helper Commands (Windows)
```batch
run.bat install         # Install dependencies
run.bat test            # Run tests
run.bat quick-start     # Run quick examples
run.bat run             # Full pipeline
run.bat run-hpo         # Full pipeline with HPO
run.bat clean           # Clean outputs
run.bat docs            # Display documentation
```

---

## 📊 Pipeline Stages

### Stage 1: Data Preprocessing
- Loads application_train.csv
- Handles missing values through intelligent imputation
- Encodes categorical features using label encoding
- Scales features to N(0,1) distribution
- Performs stratified train-test split (80-20)

### Stage 2: Baseline Model
- Trains RandomForest with balanced class weights
- Evaluates on test set
- Generates ROC/PR curves and confusion matrix
- Establishes performance benchmark

### Stage 3: Hyperparameter Optimization (Optional)
- Uses Optuna to search hyperparameter space
- 30 trials of Bayesian optimization
- 3-fold cross-validation per trial
- Median pruning to eliminate poor trials
- ~1 hour on CPU, ~15-30 min on GPU

### Stage 4: TabNet Training
- Trains TabNet with best parameters (from HPO or defaults)
- Uses validation split with early stopping
- Supports GPU acceleration
- Saves trained model

### Stage 5: Comprehensive Evaluation
- Evaluates all models (RF, TabNet, Ensemble)
- Computes ROC-AUC, PR-AUC, F1-Score, Precision, Recall, Accuracy
- Finds optimal classification threshold
- Generates comparison table and plots

---

## 🎯 Key Features

### Class Imbalance Handling
- **Stratified Splits**: Maintains class distribution in train/test
- **Focal Loss**: Focuses training on hard examples with α=0.25, γ=2.0
- **Class Weights**: Balanced weighting in RandomForest
- **Threshold Optimization**: Finds best operating point

### Hyperparameter Optimization
- **Search Space**: 8+ parameters for TabNet + Focal Loss
- **Algorithm**: Bayesian Optimization (TPE sampler)
- **Strategy**: Cross-validation with pruning
- **Objective**: Maximize ROC-AUC (threshold-independent)

### Ensemble Approach
- **RF Weight**: 0.3 (captures interpretable patterns)
- **TabNet Weight**: 0.7 (captures complex patterns)
- **Benefit**: Often outperforms individual models

### Evaluation Framework
- **Multiple Metrics**: ROC-AUC, PR-AUC, F1, Precision, Recall, Accuracy
- **Visualization**: ROC curves, PR curves, confusion matrices
- **Threshold Search**: Optimize for any metric across 100 thresholds

---

## 📝 Configuration

All parameters centralized in `config.py`:

```python
# Model parameters
MODEL_CONFIG['tabnet'] = {
    'n_d': 64,              # Decision layer width
    'n_a': 64,              # Attention layer width
    'n_steps': 3,           # Decision steps
    'gamma': 1.5,           # Relaxation parameter
    'lambda_sparse': 1e-4,  # Sparsity regularization
}

# Focal Loss parameters
FOCAL_LOSS_CONFIG = {
    'alpha': 0.25,          # Positive class weight
    'gamma': 2.0,           # Focusing parameter
}

# HPO settings
OPTUNA_CONFIG = {
    'n_trials': 30,         # Number of trials
    'timeout': 3600,        # 1 hour timeout
}
```

Modify these for your specific needs!

---

## 💡 Usage Examples

### Example 1: Run Preprocessing Only
```python
from src.preprocessing import preprocess_data
from config import DATA_DIR

data = preprocess_data(DATA_DIR)
X_train, y_train = data['X_train'], data['y_train']
print(f"Training set: {X_train.shape}")
```

### Example 2: Train Baseline
```python
from src.models import BaselineRandomForest
from src.preprocessing import preprocess_data

data = preprocess_data(DATA_DIR)
rf = BaselineRandomForest(n_estimators=100)
rf.train(data['X_train'], data['y_train'])
predictions = rf.predict_proba(data['X_test'])
```

### Example 3: Custom Loss Function
```python
import torch
from src.loss_functions import FocalLoss

loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32,))
loss = loss_fn(logits, targets)
```

### Example 4: Hyperparameter Optimization
```python
from src.hpo import run_hpo
from src.preprocessing import preprocess_data

data = preprocess_data(DATA_DIR)
results = run_hpo(
    data['X_train'], data['y_train'],
    n_trials=50,
    metric='roc_auc',
    cv_folds=5
)
best_params = results['best_params']
```

---

## 📈 Expected Performance

Based on Home Credit Default Risk dataset (~307k loans, 8% default rate):

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| Random Forest | ~0.75 | ~0.45 | ~0.55 |
| TabNet (optimized) | ~0.78 | ~0.50 | ~0.58 |
| Ensemble | ~0.79 | ~0.52 | ~0.60 |

*Actual performance depends on data quality, preprocessing, and HPO iterations*

---

## 🔧 Technical Stack

- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Deep Learning**: PyTorch, PyTorch-TabNet
- **Hyperparameter Optimization**: Optuna
- **Evaluation**: Scikit-learn metrics, Matplotlib, Seaborn
- **Utilities**: Joblib, tqdm, Python logging

---

## 📚 Documentation

- **README.md**: Comprehensive project overview
- **DOCUMENTATION.py**: Detailed architecture, design decisions, tuning guide
- **Docstrings**: Every function and class has detailed docstrings
- **Comments**: Code comments explain complex logic

---

## ⚠️ Important Notes

1. **Data Preparation**: Ensure CSV files are in `home-credit-default-risk/` directory
2. **GPU Support**: Pipeline automatically uses GPU if available
3. **Memory Requirements**: 8GB RAM minimum, 16GB+ recommended
4. **Time Requirements**:
   - Without HPO: ~5-10 minutes
   - With HPO: ~1-2 hours (CPU) or ~15-30 minutes (GPU)
5. **First Run**: Expect preprocessing to take time as data is large

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in config.py
TRAINING_CONFIG['batch_size'] = 128  # from 256
```

### Slow Training
```python
# Use fewer HPO trials
OPTUNA_CONFIG['n_trials'] = 10  # from 30
```

### Poor Performance
```python
# Increase model complexity
MODEL_CONFIG['tabnet']['n_d'] = 128  # from 64
OPTUNA_CONFIG['n_trials'] = 100  # more trials
```

---

## 📞 Support

For issues or questions:
1. Check `outputs/logs/pipeline.log` for detailed error messages
2. Review `DOCUMENTATION.py` for architecture details
3. Run `test_pipeline.py` to verify all components
4. See `quick_start.py` for working examples

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🎉 Summary

This complete pipeline provides:
- ✅ Production-ready code quality
- ✅ Modular, extensible architecture
- ✅ Comprehensive evaluation framework
- ✅ Automatic hyperparameter optimization
- ✅ Detailed documentation
- ✅ Testing suite
- ✅ Helper scripts for quick start

**Everything is ready to use. Happy modeling! 🚀**
