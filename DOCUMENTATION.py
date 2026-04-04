"""
Comprehensive documentation for the Micro-Loan Default Risk Prediction Pipeline.

This file provides detailed explanations of architecture, design decisions, and best practices.
"""

# ============================================================================
# ARCHITECTURE OVERVIEW
# ============================================================================

"""
The pipeline follows a modular, layered architecture with clear separation of concerns:

┌─────────────────────────────────────────────────────────────┐
│                        MAIN PIPELINE                         │
│             (main.py - MicroLoanPipeline class)              │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Data Pre-processing → Stage 2: Baseline Model     │
│  Stage 3: Hyperparameter Optimization → Stage 4: TabNet     │
│  Stage 5: Comprehensive Evaluation & Ensemble               │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ PRE-     │    │ MODELS   │    │ EVAL &   │
    │ PROCESS. │    │ & LOSS   │    │ HPO      │
    └──────────┘    └──────────┘    └──────────┘
         │               │               │
    ┌────┴───┐       ┌────┴───┐    ┌────┴────┐
    │ utils  │       │ models │    │ hpo     │
    │ prep   │       │ loss   │    │ eval    │
    └────────┘       └────────┘    └─────────┘
"""

# ============================================================================
# MODULE DESCRIPTIONS
# ============================================================================

"""
1. CONFIG.PY
   Purpose: Centralized configuration management
   Contains:
   - File paths and directories
   - Model hyperparameters (RandomForest, TabNet)
   - Loss function parameters (Focal Loss)
   - Optuna HPO settings
   - Training configurations
   - Evaluation settings
   
   Key Design: All parameters in one place for easy experiment variation

2. SRC/UTILS.PY
   Purpose: Low-level utilities used across modules
   Contains:
   - Logging setup
   - Data loading helpers
   - Feature encoding (label encoding, one-hot)
   - Feature scaling (StandardScaler)
   - Train-test splitting with stratification
   - Class weight computation
   
   Key Design: Functional programming style for flexibility

3. SRC/PREPROCESSING.PY
   Purpose: Complete data preprocessing pipeline
   Contains:
   - DataPreprocessor class (orchestrates preprocessing)
   - Missing value imputation
   - Categorical feature handling
   - Feature scaling
   - Data validation and logging
   
   Key Design: 
   - Object-oriented for state management
   - Fit-transform pattern consistent with sklearn
   - Full preprocessing pipeline method

4. SRC/LOSS_FUNCTIONS.PY
   Purpose: Custom PyTorch loss functions for class imbalance
   Contains:
   - FocalLoss: Main loss function (alpha, gamma parameters)
   - WeightedBCELoss: Alternative weighted loss
   - CombinedLoss: Focal + L2 regularization
   
   Key Design:
   - Inherits from nn.Module for PyTorch compatibility
   - Handles edge cases (1D input, device placement)
   - Parametric design for tuning

5. SRC/MODELS.PY
   Purpose: Model implementations
   Contains:
   - BaselineRandomForest: Sklearn-based baseline
   - TabNetModel: Wrapper around pytorch-tabnet
   - EnsembleModel: Combines two models
   
   Key Design:
   - Consistent predict/predict_proba interface
   - Save/load functionality for all models
   - Feature importance extraction
   
6. SRC/EVALUATION.PY
   Purpose: Comprehensive evaluation and metrics
   Contains:
   - MetricsEvaluator: Central evaluation class
   - Metric computation (ROC-AUC, PR-AUC, F1)
   - Threshold optimization
   - Visualization (ROC, PR, confusion matrices)
   
   Key Design:
   - Vectorized computations for speed
   - Matplotlib integration for plots
   - Threshold search across multiple metrics

7. SRC/HPO.PY
   Purpose: Optuna-based hyperparameter optimization
   Contains:
   - TabNetOptimizer: Main optimizer class
   - Objective function definition
   - Cross-validation integration
   - Trial pruning and reporting
   
   Key Design:
   - Search space: 8+ hyperparameters
   - TPE sampler with median pruning
   - Flexible metric selection
   
8. MAIN.PY
   Purpose: Pipeline orchestration
   Contains:
   - MicroLoanPipeline: Main class (5 stages)
   - Stage-by-stage execution
   - Result compilation and reporting
   - Command-line interface
   
   Key Design:
   - Logical stage separation
   - Each stage independent (can be skipped)
   - Comprehensive logging at each stage
"""

# ============================================================================
# KEY DESIGN DECISIONS
# ============================================================================

"""
1. CLASS IMBALANCE HANDLING
   
   Problem: Only ~8% of loans defaulted in Home Credit dataset
   
   Solutions Implemented:
   a) Focal Loss: Focuses training on hard examples
      - Downweights easy negatives
      - Upweights minority class examples
      - Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
      - Parameters tuned via Optuna
   
   b) Stratified Splitting: Ensures class distribution preserved
      - train_test_split with stratify=y
      - StratifiedKFold in cross-validation
   
   c) Class Weights: Weighted loss in RandomForest
      - class_weight='balanced' in RF
      - Adjusts feature importance

2. HYPERPARAMETER OPTIMIZATION
   
   Approach: Bayesian Optimization via Optuna
   
   Search Space:
   - TabNet: n_d, n_a, n_steps, gamma, lambda_sparse, learning_rate, batch_size
   - Focal Loss: alpha, gamma
   
   Strategy:
   - Tree-structured Parzen Estimator (TPE) sampler
   - Median pruning for early stopping
   - 3-5 fold cross-validation per trial
   - Maximize ROC-AUC (threshold-independent metric)
   
   Why Optuna?
   - Efficient pruning eliminates bad trials early
   - Parallel job support
   - Saves trial history for analysis

3. ENSEMBLE APPROACH
   
   Rationale:
   - RandomForest captures linear/interaction patterns
   - TabNet captures complex nonlinear patterns
   - Ensemble combines both strengths
   - Weights (0.3 RF, 0.7 TabNet) learned empirically
   
4. EVALUATION METRICS
   
   Why Multiple Metrics?
   - ROC-AUC: Threshold-independent, good for imbalanced
   - PR-AUC: More sensitive to minority class
   - F1-Score: Balances precision and recall
   - Threshold: Find point that maximizes desired metric
   
5. FEATURE PREPROCESSING
   
   Choices:
   - Label Encoding (not one-hot): TabNet works better with numeric features
   - StandardScaler: Normalizes features to mean=0, std=1
   - Missing Value Imputation: Median for numeric (robust to outliers)
   - Categorical Encoding: Most frequent value (preserves distribution)

6. MODULAR ARCHITECTURE
   
   Benefits:
   - Each module independently testable
   - Easy to swap implementations (e.g., different loss function)
   - Reusable components (e.g., use preprocessing in other projects)
   - Clear dependency graph
"""

# ============================================================================
# HYPERPARAMETER EXPLANATION
# ============================================================================

"""
TABNET PARAMETERS:

1. n_d (32-128):
   - Dimension of decision-making layer
   - Higher: More capacity, slower training
   - Optimal: 64 for most datasets
   
2. n_a (32-128):
   - Dimension of feature attention layer
   - Similar tradeoff to n_d
   - Often set equal to n_d
   
3. n_steps (2-5):
   - Number of decision steps (iterations)
   - More steps: More feature exploration
   - Computational cost scales linearly
   - Typical: 3-4 steps
   
4. gamma (1.0-2.5):
   - Relaxation parameter for feature reuse
   - Controls feature mask mixing
   - Lower: More aggressive feature reuse
   - Higher: Features more selective
   
5. lambda_sparse (1e-6 to 1e-3):
   - Sparsity regularization strength
   - Encourages selecting fewer features
   - Higher: Sparser models (faster inference)
   - Lower: More features used
   
6. learning_rate (1e-3 to 1e-1):
   - Initial learning rate for optimizer
   - Higher: Faster convergence, risk instability
   - Lower: Slower, more stable training
   - Typical: 2e-2
   
FOCAL LOSS PARAMETERS:

1. alpha (0.1 - 0.9):
   - Weighting factor for positive class
   - Default: 0.25 (empirically good for many datasets)
   - Relates to class imbalance ratio
   - alpha = 1 - pos_class_rate
   
2. gamma (1.0 - 5.0):
   - Focusing parameter
   - Higher: Focus more on hard examples
   - gamma = 0: Reduces to cross-entropy
   - gamma = 2: Recommended in literature
"""

# ============================================================================
# TRAINING PIPELINE FLOW
# ============================================================================

"""
STAGE 1: PREPROCESSING
Input: Raw CSV files
Process:
  1. Load application_train.csv
  2. Separate target (TARGET) from features
  3. Handle missing values
     - Median imputation for numeric
     - Mode imputation for categorical
  4. Encode categorical features
     - Label encoding for TabNet
  5. Scale features (StandardScaler)
  6. Stratified train-test split (80-20)
Output: X_train, X_test, y_train, y_test (all numpy arrays)

STAGE 2: BASELINE MODEL
Input: Preprocessed data
Model: RandomForestClassifier
Process:
  1. Initialize with balanced class weights
  2. Fit on training data
  3. Predict on test data
  4. Compute metrics (ROC-AUC, PR-AUC, F1, etc.)
  5. Plot evaluation curves
Output: Baseline metrics, evaluation plots

STAGE 3: HYPERPARAMETER OPTIMIZATION (OPTIONAL)
Input: Training data
Process:
  1. Define search space (8+ hyperparameters)
  2. Initialize Optuna study (TPE sampler)
  3. For each trial:
     a. Sample hyperparameters
     b. Split into 3-5 CV folds
     c. Train TabNet on each fold
     d. Evaluate on validation set
     e. Prune if score below median
  4. Track best trial and parameters
Output: Best hyperparameters, optimization report

STAGE 4: TABNET TRAINING
Input: Preprocessed data, optional best params from HPO
Process:
  1. Initialize TabNet with parameters
  2. Train on full training set
     - Validation split: 20% of train data
     - Early stopping: 15 epochs patience
     - Batch size: 256
  3. Save trained model
Output: Trained TabNet model

STAGE 5: EVALUATION & ENSEMBLE
Input: All trained models, test data
Process:
  1. Evaluate RandomForest on test set
  2. Evaluate TabNet on test set
  3. Create ensemble (weighted average)
  4. Evaluate ensemble
  5. Find optimal classification threshold
  6. Compile final comparison table
Output: 
  - Model comparison results
  - Evaluation plots (ROC, PR, CM for each model)
  - Final metrics summary
"""

# ============================================================================
# PERFORMANCE TUNING GUIDE
# ============================================================================

"""
1. MEMORY ISSUES (Out of Memory):

   Problem: CUDA out of memory or RAM overflow
   
   Solutions:
   a) Reduce batch size:
      TRAINING_CONFIG['batch_size'] = 128  # from 256
   
   b) Reduce number of HPO trials:
      OPTUNA_CONFIG['n_trials'] = 10  # from 30
   
   c) Reduce CV folds:
      evaluator.cv_folds = 3  # instead of 5
   
   d) Use CPU instead of GPU (if CUDA issues):
      - Pipeline automatically falls back to CPU
   
2. SLOW TRAINING:

   Problem: Training takes too long
   
   Solutions:
   a) Reduce model complexity:
      n_d = 32  # from 64
      n_steps = 2  # from 3
   
   b) Fewer training epochs:
      TRAINING_CONFIG['epochs'] = 50  # from 100
   
   c) Larger batch size:
      batch_size = 512  # from 256
   
   d) Skip HPO:
      Use default hyperparameters instead

3. POOR MODEL PERFORMANCE:

   Problem: Metrics (ROC-AUC, F1) are low
   
   Solutions:
   a) Increase model complexity:
      n_d = 128, n_steps = 5
   
   b) More HPO trials with longer timeout:
      n_trials = 100, timeout = 7200
   
   c) Adjust Focal Loss parameters:
      alpha = 0.3, gamma = 3.0
   
   d) Try higher Lambda_sparse:
      Encourages feature selection

4. OVERFITTING:

   Problem: Train metrics good, test metrics poor
   
   Solutions:
   a) Increase lambda_sparse:
      Encourages regularization
   
   b) Increase dropout/L2 regularization
   
   c) Reduce model complexity
   
   d) Ensemble with randomforest
      Typically more robust
"""

# ============================================================================
# EXTENDING THE PIPELINE
# ============================================================================

"""
1. ADD CUSTOM LOSS FUNCTION:

   In src/loss_functions.py:
   
   class CustomLoss(nn.Module):
       def __init__(self, ...):
           super().__init__()
           ...
       
       def forward(self, inputs, targets):
           # Your loss computation
           return loss

2. ADD NEW MODEL:

   In src/models.py:
   
   class MyModel:
       def train(self, X_train, y_train):
           pass
       
       def predict_proba(self, X):
           pass
       
       def predict(self, X):
           pass

3. ADD NEW EVALUATION METRIC:

   In src/evaluation.py compute_metrics():
   
   my_metric = my_metric_function(y_true, y_pred)
   metrics['My_Metric'] = my_metric

4. CHANGE HPO OBJECTIVE:

   In src/hpo.py objective():
   
   if metric == 'custom':
       score = custom_scoring_function(y_val, y_pred_proba)

5. USE IN PRODUCTION:

   # Load trained model
   tabnet_model = TabNetModel(...)
   tabnet_model.load(Path('models/tabnet_optimized.pkl'))
   
   # Make predictions on new data
   new_data_processed = preprocess_new_data(new_data)
   predictions = tabnet_model.predict_proba(new_data_processed)
   
   # Apply business logic
   if predictions[i] > optimal_threshold:
       flag_as_high_risk(loan_id)
"""

# ============================================================================
# BEST PRACTICES & TIPS
# ============================================================================

"""
1. ALWAYS RUN TEST_PIPELINE.PY FIRST
   - Verifies all dependencies are installed
   - Checks data files are accessible
   - Catches configuration issues early

2. START WITH QUICK_START.PY
   - Smaller scale examples
   - Faster feedback loop
   - Understand each component separately

3. MONITOR LOGS
   - Check logs/pipeline.log for detailed execution info
   - Helps diagnose issues and understand training progress

4. SAVE INTERMEDIATE RESULTS
   - Save preprocessed data for faster iteration
   - Save model checkpoints during training
   - Save HPO study for resume capability

5. VERSION YOUR EXPERIMENTS
   - Keep track of parameters that worked well
   - Record test metrics for each experiment
   - Use git to version control configs

6. USE ENSEMBLE FOR STABILITY
   - Often outperforms individual models
   - More robust to hyperparameter variations
   - Recommended for production deployments

7. UNDERSTAND YOUR BUSINESS METRICS
   - ROC-AUC optimizes ranking quality
   - F1-Score optimizes balance
   - PR-AUC best for rare event detection
   - Choose metric aligned with business goal

8. HANDLE CLASS IMBALANCE PROPERLY
   - Never use accuracy as metric
   - Use stratified splits consistently
   - Consider class weights in all models
   - Focal Loss crucial for extreme imbalance
"""

# ============================================================================
print(__doc__)
