"""
Test script to verify the pipeline components are working correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        logger.info("  ✓ PyTorch-TabNet")
    except ImportError as e:
        logger.error(f"  ✗ PyTorch-TabNet: {e}")
        return False
    
    try:
        import optuna
        logger.info(f"  ✓ Optuna {optuna.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ Optuna: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        logger.info("  ✓ Scikit-learn")
    except ImportError as e:
        logger.error(f"  ✗ Scikit-learn: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ Pandas: {e}")
        return False
    
    try:
        from src.preprocessing import DataPreprocessor
        logger.info("  ✓ Project modules")
    except ImportError as e:
        logger.error(f"  ✗ Project modules: {e}")
        return False
    
    logger.info("✓ All imports successful!")
    return True


def test_config():
    """Test that configuration is accessible."""
    logger.info("\\nTesting configuration...")
    
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR,
            MODEL_CONFIG, FOCAL_LOSS_CONFIG, OPTUNA_CONFIG
        )
        
        logger.info(f"  ✓ Project root: {PROJECT_ROOT}")
        logger.info(f"  ✓ Data directory: {DATA_DIR}")
        logger.info(f"  ✓ Output directory: {OUTPUT_DIR}")
        logger.info(f"  ✓ TabNet config: n_d={MODEL_CONFIG['tabnet']['n_d']}, "
                   f"n_a={MODEL_CONFIG['tabnet']['n_a']}")
        logger.info(f"  ✓ Focal Loss config: alpha={FOCAL_LOSS_CONFIG['alpha']}, "
                   f"gamma={FOCAL_LOSS_CONFIG['gamma']}")
        logger.info("✓ Configuration loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"  ✗ Configuration error: {e}")
        return False


def test_data_access():
    """Test that data files can be accessed."""
    logger.info("\\nTesting data access...")
    
    try:
        from config import DATA_DIR, DATA_CONFIG
        
        for name, filename in DATA_CONFIG.items():
            filepath = DATA_DIR / filename
            if filepath.exists():
                logger.info(f"  ✓ {filename} ({filepath.stat().st_size / (1024*1024):.2f} MB)")
            else:
                logger.warning(f"  ⚠ {filename} not found at {filepath}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Data access error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing module."""
    logger.info("\\nTesting preprocessing module...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.utils import load_data, get_missing_stats, compute_class_weights
        from config import DATA_DIR
        
        # Load data
        df = load_data(DATA_DIR, "application_train.csv")
        logger.info(f"  ✓ Loaded data shape: {df.shape}")
        
        # Check missing values
        missing_stats = get_missing_stats(df.drop('TARGET', axis=1))
        logger.info(f"  ✓ Found missing value stats for {len(missing_stats)} columns")
        
        # Test class weights
        y = df['TARGET']
        class_weights = compute_class_weights(y)
        logger.info(f"  ✓ Computed class weights: {class_weights}")
        
        logger.info("✓ Preprocessing module working!")
        return True
    except Exception as e:
        logger.error(f"  ✗ Preprocessing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_models():
    """Test model implementations."""
    logger.info("\\nTesting models...")
    
    try:
        from src.models import BaselineRandomForest, TabNetModel
        import numpy as np
        
        # Create dummy data
        n_samples, n_features = 100, 50
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        
        # Test RandomForest
        rf = BaselineRandomForest(n_estimators=10, max_depth=5, n_jobs=-1)
        rf.train(X[:80], y[:80])
        pred_proba = rf.predict_proba(X[80:])
        logger.info(f"  ✓ RandomForest predictions shape: {pred_proba.shape}")
        
        # Test TabNet
        tabnet = TabNetModel(n_features=n_features, n_d=32, n_a=32, n_steps=2)
        tabnet.train(
            X[:80], y[:80],
            X_val=X[80:], y_val=y[80:],
            epochs=2,
            batch_size=32,
            learning_rate=0.01,
        )
        pred_proba = tabnet.predict_proba(X[80:])
        logger.info(f"  ✓ TabNet predictions shape: {pred_proba.shape}")
        
        logger.info("✓ Models working!")
        return True
    except Exception as e:
        logger.error(f"  ✗ Models error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_evaluation():
    """Test evaluation metrics."""
    logger.info("\\nTesting evaluation metrics...")
    
    try:
        from src.evaluation import MetricsEvaluator
        import numpy as np
        
        evaluator = MetricsEvaluator()
        
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6])
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Compute metrics
        metrics = evaluator.compute_metrics(y_true, y_pred_proba, y_pred)
        logger.info(f"  ✓ Computed metrics: {list(metrics.keys())}")
        
        # Find optimal threshold
        threshold, value = evaluator.find_optimal_threshold(y_true, y_pred_proba, metric='f1')
        logger.info(f"  ✓ Optimal F1 threshold: {threshold:.3f} (value: {value:.3f})")
        
        logger.info("✓ Evaluation metrics working!")
        return True
    except Exception as e:
        logger.error(f"  ✗ Evaluation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_loss_functions():
    """Test custom loss functions."""
    logger.info("\\nTesting loss functions...")
    
    try:
        import torch
        from src.loss_functions import FocalLoss, WeightedBCELoss
        
        # Create dummy data
        batch_size = 32
        logits = torch.randn(batch_size, 1)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Test Focal Loss
        focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss_fn(logits, targets)
        logger.info(f"  ✓ Focal Loss: {loss.item():.4f}")
        
        # Test Weighted BCE Loss
        weighted_bce_fn = WeightedBCELoss(pos_weight=2.0)
        loss = weighted_bce_fn(logits, targets)
        logger.info(f"  ✓ Weighted BCE Loss: {loss.item():.4f}")
        
        logger.info("✓ Loss functions working!")
        return True
    except Exception as e:
        logger.error(f"  ✗ Loss functions error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_hpo():
    """Test HPO module."""
    logger.info("\\nTesting HPO module...")
    
    try:
        from src.hpo import TabNetOptimizer
        import numpy as np
        
        # Create dummy data
        n_samples, n_features = 100, 50
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        
        # Initialize optimizer
        optimizer = TabNetOptimizer(n_features=n_features)
        logger.info(f"  ✓ Optimizer initialized")
        
        # Note: Full HPO would take too long for tests
        logger.info("  ⚠ Skipping full HPO (would take too long for tests)")
        
        logger.info("✓ HPO module ready!")
        return True
    except Exception as e:
        logger.error(f"  ✗ HPO error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """Run all tests."""
    print("\\n" + "="*80)
    print("MICRO-LOAN DEFAULT RISK PIPELINE - COMPONENT TEST SUITE")
    print("="*80 + "\\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Access", test_data_access),
        ("Preprocessing", test_preprocessing),
        ("Models", test_models),
        ("Evaluation", test_evaluation),
        ("Loss Functions", test_loss_functions),
        ("HPO Module", test_hpo),
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print(f"\\nResult: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\\n✓ All tests passed! Pipeline is ready to use.")
        return True
    else:
        print(f"\\n✗ {total_count - passed_count} test(s) failed. Please fix issues before running pipeline.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
