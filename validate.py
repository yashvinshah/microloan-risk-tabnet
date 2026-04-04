#!/usr/bin/env python
"""
Quick import validation script to ensure all modules are syntactically correct.
Run this to verify the pipeline is ready before running the full pipeline.
"""

import sys

def validate_imports():
    """Validate all module imports."""
    print("\n" + "="*80)
    print("QUICK VALIDATION - CHECKING ALL IMPORTS")
    print("="*80 + "\n")
    
    validation_items = [
        ("Dependencies", [
            ("pandas", "import pandas as pd"),
            ("numpy", "import numpy as np"),
            ("torch", "import torch"),
            ("sklearn", "from sklearn.ensemble import RandomForestClassifier"),
            ("optuna", "import optuna"),
            ("pytorch_tabnet", "from pytorch_tabnet.tab_model import TabNetClassifier"),
        ]),
        ("Project Config", [
            ("config", "import config"),
        ]),
        ("Project Modules", [
            ("utils", "from src import utils"),
            ("preprocessing", "from src import preprocessing"),
            ("loss_functions", "from src import loss_functions"),
            ("models", "from src import models"),
            ("evaluation", "from src import evaluation"),
            ("hpo", "from src import hpo"),
        ]),
        ("Project Classes", [
            ("DataPreprocessor", "from src.preprocessing import DataPreprocessor"),
            ("BaselineRandomForest", "from src.models import BaselineRandomForest"),
            ("TabNetModel", "from src.models import TabNetModel"),
            ("EnsembleModel", "from src.models import EnsembleModel"),
            ("FocalLoss", "from src.loss_functions import FocalLoss"),
            ("MetricsEvaluator", "from src.evaluation import MetricsEvaluator"),
            ("TabNetOptimizer", "from src.hpo import TabNetOptimizer"),
        ]),
    ]
    
    total_items = 0
    passed_items = 0
    failed_items = []
    
    for category, items in validation_items:
        print(f"\\n{category}:")
        for item_name, import_stmt in items:
            total_items += 1
            try:
                exec(import_stmt)
                print(f"  ✓ {item_name}")
                passed_items += 1
            except Exception as e:
                print(f"  ✗ {item_name}: {str(e)[:60]}")
                failed_items.append((item_name, str(e)))
    
    # Summary
    print("\\n" + "="*80)
    print(f"VALIDATION SUMMARY: {passed_items}/{total_items} items passed")
    print("="*80)
    
    if failed_items:
        print("\\nFailed items:")
        for item, error in failed_items:
            print(f"  • {item}: {error[:80]}")
        return False
    else:
        print("\\n✓ All imports validated successfully!")
        print("✓ Pipeline is ready to use!")
        return True


if __name__ == "__main__":
    success = validate_imports()
    sys.exit(0 if success else 1)
