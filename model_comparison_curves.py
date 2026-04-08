"""Build combined model-comparison charts for RandomForest, TabNet, and Ensemble.

This script loads saved models from a run directory (if available) or trains new ones,
then generates:
- combined ROC curve for all models
- combined Precision-Recall curve for all models
- grouped metric bar chart for key performance metrics
- threshold sweep plots for F1, precision, and recall
- confusion matrix comparison at chosen thresholds
- optional feature importance comparison for RF and TabNet

Usage:
    # Load from existing run (fastest)
    python model_comparison_curves.py --run-dir outputs/run_20260407_054100 --output outputs/model_comparison_curves

    # Train new models (slower)
    python model_comparison_curves.py --output outputs/model_comparison_curves
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from config import DATA_DIR, MODEL_CONFIG, TRAINING_CONFIG
from src.models import BaselineRandomForest, TabNetModel, EnsembleModel
from src.preprocessing import preprocess_data

sns.set_style("whitegrid")

MODEL_NAMES = ["Random Forest", "TabNet", "Ensemble"]
METRIC_COLUMNS = ["ROC_AUC", "PR_AUC", "F1_Score", "Precision", "Recall", "Accuracy"]


def load_run_thresholds(run_dirs: List[Path]) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for run_dir in run_dirs:
        run_dir = run_dir.expanduser().resolve()
        thresholds_file = run_dir / "optimal_thresholds.txt"
        if not thresholds_file.exists():
            continue
        run_index = run_dir.name
        thresholds[run_index] = {}
        with open(thresholds_file, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^([A-Za-z ]+):\s*([0-9.]+)$", line.strip())
                if match:
                    key = match.group(1).strip()
                    thresholds[run_index][key] = float(match.group(2))
    return thresholds


def load_models_from_run(run_dir: Path, data_dir: Path):
    """Load saved models from a run directory if they exist."""
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return None, None, None, None

    rf_path = models_dir / "random_forest_baseline.pkl"
    tabnet_path = models_dir / "tabnet_optimized.pkl.zip"  # pytorch_tabnet saves as .pkl.zip

    rf = None
    tabnet = None

    if rf_path.exists():
        print(f"Loading Random Forest from {rf_path}")
        rf = BaselineRandomForest()
        rf.load(rf_path)

    if tabnet_path.exists():
        print(f"Loading TabNet from {tabnet_path}")
        # Need to know n_features, get from preprocessing
        data = preprocess_data(data_dir, test_size=0.2, random_state=42)
        n_features = data["X_train"].shape[1]
        tabnet = TabNetModel(n_features=n_features)
        tabnet.load(tabnet_path)

    ensemble = None
    if rf is not None and tabnet is not None:
        ensemble = EnsembleModel(rf, tabnet)

    return data, rf, tabnet, ensemble


def train_models(data_dir: Path, epochs: int, batch_size: int, early_stopping_patience: int):
    print("Preprocessing data...")
    data = preprocess_data(data_dir, test_size=0.2, random_state=42)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Training Random Forest baseline...")
    rf = BaselineRandomForest(**MODEL_CONFIG["random_forest"])
    rf.train(X_train, y_train)

    print("Training TabNet model...")
    tabnet = TabNetModel(
        n_features=X_train.shape[1],
        n_d=MODEL_CONFIG["tabnet"].get("n_d", 64),
        n_a=MODEL_CONFIG["tabnet"].get("n_a", 64),
        n_steps=MODEL_CONFIG["tabnet"].get("n_steps", 3),
        gamma=MODEL_CONFIG["tabnet"].get("gamma", 1.5),
        lambda_sparse=MODEL_CONFIG["tabnet"].get("lambda_sparse", 1e-4),
        mask_type=MODEL_CONFIG["tabnet"].get("mask_type", "sparsemax"),
    )
    # Use a small validation split from the test set for early stopping
    val_count = max(1, len(X_test) // 5)
    X_val = X_test[:val_count]
    y_val = y_test[:val_count]
    tabnet.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
    )

    ensemble = EnsembleModel(rf, tabnet)
    return data, rf, tabnet, ensemble


def evaluate_model(name: str, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "Model": name,
        "ROC_AUC": auc(fpr, tpr),
        "PR_AUC": auc(recall_vals, precision_vals),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Threshold": threshold,
        "FPR": fpr,
        "TPR": tpr,
        "Precision_Values": precision_vals,
        "Recall_Values": recall_vals,
        "Y_Pred_Proba": y_pred_proba,
    }
    return metrics


def evaluate_models(data, rf, tabnet, ensemble, thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    X_test = data["X_test"]
    y_test = data["y_test"]

    model_objects = {
        "Random Forest": rf,
        "TabNet": tabnet,
        "Ensemble": ensemble,
    }

    results = []
    for name, model in model_objects.items():
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model.predict(X_test).astype(float)

        results.append(evaluate_model(name, y_test, y_pred_proba, threshold=0.5))

    return pd.DataFrame(results)


def plot_combined_roc(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    for _, row in results.iterrows():
        plt.plot(row["FPR"], row["TPR"], lw=2, label=f"{row['Model']} (AUC={row['ROC_AUC']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_pr(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    for _, row in results.iterrows():
        plt.plot(row["Recall_Values"], row["Precision_Values"], lw=2, label=f"{row['Model']} (AUC={row['PR_AUC']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_bars(results: pd.DataFrame, output_path: Path) -> None:
    plot_data = results.set_index("Model")[METRIC_COLUMNS]
    plot_data = plot_data.sort_values("ROC_AUC", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot(kind="bar", ax=ax, rot=30)
    ax.set_ylabel("Metric value")
    ax.set_title("Model Performance Metrics")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_threshold_sweep(data, rf, tabnet, ensemble, output_path: Path) -> None:
    X_test = data["X_test"]
    y_test = data["y_test"]
    thresholds = np.linspace(0.0, 1.0, 101)

    records = []
    for name, model in [("Random Forest", rf), ("TabNet", tabnet), ("Ensemble", ensemble)]:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model.predict(X_test).astype(float)

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            records.append(
                {
                    "Model": name,
                    "Threshold": threshold,
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                }
            )

    threshold_df = pd.DataFrame(records)

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    for metric, ax in zip(["F1", "Precision", "Recall"], axes):
        sns.lineplot(data=threshold_df, x="Threshold", y=metric, hue="Model", ax=ax)
        ax.set_title(f"{metric} vs Threshold")
        ax.grid(alpha=0.3)
        if metric == "Recall":
            ax.set_xlabel("Threshold")
        else:
            ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrices(results: pd.DataFrame, data, output_path: Path, thresholds: Optional[Dict[str, float]] = None) -> None:
    X_test = data["X_test"]
    y_test = data["y_test"]
    models = {
        "Random Forest": results.loc[results["Model"] == "Random Forest", "Y_Pred_Proba"].iloc[0],
        "TabNet": results.loc[results["Model"] == "TabNet", "Y_Pred_Proba"].iloc[0],
        "Ensemble": results.loc[results["Model"] == "Ensemble", "Y_Pred_Proba"].iloc[0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, proba) in zip(axes, models.items()):
        threshold = 0.5
        if thresholds and name in thresholds:
            threshold = thresholds[name]
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{name} Confusion Matrix\nThreshold = {threshold:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(data, rf, tabnet, output_path: Path) -> None:
    feature_names = data["feature_names"]
    rf_importance = rf.get_feature_importance()
    tabnet_importance = tabnet.get_feature_importance()
    top_n = min(15, len(feature_names))

    rf_order = np.argsort(rf_importance)[-top_n:][::-1]
    tabnet_order = np.argsort(tabnet_importance)[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(x=rf_importance[rf_order], y=np.array(feature_names)[rf_order], ax=axes[0], palette="Blues_r")
    axes[0].set_title("Top Random Forest Feature Importances")
    axes[0].set_xlabel("Importance")

    sns.barplot(x=tabnet_importance[tabnet_order], y=np.array(feature_names)[tabnet_order], ax=axes[1], palette="Greens_r")
    axes[1].set_title("Top TabNet Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics_table(results: pd.DataFrame, output_path: Path) -> None:
    results[METRIC_COLUMNS + ["Threshold"]].to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined model comparison curves and charts.")
    parser.add_argument(
        "--output",
        default="outputs/model_comparison_curves",
        help="Output directory for generated comparison plots",
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory to load saved models from (avoids retraining)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing the raw home-credit-default-risk data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG.get("epochs", 60),
        help="Number of TabNet training epochs (only used if training)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG.get("batch_size", 256),
        help="TabNet batch size (only used if training)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TRAINING_CONFIG.get("early_stopping_patience", 7),
        help="Early stopping patience for TabNet training (only used if training)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = None
    rf = None
    tabnet = None
    ensemble = None

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        print(f"Attempting to load models from {run_dir}")
        data, rf, tabnet, ensemble = load_models_from_run(run_dir, Path(args.data_dir))
        if data is None:
            print("Could not load models from run directory, will train new ones.")

    if data is None or rf is None or tabnet is None:
        print("Training new models...")
        data, rf, tabnet, ensemble = train_models(
            Path(args.data_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
        )

    thresholds = None
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
        thresholds_by_run = load_run_thresholds(run_dirs)
        if thresholds_by_run:
            thresholds = next(iter(thresholds_by_run.values()))

    results = evaluate_models(data, rf, tabnet, ensemble, thresholds=thresholds)
    save_metrics_table(results, output_dir / "model_metric_summary.csv")

    plot_combined_roc(results, output_dir / "combined_roc_curve.png")
    plot_combined_pr(results, output_dir / "combined_pr_curve.png")
    plot_metric_bars(results, output_dir / "metric_bar_comparison.png")
    plot_threshold_sweep(data, rf, tabnet, ensemble, output_dir / "threshold_sweep_curves.png")
    plot_confusion_matrices(results, data, output_dir / "confusion_matrix_comparison.png", thresholds=thresholds)
    plot_feature_importance(data, rf, tabnet, output_dir / "feature_importance_comparison.png")

    print("Saved new model comparison visuals to:")
    for path in output_dir.glob("*.png"):
        print(f"- {path}")
    print(f"- {output_dir / 'model_metric_summary.csv'}")


if __name__ == "__main__":
    main()
