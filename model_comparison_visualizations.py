"""Create rich visual model comparison outputs from saved pipeline run artifacts.

This script reads one or more run directories under `outputs/` and creates:
- evaluation image grids for ROC, PR, and confusion matrices
- threshold comparison visuals from saved optimal threshold files
- grouped metric bar charts when a `model_comparison.csv` file is available
- a compact dashboard summarizing the available comparisons

Usage:
    python model_comparison_visualizations.py --runs outputs/run_20260407_054100 outputs/run_20260407_055441 --output outputs/visualizations
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

METRIC_ORDER = [
    "ROC_AUC",
    "PR_AUC",
    "F1_Score",
    "Precision",
    "Recall",
    "Accuracy",
]


@dataclass
class RunEvaluation:
    run_dir: Path
    name: str
    evaluation_dir: Path
    thresholds: Dict[str, float]
    metrics_df: Optional[pd.DataFrame]
    image_paths: Dict[str, Path]
    log_metrics: Dict[str, Dict[str, float]]

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunEvaluation":
        run_dir = run_dir.expanduser().resolve()
        evaluation_dir = run_dir / "evaluation"
        thresholds = cls._load_thresholds(run_dir)
        metrics_df = cls._load_metrics_df(run_dir)
        image_paths = cls._load_evaluation_images(evaluation_dir)
        log_metrics = cls._parse_logs(run_dir)
        return cls(
            run_dir=run_dir,
            name=run_dir.name,
            evaluation_dir=evaluation_dir,
            thresholds=thresholds,
            metrics_df=metrics_df,
            image_paths=image_paths,
            log_metrics=log_metrics,
        )

    @staticmethod
    def _load_thresholds(run_dir: Path) -> Dict[str, float]:
        thresholds_file = run_dir / "optimal_thresholds.txt"
        result: Dict[str, float] = {}
        if thresholds_file.exists():
            with open(thresholds_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.match(r"^([A-Za-z ]+):\s*([0-9.]+)$", line.strip())
                    if match:
                        key = match.group(1).strip()
                        value = float(match.group(2))
                        result[key] = value
        return result

    @staticmethod
    def _load_metrics_df(run_dir: Path) -> Optional[pd.DataFrame]:
        candidate = run_dir / "model_comparison.csv"
        if candidate.exists():
            df = pd.read_csv(candidate)
            return df
        return None

    @staticmethod
    def _load_evaluation_images(evaluation_dir: Path) -> Dict[str, Path]:
        images: Dict[str, Path] = {}
        if evaluation_dir.exists():
            for image_path in evaluation_dir.glob("*.png"):
                images[image_path.name] = image_path
        return images

    @staticmethod
    def _parse_logs(run_dir: Path) -> Dict[str, Dict[str, float]]:
        log_metrics: Dict[str, Dict[str, float]] = {}
        log_dir = run_dir / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                lines = text.splitlines()
                current_model = None
                for line in lines:
                    model_match = re.search(r"Evaluating ([A-Za-z0-9_ ()+-]+)", line)
                    if model_match:
                        current_model = model_match.group(1).strip()
                        log_metrics[current_model] = {}
                    metric_match = re.search(r"([A-Z][A-Za-z_ ]+):\s*([0-9.]+)", line)
                    if current_model and metric_match:
                        key = metric_match.group(1).strip().replace(" ", "_")
                        try:
                            value = float(metric_match.group(2))
                            log_metrics[current_model][key] = value
                        except ValueError:
                            continue
        return log_metrics


def plot_image_grid(run: RunEvaluation, output_dir: Path) -> Path:
    images_to_show = [
        ("roc_rf.png", "RF ROC"),
        ("roc_tabnet.png", "TabNet ROC"),
        ("roc_ensemble.png", "Ensemble ROC"),
        ("pr_rf.png", "RF PR"),
        ("pr_tabnet.png", "TabNet PR"),
        ("pr_ensemble.png", "Ensemble PR"),
        ("cm_rf_optimal.png", "RF CM Optimal"),
        ("cm_tabnet_optimal.png", "TabNet CM Optimal"),
        ("cm_ensemble_optimal.png", "Ensemble CM Optimal"),
    ]

    found_images = [(title, run.image_paths[name]) for name, title in images_to_show if name in run.image_paths]
    if not found_images:
        raise FileNotFoundError(f"No evaluation PNG images found in {run.evaluation_dir}")

    n = len(found_images)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for idx, (title, path) in enumerate(found_images):
        img = mpimg.imread(path)
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"Evaluation Image Grid: {run.name}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"evaluation_image_grid_{run.name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_threshold_comparison(runs: List[RunEvaluation], output_dir: Path) -> Optional[Path]:
    model_names = []
    threshold_values: Dict[str, List[float]] = {}
    for run in runs:
        if not run.thresholds:
            continue
        for model_name, value in run.thresholds.items():
            threshold_values.setdefault(model_name, [])
            threshold_values[model_name].append(value)
    if not threshold_values:
        return None

    x = np.arange(len(runs))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (model_name, values) in enumerate(sorted(threshold_values.items())):
        padded = values + [np.nan] * (len(runs) - len(values))
        ax.bar(x + idx * width, padded, width, label=model_name)

    ax.set_xticks(x + width * (len(threshold_values) - 1) / 2)
    ax.set_xticklabels([run.name for run in runs], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Optimal Classification Threshold")
    ax.set_title("Optimal Threshold Comparison Across Runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "threshold_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_grouped_metrics(runs: List[RunEvaluation], output_dir: Path) -> Optional[Path]:
    data_frames: List[pd.DataFrame] = []
    for run in runs:
        if run.metrics_df is None:
            continue
        df = run.metrics_df.copy()
        df["Run"] = run.name
        data_frames.append(df)

    if not data_frames:
        return None

    combined = pd.concat(data_frames, ignore_index=True)
    models = combined["Model"].unique().tolist()
    metrics = [m for m in METRIC_ORDER if m in combined.columns]
    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    total_width = 0.8
    width = total_width / len(runs)

    for idx, run in enumerate(runs):
        if run.metrics_df is None:
            continue
        values = [run.metrics_df[run.metrics_df["Model"] == model][metrics].mean().mean() for model in models]
        ax.bar(x + idx * width, values, width, label=run.name)

    ax.set_xticks(x + width * (len(runs) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Average Metric Value")
    ax.set_title("Average Model Performance Across Runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "grouped_metric_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_radar_metrics(run: RunEvaluation, output_dir: Path) -> Optional[Path]:
    if run.metrics_df is None:
        return None

    metrics = [m for m in METRIC_ORDER if m in run.metrics_df.columns]
    if not metrics:
        return None

    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model_name in run.metrics_df["Model"].unique():
        values = run.metrics_df.loc[run.metrics_df["Model"] == model_name, metrics].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Radar Comparison: {run.name}")
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    output_path = output_dir / f"radar_metrics_{run.name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_dashboard(runs: List[RunEvaluation], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(18, 12))
    rows = 3
    cols = 2

    # Top-left: threshold comparison
    threshold_path = plot_threshold_comparison(runs, output_dir)
    ax1 = fig.add_subplot(rows, cols, 1)
    if threshold_path and threshold_path.exists():
        img = mpimg.imread(threshold_path)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Threshold Comparison")
    else:
        ax1.text(0.5, 0.5, "No threshold data available", ha="center", va="center")
        ax1.axis("off")

    # Top-right: run image grid for first run
    if runs:
        image_path = plot_image_grid(runs[0], output_dir)
        ax2 = fig.add_subplot(rows, cols, 2)
        img = mpimg.imread(image_path)
        ax2.imshow(img)
        ax2.axis("off")
        ax2.set_title(f"Evaluation Grid: {runs[0].name}")
    else:
        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.text(0.5, 0.5, "No run available", ha="center", va="center")
        ax2.axis("off")

    # Bottom-left: grouped metric comparison
    grouped_path = plot_grouped_metrics(runs, output_dir)
    ax3 = fig.add_subplot(rows, cols, 3)
    if grouped_path and grouped_path.exists():
        img = mpimg.imread(grouped_path)
        ax3.imshow(img)
        ax3.axis("off")
        ax3.set_title("Grouped Metric Comparison")
    else:
        ax3.text(0.5, 0.5, "No metric CSV data available", ha="center", va="center")
        ax3.axis("off")

    # Bottom-right: radar chart for first run if available
    ax4 = fig.add_subplot(rows, cols, 4, polar=True)
    if runs and runs[0].metrics_df is not None:
        run = runs[0]
        metrics = [m for m in METRIC_ORDER if m in run.metrics_df.columns]
        if metrics:
            labels = metrics
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            for model_name in run.metrics_df["Model"].unique():
                values = run.metrics_df.loc[run.metrics_df["Model"] == model_name, metrics].mean().tolist()
                values += values[:1]
                ax4.plot(angles, values, label=model_name)
                ax4.fill(angles, values, alpha=0.12)
            ax4.set_thetagrids(np.degrees(angles[:-1]), labels)
            ax4.set_title(f"Radar Comparison: {run.name}")
            ax4.set_ylim(0, 1)
            ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        else:
            ax4.text(0.5, 0.5, "No metrics available", ha="center", va="center")
            ax4.axis("off")
    else:
        ax4.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax4.axis("off")

    fig.suptitle("Model Comparison Dashboard", fontsize=20)
    output_path = output_dir / "model_comparison_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build comprehensive model comparison visualizations from saved runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run directories to compare (e.g. outputs/run_20260407_054100)",
    )
    parser.add_argument(
        "--output",
        default="outputs/model_comparison_visualizations",
        help="Directory to save generated comparison images",
    )

    args = parser.parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_evaluations: List[RunEvaluation] = []
    for run_arg in args.runs:
        run_path = Path(run_arg).expanduser().resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")
        run_evaluations.append(RunEvaluation.from_run_dir(run_path))

    created_images: List[Path] = []
    for run in run_evaluations:
        try:
            created_images.append(plot_image_grid(run, output_dir))
        except FileNotFoundError:
            pass
        if run.thresholds:
            created_images.append(plot_threshold_comparison([run], output_dir))
        if run.metrics_df is not None:
            created_images.append(plot_radar_metrics(run, output_dir))

    comparison_path = plot_threshold_comparison(run_evaluations, output_dir)
    if comparison_path:
        created_images.append(comparison_path)

    grouped_path = plot_grouped_metrics(run_evaluations, output_dir)
    if grouped_path:
        created_images.append(grouped_path)

    dashboard_path = build_dashboard(run_evaluations, output_dir)
    created_images.append(dashboard_path)

    print("Saved comparison visualizations:")
    for path in created_images:
        if path is not None:
            print(f"- {path}")


if __name__ == "__main__":
    main()
