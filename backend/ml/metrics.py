"""Metric computation + saving. Pure functions, no model or web dependencies."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no interactive display (works on servers/Colab)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)


def compute_metrics(y_true, y_pred, labels: list[str], label_ids=None) -> dict:
    """Headline numbers that answer the research question.

    - accuracy: overall fraction correct.
    - macro_f1: average F1 across classes, treating each class equally. This is
      the honest metric under class imbalance — a model that only predicts the
      majority class scores high accuracy but low macro-F1.
    - per_class: precision/recall/F1 for every emotion (great results-table).

    `label_ids` restricts scoring to a subset of classes (their integer ids). This
    is needed for external benchmarks that only cover some Ekman emotions — macro-F1
    is then averaged over the classes the dataset actually uses, not all seven.
    """
    if label_ids is None:
        label_ids = list(range(len(labels)))
    names = [labels[i] for i in label_ids]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)), 4
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=label_ids, target_names=names,
            output_dict=True, zero_division=0,
        ),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {path}")


def save_confusion_matrix(y_true, y_pred, labels: list[str], path: str | Path, label_ids=None) -> None:
    """Confusion matrix heatmap: rows = true emotion, cols = predicted. The
    diagonal is correct predictions; off-diagonal shows which emotions the model
    confuses (e.g. fear mistaken for surprise). Strong dissertation figure.

    `label_ids` restricts the matrix to a subset of classes (for external
    benchmarks that don't cover all seven Ekman emotions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if label_ids is None:
        label_ids = list(range(len(labels)))
    labels = [labels[i] for i in label_ids]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MoodLens — Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")
