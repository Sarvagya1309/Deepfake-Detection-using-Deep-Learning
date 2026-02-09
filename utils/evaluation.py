"""
Evaluation utilities for deepfake detection.

Includes:
- Frame-level metric computation
- Threshold tuning based on validation data
- Confusion matrix and ROC plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)


def tune_frame_threshold(probs, labels, min_t=0.05, max_t=0.95, step=0.01):
    """
    Finds the optimal decision threshold that maximizes F1-score.

    Args:
        probs (np.ndarray): predicted probabilities for fake class
        labels (np.ndarray): ground truth labels
    """
    best_t, best_f1 = 0.5, -1.0

    for t in np.arange(min_t, max_t + step, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def compute_frame_metrics(labels, probs, threshold=0.5):
    """
    Computes frame-level evaluation metrics.

    Returns:
        dict containing accuracy, f1, auc, confusion matrix, and report
    """
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(
            labels, preds, target_names=["Real", "Fake"]
        )
    }

    return metrics


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots a confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(labels, probs, title="ROC Curve"):
    """
    Plots ROC curve and computes AUC.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc

