"""
Metrics calculation module for classification performance evaluation.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str] = ["masculine", "feminine"]
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics.

    Args:
        y_true (np.ndarray): True class labels (0 or 1).
        y_pred (np.ndarray): Predicted class labels (0 or 1).
        y_scores (np.ndarray): Predicted probability scores for positive class (1).
        class_names (List[str]): List of class names.

    Returns:
        Dict[str, Any]: Dictionary containing calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_scores)
    except Exception:
        auc_score = 0.5

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc_score),
        "confusion_matrix": cm.tolist(),
        "classification_report_str": report_str,
        "classification_report_dict": report_dict
    }
