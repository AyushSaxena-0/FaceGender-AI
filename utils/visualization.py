"""
Visualization utility module for evaluation metrics, training history, and Grad-CAM maps.
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import cv2

plt.style.use("dark_background")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots training and validation loss and accuracy curves over epochs.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path (str, optional): Path to save the plot image.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#4EA8DE", linewidth=2)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#F72585", linewidth=2, linestyle="--")
    ax1.set_title("Loss History", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history.get("train_acc", []), label="Train Accuracy", color="#4CC9F0", linewidth=2)
    ax2.plot(epochs, history.get("val_acc", []), label="Val Accuracy", color="#7209B7", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy History", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Masculine", "Feminine"],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): Class labels.
        save_path (str, optional): Save path.

    Returns:
        plt.Figure: Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="magma",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar=True
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Actual Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, float]:
    """
    Plots Receiver Operating Characteristic (ROC) curve and returns AUC score.

    Args:
        y_true (np.ndarray): Binary true labels.
        y_scores (np.ndarray): Predicted probability scores for positive class.
        save_path (str, optional): Save path.

    Returns:
        Tuple[plt.Figure, float]: Figure object and calculated AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4CC9F0", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#7209B7", lw=2, label="Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def overlay_gradcam(
    original_img_rgb: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap over the original RGB image.

    Args:
        original_img_rgb (np.ndarray): Original RGB image (H, W, 3) in uint8 [0, 255].
        cam_heatmap (np.ndarray): Grad-CAM map in range [0, 1] (H, W).
        alpha (float): Transparency factor for heatmap overlay.

    Returns:
        np.ndarray: Blended RGB image.
    """
    h, w, _ = original_img_rgb.shape
    heatmap_resized = cv2.resize(cam_heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended
