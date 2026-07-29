"""
Automatic Error Analysis Engine.
Saves misclassified validation/test images into outputs/misclassified/
and generates detailed error breakdown reports.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from torch.utils.data import DataLoader

from utils.logger import setup_logger
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from training.metrics import compute_metrics

logger = setup_logger("ErrorAnalysis")


def perform_error_analysis(
    trainer,
    val_loader: DataLoader,
    output_dir: str = "outputs/misclassified",
    class_names: List[str] = ["masculine", "feminine"]
) -> Dict[str, Any]:
    """
    Performs error analysis on validation/test set images, identifies misclassified images,
    saves annotated image crops to output_dir, and generates confusion matrix plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.eval()

    misclassified_records = []
    y_true_list = []
    y_pred_list = []
    y_score_list = []

    ds = val_loader.dataset
    device = trainer.device

    logger.info(f"--- STARTING AUTOMATIC ERROR ANALYSIS ON {len(ds)} SAMPLES ---")

    for i in range(len(ds)):
        tensor_img, target_idx = ds[i]
        input_tensor = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = trainer.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx]) * 100.0

        y_true_list.append(target_idx)
        y_pred_list.append(pred_idx)
        y_score_list.append(probs[1])

        # If prediction is wrong:
        if pred_idx != target_idx:
            img_path, _ = ds.samples[i]
            actual_name = class_names[target_idx]
            pred_name = class_names[pred_idx]

            filename = os.path.basename(img_path)
            save_name = f"misclass_{i+1:03d}_actual_{actual_name}_pred_{pred_name}_conf_{int(conf)}.jpg"
            save_path = os.path.join(output_dir, save_name)

            # Load original image for saving
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                # Annotate image with red text banner
                annotated = orig_img.copy()
                label_str = f"ACTUAL: {actual_name.upper()} | PRED: {pred_name.upper()} ({conf:.1f}%)"
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 200), -1)
                cv2.putText(annotated, label_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(save_path, annotated)

            misclassified_records.append({
                "sample_index": i,
                "original_path": img_path,
                "saved_path": save_path,
                "actual_label": actual_name,
                "predicted_label": pred_name,
                "confidence_percent": round(conf, 2)
            })

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_scores = np.array(y_score_list)

    metrics = compute_metrics(y_true, y_pred, y_scores, class_names)

    # Save Confusion Matrix
    cm_path = os.path.join("outputs", "confusion_matrix_error_analysis.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    df_err = pd.DataFrame(misclassified_records)
    csv_path = os.path.join("outputs", "misclassified_summary.csv")
    df_err.to_csv(csv_path, index=False)

    logger.info(f"Analysis complete. Total misclassified images: {len(misclassified_records)} / {len(ds)}")
    logger.info(f"Saved misclassified images to: {output_dir}")
    logger.info(f"Saved summary CSV to: {csv_path}")

    return {
        "misclassified_count": len(misclassified_records),
        "total_samples": len(ds),
        "metrics": metrics,
        "misclassified_records": misclassified_records
    }
