"""
Misclassified Images Logger Unit.
Saves top-100 incorrect predictions into outputs/misclassified/
annotated with Prediction, Ground Truth, and Confidence.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from utils.logger import setup_logger

logger = setup_logger("ErrorLogger")


def log_misclassified_images(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = "outputs/misclassified",
    max_images: int = 100,
    class_names: List[str] = ["masculine", "feminine"]
) -> Dict[str, Any]:
    """
    Saves up to max_images misclassified predictions with annotated banners.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    misclassified_records = []
    ds = val_loader.dataset
    saved_count = 0

    label_map_str = {0: "Masculine-presenting", 1: "Feminine-presenting"}

    for i in range(len(ds)):
        if saved_count >= max_images:
            break

        tensor_img, target_idx = ds[i]
        input_tensor = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx]) * 100.0

        if pred_idx != target_idx:
            saved_count += 1
            img_path = ds.samples[i][0] if hasattr(ds, 'samples') else f"sample_{i}.jpg"
            ground_truth = label_map_str.get(target_idx, class_names[target_idx])
            prediction = label_map_str.get(pred_idx, class_names[pred_idx])

            save_filename = f"error_{saved_count:03d}_true_{ground_truth[:3]}_pred_{prediction[:3]}_conf_{int(conf)}.jpg"
            save_path = os.path.join(output_dir, save_filename)

            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                annotated = orig_img.copy()
                banner_text = f"GT: {ground_truth} | PRED: {prediction} ({conf:.1f}%)"
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), (0, 0, 220), -1)
                cv2.putText(
                    annotated, banner_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imwrite(save_path, annotated)

            misclassified_records.append({
                "sample_index": i,
                "file_path": img_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence_percent": round(conf, 2),
                "saved_path": save_path
            })

    csv_path = os.path.join("outputs", "misclassified_report.csv")
    pd.DataFrame(misclassified_records).to_csv(csv_path, index=False)
    logger.info(f"Saved {saved_count} misclassified images to '{output_dir}'. Summary log saved to '{csv_path}'.")

    return {
        "misclassified_count": saved_count,
        "records": misclassified_records
    }
