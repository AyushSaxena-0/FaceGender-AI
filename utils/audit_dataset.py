"""
Dataset Audit, Verification, and Leakage Detection Utility.
"""

import os
import glob
import hashlib
from typing import Dict, List, Tuple, Any
from PIL import Image
from utils.logger import setup_logger

logger = setup_logger("DatasetAudit")


def get_image_hash(img_path: str) -> str:
    """Computes MD5 hash of an image file to detect exact duplicates."""
    hasher = hashlib.md5()
    with open(img_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def audit_dataset(dataset_dir: str = "dataset", class_names: List[str] = ["masculine", "feminine"]) -> Dict[str, Any]:
    """
    Performs a thorough audit of the dataset structure, corruption, class balance, and data leakage.
    """
    logger.info(f"--- STARTING DATASET AUDIT ON '{dataset_dir}' ---")
    splits = ["train", "validation", "test"]
    audit_results = {
        "label_mapping": {idx: name for idx, name in enumerate(class_names)},
        "split_stats": {},
        "corrupted_files": [],
        "duplicate_hashes": {},
        "data_leakage": [],
        "total_images": 0
    }

    hash_to_path: Dict[str, str] = {}

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        audit_results["split_stats"][split] = {name: 0 for name in class_names}

        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            image_files = glob.glob(os.path.join(class_dir, "*.*"))
            for img_path in image_files:
                audit_results["total_images"] += 1

                # 1. Check Corruption
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        _ = img.convert("RGB").size
                except Exception as e:
                    logger.error(f"Corrupted image detected: {img_path} ({e})")
                    audit_results["corrupted_files"].append((img_path, str(e)))
                    continue

                # 2. Check Class Stats
                audit_results["split_stats"][split][class_name] += 1

                # 3. Check Duplicate & Data Leakage
                img_hash = get_image_hash(img_path)
                if img_hash in hash_to_path:
                    previous_path = hash_to_path[img_hash]
                    prev_split = previous_path.split(os.sep)[1] if os.sep in previous_path else "unknown"
                    if prev_split != split:
                        leakage_msg = f"Data Leakage: Image {img_path} (in {split}) is identical to {previous_path} (in {prev_split})"
                        logger.warning(leakage_msg)
                        audit_results["data_leakage"].append(leakage_msg)
                    else:
                        audit_results["duplicate_hashes"].setdefault(img_hash, []).extend([previous_path, img_path])
                else:
                    hash_to_path[img_hash] = img_path

    # Print Summary Table
    logger.info("=== DATALOADER LABEL MAPPING ===")
    for idx, name in audit_results["label_mapping"].items():
        logger.info(f"Index {idx} -> Class '{name}'")

    logger.info("=== DATASET CLASS DISTRIBUTION ===")
    for split, counts in audit_results["split_stats"].items():
        logger.info(f"Split '{split}': {counts}")

    logger.info(f"Total Images Audited: {audit_results['total_images']}")
    logger.info(f"Corrupted Files Count: {len(audit_results['corrupted_files'])}")
    logger.info(f"Data Leakage Incidents: {len(audit_results['data_leakage'])}")
    logger.info("--- DATASET AUDIT COMPLETE ---")

    return audit_results


if __name__ == "__main__":
    audit_dataset()
