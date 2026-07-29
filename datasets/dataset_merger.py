"""
Multi-Dataset Merger & Deduplication Unit (FairFace + CelebA + UTKFace).
Splits merged samples into 80% Train, 10% Validation, and 10% Test splits.
"""

import os
import glob
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
from data.generate_sample_data import generate_dataset_structure

logger = setup_logger("DatasetMerger")

LABEL_MAP = {
    "0": 0, "1": 1,
    "male": 0, "female": 1,
    "m": 0, "f": 1,
    "masculine": 0, "feminine": 1,
    "Male": 0, "Female": 1,
    "M": 0, "F": 1
}


def get_image_hash(filepath: str) -> str:
    """Computes MD5 hash of an image for deduplication."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return ""


def parse_utkface_filenames(utk_dir: str) -> List[Tuple[str, int]]:
    """
    Parses UTKFace filenames formatted as: [age]_[gender]_[race]_[date].jpg
    Gender: 0 = Male, 1 = Female
    """
    samples = []
    if not os.path.exists(utk_dir):
        return samples

    files = glob.glob(os.path.join(utk_dir, "*.jpg")) + glob.glob(os.path.join(utk_dir, "*.png"))
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        if len(parts) >= 2 and parts[1] in ["0", "1"]:
            gender_idx = int(parts[1])  # 0 -> Male, 1 -> Female
            samples.append((f, gender_idx))
    return samples


def parse_celeba_csv(celeba_csv: str, img_dir: str) -> List[Tuple[str, int]]:
    """
    Parses CelebA list_attr_celeba.csv containing 'Male' column (+1 for Male, -1 for Female).
    """
    samples = []
    if not os.path.exists(celeba_csv):
        return samples

    try:
        df = pd.read_csv(celeba_csv)
        if "Male" in df.columns and "image_id" in df.columns:
            for idx, row in df.iterrows():
                fname = row["image_id"]
                full_path = os.path.join(img_dir, fname)
                if os.path.exists(full_path):
                    male_val = row["Male"]
                    gender_idx = 0 if male_val == 1 else 1
                    samples.append((full_path, gender_idx))
    except Exception as e:
        logger.warning(f"Error parsing CelebA CSV: {e}")

    return samples


def merge_and_split_datasets(
    output_meta_dir: str = "datasets/merged",
    dataset_paths: Dict[str, Any] = None
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Merges FairFace, CelebA, UTKFace, and local datasets, removes duplicate image hashes,
    validates labels, and splits into 80% Train, 10% Validation, 10% Test.

    Returns:
        Tuple[List, List, List]: (train_samples, val_samples, test_samples)
    """
    os.makedirs(output_meta_dir, exist_ok=True)
    all_raw_samples: List[Tuple[str, int, str]] = []  # (filepath, label_idx, source)

    # 1. Parse local directory structure
    for split in ["train", "validation", "test"]:
        for cidx, cname in [(0, "masculine"), (1, "feminine")]:
            folder = os.path.join("dataset", split, cname)
            if os.path.exists(folder):
                for f in glob.glob(os.path.join(folder, "*.*")):
                    all_raw_samples.append((f, cidx, "local"))

    # 2. Parse FairFace if CSV available
    if os.path.exists("data/fairface_train.csv"):
        try:
            df_ff = pd.read_csv("data/fairface_train.csv")
            for idx, row in df_ff.iterrows():
                f = str(row.get("file", row.get("file_name", "")))
                g = str(row.get("gender", "")).strip()
                if g in LABEL_MAP and os.path.exists(f):
                    all_raw_samples.append((f, LABEL_MAP[g], "fairface"))
        except Exception:
            pass

    # 3. Parse UTKFace if directory exists
    utk_samples = parse_utkface_filenames("datasets/utkface")
    for f, idx in utk_samples:
        all_raw_samples.append((f, idx, "utkface"))

    # 4. Parse CelebA if files exist
    celeba_samples = parse_celeba_csv("datasets/celeba/list_attr_celeba.csv", "datasets/celeba/images")
    for f, idx in celeba_samples:
        all_raw_samples.append((f, idx, "celeba"))

    # If no samples found, generate sample dataset automatically
    if len(all_raw_samples) == 0:
        logger.info("No raw datasets detected. Generating sample dataset automatically...")
        generate_dataset_structure("dataset")
        for split in ["train", "validation", "test"]:
            for cidx, cname in [(0, "masculine"), (1, "feminine")]:
                folder = os.path.join("dataset", split, cname)
                if os.path.exists(folder):
                    for f in glob.glob(os.path.join(folder, "*.*")):
                        all_raw_samples.append((f, cidx, "local"))

    logger.info(f"Total raw dataset entries harvested: {len(all_raw_samples)}")

    # Deduplicate image hashes across datasets
    unique_hashes = set()
    dedup_samples: List[Tuple[str, int]] = []
    duplicate_count = 0

    for fpath, label_idx, source in all_raw_samples:
        img_h = get_image_hash(fpath)
        if img_h and img_h not in unique_hashes:
            unique_hashes.add(img_h)
            dedup_samples.append((fpath, label_idx))
        else:
            duplicate_count += 1

    logger.info(f"Deduplication complete. Duplicates removed: {duplicate_count}. Unique samples: {len(dedup_samples)}")

    filepaths = [s[0] for s in dedup_samples]
    labels = [s[1] for s in dedup_samples]

    # Split 80% Train, 10% Validation, 10% Test
    if len(dedup_samples) >= 10:
        train_files, temp_files, train_lbls, temp_lbls = train_test_split(
            filepaths, labels, test_size=0.20, random_state=42, stratify=labels
        )
        val_files, test_files, val_lbls, test_lbls = train_test_split(
            temp_files, temp_lbls, test_size=0.50, random_state=42, stratify=temp_lbls
        )
    else:
        train_files, val_files, test_files = filepaths, filepaths, filepaths
        train_lbls, val_lbls, test_lbls = labels, labels, labels

    train_samples = list(zip(train_files, train_lbls))
    val_samples = list(zip(val_files, val_lbls))
    test_samples = list(zip(test_files, test_lbls))

    logger.info(f"Merged Dataset Split Summary:")
    logger.info(f"  • Train Samples (80%):      {len(train_samples)}")
    logger.info(f"  • Validation Samples (10%): {len(val_samples)}")
    logger.info(f"  • Test Samples (10%):       {len(test_samples)}")

    return train_samples, val_samples, test_samples
