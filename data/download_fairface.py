"""
FairFace Dataset Helper Script.
Provides automatic download utility and CSV structure generator for FairFace dataset.
"""

import os
import urllib.request
import pandas as pd
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger("FairFaceDownloader")

FAIRFACE_TRAIN_CSV_URL = "https://raw.githubusercontent.com/dssg/fairface/master/fairface_label_train.csv"
FAIRFACE_VAL_CSV_URL = "https://raw.githubusercontent.com/dssg/fairface/master/fairface_label_val.csv"


def download_fairface_csvs(target_dir: str = "data"):
    """
    Downloads official FairFace label CSV files.
    """
    os.makedirs(target_dir, exist_ok=True)
    train_csv_path = os.path.join(target_dir, "fairface_train.csv")
    val_csv_path = os.path.join(target_dir, "fairface_val.csv")

    logger.info("Downloading official FairFace label metadata CSVs...")

    try:
        if not os.path.exists(train_csv_path):
            urllib.request.urlretrieve(FAIRFACE_TRAIN_CSV_URL, train_csv_path)
            logger.info(f"Downloaded FairFace train CSV to: {train_csv_path}")

        if not os.path.exists(val_csv_path):
            urllib.request.urlretrieve(FAIRFACE_VAL_CSV_URL, val_csv_path)
            logger.info(f"Downloaded FairFace val CSV to: {val_csv_path}")

    except Exception as e:
        logger.warning(f"Could not auto-download FairFace CSVs: {e}")


def generate_sample_fairface_structure(target_dir: str = "data"):
    """
    Generates synthetic FairFace metadata CSVs pointing to local synthetic dataset samples.
    """
    os.makedirs(target_dir, exist_ok=True)
    train_csv_path = os.path.join(target_dir, "fairface_train.csv")
    val_csv_path = os.path.join(target_dir, "fairface_val.csv")

    logger.info("Generating FairFace metadata CSVs mapping local dataset structure...")

    train_data = []
    val_data = []

    for split, data_list in [("train", train_data), ("validation", val_data)]:
        split_dir = os.path.join("dataset", split)
        if not os.path.exists(split_dir):
            continue
        for gender, raw_gender in [("masculine", "Male"), ("feminine", "Female")]:
            gender_dir = os.path.join(split_dir, gender)
            if not os.path.exists(gender_dir):
                continue
            for img_name in os.listdir(gender_dir):
                rel_path = os.path.join("dataset", split, gender, img_name)
                data_list.append({
                    "file": rel_path,
                    "age": "20-29",
                    "gender": raw_gender,
                    "race": "East Asian",
                    "service_test": False
                })

    if train_data and not os.path.exists(train_csv_path):
        pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
        logger.info(f"Created sample FairFace train CSV at: {train_csv_path}")

    if val_data and not os.path.exists(val_csv_path):
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        logger.info(f"Created sample FairFace val CSV at: {val_csv_path}")


if __name__ == "__main__":
    download_fairface_csvs()
    generate_sample_fairface_structure()
