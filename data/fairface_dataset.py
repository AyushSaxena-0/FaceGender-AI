"""
FairFace Dataset Loader & Preprocessing Unit.
Parses FairFace CSV annotations (fairface_train.csv, fairface_val.csv) for balanced,
bias-reduced facial presentation classification.
"""

import os
import cv2
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.logger import setup_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = setup_logger("FairFaceDataset")


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Applies CLAHE on L channel in LAB color space for RGB image."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)


def center_crop_square(image_rgb: np.ndarray) -> np.ndarray:
    """Crops central square to maintain facial aspect ratio."""
    h, w, _ = image_rgb.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]


def get_fairface_transforms(split: str = "train", img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """Returns Albumentations composition for FairFace training/eval."""
    if split == "train":
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class FairFaceDataset(Dataset):
    """
    PyTorch Dataset for FairFace (108,501 images across race, age, and sex).
    Reads annotations from FairFace CSV format:
      file, age, gender, race, service_test
    Maps:
      'Male' -> 0 (masculine)
      'Female' -> 1 (feminine)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str = "",
        img_size: Tuple[int, int] = (224, 224),
        split: str = "train",
        apply_clahe: bool = True
    ):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.split = split
        self.apply_clahe = apply_clahe
        self.transform = get_fairface_transforms(split, img_size)

        self.label_map = {"Male": 0, "Female": 1, "masculine": 0, "feminine": 1}
        self.samples: List[Tuple[str, int]] = []
        self.demographics_stats: Dict[str, Any] = {}

        self._load_fairface_csv()

    def _load_fairface_csv(self):
        if not os.path.exists(self.csv_path):
            logger.warning(f"FairFace CSV not found at '{self.csv_path}'. Sample FairFace dataset structure can be initialized.")
            return

        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded FairFace CSV '{self.csv_path}' with {len(df)} entries.")

        # Identify gender column name
        gender_col = "gender" if "gender" in df.columns else ("sex" if "sex" in df.columns else None)
        file_col = "file" if "file" in df.columns else ("file_name" if "file_name" in df.columns else "image")

        if gender_col is None or file_col is None:
            logger.error(f"Invalid FairFace CSV format. Missing gender or file column in {df.columns.tolist()}")
            return

        valid_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            rel_file = str(row[file_col])
            raw_gender = str(row[gender_col]).strip()

            if raw_gender not in self.label_map:
                skipped_count += 1
                continue

            target_idx = self.label_map[raw_gender]

            # Determine full path
            full_path = rel_file if os.path.isabs(rel_file) else os.path.join(self.img_dir, rel_file)
            if not os.path.exists(full_path) and not os.path.exists(rel_file):
                skipped_count += 1
                continue

            actual_path = full_path if os.path.exists(full_path) else rel_file
            self.samples.append((actual_path, target_idx))
            valid_count += 1

        logger.info(f"FairFace split '{self.split}': Loaded {valid_count} samples. Skipped {skipped_count} missing/invalid rows.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            square_img = center_crop_square(image_np)
            if self.apply_clahe:
                square_img = apply_clahe_rgb(square_img)

            augmented = self.transform(image=square_img)
            return augmented["image"], target

        except Exception as e:
            logger.error(f"Error loading FairFace image {img_path}: {e}")
            dummy = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy, target


def create_fairface_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Creates PyTorch DataLoaders for FairFace train, val, and test splits."""
    ds_cfg = config.get("dataset", {})
    tr_cfg = config.get("training", {})
    fairface_cfg = config.get("fairface", {})

    train_csv = fairface_cfg.get("train_csv", "data/fairface_train.csv")
    val_csv = fairface_cfg.get("val_csv", "data/fairface_val.csv")
    img_dir = fairface_cfg.get("img_dir", "data/fairface_images")
    input_size = tuple(ds_cfg.get("input_size", [224, 224]))

    batch_size = tr_cfg.get("batch_size", 32)
    num_workers = tr_cfg.get("num_workers", 2)

    train_ds = FairFaceDataset(train_csv, img_dir, input_size, "train")
    val_ds = FairFaceDataset(val_csv, img_dir, input_size, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, None
