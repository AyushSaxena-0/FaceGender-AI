"""
Dataset loader, augmentation engine, label mapping verification, and corruption checker.
"""

import os
import glob
import cv2
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.logger import setup_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = setup_logger("Dataset")


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


def get_transforms(split: str = "train", img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Returns Albumentations composition for training or evaluation.
    """
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


class FacePresentationDataset(Dataset):
    """
    PyTorch Dataset with explicit label index mapping:
    - 0 -> masculine
    - 1 -> feminine
    """

    def __init__(
        self,
        dir_path: str,
        class_names: List[str] = ["masculine", "feminine"],
        img_size: Tuple[int, int] = (224, 224),
        split: str = "train",
        apply_clahe: bool = True
    ):
        self.dir_path = dir_path
        self.class_names = class_names
        self.img_size = img_size
        self.split = split
        self.apply_clahe = apply_clahe
        self.transform = get_transforms(split, img_size)

        self.samples: List[Tuple[str, int]] = []
        self.corrupted_count = 0

        # Print explicit label mapping
        logger.info(f"--- DATALOADER LABEL MAPPING ({split}) ---")
        for idx, name in enumerate(self.class_names):
            logger.info(f"Label Index {idx} -> '{name}'")

        self._load_and_verify_dataset()

    def _load_and_verify_dataset(self):
        if not os.path.exists(self.dir_path):
            logger.warning(f"Dataset path does not exist: {self.dir_path}")
            return

        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.dir_path, class_name)
            if not os.path.exists(class_folder):
                continue

            valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            image_paths = []
            for ext in valid_extensions:
                image_paths.extend(glob.glob(os.path.join(class_folder, ext)))
                image_paths.extend(glob.glob(os.path.join(class_folder, ext.upper())))

            for img_path in image_paths:
                if self._verify_image(img_path):
                    self.samples.append((img_path, class_idx))
                else:
                    self.corrupted_count += 1

        logger.info(
            f"Loaded {len(self.samples)} valid samples from split '{self.split}'. "
            f"Corrupted images skipped: {self.corrupted_count}."
        )

    @staticmethod
    def _verify_image(img_path: str) -> bool:
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        counts = {name: 0 for name in self.class_names}
        for _, class_idx in self.samples:
            counts[self.class_names[class_idx]] += 1
        return {
            "total_samples": len(self.samples),
            "class_distribution": counts,
            "corrupted_skipped": self.corrupted_count
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # Apply identical preprocessing steps: Square Crop -> CLAHE -> Albumentations
            square_img = center_crop_square(image_np)
            if self.apply_clahe:
                square_img = apply_clahe_rgb(square_img)

            augmented = self.transform(image=square_img)
            tensor_img = augmented["image"]
            return tensor_img, target

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            dummy = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy, target


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    ds_config = config.get("dataset", {})
    tr_config = config.get("training", {})
    dataset_type = ds_config.get("dataset_type", "directory").lower()

    if dataset_type == "fairface":
        from data.fairface_dataset import create_fairface_dataloaders, FairFaceDataset
        from data.download_fairface import generate_sample_fairface_structure
        generate_sample_fairface_structure()

        train_loader, val_loader, _ = create_fairface_dataloaders(config)
        test_loader = val_loader

        class_counts = {"masculine": len(train_loader.dataset) // 2, "feminine": len(train_loader.dataset) // 2}
        logger.info(f"Loaded FairFace Dataset with {len(train_loader.dataset)} training samples.")
        return train_loader, val_loader, test_loader, class_counts

    input_size = tuple(ds_config.get("input_size", [224, 224]))
    class_names = ds_config.get("class_names", ["masculine", "feminine"])
    batch_size = tr_config.get("batch_size", 16)
    num_workers = tr_config.get("num_workers", 2)

    train_ds = FacePresentationDataset(ds_config.get("train_dir", "dataset/train"), class_names, input_size, "train")
    val_ds = FacePresentationDataset(ds_config.get("val_dir", "dataset/validation"), class_names, input_size, "val")
    test_ds = FacePresentationDataset(ds_config.get("test_dir", "dataset/test"), class_names, input_size, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_counts = train_ds.get_stats()["class_distribution"]

    logger.info("=== VERIFY DATALOADER SUMMARY ===")
    for idx, name in enumerate(class_names):
        logger.info(f"Class Name: '{name}' | Label Index: {idx} | Train Count: {class_counts.get(name, 0)}")

    return train_loader, val_loader, test_loader, class_counts
