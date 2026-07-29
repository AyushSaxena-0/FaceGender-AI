"""
PyTorch Multi-Dataset Loader with Comprehensive Data Augmentations.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Any
from utils.logger import setup_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = setup_logger("MultiDataset")


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)


def center_crop_square(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]


def get_rtx4050_augmentations(split: str = "train", img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Returns full suite of image augmentations:
    - RandomResizedCrop / Resize
    - RandomHorizontalFlip
    - RandomRotate90 & ShiftScaleRotate (RandomAffine)
    - ColorJitter
    - GaussianBlur
    - CoarseDropout (RandomErasing)
    - ImageNet Normalization
    """
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.85, 1.0), p=0.5),
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
            A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class RTX4050FaceDataset(Dataset):
    """
    PyTorch Dataset wrapper for merged face samples list [(filepath, label_idx), ...].
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        img_size: Tuple[int, int] = (224, 224),
        split: str = "train",
        apply_clahe: bool = True
    ):
        self.samples = samples
        self.img_size = img_size
        self.split = split
        self.apply_clahe = apply_clahe
        self.transform = get_rtx4050_augmentations(split, img_size)

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
            logger.error(f"Error loading sample {img_path}: {e}")
            dummy = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return dummy, target
