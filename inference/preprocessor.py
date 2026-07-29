"""
Image Preprocessing Pipeline for face normalization and tensor conversion.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional


class FacePreprocessor:
    """
    Standardized face preprocessing unit:
    - CLAHE / Brightness normalization
    - Square Center Crop
    - Bilinear / Bicubic Resizing
    - Pixel Scaling & Normalization (ImageNet standards)
    - PyTorch Tensor Conversion
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        brightness_norm: bool = True,
        apply_clahe: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.brightness_norm = brightness_norm
        self.apply_clahe = apply_clahe
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def normalize_brightness(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on L channel in LAB color space.
        """
        if not self.apply_clahe:
            return image_rgb

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l_channel)

        lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
        rgb_equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
        return rgb_equalized

    def center_crop_square(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Crops central square from input image to preserve facial aspect ratio.
        """
        h, w, _ = image_rgb.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        return image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def preprocess_image(self, face_rgb: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Preprocesses cropped face RGB image into preprocessed RGB numpy array and PyTorch tensor (1, 3, H, W).

        Args:
            face_rgb (np.ndarray): Cropped face RGB image.

        Returns:
            Tuple[np.ndarray, torch.Tensor]:
                - Preprocessed RGB image (H, W, 3) in uint8 [0, 255]
                - Normalized tensor of shape (1, 3, target_h, target_w)
        """
        # Step 1: Brightness Normalization / CLAHE
        norm_img = self.normalize_brightness(face_rgb)

        # Step 2: Center Crop Square
        square_img = self.center_crop_square(norm_img)

        # Step 3: Resize to target dimension
        resized_img = cv2.resize(square_img, self.target_size, interpolation=cv2.INTER_AREA)

        # Step 4: Scale to [0, 1] and Normalize
        img_float = resized_img.astype(np.float32) / 255.0
        normalized_float = (img_float - self.mean) / self.std

        # Step 5: Convert (H, W, C) to (1, C, H, W) Tensor
        tensor = torch.from_numpy(normalized_float.transpose(2, 0, 1)).unsqueeze(0).float()

        return resized_img, tensor
