"""
Synthetic Dataset Generator for instant training, testing, and debugging.
Creates distinct, visually separable facial presentation samples.
"""

import os
import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("SampleDataGenerator")


def draw_synthetic_face(presentation: str = "masculine", seed: int = 42) -> np.ndarray:
    """
    Renders a synthetic facial image with clear, distinct visual presentation cues
    visible inside the face crop region.

    Args:
        presentation (str): 'masculine' or 'feminine'.
        seed (int): Random seed for reproducible visual attributes.

    Returns:
        np.ndarray: RGB image array of shape (256, 256, 3).
    """
    np.random.seed(seed)
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 1. Background gradient (RGB)
    bg_color1 = np.array([np.random.randint(20, 50), np.random.randint(30, 60), np.random.randint(50, 90)])
    bg_color2 = np.array([np.random.randint(100, 140), np.random.randint(110, 150), np.random.randint(130, 180)])
    for y in range(256):
        alpha = y / 256.0
        color = (1 - alpha) * bg_color1 + alpha * bg_color2
        img[y, :] = color.astype(np.uint8)

    # 2. Skin tone (RGB: R, G, B)
    r_val = np.random.randint(210, 255)
    g_val = np.random.randint(170, 210)
    b_val = np.random.randint(140, 180)
    skin_tone = (r_val, g_val, b_val)

    face_center = (128, 132)

    if presentation == "masculine":
        # Square / Angular Jawline
        face_axes = (62 + np.random.randint(-2, 3), 78 + np.random.randint(-2, 3))
        cv2.ellipse(img, face_center, face_axes, 0, 0, 360, skin_tone, -1)
        # Add angular chin box
        cv2.rectangle(img, (108, 185), (148, 205), skin_tone, -1)
    else:
        # Tapered Oval Face & Soft Chin
        face_axes = (58 + np.random.randint(-2, 3), 82 + np.random.randint(-2, 3))
        cv2.ellipse(img, face_center, face_axes, 0, 0, 360, skin_tone, -1)

    # 3. Eyes (RGB)
    iris_color = (np.random.randint(20, 50), np.random.randint(30, 80), np.random.randint(20, 60))
    l_eye = (98, 118)
    r_eye = (158, 118)
    cv2.circle(img, l_eye, 11, (250, 250, 250), -1)
    cv2.circle(img, r_eye, 11, (250, 250, 250), -1)
    cv2.circle(img, l_eye, 5, iris_color, -1)
    cv2.circle(img, r_eye, 5, iris_color, -1)

    # 4. Eyebrows
    brow_color = (30, 25, 20)
    if presentation == "masculine":
        # Straight, thick eyebrows
        cv2.line(img, (84, 102), (114, 103), brow_color, 5)
        cv2.line(img, (142, 103), (172, 102), brow_color, 5)
    else:
        # Curved, arched thin eyebrows
        pts_left = np.array([[84, 105], [99, 98], [114, 104]], np.int32)
        pts_right = np.array([[142, 104], [157, 98], [172, 105]], np.int32)
        cv2.polylines(img, [pts_left], False, brow_color, 2, cv2.LINE_AA)
        cv2.polylines(img, [pts_right], False, brow_color, 2, cv2.LINE_AA)

    # 5. Nose
    nose_color = (int(r_val * 0.8), int(g_val * 0.8), int(b_val * 0.8))
    cv2.line(img, (128, 120), (128, 146), nose_color, 2)
    cv2.circle(img, (124, 147), 2, nose_color, -1)
    cv2.circle(img, (132, 147), 2, nose_color, -1)

    # 6. Mouth & Lips
    if presentation == "feminine":
        # Fuller pink/red lip
        lip_color = (220, 70, 110)
        cv2.ellipse(img, (128, 168), (18, 9), 0, 0, 360, lip_color, -1)
        cv2.line(img, (110, 168), (146, 168), (140, 30, 60), 1)
    else:
        # Neutral natural lip line
        lip_color = (int(r_val * 0.85), int(g_val * 0.75), int(b_val * 0.75))
        cv2.ellipse(img, (128, 168), (16, 4), 0, 0, 360, lip_color, -1)

    # 7. Hair & Forehead presentation (Visible within face crop)
    hair_color = (np.random.randint(15, 45), np.random.randint(15, 40), np.random.randint(15, 40))
    if presentation == "masculine":
        # Short cropped hair top
        cv2.ellipse(img, (128, 62), (68, 28), 0, 180, 360, hair_color, -1)
        # Sideburns
        cv2.rectangle(img, (64, 90), (72, 130), hair_color, -1)
        cv2.rectangle(img, (184, 90), (192, 130), hair_color, -1)
    else:
        # Swept bangs covering upper forehead & long side tresses
        cv2.ellipse(img, (128, 65), (72, 36), 0, 180, 360, hair_color, -1)
        # Swept bangs onto forehead
        cv2.ellipse(img, (110, 80), (35, 18), 25, 0, 360, hair_color, -1)
        # Side tresses framing face
        cv2.rectangle(img, (58, 75), (78, 220), hair_color, -1)
        cv2.rectangle(img, (178, 75), (198, 220), hair_color, -1)

    return img


def generate_dataset_structure(base_dir: str = "dataset", samples_per_class: Dict[str, int] = None):
    """
    Generates synthetic training, validation, and test datasets in RGB format.
    """
    if samples_per_class is None:
        samples_per_class = {"train": 50, "validation": 15, "test": 15}

    splits = ["train", "validation", "test"]
    classes = ["masculine", "feminine"]

    logger.info(f"Generating synthetic dataset structure inside '{base_dir}'...")

    counter = 0
    for split in splits:
        num_samples = samples_per_class.get(split, 15)
        for class_name in classes:
            target_dir = os.path.join(base_dir, split, class_name)
            os.makedirs(target_dir, exist_ok=True)

            for i in range(num_samples):
                counter += 1
                img_rgb = draw_synthetic_face(presentation=class_name, seed=counter * 13 + i)
                # Convert RGB to BGR for cv2.imwrite saving
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                filename = os.path.join(target_dir, f"sample_{class_name}_{i + 1:03d}.jpg")
                cv2.imwrite(filename, img_bgr)

    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    generate_dataset_structure()
