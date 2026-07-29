"""
End-to-End Face Presentation Inference Pipeline with Explainability (Grad-CAM & Score-CAM),
Temperature Calibration, Top-2 Probabilities, and Low-Confidence Uncertainty Handling.
"""

import os
import time
import yaml
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from models.classifier import FacePresentationClassifierModel, get_target_layer_for_gradcam
from models.explainability import GradCAM, ScoreCAM
from models.calibration import calibrate_probabilities
from inference.face_detector import FaceDetector
from inference.preprocessor import FacePreprocessor
from utils.logger import setup_logger
from utils.visualization import overlay_gradcam

logger = setup_logger("InferencePipeline")

DISCLAIMER_TEXT = (
    "IMPORTANT DISCLAIMER: This model estimates facial presentation based only on visual appearance. "
    "It does NOT determine or verify a person's actual gender identity."
)


class FacePresentationPipeline:
    """
    Production Inference Pipeline:
    - Dynamic Label Mapping (0 -> Masculine-presenting, 1 -> Feminine-presenting)
    - Multi-Engine Face Detection (MediaPipe / RetinaFace / OpenCV)
    - Face Cropping & Eye Landmark Alignment
    - Identical Preprocessing (Square Center Crop + CLAHE Brightness Normalization)
    - Model Inference (CUDA / CPU) with raw logits output
    - Temperature Scaling Probability Calibration
    - Top-2 Class Probabilities Display
    - Low-Confidence Rejection (<70%)
    - Explainability Engine (Grad-CAM & Score-CAM)
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        weights_path: Optional[str] = None,
        temperature: float = 1.2
    ):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.temperature = temperature

        ds_cfg = self.config.get("dataset", {})
        det_cfg = self.config.get("face_detection", {})
        pre_cfg = self.config.get("preprocessing", {})
        model_cfg = self.config.get("model", {})
        paths_cfg = self.config.get("paths", {})

        self.class_names = ds_cfg.get("class_names", ["masculine", "feminine"])
        self.target_size = tuple(ds_cfg.get("input_size", [224, 224]))

        # Dynamic Label Mapping
        self.label_mapping = {
            0: "Masculine-presenting" if self.class_names[0] == "masculine" else "Feminine-presenting",
            1: "Feminine-presenting" if self.class_names[1] == "feminine" else "Masculine-presenting"
        }

        logger.info("=== VERIFY LABEL MAPPING IN PIPELINE ===")
        for idx, label_str in self.label_mapping.items():
            logger.info(f"Index {idx} -> {label_str}")

        # Device Setup
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        except (OSError, RuntimeError) as err:
            logger.warning(f"CUDA initialization encountered system DLL error ({err}). Falling back to CPU.")
            self.device = torch.device("cpu")

        logger.info(f"Inference Pipeline initialized on device: {self.device}")

        # Face Detector & Preprocessor
        self.detector = FaceDetector(
            detector_type=det_cfg.get("detector", "mediapipe"),
            min_confidence=det_cfg.get("min_detection_confidence", 0.5)
        )
        self.preprocessor = FacePreprocessor(
            target_size=self.target_size,
            brightness_norm=pre_cfg.get("brightness_normalization", True),
            apply_clahe=pre_cfg.get("apply_clahe", True)
        )

        # Model Setup
        self.model = FacePresentationClassifierModel(
            backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
            num_classes=model_cfg.get("num_classes", 2),
            pretrained=False
        ).to(self.device)

        actual_weights_path = weights_path or paths_cfg.get("best_model_path", "weights/best_model.pth")
        if os.path.exists(actual_weights_path):
            try:
                checkpoint = torch.load(actual_weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded weights from: {actual_weights_path}")
            except Exception as e:
                logger.error(f"Failed to load weights from {actual_weights_path}: {e}. Using initialized weights.")
        else:
            logger.warning(f"No weights file found at {actual_weights_path}. Model running with default initialization.")

        self.model.eval()

        # Explainability Modules
        try:
            target_layer = get_target_layer_for_gradcam(self.model)
            self.gradcam = GradCAM(self.model, target_layer)
            self.scorecam = ScoreCAM(self.model, target_layer)
        except Exception as e:
            logger.warning(f"Could not initialize explainability modules: {e}")
            self.gradcam = None
            self.scorecam = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def predict(
        self,
        image_input: Any,
        face_index: int = 0,
        explainability_method: str = "gradcam"
    ) -> Dict[str, Any]:
        """
        Runs complete inference with temperature calibration, explainability, and uncertainty handling.
        """
        t_start = time.time()

        # 1. Image decoding
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return {"success": False, "message": f"File path does not exist: {image_input}", "disclaimer": DISCLAIMER_TEXT}
                pil_img = Image.open(image_input).convert("RGB")
                img_rgb = np.array(pil_img)
            elif isinstance(image_input, Image.Image):
                img_rgb = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    img_rgb = image_input
                else:
                    return {"success": False, "message": "Invalid numpy array format. Expected (H, W, 3) RGB.", "disclaimer": DISCLAIMER_TEXT}
            else:
                return {"success": False, "message": "Unsupported image input format.", "disclaimer": DISCLAIMER_TEXT}
        except Exception as e:
            return {"success": False, "message": f"Error decoding image: {str(e)}", "disclaimer": DISCLAIMER_TEXT}

        # 2. Face Detection
        faces = self.detector.detect_faces(img_rgb)
        face_count = len(faces)

        if face_count == 0:
            return {
                "success": False,
                "disclaimer": DISCLAIMER_TEXT,
                "face_detected": False,
                "face_count": 0,
                "message": "No face detected. Please upload a clearer image."
            }

        target_idx = max(0, min(face_index, face_count - 1))
        selected_face = faces[target_idx]

        annotated_img = self.detector.draw_bounding_boxes(img_rgb, faces, selected_index=target_idx)
        cropped_face = self.detector.crop_and_align_face(img_rgb, selected_face)

        # 3. Preprocessing (Identical to training: Crop -> CLAHE -> Resize -> Normalize)
        preprocessed_rgb, tensor_input = self.preprocessor.preprocess_image(cropped_face)
        tensor_input = tensor_input.to(self.device)

        # 4. Forward Pass & Raw Logits
        with torch.no_grad():
            raw_logits = self.model(tensor_input).squeeze(0).cpu().numpy()

        # 5. Temperature Scaling Probability Calibration
        calibrated_probs = calibrate_probabilities(raw_logits, temperature=self.temperature)

        pred_class_idx = int(np.argmax(calibrated_probs))
        confidence = float(calibrated_probs[pred_class_idx]) * 100.0
        pred_label_str = self.label_mapping[pred_class_idx]

        # Top-2 Class Probabilities Dictionary
        top2_probabilities = {
            self.label_mapping[0]: round(float(calibrated_probs[0]) * 100.0, 2),
            self.label_mapping[1]: round(float(calibrated_probs[1]) * 100.0, 2)
        }

        # Low Confidence Uncertainty Handling (< 70%)
        is_uncertain = confidence < 70.0
        status_message = "Analysis completed successfully."
        if is_uncertain:
            status_message = f"Prediction uncertain (Confidence {confidence:.1f}% < 70%). Please upload a clearer image."

        # Debug Logs
        logger.info(f"=== VERIFY INFERENCE DEBUG LOGS ===")
        logger.info(f"Raw Logits: {raw_logits.tolist()}")
        logger.info(f"Calibrated Probabilities: {calibrated_probs.tolist()}")
        logger.info(f"Predicted Class Index: {pred_class_idx} -> Label: '{pred_label_str}'")
        logger.info(f"Confidence: {confidence:.2f}% | Is Uncertain: {is_uncertain}")

        # 6. Explainability Heatmap Generation (Grad-CAM / Score-CAM)
        cam_overlay = preprocessed_rgb
        if explainability_method.lower() == "scorecam" and self.scorecam is not None:
            try:
                heatmap = self.scorecam.generate_heatmap(tensor_input, target_class=pred_class_idx)
                cam_overlay = overlay_gradcam(preprocessed_rgb, heatmap, alpha=0.5)
            except Exception as e:
                logger.warning(f"Score-CAM error: {e}")
        elif self.gradcam is not None:
            try:
                with torch.enable_grad():
                    heatmap = self.gradcam.generate_heatmap(tensor_input, target_class=pred_class_idx)
                cam_overlay = overlay_gradcam(preprocessed_rgb, heatmap, alpha=0.5)
            except Exception as e:
                logger.warning(f"Grad-CAM error: {e}")

        t_total_ms = (time.time() - t_start) * 1000.0

        return {
            "success": True,
            "disclaimer": DISCLAIMER_TEXT,
            "face_detected": True,
            "face_count": face_count,
            "selected_face_index": target_idx,
            "all_faces": faces,
            "annotated_image": annotated_img,
            "cropped_face": cropped_face,
            "preprocessed_face": preprocessed_rgb,
            "raw_logits": raw_logits.tolist(),
            "calibrated_probabilities": calibrated_probs.tolist(),
            "prediction": pred_label_str,
            "confidence_percentage": round(confidence, 2),
            "class_probabilities": top2_probabilities,
            "is_uncertain": is_uncertain,
            "inference_time_ms": round(t_total_ms, 2),
            "bounding_box": selected_face["box"],
            "explainability_overlay": cam_overlay,
            "message": status_message
        }
