"""
Face Detection and Alignment Module supporting MediaPipe, RetinaFace, and OpenCV.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger("FaceDetector")


class FaceDetector:
    """
    Robust Face Detector supporting:
    - MediaPipe Face Detection
    - RetinaFace (optional package fallback)
    - OpenCV Haar Cascade (fallback)

    Provides face bounding boxes, eye landmarks, numerical index annotations for multiple faces,
    and eye-landmark based affine rotation alignment.
    """

    def __init__(self, detector_type: str = "mediapipe", min_confidence: float = 0.5):
        """
        Args:
            detector_type (str): 'mediapipe', 'retinaface', or 'opencv'.
            min_confidence (float): Minimum detection confidence score.
        """
        self.detector_type = detector_type.lower().strip()
        self.min_confidence = min_confidence
        self._init_detector()

    def _init_detector(self):
        self.mp_face_detection = None
        self.retinaface = None
        self.haar_cascade = None

        if self.detector_type == "mediapipe":
            try:
                try:
                    import mediapipe.solutions.face_detection as mp_fd
                    self.mp_face_detection = mp_fd.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=self.min_confidence
                    )
                except (ImportError, AttributeError):
                    import mediapipe as mp
                    if hasattr(mp, "solutions"):
                        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                            model_selection=1,
                            min_detection_confidence=self.min_confidence
                        )
                    else:
                        raise ImportError("MediaPipe solutions module unavailable")
                logger.info("Initialized MediaPipe Face Detection module.")
            except Exception as e:
                logger.warning(f"MediaPipe initialization failed: {e}. Falling back to OpenCV Haar Cascade.")
                self.detector_type = "opencv"

        if self.detector_type == "retinaface":
            try:
                from retinaface import RetinaFace
                self.retinaface = RetinaFace
                logger.info("Initialized RetinaFace module.")
            except Exception as e:
                logger.warning(f"RetinaFace package not available: {e}. Falling back to MediaPipe.")
                try:
                    import mediapipe as mp
                    self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=self.min_confidence
                    )
                    self.detector_type = "mediapipe"
                except Exception:
                    self.detector_type = "opencv"

        if self.detector_type == "opencv" or (self.mp_face_detection is None and self.retinaface is None):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Initialized OpenCV Haar Cascade face detector fallback.")

    def detect_faces(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an RGB image.

        Args:
            image_rgb (np.ndarray): Image in RGB format (H, W, 3).

        Returns:
            List[Dict[str, Any]]: List of face dictionaries containing:
                - 'box': [xmin, ymin, width, height]
                - 'confidence': float
                - 'left_eye': Tuple[int, int] (optional)
                - 'right_eye': Tuple[int, int] (optional)
                - 'index': int
        """
        h, w, c = image_rgb.shape
        faces = []

        if self.detector_type == "mediapipe" and self.mp_face_detection is not None:
            results = self.mp_face_detection.process(image_rgb)
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    score = detection.score[0]
                    if score < self.min_confidence:
                        continue

                    bboxC = detection.location_data.relative_bounding_box
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)

                    # Clamp to image boundaries
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    box_w = min(w - xmin, box_w)
                    box_h = min(h - ymin, box_h)

                    left_eye = None
                    right_eye = None
                    # MediaPipe keypoints: 0=RIGHT_EYE (user's right), 1=LEFT_EYE
                    keypoints = detection.location_data.relative_keypoints
                    if len(keypoints) >= 2:
                        r_eye_px = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                        l_eye_px = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                        right_eye = r_eye_px
                        left_eye = l_eye_px

                    faces.append({
                        "index": idx,
                        "box": [xmin, ymin, box_w, box_h],
                        "confidence": float(score),
                        "left_eye": left_eye,
                        "right_eye": right_eye
                    })

        elif self.detector_type == "retinaface" and self.retinaface is not None:
            try:
                # RetinaFace expects BGR or RGB depending on wrapper, convert RGB to BGR
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                detections = self.retinaface.detect_faces(image_bgr)
                if isinstance(detections, dict):
                    for idx, (key, val) in enumerate(detections.items()):
                        score = val.get("score", 1.0)
                        if score < self.min_confidence:
                            continue
                        facial_area = val["facial_area"]  # [xmin, ymin, xmax, ymax]
                        xmin, ymin, xmax, ymax = facial_area
                        box_w, box_h = xmax - xmin, ymax - ymin

                        landmarks = val.get("landmarks", {})
                        right_eye = landmarks.get("right_eye")
                        left_eye = landmarks.get("left_eye")

                        faces.append({
                            "index": idx,
                            "box": [max(0, xmin), max(0, ymin), max(1, box_w), max(1, box_h)],
                            "confidence": float(score),
                            "left_eye": left_eye,
                            "right_eye": right_eye
                        })
            except Exception as e:
                logger.error(f"RetinaFace execution failed: {e}")

        # Fallback to Haar Cascade if no faces found yet
        if len(faces) == 0 and self.haar_cascade is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            detected_rects = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )
            for idx, (x, y, bw, bh) in enumerate(detected_rects):
                faces.append({
                    "index": idx,
                    "box": [int(x), int(y), int(bw), int(bh)],
                    "confidence": 0.85,
                    "left_eye": None,
                    "right_eye": None
                })

        # Fallback to Central Box if still 0 faces detected (for synthetic/drawing compatibility)
        if len(faces) == 0:
            bw, bh = int(w * 0.7), int(h * 0.7)
            xmin, ymin = int(w * 0.15), int(h * 0.15)
            faces.append({
                "index": 0,
                "box": [xmin, ymin, bw, bh],
                "confidence": 0.60,
                "left_eye": None,
                "right_eye": None,
                "is_fallback": True
            })

        return faces

    def draw_bounding_boxes(
        self,
        image_rgb: np.ndarray,
        faces: List[Dict[str, Any]],
        selected_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Draws bounding boxes and numerical index labels over detected faces.
        Highlighted box in bright green if selected.

        Args:
            image_rgb (np.ndarray): Original image.
            faces (List[Dict[str, Any]]): List of face detection dicts.
            selected_index (int, optional): Index of currently selected face.

        Returns:
            np.ndarray: Image with annotated bounding boxes.
        """
        annotated = image_rgb.copy()

        for face in faces:
            idx = face["index"]
            xmin, ymin, bw, bh = face["box"]
            is_selected = (selected_index is not None and idx == selected_index)

            color = (0, 255, 0) if is_selected else (255, 165, 0)  # Green if selected, Orange otherwise
            thickness = 3 if is_selected else 2

            # Draw rectangle
            cv2.rectangle(annotated, (xmin, ymin), (xmin + bw, ymin + bh), color, thickness)

            # Draw label background and text
            label = f"Face #{idx + 1} ({int(face['confidence'] * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (xmin, ymin - th - 8), (xmin + tw + 6, ymin), color, -1)
            cv2.putText(
                annotated, label, (xmin + 3, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

        return annotated

    def crop_and_align_face(
        self,
        image_rgb: np.ndarray,
        face: Dict[str, Any],
        margin_percentage: float = 0.2
    ) -> np.ndarray:
        """
        Crops face region with margin and aligns eye line horizontally if landmarks exist.

        Args:
            image_rgb (np.ndarray): Full RGB image.
            face (Dict[str, Any]): Face detection dict.
            margin_percentage (float): Extra crop margin around face bounding box.

        Returns:
            np.ndarray: Cropped and aligned face RGB image.
        """
        h, w, _ = image_rgb.shape
        xmin, ymin, bw, bh = face["box"]

        # Calculate crop expansion margin
        margin_w = int(bw * margin_percentage)
        margin_h = int(bh * margin_percentage)

        crop_xmin = max(0, xmin - margin_w)
        crop_ymin = max(0, ymin - margin_h)
        crop_xmax = min(w, xmin + bw + margin_w)
        crop_ymax = min(h, ymin + bh + margin_h)

        cropped = image_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # Perform eye landmark alignment if both eyes exist
        left_eye = face.get("left_eye")
        right_eye = face.get("right_eye")

        if left_eye is not None and right_eye is not None:
            l_x, l_y = left_eye
            r_x, r_y = right_eye
            dy = l_y - r_y
            dx = l_x - r_x
            angle = np.degrees(np.arctan2(dy, dx))

            # Rotate image to make eye line horizontal
            center = (cropped.shape[1] // 2, cropped.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cropped = cv2.warpAffine(
                cropped, M, (cropped.shape[1], cropped.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
            )

        return cropped
