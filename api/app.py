"""
FastAPI Production REST API for Face Presentation Classifier.
"""

import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from inference.pipeline import FacePresentationPipeline, DISCLAIMER_TEXT
from utils.logger import setup_logger

logger = setup_logger("FastAPIApp")

app = FastAPI(
    title="Face Presentation Classifier API",
    description=(
        "Production AI REST API that estimates facial presentation (Masculine-presenting or Feminine-presenting) "
        "based on visual features. " + DISCLAIMER_TEXT
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: FacePresentationPipeline = None


@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Initializing Face Presentation Classifier Pipeline for FastAPI...")
    pipeline = FacePresentationPipeline(config_path="configs/config.yaml")


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Face Presentation Classifier API",
        "version": "1.1.0",
        "disclaimer": DISCLAIMER_TEXT,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["Health"])
@app.post("/health", tags=["Health"])
def health_check():
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "healthy",
        "disclaimer": DISCLAIMER_TEXT,
        "cuda_available": cuda_available,
        "device": device_name,
        "label_mapping": pipeline.label_mapping if pipeline else {},
        "model_backbone": pipeline.config.get("model", {}).get("backbone", "efficientnet_b0") if pipeline else "N/A"
    }


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(...),
    face_index: int = Form(0),
    explainability_method: str = Form("gradcam")
):
    """
    Accepts an uploaded face image and returns prediction details, Top-2 class probabilities,
    raw logits, calibrated probabilities, bounding box, and uncertainty status.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Supported formats: JPG, JPEG, PNG, WEBP."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable image file: {str(e)}"
        )

    result = pipeline.predict(img_np, face_index=face_index, explainability_method=explainability_method)

    if not result["success"]:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "disclaimer": DISCLAIMER_TEXT,
                "message": result["message"],
                "face_detected": False
            }
        )

    return {
        "success": True,
        "disclaimer": DISCLAIMER_TEXT,
        "prediction": result["prediction"],
        "confidence_percentage": result["confidence_percentage"],
        "class_probabilities": result["class_probabilities"],
        "raw_logits": result["raw_logits"],
        "calibrated_probabilities": result["calibrated_probabilities"],
        "label_mapping": pipeline.label_mapping,
        "is_uncertain": result["is_uncertain"],
        "inference_time_ms": result["inference_time_ms"],
        "face_count": result["face_count"],
        "selected_face_index": result["selected_face_index"],
        "bounding_box": result["bounding_box"],
        "message": result["message"]
    }
