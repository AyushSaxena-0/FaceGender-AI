# Face Presentation Classifier

> **IMPORTANT DISCLAIMER**  
> *This model estimates facial presentation based only on visual appearance (`Masculine-presenting` or `Feminine-presenting`). It does **NOT** determine or verify a person's actual gender identity.*

---

## Project Overview

**Face Presentation Classifier** is a full-stack, production-ready Deep Learning & Computer Vision application designed to estimate visual facial presentation from images or live webcam streams. Built with **PyTorch**, **Torchvision**, **MediaPipe**, **FastAPI**, and **Gradio**, it includes end-to-end data generation, preprocessing, model training, evaluation, explainability (Grad-CAM), model export (ONNX / TorchScript), and REST API deployment.
<img width="1886" height="860" alt="image" src="https://github.com/user-attachments/assets/05f05fa2-f30f-4759-963d-9187311161c0" />
<img width="1895" height="857" alt="image" src="https://github.com/user-attachments/assets/9cc823ce-6e84-49e2-b8b1-4f8fd6209cc3" />


---

## Key Features

- 📷 **Dual Input Modes**: Live webcam capture with preview and crop or local image upload (`.jpg`, `.jpeg`, `.png`, `.webp`).
- 🔍 **Multi-Engine Face Detection**: Supports **MediaPipe Face Detection**, **RetinaFace**, and **OpenCV Haar Cascades** fallback.
- 👥 **Multi-Face Handling**: Automatically detects all faces, draws bounding boxes with index labels, and lets the user choose which face to analyze.
- 📐 **Preprocessing & Alignment**:
  - Eye landmark horizontal alignment
  - CLAHE & Contrast Limited Brightness Normalization
  - Square Center Crop & ImageNet scaling
  - **Multiple Model Backbones**:
  - `EfficientNet-B0` (Default)
  - `EfficientNet-V2-S`
  - `ResNet50`
  - `MobileNetV3-Large`
  - `ConvNeXt-Tiny`
  - **Advanced PyTorch Training Engine**:
  - Automatic Mixed Precision (AMP)
  - Cosine Annealing / ReduceLROnPlateau Scheduler
  - Label Smoothing Cross Entropy
  - Class Weight Balancing & Gradient Clipping
  - Early Stopping & Best Checkpoint Preservation
  - TensorBoard Dashboard Logging
-  **Data Augmentation**: Powered by **Albumentations** (Horizontal Flip, Rotation, Color Jitter, Gaussian Blur, Coarse Dropout, Shift-Scale-Rotate).
-  **Grad-CAM Explainability**: Visual attention heatmaps displaying which facial regions influenced the model's decision.
-  **Modern Gradio Web UI**: Responsive dark theme with live preview, prediction card, confidence bar, inference timing meter, and reset controls.
-  **FastAPI REST Service**: Production `POST /predict` and `GET/POST /health` REST endpoints.
-  **Export Engine**: Export trained models to **TorchScript (.pt)**, **ONNX (.onnx)**, and **TensorRT (.engine)**.

---

##  Repository Structure

```
FacePresentationClassifier/
├── configs/
│   └── config.yaml           # Central configuration file
├── data/
│   ├── dataset.py            # PyTorch Dataset, Albumentations, Corruption Checker
│   └── generate_sample_data.py # Synthetic face dataset generator
├── models/
│   ├── classifier.py         # Dynamic PyTorch Multi-Backbone Classifier
│   ├── gradcam.py            # Grad-CAM explainability module
│   └── exporter.py           # Exporter to TorchScript, ONNX, and TensorRT
├── training/
│   ├── trainer.py            # PyTorch training engine with AMP & TensorBoard
│   ├── metrics.py            # Accuracy, Precision, Recall, F1, ROC/AUC, Confusion Matrix
│   └── train.py              # Main training execution script
├── inference/
│   ├── face_detector.py      # MediaPipe, RetinaFace & OpenCV face detector
│   ├── preprocessor.py       # CLAHE brightness normalization & alignment
│   └── pipeline.py           # End-to-end inference engine
├── utils/
│   ├── logger.py             # Stream & File logger
│   └── visualization.py      # Plotter for training curves, ROC, Confusion Matrix, Grad-CAM
├── api/
│   └── app.py                # FastAPI REST API
├── gradio_app/
│   └── app.py                # Gradio dark-themed web application
├── main.py                   # Unified CLI runner
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production Docker container setup
└── README.md                 # Documentation
```

---

## Installation

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-org/FacePresentationClassifier.git
cd FacePresentationClassifier

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation & Synthetic Generator

The dataset should follow this directory structure:

```
dataset/
├── train/
│   ├── masculine/
│   └── feminine/
├── validation/
│   ├── masculine/
│   └── feminine/
└── test/
    ├── masculine/
    └── feminine/
```

### Automatic Synthetic Data Generation
If no dataset is present, the app automatically generates a synthetic face dataset for immediate out-of-the-box training:

```bash
python main.py generate-data
```

---

## Quick Start & Usage

### 1. Launch Gradio Web Interface
```bash
python main.py web
```
Open your browser at `http://localhost:7860`.

### 2. Launch FastAPI REST Server
```bash
python main.py api
```
Access interactive API docs (Swagger UI) at `http://localhost:8000/docs`.

### 3. Train Model
```bash
python main.py train --config configs/config.yaml
```

### 4. Export Model to ONNX & TorchScript
```bash
python main.py export --config configs/config.yaml
```

---

## 📡 REST API Documentation

### `POST /predict`
Uploads an image file and returns prediction details.

#### Request (Multipart Form-Data)
- `file`: Image file (`.jpg`, `.jpeg`, `.png`, `.webp`)
- `face_index`: (Optional, int) Index of target face if multiple faces exist (default: `0`).

#### Response Example
```json
{
  "success": true,
  "disclaimer": "IMPORTANT DISCLAIMER: This model estimates facial presentation based only on visual appearance. It does NOT determine or verify a person's actual gender identity.",
  "prediction": "Masculine-presenting",
  "confidence_percentage": 98.42,
  "class_probabilities": {
    "Masculine-presenting": 98.42,
    "Feminine-presenting": 1.58
  },
  "inference_time_ms": 28.54,
  "face_count": 1,
  "selected_face_index": 0,
  "bounding_box": [110, 85, 140, 150]
}
```

### `GET /health`
Returns system health, CUDA availability, and device information.

---

## Docker Deployment

Build and run using Docker:

```bash
# Build Docker image
docker build -t face-presentation-classifier .

# Run Gradio Web UI container
docker run -p 7860:7860 face-presentation-classifier

# Run FastAPI REST server container
docker run -p 8000:8000 face-presentation-classifier python main.py api
```

---

## Configuration (`configs/config.yaml`)

```yaml
dataset:
  path: "dataset"
  input_size: [224, 224]
  class_names: ["masculine", "feminine"]

model:
  backbone: "efficientnet_b0" # Options: efficientnet_b0, resnet50, mobilenet_v3_large, convnext_tiny
  pretrained: true
  dropout_rate: 0.3

face_detection:
  detector: "mediapipe" # Options: mediapipe, retinaface, opencv
  min_detection_confidence: 0.5

training:
  batch_size: 16
  epochs: 15
  learning_rate: 0.0003
  optimizer: "adamw"
  use_amp: true
```

---

## Evaluation Metrics

- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **ROC Curve & AUC Score**
- **Confusion Matrix Heatmap**
- **Grad-CAM Attention Heatmap**

---

## Future Improvements

1. Integration with Vision Transformer (ViT) backbones.
2. TensorRT C++ CNI integration for ultra-low latency edge deployments.
3. Multi-camera RTSP streaming for real-time video feeds.

---

## 📄 License
MIT License.
