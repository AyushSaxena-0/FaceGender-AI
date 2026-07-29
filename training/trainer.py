"""
PyTorch Training Loop Engine with Mixed Precision, Early Stopping, and TensorBoard logging.
"""

import os
import time
import yaml
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.classifier import FacePresentationClassifierModel
from training.metrics import compute_metrics
from utils.logger import setup_logger
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve

logger = setup_logger("Trainer")


class Trainer:
    """
    Production-grade PyTorch Trainer.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})
        self.tr_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.ds_cfg = config.get("dataset", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device} for training.")

        # Initialize Directories
        self.weights_dir = self.paths.get("weights_dir", "weights")
        self.logs_dir = self.paths.get("logs_dir", "logs")
        self.output_dir = self.paths.get("output_dir", "outputs")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_dir, "tensorboard"))

        # Model Initialization
        self.model = FacePresentationClassifierModel(
            backbone_name=self.model_cfg.get("backbone", "efficientnet_b0"),
            num_classes=self.model_cfg.get("num_classes", 2),
            pretrained=self.model_cfg.get("pretrained", True),
            dropout_rate=self.model_cfg.get("dropout_rate", 0.3),
            freeze_backbone=self.model_cfg.get("freeze_backbone", False)
        ).to(self.device)

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
        opt_name = self.tr_cfg.get("optimizer", "adamw").lower()
        if opt_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # Learning Rate Scheduler
        epochs = self.tr_cfg.get("epochs", 15)
        sched_name = self.tr_cfg.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched_name == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # Mixed Precision Scaler
        self.use_amp = self.tr_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                grad_clip = self.tr_cfg.get("grad_clip_norm", 1.0)
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()

                grad_clip = self.tr_cfg.get("grad_clip_norm", 1.0)
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += images.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        for images, targets in val_loader:
            images, targets = images.to(self.device), targets.to(self.device)

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
            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class (feminine index 1)

        eval_loss = running_loss / max(total, 1)
        eval_acc = (correct / max(total, 1)) * 100.0

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_scores = np.array(all_probs)

        metrics = compute_metrics(y_true, y_pred, y_scores, self.ds_cfg.get("class_names", ["masculine", "feminine"]))
        return eval_loss, eval_acc, metrics, y_true, y_pred, y_scores

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        epochs = self.tr_cfg.get("epochs", 15)
        patience = self.tr_cfg.get("early_stopping_patience", 5)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.weights_dir, "best_model.pth")

        start_time = time.time()
        logger.info("================== STARTING TRAINING DASHBOARD ==================")

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_metrics, y_true, y_pred, y_scores = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            ep_time = time.time() - ep_start
            total_elapsed = time.time() - start_time
            eta_seconds = (epochs - epoch) * (total_elapsed / epoch)
            gpu_mem = f"{torch.cuda.memory_allocated() / 1e6:.1f}MB" if torch.cuda.is_available() else "N/A"

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"F1: {val_metrics['f1_score']:.4f} | LR: {current_lr:.6f} | "
                f"GPU: {gpu_mem} | Time: {ep_time:.1f}s | ETA: {eta_seconds / 60:.1f}m"
            )

            # Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                class_names_list = self.ds_cfg.get("class_names", ["masculine", "feminine"])
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "metrics": val_metrics,
                    "config": self.config,
                    "class_names": class_names_list,
                    "index_to_class": {idx: name for idx, name in enumerate(class_names_list)},
                    "class_to_index": {name: idx for idx, name in enumerate(class_names_list)},
                    "backbone": self.model_cfg.get("backbone", "efficientnet_b0"),
                    "num_classes": self.model_cfg.get("num_classes", 2),
                    "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"--> Saved new best checkpoint to: {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without validation loss improvement.")
                    break

        self.writer.close()

        # Save Plots
        plot_training_history(history, os.path.join(self.output_dir, "training_history.png"))
        plot_confusion_matrix(y_true, y_pred, self.ds_cfg.get("class_names", ["masculine", "feminine"]), os.path.join(self.output_dir, "confusion_matrix.png"))
        plot_roc_curve(y_true, y_scores, os.path.join(self.output_dir, "roc_curve.png"))

        logger.info("Training process completed successfully.")
        return history
