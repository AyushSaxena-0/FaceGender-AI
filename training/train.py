"""
Main Training Pipeline Optimized for NVIDIA RTX 4050 Laptop GPU (6GB VRAM).
"""

import os
import yaml
import time
import argparse
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset_merger import merge_and_split_datasets
from datasets.multi_dataset import RTX4050FaceDataset
from training.rtx4050_trainer import RTX4050Trainer
from validation.error_logger import log_misclassified_images
from validation.benchmark import benchmark_inference
from models.exporter import ModelExporter
from utils.logger import setup_logger
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve

logger = setup_logger("TrainPipeline")


def run_rtx4050_training(config_path: str = "configs/config.yaml"):
    """
    Executes production training pipeline specifically optimized for RTX 4050 GPU (6GB VRAM).
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tr_cfg = config.get("training", {})
    ds_cfg = config.get("dataset", {})
    paths = config.get("paths", {})

    input_size = tuple(ds_cfg.get("input_size", [224, 224]))
    suggested_batch_size = tr_cfg.get("batch_size", 32)
    epochs = tr_cfg.get("epochs", 50)
    patience = tr_cfg.get("early_stopping_patience", 8)

    # 1. Merge and split datasets (80% Train, 10% Val, 10% Test)
    train_samples, val_samples, test_samples = merge_and_split_datasets()

    # 2. Automatic Batch Size Finder & OOM Fallback (32 -> 24 -> 16 -> 8)
    batch_size_candidates = [suggested_batch_size, 24, 16, 8]
    actual_batch_size = suggested_batch_size
    train_loader = None
    val_loader = None
    test_loader = None

    num_workers = min(8, psutil.cpu_count(logical=True) or 2)

    for b_size in batch_size_candidates:
        try:
            logger.info(f"Attempting DataLoader setup with Batch Size = {b_size}...")

            tr_dataset = RTX4050FaceDataset(train_samples, input_size, "train")
            va_dataset = RTX4050FaceDataset(val_samples, input_size, "val")
            te_dataset = RTX4050FaceDataset(test_samples, input_size, "test")

            tr_loader = DataLoader(
                tr_dataset, batch_size=b_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            va_loader = DataLoader(
                va_dataset, batch_size=b_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            te_loader = DataLoader(
                te_dataset, batch_size=b_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )

            # Test a dummy batch on GPU to verify VRAM fit
            trainer_test = RTX4050Trainer(config)
            dummy_img, dummy_target = next(iter(tr_loader))
            dummy_img = dummy_img.to(trainer_test.device, non_blocking=True)
            dummy_target = dummy_target.to(trainer_test.device, non_blocking=True)

            with torch.amp.autocast('cuda') if trainer_test.use_amp else torch.no_grad():
                dummy_out = trainer_test.model(dummy_img)
                dummy_loss = trainer_test.criterion(dummy_out, dummy_target)

            actual_batch_size = b_size
            train_loader = tr_loader
            val_loader = va_loader
            test_loader = te_loader
            logger.info(f"Successfully validated Batch Size = {actual_batch_size} on RTX 4050 GPU!")
            break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA Out Of Memory or setup error with Batch Size {b_size}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if train_loader is None:
        logger.error("Failed to initialize DataLoaders even with minimum batch size 8.")
        return

    # Update config with actual batch size
    config["training"]["batch_size"] = actual_batch_size
    trainer = RTX4050Trainer(config)

    # TensorBoard SummaryWriter
    tb_dir = os.path.join(paths.get("logs_dir", "logs"), "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    best_acc_model_path = os.path.join(trainer.weights_dir, "best_accuracy_model.pth")
    best_loss_model_path = os.path.join(trainer.weights_dir, "best_loss_model.pth")
    last_checkpoint_path = os.path.join(trainer.weights_dir, "last_checkpoint.pth")

    start_time = time.time()
    logger.info("================== RTX 4050 TRAINING DASHBOARD ==================")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc, speed_fps = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc, val_metrics, y_true, y_pred, y_scores = trainer.evaluate(val_loader)

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Metrics/Precision", val_metrics["precision"], epoch)
        writer.add_scalar("Metrics/Recall", val_metrics["recall"], epoch)
        writer.add_scalar("Metrics/F1_Score", val_metrics["f1_score"], epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        if torch.cuda.is_available():
            vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            vram_str = f"{vram_allocated_mb:.1f} MB"
            writer.add_scalar("GPU/VRAM_MB", vram_allocated_mb, epoch)
        else:
            vram_str = "N/A"

        ep_time = time.time() - ep_start
        total_elapsed = time.time() - start_time
        eta_seconds = (epochs - epoch) * (total_elapsed / epoch)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"F1: {val_metrics['f1_score']:.4f} | Speed: {speed_fps:.1f} img/s | "
            f"VRAM: {vram_str} | Time: {ep_time:.1f}s | ETA: {eta_seconds / 60:.1f}m"
        )

        # Save Checkpoint: Best Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "metrics": val_metrics,
                "config": config
            }
            torch.save(ckpt, best_acc_model_path)
            torch.save(ckpt, paths.get("best_model_path", "weights/best_model.pth"))
            logger.info(f"--> Saved Best Accuracy Checkpoint ({best_val_acc:.2f}%) to: {best_acc_model_path}")
        else:
            patience_counter += 1

        # Save Checkpoint: Best Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_loss = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": config
            }
            torch.save(ckpt_loss, best_loss_model_path)

        # Save Last Checkpoint
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config
        }
        torch.save(last_ckpt, last_checkpoint_path)

        # Goal Check: Val Acc >= 95% OR Early Stopping
        if best_val_acc >= 95.0:
            logger.info(f"🎉 TARGET REACHED: Validation Accuracy achieved {best_val_acc:.2f}% >= 95.0%!")
            break

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without accuracy improvement.")
            break

    writer.close()

    # Save Plots
    plot_training_history(history, os.path.join(trainer.output_dir, "training_history.png"))
    plot_confusion_matrix(y_true, y_pred, ds_cfg.get("class_names", ["masculine", "feminine"]), os.path.join(trainer.output_dir, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_scores, os.path.join(trainer.output_dir, "roc_curve.png"))

    # Log up to 100 misclassified validation images
    log_misclassified_images(trainer.model, val_loader, trainer.device, os.path.join(trainer.output_dir, "misclassified"), max_images=100)

    # Benchmark Inference
    benchmark_inference(trainer.model, input_size, str(trainer.device))

    # Export Model (TorchScript & ONNX)
    exporter = ModelExporter(trainer.model, input_size, str(trainer.device))
    exporter.export_torchscript(paths.get("torchscript_model_path", "weights/model.pt"))
    exporter.export_onnx(paths.get("onnx_model_path", "weights/model.onnx"))

    # Automatic Diagnostic Feedback if Val Acc < 95%
    if best_val_acc < 95.0:
        logger.info("\n==================================================")
        logger.info("       AUTOMATIC DIAGNOSTIC FEEDBACK REPORT       ")
        logger.info("==================================================")
        logger.info(f"Current Best Validation Accuracy: {best_val_acc:.2f}% (Below 95.0% target)")
        logger.info("Analysis:")
        logger.info("  1. Dataset Quality: Ensure images have high facial clarity without extreme occlusion.")
        logger.info("  2. Learning Curves: Check if training loss is still decreasing (underfitting) or validation loss diverged (overfitting).")
        logger.info("  3. Class Imbalance: Verify equal distribution between masculine and feminine samples.")
        logger.info("Recommendations:")
        logger.info("  • Increase epochs to 50 or fine-tune backbone learning rate to 1e-4.")
        logger.info("  • Expand dataset size using FairFace / CelebA / UTKFace mergers.")
        logger.info("==================================================")

    logger.info("RTX 4050 Training Pipeline execution completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Face Presentation Classifier on RTX 4050 GPU")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_rtx4050_training(args.config)
