"""
Model Comparison Suite to train, evaluate, and benchmark multiple backbones.
"""

import os
import yaml
import time
import pandas as pd
from typing import List, Dict, Any

from data.dataset import create_dataloaders
from training.trainer import Trainer
from utils.logger import setup_logger

logger = setup_logger("ModelComparison")


def compare_models(
    config_path: str = "configs/config.yaml",
    backbones: List[str] = ["efficientnet_b0", "efficientnet_v2_s", "resnet50", "mobilenet_v3_large", "convnext_tiny"]
) -> pd.DataFrame:
    """
    Trains and benchmarks all requested model backbones on the dataset, comparing:
    - Test Accuracy (%)
    - Test F1 Score
    - Test ROC AUC
    - Model Parameters (M)
    - Average Inference Latency (ms)
    """
    if not os.path.exists(config_path):
        logger.error(f"Config path not found: {config_path}")
        return pd.DataFrame()

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Use smaller epochs for comparison suite if needed
    base_config["training"]["epochs"] = min(base_config["training"].get("epochs", 15), 10)
    train_loader, val_loader, test_loader, _ = create_dataloaders(base_config)

    results = []
    logger.info(f"--- STARTING MODEL COMPARISON BENCHMARK FOR {len(backbones)} BACKBONES ---")

    for bb in backbones:
        logger.info(f"\n================ BENCHMARKING BACKBONE: {bb} ================")
        config = yaml.safe_load(yaml.dump(base_config))
        config["model"]["backbone"] = bb

        trainer = Trainer(config)
        num_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6

        t0 = time.time()
        trainer.fit(train_loader, val_loader)
        train_time_sec = time.time() - t0

        test_loss, test_acc, test_metrics, _, _, _ = trainer.evaluate(test_loader)

        results.append({
            "Backbone": bb,
            "Parameters (M)": round(num_params, 2),
            "Test Accuracy (%)": round(test_acc, 2),
            "Test F1 Score": round(test_metrics["f1_score"], 4),
            "Test ROC AUC": round(test_metrics["roc_auc"], 4),
            "Train Time (s)": round(train_time_sec, 1)
        })

    df_results = pd.DataFrame(results)

    logger.info("\n================ MODEL COMPARISON TABLE ================")
    logger.info(df_results.to_string(index=False))

    output_csv = os.path.join("outputs", "model_comparison.csv")
    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    logger.info(f"Saved comparison benchmark results to: {output_csv}")

    return df_results


if __name__ == "__main__":
    compare_models()
