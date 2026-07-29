"""
NVIDIA RTX 4050 Hardware & VRAM Profiling Utility.
"""

import torch
import psutil
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger("HardwareInfo")


def profile_hardware() -> Dict[str, Any]:
    """
    Profiles hardware capability with specific focus on NVIDIA RTX 4050 Laptop GPU (6GB VRAM).

    Returns:
        Dict[str, Any]: Detailed hardware information dictionary.
    """
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    cuda_version = "N/A"
    cudnn_version = "N/A"
    total_vram_mb = 0.0
    used_vram_mb = 0.0
    free_vram_mb = 0.0

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "N/A"
        if torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())

        total_vram = torch.cuda.get_device_properties(0).total_memory
        total_vram_mb = total_vram / (1024 ** 2)
        allocated_vram = torch.cuda.memory_allocated(0)
        reserved_vram = torch.cuda.memory_reserved(0)
        used_vram_mb = max(allocated_vram, reserved_vram) / (1024 ** 2)
        free_vram_mb = total_vram_mb - used_vram_mb

    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    logger.info("==================================================")
    logger.info("        RTX 4050 HARDWARE PROFILER REPORT         ")
    logger.info("==================================================")
    logger.info(f"Target GPU:         {device_name}")
    logger.info(f"CUDA Available:     {cuda_available}")
    logger.info(f"CUDA Version:       {cuda_version}")
    logger.info(f"PyTorch Version:    {torch.__version__}")
    logger.info(f"cuDNN Version:      {cudnn_version}")
    logger.info(f"Total VRAM:         {total_vram_mb / 1024:.2f} GB ({total_vram_mb:.1f} MB)")
    logger.info(f"Used VRAM:          {used_vram_mb / 1024:.2f} GB ({used_vram_mb:.1f} MB)")
    logger.info(f"Free VRAM:          {free_vram_mb / 1024:.2f} GB ({free_vram_mb:.1f} MB)")
    logger.info(f"System CPU Cores:   {cpu_cores}")
    logger.info(f"System RAM:         {ram_gb:.2f} GB")
    logger.info("==================================================")

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cuda_version": cuda_version,
        "pytorch_version": torch.__version__,
        "cudnn_version": cudnn_version,
        "total_vram_mb": total_vram_mb,
        "used_vram_mb": used_vram_mb,
        "free_vram_mb": free_vram_mb,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb
    }


if __name__ == "__main__":
    profile_hardware()
