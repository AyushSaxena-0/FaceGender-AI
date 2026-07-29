"""
Inference Latency, FPS, and Memory Usage Benchmarking Utility.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger("Benchmark")


def benchmark_inference(
    model: torch.nn.Module,
    input_size: Tuple[int, int] = (224, 224),
    device_str: str = "cuda",
    num_warmup: int = 20,
    num_iters: int = 100
) -> Dict[str, Any]:
    """
    Measures Average Inference Time (ms), Frames Per Second (FPS), GPU Memory (MB), and CPU Memory (MB).
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    # Warmup iterations
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark loop
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_iters):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    avg_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / max(avg_latency_ms, 1e-4)

    gpu_mem_mb = 0.0
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    process = psutil.Process()
    cpu_mem_mb = process.memory_info().rss / (1024 ** 2)

    logger.info("==================================================")
    logger.info("           INFERENCE BENCHMARK REPORT             ")
    logger.info("==================================================")
    logger.info(f"Target Device:          {device}")
    logger.info(f"Input Dimension:        3 x {input_size[0]} x {input_size[1]}")
    logger.info(f"Avg Inference Time:     {avg_latency_ms:.2f} ms")
    logger.info(f"P95 Inference Time:     {p95_latency_ms:.2f} ms")
    logger.info(f"Throughput (FPS):       {fps:.1f} FPS")
    logger.info(f"Peak GPU Memory:        {gpu_mem_mb:.1f} MB")
    logger.info(f"CPU RAM Memory (RSS):   {cpu_mem_mb:.1f} MB")
    logger.info("==================================================")

    return {
        "device": str(device),
        "input_size": input_size,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "fps": round(fps, 1),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "cpu_memory_mb": round(cpu_mem_mb, 1)
    }
