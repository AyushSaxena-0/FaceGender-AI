"""
Model Export Engine supporting TorchScript, ONNX, and TensorRT export.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger("Exporter")


class ModelExporter:
    """
    Exports trained PyTorch models to production deployment formats:
    - TorchScript (.pt)
    - ONNX (.onnx)
    - TensorRT (.engine / warning fallback)
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, int] = (224, 224), device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self._strip_hooks(self.model)
        self.input_size = input_size
        self.device = device
        self.dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    @staticmethod
    def _strip_hooks(module: nn.Module):
        """Removes any registered PyTorch hooks from model modules for clean tracing."""
        for m in module.modules():
            m._forward_hooks.clear()
            m._backward_hooks.clear()
            m._forward_pre_hooks.clear()
            if hasattr(m, '_is_full_backward_hook'):
                m._is_full_backward_hook = None

    def export_torchscript(self, output_path: str) -> str:
        """
        Exports model to TorchScript (JIT trace).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            traced_script_module = torch.jit.trace(self.model, self.dummy_input)
            traced_script_module.save(output_path)
            logger.info(f"Successfully exported TorchScript model to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export TorchScript model: {e}")
            raise e

    def export_onnx(self, output_path: str, opset_version: int = 14) -> Optional[str]:
        """
        Exports model to ONNX format with dynamic batch dimensions.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            try:
                torch.onnx.export(
                    self.model,
                    self.dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"}
                    }
                )
            except Exception as e1:
                logger.warning(f"Standard ONNX export encountered error ({e1}). Attempting legacy mode export...")
                torch.onnx.export(
                    self.model,
                    self.dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    dynamo=False
                )
            logger.info(f"Successfully exported ONNX model to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return None

    def export_tensorrt(self, output_path: str) -> Optional[str]:
        """
        Exports model to TensorRT engine format if TensorRT environment is available.
        """
        try:
            import tensorrt as trt
            logger.info("TensorRT package detected. Proceeding with ONNX-to-TensorRT compilation...")
            # If TensorRT Python API is installed, parse ONNX and build engine
            onnx_path = output_path.replace(".engine", ".onnx")
            if not os.path.exists(onnx_path):
                self.export_onnx(onnx_path)

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TensorRT ONNX Parser Error: {parser.get_error(error)}")
                    return None

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
            engine_bytes = builder.build_serialized_network(network, config)

            if engine_bytes is not None:
                with open(output_path, "wb") as f:
                    f.write(engine_bytes)
                logger.info(f"Successfully compiled TensorRT engine to: {output_path}")
                return output_path
            else:
                logger.warning("TensorRT engine compilation failed.")
                return None

        except ImportError:
            logger.warning("TensorRT python package is not installed in this environment. TensorRT export skipped.")
            return None
        except Exception as e:
            logger.error(f"Error during TensorRT conversion: {e}")
            return None
