"""
Grad-CAM Explainability Module for face feature visual verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM class to produce activation heatmaps for convolutional layers.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model (nn.Module): PyTorch model.
            target_layer (nn.Module): Target convolutional layer to extract activations/gradients from.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def remove_hooks(self):
        """Removes registered PyTorch forward and backward hooks."""
        if hasattr(self, 'forward_handle') and self.forward_handle:
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle') and self.backward_handle:
            self.backward_handle.remove()

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates a normalized 2D heatmap in [0, 1] for a single input image tensor (1, C, H, W).

        Args:
            input_tensor (torch.Tensor): Input tensor (1, C, H, W).
            target_class (int, optional): Index of the target class. If None, uses max logit class.

        Returns:
            np.ndarray: Normalized 2D heatmap array of shape (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Return empty map if hooks failed
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32)

        gradients = self.gradients.data.cpu().numpy()[0]  # (C, H_feat, W_feat)
        activations = self.activations.data.cpu().numpy()[0]  # (C, H_feat, W_feat)

        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU on map
        if cam.max() > 0:
            cam = cam / cam.max()  # Normalize to [0, 1]

        # Resize to match input image spatial dimensions
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (target_w, target_h))
        return cam_resized
