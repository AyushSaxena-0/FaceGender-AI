"""
Explainability Module containing Grad-CAM and Score-CAM implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        if hasattr(self, 'forward_handle') and self.forward_handle:
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle') and self.backward_handle:
            self.backward_handle.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32)

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        return cv2.resize(cam, (target_w, target_h))


class ScoreCAM:
    """
    Score-CAM (Score-Weighted Visual Explanations without Gradients).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, max_channels: int = 32):
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max_channels
        self.activations = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output

    def remove_hooks(self):
        if hasattr(self, 'forward_handle') and self.forward_handle:
            self.forward_handle.remove()

    @torch.no_grad()
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()

        # Baseline forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        if self.activations is None:
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32)

        acts = self.activations.data.clone()[0]  # (C, H_feat, W_feat)
        num_channels = acts.shape[0]
        b, c, target_h, target_w = input_tensor.shape

        # Select top channels by activation norm to save computation
        channel_norms = acts.view(num_channels, -1).norm(dim=1)
        _, top_indices = torch.topk(channel_norms, min(self.max_channels, num_channels))

        weights = []
        cams = []

        for idx in top_indices:
            act_map = acts[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, H_feat, W_feat)
            act_map_resized = F.interpolate(act_map, size=(target_h, target_w), mode="bilinear", align_corners=False)

            # Normalize activation map to [0, 1]
            min_val, max_val = act_map_resized.min(), act_map_resized.max()
            if max_val > min_val:
                act_map_norm = (act_map_resized - min_val) / (max_val - min_val)
            else:
                act_map_norm = act_map_resized

            # Mask input image
            masked_input = input_tensor * act_map_norm
            masked_output = self.model(masked_input)
            masked_score = F.softmax(masked_output, dim=1)[0, target_class].item()

            weights.append(masked_score)
            cams.append(act_map_norm.squeeze().cpu().numpy())

        weights = np.array(weights)
        weights = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))  # Softmax weights

        score_cam = np.zeros((target_h, target_w), dtype=np.float32)
        for w, cam in zip(weights, cams):
            score_cam += w * cam

        score_cam = np.maximum(score_cam, 0)
        if score_cam.max() > 0:
            score_cam = score_cam / score_cam.max()

        return score_cam
