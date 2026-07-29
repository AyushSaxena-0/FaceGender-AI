"""
Temperature Scaling and Probability Calibration Engine.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict


class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to calibrate model output probabilities.
    Logits are divided by T (Temperature) before Softmax:
        p = Softmax(logits / T)
    """

    def __init__(self, initial_temperature: float = 1.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits with temperature parameter.
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(self, val_logits: torch.Tensor, val_targets: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """
        Learns optimal temperature parameter T on validation set logits using LBFGS optimizer.
        """
        self.temperature.requires_grad = True
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature.requires_grad = False
        return float(self.temperature.item())


def calibrate_probabilities(logits_np: np.ndarray, temperature: float = 1.2) -> np.ndarray:
    """
    Applies temperature scaling to raw logits numpy array and returns calibrated softmax probabilities.

    Args:
        logits_np (np.ndarray): Shape (num_classes,) or (1, num_classes).
        temperature (float): Temperature scaling parameter T > 0.

    Returns:
        np.ndarray: Calibrated softmax probabilities summing to 1.0.
    """
    scaled_logits = logits_np / max(temperature, 1e-4)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)
