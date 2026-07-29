"""
Neural Network Classifier Module supporting multiple backbones.
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as models


class FacePresentationClassifierModel(nn.Module):
    """
    Modular PyTorch Neural Network supporting multiple standard backbones:
    - EfficientNet-B0
    - EfficientNet-V2-S
    - ResNet50
    - MobileNetV3-Large
    - ConvNeXt-Tiny
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone_name (str): Backbone name ('efficientnet_b0', 'resnet50', 'mobilenet_v3_large', 'convnext_tiny', 'efficientnet_v2_s').
            num_classes (int): Number of output classes (2 for masculine/feminine).
            pretrained (bool): Whether to use ImageNet pretrained weights.
            dropout_rate (float): Dropout probability before classifier layer.
            freeze_backbone (bool): Whether to freeze feature extractor layers.
        """
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.num_classes = num_classes

        # Build feature extractor and retrieve feature dimension
        self.feature_extractor, in_features = self._build_backbone(pretrained)

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Build classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, num_classes)
        )

    def _build_backbone(self, pretrained: bool):
        weights_arg = "DEFAULT" if pretrained else None

        if self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            return base, in_features

        elif self.backbone_name in ["efficientnet_v2", "efficientnet_v2_s"]:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_v2_s(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            return base, in_features

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            return base, in_features

        elif self.backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            return base, in_features

        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features
            base.classifier = nn.Identity()
            return base, in_features

        else:
            # Fallback to EfficientNet-B0
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            return base, in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        features = self.feature_extractor(x)
        # Handle ConvNeXt 4D output if identity returns (B, C, 1, 1)
        if len(features.shape) == 4:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits


def get_target_layer_for_gradcam(model: FacePresentationClassifierModel) -> nn.Module:
    """
    Returns the target convolutional layer for Grad-CAM visualization based on backbone architecture.
    """
    name = model.backbone_name
    base = model.feature_extractor

    if "efficientnet" in name:
        return base.features[-1]
    elif "resnet" in name:
        return base.layer4[-1]
    elif "mobilenet" in name:
        return base.features[-1]
    elif "convnext" in name:
        return base.features[-1]
    else:
        # Fallback to last child of feature extractor
        return list(base.children())[-1]
