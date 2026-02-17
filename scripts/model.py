"""
ResNet-18 model builder.
Single model on one device; no pipeline/split logic.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from .config import DEVICE


def build_resnet18(
    pretrained: bool = True,
    num_classes: int = 10,
) -> nn.Module:
    """
    Build ResNet-18 and move to config device.

    Args:
        pretrained: If True, use ImageNet pretrained weights (then replace fc for num_classes).
        num_classes: Output classes (10 for CIFAR-10, 1000 for ImageNet).

    Returns:
        Model in eval or train mode on DEVICE. Forward input: (B, 3, 224, 224) -> (B, num_classes).
    """
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None, num_classes=num_classes)

    model = model.to(DEVICE)
    return model
