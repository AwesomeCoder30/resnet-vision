import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from .config import DEVICE


def build_resnet18(pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None, num_classes=num_classes)

    model = model.to(DEVICE)
    return model
