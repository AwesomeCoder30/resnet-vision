"""
Configuration: hyperparameters, paths, and device.
Single source of truth for the ResNet-18 vision project.
"""

import torch

# Reproducibility
RANDOM_SEED = 42

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

# Dataset: "cifar10" (portable, no path) or "imagenet" (set DATA_ROOT)
DATASET = "cifar10"
DATA_ROOT = None  # For ImageFolder; e.g. "/path/to/ILSVRC/Data/CLS-LOC"

# Eval / throughput
WARM_UP_BATCHES = 5
COOL_DOWN_BATCHES = 5
NUM_WORKERS = 4
PIN_MEMORY = True

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds and cuDNN for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
