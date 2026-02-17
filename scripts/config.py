import torch

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
DATASET = "cifar10"
DATA_ROOT = None  # ImageFolder root if DATASET == "imagenet"
WARM_UP_BATCHES = 5
COOL_DOWN_BATCHES = 5
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix seeds and cuDNN for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
