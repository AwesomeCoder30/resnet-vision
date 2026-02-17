from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

from .config import (
    BATCH_SIZE,
    DATA_ROOT,
    DATASET,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_cifar10_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_loader():
    if DATASET == "cifar10":
        transform = get_cifar10_transform()
        dataset = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
    elif DATASET == "imagenet" and DATA_ROOT:
        transform = get_imagenet_transform()
        dataset = ImageFolder(root=f"{DATA_ROOT}/train", transform=transform)
    else:
        raise ValueError(
            f"DATASET must be 'cifar10' or 'imagenet' (with DATA_ROOT). Got {DATASET!r}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return loader


def get_eval_loader(subset_size: Optional[int] = None):
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    if DATASET == "cifar10":
        transform = get_cifar10_transform()
        dataset = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif DATASET == "imagenet" and DATA_ROOT:
        transform = get_imagenet_transform()
        dataset = ImageFolder(root=f"{DATA_ROOT}/val", transform=transform)
    else:
        raise ValueError(
            f"DATASET must be 'cifar10' or 'imagenet' (with DATA_ROOT). Got {DATASET!r}."
        )

    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
        dataset = Subset(dataset, indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return loader
