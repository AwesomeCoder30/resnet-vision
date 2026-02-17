"""
Training loop: forward, loss, backward, step.
Demonstrates tensor flow and gradient updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import DEVICE, LEARNING_RATE, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> dict:
    """
    Run one training epoch. Returns metrics dict (loss, optional accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total if total else 0.0
    return {"epoch": epoch, "train_loss": avg_loss, "train_acc": acc}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
) -> list[dict]:
    """
    Full training run. Returns list of per-epoch metrics.
    """
    set_seed()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    for epoch in range(1, num_epochs + 1):
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        history.append(metrics)
        print(
            f"Epoch {epoch}/{num_epochs}  loss: {metrics['train_loss']:.4f}  "
            f"acc: {metrics['train_acc']:.2f}%"
        )
    return history
