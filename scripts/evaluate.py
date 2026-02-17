"""
Evaluation: accuracy, throughput, and latency.
Uses warm-up/cool-down batches for steady-state throughput.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    COOL_DOWN_BATCHES,
    DEVICE,
    WARM_UP_BATCHES,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> dict:
    """
    Run evaluation. Returns dict with accuracy, throughput_images_per_sec,
    total_time_sec, steady_state_throughput (excluding warm-up/cool-down).
    """
    model.eval()
    correct = 0
    total = 0

    num_batches = len(loader)
    batches_to_measure = max(
        0,
        num_batches - WARM_UP_BATCHES - COOL_DOWN_BATCHES,
    )

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        steady_start = torch.cuda.Event(enable_timing=True)
        steady_end = torch.cuda.Event(enable_timing=True)
    else:
        start_time = None
        steady_start_time = None
        steady_end_time = None

    with torch.no_grad():
        if torch.cuda.is_available():
            start_event.record()
        else:
            start_time = time.time()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if batch_idx == WARM_UP_BATCHES and torch.cuda.is_available():
                steady_start.record()
            elif batch_idx == WARM_UP_BATCHES and not torch.cuda.is_available():
                steady_start_time = time.time()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (
                batch_idx == num_batches - COOL_DOWN_BATCHES - 1
                and torch.cuda.is_available()
            ):
                steady_end.record()
            elif (
                batch_idx == num_batches - COOL_DOWN_BATCHES - 1
                and not torch.cuda.is_available()
            ):
                steady_end_time = time.time()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            total_time_sec = start_event.elapsed_time(end_event) / 1000.0
            if batches_to_measure > 0:
                steady_time_sec = steady_start.elapsed_time(steady_end) / 1000.0
                steady_throughput = (batches_to_measure * BATCH_SIZE) / steady_time_sec
            else:
                steady_throughput = 0.0
        else:
            total_time_sec = time.time() - start_time
            if batches_to_measure > 0 and steady_start_time is not None and steady_end_time is not None:
                steady_time_sec = steady_end_time - steady_start_time
                steady_throughput = (batches_to_measure * BATCH_SIZE) / steady_time_sec
            else:
                steady_throughput = 0.0

    accuracy = 100.0 * correct / total if total else 0.0
    overall_throughput = total / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "accuracy": accuracy,
        "total_samples": total,
        "total_time_sec": total_time_sec,
        "throughput_images_per_sec": overall_throughput,
        "steady_state_throughput_images_per_sec": steady_throughput,
        "batch_size": BATCH_SIZE,
        "batches_to_measure": batches_to_measure,
    }
