from __future__ import annotations

import csv
import json
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
RESULTS_DIR = _PROJECT_ROOT / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_train_metrics(history: list[dict]) -> Path:
    ensure_results_dir()
    path = RESULTS_DIR / "train_metrics.csv"
    if not history:
        return path
    fieldnames = ["epoch", "train_loss", "train_acc"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in fieldnames if k in row})
    return path


def save_eval_metrics(metrics: dict) -> Path:
    ensure_results_dir()
    csv_path = RESULTS_DIR / "eval_metrics.csv"
    fieldnames = [
        "accuracy",
        "total_samples",
        "total_time_sec",
        "throughput_images_per_sec",
        "steady_state_throughput_images_per_sec",
        "batch_size",
        "batches_to_measure",
    ]
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: metrics.get(k) for k in fieldnames})

    json_path = RESULTS_DIR / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return csv_path


def checkpoint_path(epoch: int) -> Path:
    ensure_results_dir()
    return RESULTS_DIR / f"checkpoint_epoch{epoch}.pt"
