# ResNet Vision

A minimal vision project for training, evaluating, and measuring a ResNet-18 model: tensors, data loading, training loop (forward, loss, backward, step), and evaluation with accuracy and throughput metrics.

## Project structure

```
resnet_vision/                    # Project root
├── .gitignore
├── README.md
├── requirements.txt
├── run.py                        # CLI entrypoint (only file at root)
├── scripts/                      # Python package
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, device, seed
│   ├── data.py                   # Datasets and DataLoaders
│   ├── model.py                  # build_resnet18()
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Eval loop (accuracy, throughput)
│   └── metrics.py                # Save metrics and checkpoint paths
└── results/
    └── resnet18_inference_results.txt   # Prior-run results
```

## Setup

```bash
cd resnet_vision
pip install -r requirements.txt
```

## How to run

From the project root (`resnet_vision/`):

- **Train** (default: CIFAR-10, 3 epochs):
  ```bash
  python run.py train
  ```
- **Train then evaluate**:
  ```bash
  python run.py train --evaluate
  ```
- **Evaluate** (pretrained, no checkpoint):
  ```bash
  python run.py evaluate
  ```
- **Evaluate with checkpoint**:
  ```bash
  python run.py evaluate --checkpoint results/checkpoint_epoch3.pt
  ```

Configuration: `scripts/config.py` (batch size, epochs, dataset, `DATA_ROOT`). Default is CIFAR-10. For ImageNet, set `DATASET = "imagenet"` and `DATA_ROOT` to your ImageNet root.

## Results

**Results in `results/` are from prior runs.** No new execution is required to view them.

| File | Description |
|------|-------------|
| `results/resnet18_inference_results.txt` | Accuracy (78.37%), throughput (images/sec), inference time, GPU memory from existing ResNet-18 runs. |

Running `run.py train` or `run.py evaluate` will also write `train_metrics.csv`, `eval_metrics.csv`, `eval_metrics.json`, and checkpoints under `results/` (these are gitignored).

## Code layout

| Module | Role |
|--------|------|
| **scripts/config.py** | Single source of truth: seed, device, batch size, epochs, LR, dataset choice, paths, warm-up/cool-down. |
| **scripts/data.py** | `get_train_loader()`, `get_eval_loader(subset_size?)`. CIFAR-10 or ImageFolder. Batch shapes: images `(B, 3, H, W)`, labels `(B,)`. |
| **scripts/model.py** | `build_resnet18(pretrained, num_classes)` → model on device. |
| **scripts/train.py** | `train_one_epoch()`, `train()`: forward, loss, backward, step; returns per-epoch metrics. |
| **scripts/evaluate.py** | `evaluate(model, loader)`: accuracy, throughput, timing (steady-state excludes warm-up/cool-down). |
| **scripts/metrics.py** | `save_train_metrics()`, `save_eval_metrics()`, `checkpoint_path(epoch)`. |
| **run.py** (root) | CLI: `train`, `evaluate`, `--evaluate`, `--checkpoint`. |
