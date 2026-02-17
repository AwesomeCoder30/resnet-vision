import argparse
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts import (
    build_resnet18,
    checkpoint_path,
    get_eval_loader,
    get_train_loader,
    evaluate,
    save_eval_metrics,
    save_train_metrics,
    train,
)
from scripts.config import DATASET, DEVICE, NUM_EPOCHS, set_seed


def _num_classes() -> int:
    if DATASET == "cifar10":
        return 10
    if DATASET == "imagenet":
        return 1000
    return 10


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-18 train / evaluate")
    parser.add_argument("mode", choices=["train", "evaluate"], help="train or evaluate")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="after training, run evaluation (only with mode=train)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to checkpoint for evaluate (optional)",
    )
    args = parser.parse_args()

    set_seed()
    num_classes = _num_classes()

    if args.mode == "train":
        train_loader = get_train_loader()
        model = build_resnet18(pretrained=True, num_classes=num_classes)
        history = train(model, train_loader, NUM_EPOCHS)
        save_train_metrics(history)
        ckpt = checkpoint_path(NUM_EPOCHS)
        torch.save({"model_state_dict": model.state_dict(), "epoch": NUM_EPOCHS}, ckpt)
        print(f"Saved {ckpt}")

        if args.evaluate:
            eval_loader = get_eval_loader()
            metrics = evaluate(model, eval_loader)
            save_eval_metrics(metrics)
            print(f"Eval accuracy: {metrics['accuracy']:.2f}%")
            print(f"Throughput: {metrics['throughput_images_per_sec']:.2f} images/sec")

    else:
        eval_loader = get_eval_loader()
        model = build_resnet18(pretrained=True, num_classes=num_classes)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        model = model.to(DEVICE)
        metrics = evaluate(model, eval_loader)
        save_eval_metrics(metrics)
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Throughput: {metrics['throughput_images_per_sec']:.2f} images/sec")


if __name__ == "__main__":
    main()
