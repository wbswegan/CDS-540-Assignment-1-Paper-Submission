import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import EXPERIMENTS, TEST_TRANSFORM_SPEC, Config, resolve_experiment_name
from datasets import create_dataloaders
from model import create_model
from plots import plot_training_curves
from utils import ensure_project_dirs, get_device, set_seed


def _mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """
    Apply MixUp on one mini-batch.
    Returns mixed images and paired labels for MixUp loss computation.
    """
    if alpha <= 0.0:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.0,
):
    """
    One epoch over a loader.
    - training mode: optimizer is provided
    - eval mode: optimizer is None
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if training and use_mixup:
                images, labels_a, labels_b, lam = _mixup_batch(images, labels, mixup_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)

            if training:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)

        if training and use_mixup:
            # For MixUp, accuracy is the weighted match for the two target sets.
            correct_a = (preds == labels_a).sum().item()
            correct_b = (preds == labels_b).sum().item()
            total_correct += lam * correct_a + (1.0 - lam) * correct_b
        else:
            total_correct += (preds == labels).sum().item()

        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_best_checkpoint(
    checkpoint_path: str,
    cfg: Config,
    class_names: list[str],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """
    Load the best checkpoint and compute final test metrics.
    """
    best_model = create_model(cfg.num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model=best_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        use_mixup=False,
        mixup_alpha=0.0,
    )
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train one CIFAR-10 experiment (PyTorch ResNet-18).")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config()
    experiment_name = resolve_experiment_name(args.experiment)
    exp_cfg = EXPERIMENTS[experiment_name]

    # Reproducibility setup for fair experiment comparison.
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    device = get_device(args.device)

    # Same split/batch settings across experiments; only augmentation policy changes.
    data = create_dataloaders(
        cfg=cfg,
        experiment_group=experiment_name,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = create_model(cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    checkpoint_path = os.path.join(cfg.models_dir, f"resnet18_{experiment_name}.pt")

    # Save a clear configuration record for report writing and reproducibility.
    experiment_config_path = os.path.join(cfg.metrics_dir, f"experiment_{experiment_name}.json")
    with open(experiment_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_group": experiment_name,
                "train_transform": exp_cfg["train_transform"],
                "test_transform": TEST_TRANSFORM_SPEC,
                "mixup": {
                    "enabled": bool(exp_cfg["use_mixup"]),
                    "alpha": float(exp_cfg["mixup_alpha"]),
                },
                "fixed_settings": {
                    "model": cfg.model_name,
                    "optimizer": cfg.optimizer_name,
                    "loss": "CrossEntropyLoss",
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "batch_size": cfg.batch_size,
                    "epochs": cfg.epochs,
                    "seed": cfg.seed,
                    "val_size": cfg.val_size,
                },
            },
            f,
            indent=2,
        )

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            use_mixup=bool(exp_cfg["use_mixup"]),
            mixup_alpha=float(exp_cfg["mixup_alpha"]),
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=data.val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_mixup=False,
            mixup_alpha=0.0,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"[{experiment_name}] Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Best model is selected by validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment_group": experiment_name,
                    "best_epoch": best_epoch,
                    "val_accuracy": best_val_acc,
                    "class_names": data.class_names,
                },
                checkpoint_path,
            )

    # Per-epoch training metrics.
    metrics_path = os.path.join(cfg.metrics_dir, f"train_{experiment_name}.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    # Final test metric from the best checkpoint.
    final_test_loss, final_test_acc = evaluate_best_checkpoint(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        class_names=data.class_names,
        test_loader=data.test_loader,
        device=device,
    )
    summary_path = os.path.join(cfg.metrics_dir, f"final_{experiment_name}.csv")
    pd.DataFrame(
        [
            {
                "experiment_group": experiment_name,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
                "final_test_loss": final_test_loss,
                "final_test_accuracy": final_test_acc,
            }
        ]
    ).to_csv(summary_path, index=False)

    curve_path = os.path.join(cfg.figures_dir, f"training_curve_{experiment_name}.png")
    plot_training_curves(
        metrics_csv=metrics_path,
        save_path=curve_path,
        title=f"ResNet-18 ({experiment_name})",
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved per-epoch metrics: {metrics_path}")
    print(f"Saved final test metrics: {summary_path}")
    print(f"Saved curve: {curve_path}")
    print(f"Saved experiment config: {experiment_config_path}")


if __name__ == "__main__":
    main()
