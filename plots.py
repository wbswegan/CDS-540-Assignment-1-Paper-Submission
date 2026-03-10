from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CIFAR10_MEAN, CIFAR10_STD


def plot_training_curves(metrics_csv: str, save_path: str, title: str) -> None:
    df = pd.read_csv(metrics_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _denormalize_image(image_chw: np.ndarray) -> np.ndarray:
    """
    Convert normalized CHW tensor image back to displayable HWC image in [0, 1].
    """
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(3, 1, 1)
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def plot_misclassified_grid(
    samples: list[dict],
    class_names: Sequence[str],
    save_path: str,
    max_samples: int = 20,
) -> None:
    """
    Save a grid of misclassified samples with true/pred labels.
    Each sample dict requires:
    - image_tensor
    - true_label
    - pred_label
    """
    n = min(len(samples), max_samples)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No misclassified samples", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue

        item = samples[i]
        image = item["image_tensor"].detach().cpu().numpy()
        image = _denormalize_image(image)

        true_label = int(item["true_label"])
        pred_label = int(item["pred_label"])
        sample_index = int(item.get("sample_index", -1))

        ax.imshow(image)
        ax.set_title(
            f"idx:{sample_index}\nT:{class_names[true_label]} P:{class_names[pred_label]}",
            fontsize=7,
        )
        ax.axis("off")

    fig.suptitle("Misclassified Test Samples")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
