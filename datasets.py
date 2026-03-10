from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import CIFAR10_MEAN, CIFAR10_STD, Config


class CIFARSubset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: list[int],
        indices: np.ndarray,
        transform: Callable | None,
    ) -> None:
        self.images = images
        self.targets = targets
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        image = Image.fromarray(self.images[idx])
        label = int(self.targets[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]


def _train_transform(name: str) -> transforms.Compose:
    if name == "baseline":
        aug = []
    elif name == "basic_aug":
        aug = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4)]
    elif name == "stronger_aug":
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
    elif name == "advanced_aug":
        aug = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4)]
    else:
        raise ValueError(f"Unknown experiment group: {name}")

    return transforms.Compose(
        aug + [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def _split_indices(total_size: int, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return train_idx, val_idx


def create_dataloaders(
    cfg: Config,
    experiment_group: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> LoaderBundle:
    batch_size = batch_size or cfg.batch_size
    num_workers = cfg.num_workers if num_workers is None else num_workers

    train_base = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True)
    train_idx, val_idx = _split_indices(len(train_base.targets), cfg.val_size, cfg.seed)

    train_set = CIFARSubset(
        images=train_base.data,
        targets=list(train_base.targets),
        indices=train_idx,
        transform=_train_transform(experiment_group),
    )
    val_set = CIFARSubset(
        images=train_base.data,
        targets=list(train_base.targets),
        indices=val_idx,
        transform=_eval_transform(),
    )

    test_base = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True)
    test_set = CIFARSubset(
        images=test_base.data,
        targets=list(test_base.targets),
        indices=np.arange(len(test_base.targets)),
        transform=_eval_transform(),
    )

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=list(train_base.classes),
    )
