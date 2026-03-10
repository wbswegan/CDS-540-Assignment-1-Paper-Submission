from dataclasses import dataclass


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TEST_TRANSFORM_SPEC = [
    "ToTensor()",
    "Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))",
]

EXPERIMENTS = {
    "baseline": {
        "train_transform": [
            "ToTensor()",
            "Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))",
        ],
        "use_mixup": False,
        "mixup_alpha": 0.0,
    },
    "basic_aug": {
        "train_transform": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomCrop(size=32, padding=4)",
            "ToTensor()",
            "Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))",
        ],
        "use_mixup": False,
        "mixup_alpha": 0.0,
    },
    "stronger_aug": {
        "train_transform": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomCrop(size=32, padding=4)",
            "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)",
            "ToTensor()",
            "Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))",
        ],
        "use_mixup": False,
        "mixup_alpha": 0.0,
    },
    "advanced_aug": {
        "train_transform": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomCrop(size=32, padding=4)",
            "ToTensor()",
            "Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))",
        ],
        "use_mixup": True,
        "mixup_alpha": 0.2,
    },
}

EXPERIMENT_ALIASES = {
    "baseline": "baseline",
    "basic": "basic_aug",
    "basic_aug": "basic_aug",
    "stronger": "stronger_aug",
    "stronger_aug": "stronger_aug",
    "advanced": "advanced_aug",
    "advanced_aug": "advanced_aug",
    "advanced_mixup": "advanced_aug",
}


@dataclass
class Config:
    seed: int = 42
    num_classes: int = 10

    data_dir: str = "data"
    output_dir: str = "outputs"
    models_dir: str = "outputs/models"
    metrics_dir: str = "outputs/metrics"
    figures_dir: str = "outputs/figures"
    predictions_dir: str = "outputs/predictions"

    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 10

    learning_rate: float = 0.001
    weight_decay: float = 5e-4

    val_size: int = 5000
    model_name: str = "resnet18"
    optimizer_name: str = "Adam"
    experiment_groups: tuple[str, ...] = ("baseline", "basic_aug", "stronger_aug", "advanced_aug")


def resolve_experiment_name(name: str) -> str:
    key = name.strip().lower()
    if key not in EXPERIMENT_ALIASES:
        available = ", ".join(EXPERIMENT_ALIASES.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available options: {available}")
    return EXPERIMENT_ALIASES[key]
