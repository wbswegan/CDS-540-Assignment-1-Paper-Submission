import os
import random
from typing import Optional

import numpy as np
import torch

from config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def ensure_project_dirs(cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)
    os.makedirs(cfg.predictions_dir, exist_ok=True)
