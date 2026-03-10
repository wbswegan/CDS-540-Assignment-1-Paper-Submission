# CIFAR-10 Data Augmentation Study (PyTorch)

## Problem Statement
This project compares multiple data augmentation strategies for small-scale image classification on CIFAR-10 using a fixed training pipeline.  
Only augmentation settings change across experiments; model and core training setup remain fixed.

## Dataset
- Dataset: `torchvision.datasets.CIFAR10`
- Download: automatic on first run
- Location: `data/`
- Classes: 10

## Experiment Groups
- `baseline`: normalization only
- `basic_aug`: random horizontal flip + random crop
- `stronger_aug`: random horizontal flip + random crop + color jitter
- `advanced_aug`: random horizontal flip + random crop + mixup (training-time batch augmentation)

## File Structure
```text
cifar10-augmentation-study/
  config.py
  datasets.py
  model.py
  utils.py
  plots.py
  train.py
  evaluate.py
  requirements.txt
  README.md
  data/
  outputs/
    models/
    metrics/
    figures/
    predictions/
```

## Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Experiments
Train each group:
```bash
python train.py --experiment baseline
python train.py --experiment basic_aug
python train.py --experiment stronger_aug
python train.py --experiment advanced_aug
```

Evaluate each group:
```bash
python evaluate.py --experiment baseline
python evaluate.py --experiment basic_aug
python evaluate.py --experiment stronger_aug
python evaluate.py --experiment advanced_aug
```

Evaluate all trained groups in one command:
```bash
python evaluate.py --all
```

## Output Locations
- Model checkpoints: `outputs/models/`
- Per-epoch train/validation metrics CSV: `outputs/metrics/train_<group>.csv`
- Final test metrics CSV: `outputs/metrics/final_<group>.csv`
- Per-class precision/recall/F1 CSV: `outputs/metrics/per_class_<group>.csv`
- Confusion matrix CSV: `outputs/metrics/confusion_<group>.csv`
- Cross-experiment comparison CSV: `outputs/metrics/comparison_all_experiments.csv`
- Final technical summary: `outputs/metrics/final_experiment_summary.txt`
- Training curves image: `outputs/figures/training_curve_<group>.png`
- Confusion matrix image: `outputs/figures/confusion_<group>.png`
- Misclassification outputs: `outputs/predictions/`
