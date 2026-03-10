import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from config import Config, resolve_experiment_name
from datasets import create_dataloaders
from model import create_model
from plots import plot_confusion_matrix, plot_misclassified_grid
from utils import ensure_project_dirs, get_device, set_seed


@torch.no_grad()
def evaluate_one_experiment(
    cfg: Config,
    experiment_name: str,
    device: torch.device,
    checkpoint_path: str | None,
    max_misclassified_plot: int,
):
    """
    Evaluate one trained model on CIFAR-10 test set and save:
    - test metrics
    - per-class precision/recall/F1
    - confusion matrix CSV + image
    """
    data = create_dataloaders(
        cfg=cfg,
        experiment_group=experiment_name,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    ckpt_path = checkpoint_path or os.path.join(cfg.models_dir, f"resnet18_{experiment_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = create_model(cfg.num_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    misclassified_rows: list[dict] = []
    misclassified_for_plot: list[dict] = []

    for images, labels, indices in data.test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        labels_cpu = labels.cpu()
        preds_cpu = preds.cpu()
        indices_cpu = indices.cpu()
        mismatches = labels_cpu != preds_cpu
        mismatch_positions = torch.where(mismatches)[0].tolist()

        for pos in mismatch_positions:
            t = int(labels_cpu[pos].item())
            p = int(preds_cpu[pos].item())
            idx = int(indices_cpu[pos].item())

            misclassified_rows.append(
                {
                    "sample_index": idx,
                    "true_label": t,
                    "pred_label": p,
                    "true_name": data.class_names[t],
                    "pred_name": data.class_names[p],
                }
            )

            if len(misclassified_for_plot) < max_misclassified_plot:
                misclassified_for_plot.append(
                    {
                        "sample_index": idx,
                        "true_label": t,
                        "pred_label": p,
                        "image_tensor": images[pos].detach().cpu(),
                    }
                )

    test_loss = total_loss / total_samples
    test_accuracy = total_correct / total_samples

    # Global test metrics for this experiment.
    eval_metrics_path = os.path.join(cfg.metrics_dir, f"eval_{experiment_name}.csv")
    pd.DataFrame(
        [
            {
                "experiment_group": experiment_name,
                "checkpoint": ckpt_path,
                "num_samples": total_samples,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        ]
    ).to_csv(eval_metrics_path, index=False)

    # Per-class precision/recall/F1 for detailed report analysis.
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(cfg.num_classes)),
        zero_division=0,
    )
    per_class_df = pd.DataFrame(
        {
            "class_id": list(range(cfg.num_classes)),
            "class_name": data.class_names,
            "precision": precisions,
            "recall": recalls,
            "f1_score": f1_scores,
            "support": supports,
        }
    )
    per_class_path = os.path.join(cfg.metrics_dir, f"per_class_{experiment_name}.csv")
    per_class_df.to_csv(per_class_path, index=False)

    # Confusion matrix saved as CSV and image.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_classes)))
    cm_df = pd.DataFrame(cm, index=data.class_names, columns=data.class_names)
    cm_csv_path = os.path.join(cfg.metrics_dir, f"confusion_{experiment_name}.csv")
    cm_df.to_csv(cm_csv_path)

    cm_fig_path = os.path.join(cfg.figures_dir, f"confusion_{experiment_name}.png")
    plot_confusion_matrix(
        cm=cm,
        class_names=data.class_names,
        save_path=cm_fig_path,
        title=f"ResNet-18 ({experiment_name})",
    )

    # Raw misclassified sample list from predictions.
    misclassified_df = pd.DataFrame(
        misclassified_rows,
        columns=["sample_index", "true_label", "pred_label", "true_name", "pred_name"],
    )
    misclassified_csv_path = os.path.join(cfg.predictions_dir, f"misclassified_{experiment_name}.csv")
    misclassified_df.to_csv(misclassified_csv_path, index=False)

    # Small factual summary of most common confusion pairs.
    pair_summary_path = os.path.join(cfg.predictions_dir, f"common_confusions_{experiment_name}.csv")
    if not misclassified_df.empty:
        pair_summary_df = (
            misclassified_df.groupby(["true_name", "pred_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
    else:
        pair_summary_df = pd.DataFrame(columns=["true_name", "pred_name", "count"])
    pair_summary_df.to_csv(pair_summary_path, index=False)

    # Figure with a small set of misclassified images.
    misclassified_fig_path = os.path.join(cfg.predictions_dir, f"misclassified_{experiment_name}.png")
    plot_misclassified_grid(
        samples=misclassified_for_plot,
        class_names=data.class_names,
        save_path=misclassified_fig_path,
        max_samples=max_misclassified_plot,
    )

    macro_precision = float(per_class_df["precision"].mean())
    macro_recall = float(per_class_df["recall"].mean())
    macro_f1 = float(per_class_df["f1_score"].mean())

    print(
        f"[{experiment_name}] test_acc={test_accuracy:.4f} "
        f"macro_precision={macro_precision:.4f} macro_recall={macro_recall:.4f} macro_f1={macro_f1:.4f}"
    )
    print(f"Saved: {eval_metrics_path}")
    print(f"Saved: {per_class_path}")
    print(f"Saved: {cm_csv_path}")
    print(f"Saved: {cm_fig_path}")
    print(f"Saved: {misclassified_csv_path}")
    print(f"Saved: {pair_summary_path}")
    print(f"Saved: {misclassified_fig_path}")


def save_experiment_comparison(cfg: Config) -> str:
    """
    Build one comparison table across experiment groups from already computed CSV files.
    This table only includes actual available results.
    """
    rows: list[dict] = []

    for exp in cfg.experiment_groups:
        eval_path = Path(cfg.metrics_dir) / f"eval_{exp}.csv"
        class_path = Path(cfg.metrics_dir) / f"per_class_{exp}.csv"
        train_path = Path(cfg.metrics_dir) / f"train_{exp}.csv"

        if not eval_path.exists() or not class_path.exists():
            continue

        eval_df = pd.read_csv(eval_path)
        class_df = pd.read_csv(class_path)

        row = {
            "experiment_group": exp,
            "test_accuracy": float(eval_df.loc[0, "test_accuracy"]),
            "test_loss": float(eval_df.loc[0, "test_loss"]),
            "macro_precision": float(class_df["precision"].mean()),
            "macro_recall": float(class_df["recall"].mean()),
            "macro_f1": float(class_df["f1_score"].mean()),
        }

        # Add best validation accuracy from training log if available.
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            row["best_val_accuracy"] = float(train_df["val_accuracy"].max())
        else:
            row["best_val_accuracy"] = float("nan")

        rows.append(row)

    comparison_path = os.path.join(cfg.metrics_dir, "comparison_all_experiments.csv")

    if rows:
        comparison_df = pd.DataFrame(rows).sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)
        comparison_df.to_csv(comparison_path, index=False)
    else:
        # Keep behavior explicit if no results exist yet.
        pd.DataFrame(
            columns=[
                "experiment_group",
                "test_accuracy",
                "test_loss",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "best_val_accuracy",
            ]
        ).to_csv(comparison_path, index=False)

    print(f"Saved: {comparison_path}")
    return comparison_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 experiments and generate result visualizations.")
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all configured experiment groups. --checkpoint is ignored in this mode.",
    )
    parser.add_argument(
        "--max-misclassified-plot",
        type=int,
        default=20,
        help="Maximum number of misclassified images to show in the saved grid figure.",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)
    set_seed(cfg.seed)
    device = get_device(args.device)

    if args.all:
        for exp in cfg.experiment_groups:
            evaluate_one_experiment(
                cfg=cfg,
                experiment_name=exp,
                device=device,
                checkpoint_path=None,
                max_misclassified_plot=args.max_misclassified_plot,
            )
    else:
        experiment_name = resolve_experiment_name(args.experiment)
        evaluate_one_experiment(
            cfg=cfg,
            experiment_name=experiment_name,
            device=device,
            checkpoint_path=args.checkpoint,
            max_misclassified_plot=args.max_misclassified_plot,
        )

    save_experiment_comparison(cfg)


if __name__ == "__main__":
    main()
