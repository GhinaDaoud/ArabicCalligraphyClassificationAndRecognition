from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import CalligraphyStyleDataset
from src.datasets.transforms import build_eval_transforms
from src.evaluation.confusion_matrix import (
    maybe_plot_confusion_matrix,
    write_confusion_matrix_csv,
    write_confusion_matrix_markdown,
)
from src.evaluation.metrics import (
    classification_metrics_from_confusion,
    confusion_matrix,
    per_class_report_from_confusion,
)
from src.models.local_global_model import (
    ResNet18GlobalOnlyClassifier,
    ResNet18LocalGlobalClassifier,
)
from src.utils.checkpoint import load_checkpoint


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to load config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def build_model(model_cfg: dict):
    name = model_cfg["name"]
    kwargs = {
        "num_classes": model_cfg["num_classes"],
        "pretrained": model_cfg.get("pretrained", True),
        "dropout": model_cfg.get("dropout", 0.3),
    }
    if name == "resnet18_local_global":
        return ResNet18LocalGlobalClassifier(**kwargs)
    if name == "resnet18_global_only":
        return ResNet18GlobalOnlyClassifier(**kwargs)
    raise ValueError(f"Unsupported model: {name}")


def build_eval_dataset(config: dict, split_name: str) -> CalligraphyStyleDataset:
    data_cfg = config["data"]
    split_key = f"{split_name}_csv"
    csv_path = resolve_path(data_cfg.get(split_key))
    if not csv_path:
        raise ValueError(f"Config does not define {split_key}.")

    return CalligraphyStyleDataset(
        csv_path=csv_path,
        image_root=resolve_path(data_cfg["image_root"]),
        transform=build_eval_transforms(),
    )


def ordered_class_names(class_to_index: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])]


@torch.no_grad()
def collect_predictions(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    return predictions, targets


def write_per_class_report_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "true_positive",
                "false_positive",
                "false_negative",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_report_markdown(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# Per-Class Report",
        "",
        "| class_name | support | precision | recall | f1 | accuracy | tp | fp | fn |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['support']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['accuracy']:.4f} | {row['true_positive']} | {row['false_positive']} | "
            f"{row['false_negative']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_markdown(path: Path, metrics: dict[str, float], checkpoint_path: Path, split_name: str) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- split: {split_name}",
        f"- checkpoint: {checkpoint_path}",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- precision_macro: {metrics['precision_macro']:.4f}",
        f"- recall_macro: {metrics['recall_macro']:.4f}",
        f"- f1_macro: {metrics['f1_macro']:.4f}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "resnet18_local_global.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. Defaults to best.pt under the configured output root.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = build_eval_dataset(config, split_name=args.split)
    class_names = ordered_class_names(dataset.class_to_index)
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = len(class_names)
    model_cfg["pretrained"] = False

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)

    output_root = Path(resolve_path(config["paths"]["output_root"]))
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = output_root / "checkpoints" / "best.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(checkpoint_path, model=model, map_location=device)

    predictions, targets = collect_predictions(model=model, dataloader=dataloader, device=device)
    matrix = confusion_matrix(predictions, targets, num_classes=len(class_names))
    metrics = classification_metrics_from_confusion(matrix)
    per_class_rows = per_class_report_from_confusion(matrix, class_names=class_names)

    eval_root = output_root / "evaluation" / f"{args.split}_{checkpoint_path.stem}"
    eval_root.mkdir(parents=True, exist_ok=True)

    metrics_json = eval_root / "metrics.json"
    metrics_md = eval_root / "metrics.md"
    per_class_csv = eval_root / "per_class_report.csv"
    per_class_md = eval_root / "per_class_report.md"
    matrix_csv = eval_root / "confusion_matrix.csv"
    matrix_md = eval_root / "confusion_matrix.md"
    matrix_png = eval_root / "confusion_matrix.png"

    metrics_payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(dataset),
        **metrics,
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_metrics_markdown(metrics_md, metrics_payload, checkpoint_path=checkpoint_path, split_name=args.split)
    write_per_class_report_csv(per_class_csv, per_class_rows)
    write_per_class_report_markdown(per_class_md, per_class_rows)
    write_confusion_matrix_csv(matrix_csv, matrix, class_names=class_names)
    write_confusion_matrix_markdown(matrix_md, matrix, class_names=class_names)
    plotted = maybe_plot_confusion_matrix(matrix_png, matrix, class_names=class_names)

    print("Evaluation complete")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Metrics JSON: {metrics_json}")
    print(f"  Per-class report: {per_class_csv}")
    print(f"  Confusion matrix CSV: {matrix_csv}")
    if plotted:
        print(f"  Confusion matrix PNG: {matrix_png}")


if __name__ == "__main__":
    main()
