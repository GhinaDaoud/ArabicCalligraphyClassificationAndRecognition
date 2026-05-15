from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple training runs from summary.json files.")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories under classification_v2/outputs/training.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("classification_v2/artifacts"),
        help="Output directory for comparison report files.",
    )
    p.add_argument("--name", type=str, default="run_comparison", help="Base name for output files.")
    return p.parse_args(argv)


def run_compare_runs(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    missing_inputs: list[str] = []
    for rd in args.run_dirs:
        candidate = Path(rd)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate

        # Accept either:
        # 1) direct summary.json path
        # 2) run directory containing summary.json
        if candidate.is_file() and candidate.name.lower() == "summary.json":
            summary_path = candidate
            run_dir = candidate.parent
        else:
            run_dir = candidate
            summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            missing_inputs.append(str(candidate))
            continue

        s = _read_json(summary_path)

        rows.append(
            {
                "run_name": str(s.get("run_name", run_dir.name)),
                "run_dir": str(run_dir),
                "best_epoch": int(float(s.get("best", {}).get("best_epoch", -1))),
                "val_macro_f1": float(s.get("val", {}).get("macro_f1", 0.0)),
                "val_accuracy": float(s.get("val", {}).get("accuracy", 0.0)),
                "val_loss": float(s.get("val", {}).get("loss", 0.0)),
                "test_macro_f1": float(s.get("test", {}).get("macro_f1", 0.0)),
                "test_accuracy": float(s.get("test", {}).get("accuracy", 0.0)),
                "test_loss": float(s.get("test", {}).get("loss", 0.0)),
                "checkpoint_path": str(s.get("best", {}).get("checkpoint_path", "")),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No valid summaries found. Missing summary.json for inputs:\n- "
            + "\n- ".join(missing_inputs)
        )

    rows.sort(key=lambda r: r["test_macro_f1"], reverse=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}.csv"
    md_path = out_dir / f"{args.name}.md"

    fieldnames = [
        "run_name",
        "run_dir",
        "best_epoch",
        "val_macro_f1",
        "val_accuracy",
        "val_loss",
        "test_macro_f1",
        "test_accuracy",
        "test_loss",
        "checkpoint_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Run Comparison",
        "",
        "| run_name | best_epoch | val_macro_f1 | test_macro_f1 | val_acc | test_acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['best_epoch']} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | "
            f"{r['val_accuracy']:.4f} | {r['test_accuracy']:.4f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    run_compare_runs()
