from __future__ import annotations

import argparse
import sys

from data.build_manifest import main as build_manifest_main
from data.build_splits import main as build_splits_main
from data.preprocess_dataset import main as preprocess_dataset_main
from training.compare_runs import run_compare_runs
from training.evaluate import run_evaluate
from training.plot_metrics import run_plot_metrics
from training.predict_single import run_predict_single
from training.train import run_train

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="classification_v2 command entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("phase1-check", help="Verify V2 scaffold exists.")
    sub.add_parser("phase2-manifest", help="Build final manifest (Phase 2).")
    sub.add_parser("phase3-split", help="Build grouped splits (Phase 3).")
    sub.add_parser("phase4-preprocess", help="Preprocess split images (Phase 4).")
    sub.add_parser("phase5-train", help="Run EfficientNet-B0 training (Phase 5).")
    sub.add_parser("phase6-eval", help="Evaluate a trained checkpoint (Phase 6).")
    sub.add_parser("phase7-plot", help="Plot losses and metrics from a run directory.")
    sub.add_parser("phase8-compare", help="Compare multiple run summaries.")
    sub.add_parser("phase9-predict", help="Run single-image inference.")
    return parser


def run() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(sys.argv[1:])

    if args.command == "phase1-check":
        print("Phase 1 scaffold is active under classification_v2/.")
        return
    if args.command == "phase2-manifest":
        build_manifest_main(remaining)
        return
    if args.command == "phase3-split":
        build_splits_main(remaining)
        return
    if args.command == "phase4-preprocess":
        preprocess_dataset_main(remaining)
        return
    if args.command == "phase5-train":
        run_train(remaining)
        return
    if args.command == "phase6-eval":
        run_evaluate(remaining)
        return
    if args.command == "phase7-plot":
        run_plot_metrics(remaining)
        return
    if args.command == "phase8-compare":
        run_compare_runs(remaining)
        return
    if args.command == "phase9-predict":
        run_predict_single(remaining)
        return

    print(f"{args.command} is not implemented yet.")


if __name__ == "__main__":
    run()
