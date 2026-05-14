from __future__ import annotations

import argparse
import sys

from data.build_manifest import main as build_manifest_main
from data.build_splits import main as build_splits_main
from data.preprocess_dataset import main as preprocess_dataset_main
from training.train import run_train

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="classification_v2 command entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("phase1-check", help="Verify V2 scaffold exists.")
    sub.add_parser("phase2-manifest", help="Build final manifest (Phase 2).")
    sub.add_parser("phase3-split", help="Build grouped splits (Phase 3).")
    sub.add_parser("phase4-preprocess", help="Preprocess split images (Phase 4).")
    sub.add_parser("phase5-train", help="Run EfficientNet-B0 training (Phase 5).")
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

    print(f"{args.command} is not implemented yet.")


if __name__ == "__main__":
    run()
