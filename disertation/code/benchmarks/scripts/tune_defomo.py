#!/usr/bin/env python3
"""Tune DeFOMO on locked validation splits without evaluating either test set."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fakeddit", "checked"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 2e-5, 5e-5])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.01])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/tuning"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-tsv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split-manifest-root", type=Path, default=None)
    parser.add_argument("--text-model", default="distilbert-base-uncased")
    parser.add_argument("--vit-model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--neighbor-fanouts", nargs="+", type=int, default=[15, 10])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checked-train-csv", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/train.csv"))
    parser.add_argument("--checked-val-csv", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/val.csv"))
    parser.add_argument("--checked-test-csv", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/test.csv"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    search_root = args.output_root / args.dataset / "defomo"
    trials: list[dict[str, object]] = []
    grid = itertools.product(args.learning_rates, args.weight_decays, args.batch_sizes)
    for index, (learning_rate, weight_decay, batch_size) in enumerate(grid):
        trial_root = search_root / f"trial_{index:03d}"
        if args.dataset == "fakeddit":
            if not args.train_tsv or not args.image_root:
                parser.error("Fakeddit tuning requires --train-tsv and --image-root")
            result_path = trial_root / f"seed_{args.seed}" / "result.json"
            command = [
                sys.executable,
                str(repo_root / "benchmarks" / "scripts" / "run_defomo_fakeddit.py"),
                "--seeds", str(args.seed),
                "--gpus", "1",
                "--train-tsv", str(args.train_tsv.resolve()),
                "--image-root", str(args.image_root.resolve()),
                "--text-model", args.text_model,
                "--vit-model", args.vit_model,
                "--output-root", str(trial_root.resolve()),
                "--epochs", str(args.epochs),
                "--learning-rate", str(learning_rate),
                "--weight-decay", str(weight_decay),
                "--per-gpu-batch-size", str(batch_size),
                "--mixed-precision", args.mixed_precision,
                "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                "--neighbor-fanouts", *[str(value) for value in args.neighbor_fanouts],
                "--num-workers", str(args.num_workers),
                "--validation-only",
            ]
            if args.split_manifest_root:
                command.extend(["--split-manifest-root", str(args.split_manifest_root.resolve())])
            returncode = run(command, repo_root, trial_root / "tuning.log", args.dry_run)
            metrics_path = result_path
        else:
            command = [
                sys.executable,
                "train.py",
                "--train-csv", str(args.checked_train_csv.resolve()),
                "--val-csv", str(args.checked_val_csv.resolve()),
                "--test-csv", str(args.checked_test_csv.resolve()),
                "--output-root", str(trial_root.resolve()),
                "--ablation", "text_image_graph",
                "--seeds", str(args.seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(batch_size),
                "--eval-batch-size", str(max(8, batch_size)),
                "--learning-rate", str(learning_rate),
                "--weight-decay", str(weight_decay),
                "--use_class_weights",
                "--use_threshold_tuning",
                "--fair_mode",
                "--strict_fair_audit",
                "--metadata_mode", "safe",
                "--graph_variant", "full_safe",
                "--use_response_graph",
                "--disable_user_graph",
                "--validation-only",
            ]
            returncode = run(command, repo_root / "checked", trial_root / "tuning.log", args.dry_run)
            candidates = sorted(trial_root.glob(f"checked_text_image_graph_*/seed_{args.seed}/metrics.json"))
            metrics_path = candidates[-1] if candidates else None

        score = None
        if not args.dry_run and returncode == 0 and metrics_path and Path(metrics_path).exists():
            payload = json.loads(Path(metrics_path).read_text(encoding="utf-8-sig"))
            score = float(payload["validation"]["macro_f1"])
        trials.append(
            {
                "trial": index,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "validation_macro_f1": score,
                "returncode": returncode,
                "result": str(metrics_path) if metrics_path else None,
            }
        )

    completed = [trial for trial in trials if trial["validation_macro_f1"] is not None]
    best = max(completed, key=lambda item: item["validation_macro_f1"]) if completed else None
    output = {
        "dataset": args.dataset,
        "model": "defomo",
        "selection_seed": args.seed,
        "selection_metric": "validation_macro_f1",
        "test_evaluated_during_search": False,
        "best": best,
        "trials": trials,
    }
    search_root.mkdir(parents=True, exist_ok=True)
    (search_root / "best_hyperparameters.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
