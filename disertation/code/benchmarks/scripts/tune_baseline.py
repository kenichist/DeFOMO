#!/usr/bin/env python3
"""Grid-search one baseline using validation macro-F1 only."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


MODELS = [
    "distilbert_text",
    "bert_text",
    "roberta_text",
    "resnet50_image",
    "bert_resnet_concat",
    "debertnext_matched",
    "clip_linear",
    "spotfake_faknow",
    "cafe_faknow",
]
FROZEN = {"clip_linear", "spotfake_faknow", "cafe_faknow"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fakeddit", "checked"], required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 2e-5, 5e-5])
    parser.add_argument("--head-learning-rates", nargs="+", type=float, default=[3e-4, 1e-3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/tuning"))
    parser.add_argument("--feature-cache-root", type=Path, default=Path("benchmarks/feature_cache/frozen"))
    parser.add_argument("--allow-uncached-frozen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    trainer = repo_root / "benchmarks" / "src" / "train_baseline.py"
    search_root = args.output_root / args.dataset / args.model
    trials = []
    grid = itertools.product(args.learning_rates, args.head_learning_rates, args.weight_decays, args.batch_sizes)
    for index, (lr, head_lr, weight_decay, batch_size) in enumerate(grid):
        trial_root = search_root / f"trial_{index:03d}"
        result_path = trial_root / args.dataset / args.model / f"seed_{args.seed}" / "result.json"
        command = [
            sys.executable,
            str(trainer),
            "--dataset", args.dataset,
            "--model", args.model,
            "--seed", str(args.seed),
            "--prepared-root", str(args.prepared_root),
            "--output-root", str(trial_root),
            "--epochs", str(args.epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(lr),
            "--head-learning-rate", str(head_lr),
            "--weight-decay", str(weight_decay),
            "--num-workers", str(args.num_workers),
            "--validation-only",
        ]
        if args.model in FROZEN:
            command.append("--freeze-encoders")
            command.extend(["--feature-cache-root", str(args.feature_cache_root)])
            if not args.allow_uncached_frozen:
                command.append("--require-feature-cache")
        print(" ".join(command), flush=True)
        if args.dry_run:
            score = None
            returncode = 0
        else:
            run_dir = result_path.parent
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "tuning.log").open("w", encoding="utf-8") as log:
                returncode = subprocess.run(
                    command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
            score = None
            if returncode == 0 and result_path.exists():
                score = json.loads(result_path.read_text(encoding="utf-8"))["validation"]["macro_f1"]
        trials.append(
            {
                "trial": index,
                "learning_rate": lr,
                "head_learning_rate": head_lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "validation_macro_f1": score,
                "returncode": returncode,
                "result": str(result_path),
            }
        )
    completed = [trial for trial in trials if trial["validation_macro_f1"] is not None]
    best = max(completed, key=lambda trial: trial["validation_macro_f1"]) if completed else None
    output = {
        "dataset": args.dataset,
        "model": args.model,
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
