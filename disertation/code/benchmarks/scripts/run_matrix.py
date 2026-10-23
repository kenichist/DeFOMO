#!/usr/bin/env python3
"""Run the matched benchmark over models, datasets, and optimization seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [13, 21, 42, 87, 101]
DEFAULT_MODELS = [
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
FROZEN_BY_DEFAULT = {"clip_linear", "spotfake_faknow", "cafe_faknow"}


def load_tuned_hyperparameters(tuning_root: Path, dataset: str, model: str) -> dict[str, object]:
    path = tuning_root / dataset / model / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tuned configuration for {dataset}/{model}: {path}. "
            "Run tune_baseline.py first or omit --use-tuned-hyperparameters."
        )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    best = payload.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"No completed best trial in {path}")
    required = {"learning_rate", "head_learning_rate", "weight_decay", "batch_size"}
    missing = required.difference(best)
    if missing:
        raise ValueError(f"Missing fields in {path}: {sorted(missing)}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["fakeddit", "checked"], default=["fakeddit", "checked"])
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--tuning-root", type=Path, default=Path("benchmarks/tuning"))
    parser.add_argument(
        "--use-tuned-hyperparameters",
        action="store_true",
        help="Require and apply each dataset/model best_hyperparameters.json file.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-cache-root", type=Path, default=Path("benchmarks/feature_cache/frozen"))
    parser.add_argument("--allow-uncached-frozen", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    trainer = repo_root / "benchmarks" / "src" / "train_baseline.py"
    tasks = []
    for dataset in args.datasets:
        for model in args.models:
            tuned = (
                load_tuned_hyperparameters(args.tuning_root, dataset, model)
                if args.use_tuned_hyperparameters
                else None
            )
            for seed in args.seeds:
                run_dir = args.output_root / dataset / model / f"seed_{seed}"
                result_path = run_dir / "result.json"
                if result_path.exists() and not args.rerun_complete:
                    tasks.append({"dataset": dataset, "model": model, "seed": seed, "status": "skipped_complete"})
                    continue
                command = [
                    sys.executable,
                    str(trainer),
                    "--dataset", dataset,
                    "--model", model,
                    "--seed", str(seed),
                    "--prepared-root", str(args.prepared_root),
                    "--output-root", str(args.output_root),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(tuned["batch_size"] if tuned else args.batch_size),
                    "--num-workers", str(args.num_workers),
                ]
                if tuned:
                    command.extend(
                        [
                            "--learning-rate", str(tuned["learning_rate"]),
                            "--head-learning-rate", str(tuned["head_learning_rate"]),
                            "--weight-decay", str(tuned["weight_decay"]),
                        ]
                    )
                if model in FROZEN_BY_DEFAULT:
                    command.append("--freeze-encoders")
                    command.extend(["--feature-cache-root", str(args.feature_cache_root)])
                    if not args.allow_uncached_frozen:
                        command.append("--require-feature-cache")
                if args.smoke_limit:
                    command.extend(["--smoke-limit", str(args.smoke_limit)])
                print(" ".join(command), flush=True)
                if args.dry_run:
                    tasks.append({"dataset": dataset, "model": model, "seed": seed, "status": "dry_run", "command": command})
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "console.log"
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=False)
                status = "complete" if completed.returncode == 0 else "failed"
                task = {"dataset": dataset, "model": model, "seed": seed, "status": status, "returncode": completed.returncode}
                tasks.append(task)
                if completed.returncode != 0:
                    (run_dir / "failure.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
                    print(f"FAILED: inspect {log_path}", file=sys.stderr, flush=True)
    manifest = {"tasks": tasks}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "matrix_status.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    failures = [task for task in tasks if task["status"] == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} benchmark jobs failed; inspect matrix_status.json and seed logs")


if __name__ == "__main__":
    main()
