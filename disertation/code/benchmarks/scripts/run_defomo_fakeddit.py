#!/usr/bin/env python3
"""Run DeFOMO Fakeddit training and held-out evaluation for fixed split seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [13, 21, 42, 87, 101]


def completed_training_artifacts(
    best_checkpoint: Path,
    training_summary: Path,
    expected_seed: int,
) -> bool:
    """Return true only when both the selected checkpoint and validation summary are usable."""
    if (
        not best_checkpoint.is_file()
        or best_checkpoint.stat().st_size == 0
        or not training_summary.is_file()
    ):
        return False
    try:
        payload = json.loads(training_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("dataset") == "fakeddit"
        and payload.get("model") == "defomo"
        and payload.get("seed") == expected_seed
        and isinstance(payload.get("validation"), dict)
        and "macro_f1" in payload["validation"]
    )


def run_command(command, workdir: Path, log_path: Path, dry_run: bool) -> int:
    print(" ".join(str(part) for part in command), flush=True)
    if dry_run:
        return 0
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--train-tsv", type=Path, required=True)
    test_source = parser.add_mutually_exclusive_group()
    test_source.add_argument("--test-tsv", type=Path, default=None)
    test_source.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split-manifest-root", type=Path, default=None)
    parser.add_argument("--text-model", default="distilbert-base-uncased")
    parser.add_argument("--vit-model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/results/fakeddit/defomo"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-num-workers", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--neighbor-fanouts", nargs="+", type=int, default=[15, 10])
    parser.add_argument("--neighbor-workers", type=int, default=0)
    parser.add_argument("--graph-feature-cache-dir", type=Path, default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--rerun-training",
        action="store_true",
        help="Retrain even when a complete validation summary and best checkpoint already exist",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.gpus != 1:
        parser.error("DeFOMO graph runs use one GPU per seed; schedule seeds independently across GPUs.")
    split_manifest_root = args.split_manifest_root or (repo_root / "benchmarks" / "manifests" / "fakeddit")
    test_csv = args.test_csv
    if args.test_tsv is None and test_csv is None:
        test_csv = repo_root / "benchmarks" / "portable_splits" / "fakeddit" / "test.csv"
    if args.test_tsv is not None and not args.test_tsv.is_file():
        parser.error(f"--test-tsv does not exist: {args.test_tsv}")
    if test_csv is not None and not test_csv.is_file():
        parser.error(f"portable Fakeddit test CSV does not exist: {test_csv}")
    graph_feature_cache_dir = args.graph_feature_cache_dir or (repo_root / "benchmarks" / "feature_cache" / "defomo_graph")
    fakeddit_root = repo_root / "fakeddit"
    training_script = fakeddit_root / "train_multimodal_ee.py"
    evaluation_script = fakeddit_root / "test_multimodal_ee.py"
    statuses = []
    for seed in args.seeds:
        run_dir = args.output_root / f"seed_{seed}"
        result_path = run_dir / "result.json"
        if result_path.exists() and not args.rerun_complete:
            statuses.append({"seed": seed, "status": "skipped_complete"})
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best_model.pt"
        last_checkpoint = run_dir / "last_checkpoint.pt"
        training_summary = run_dir / ("result.json" if args.validation_only else "validation_result.json")
        train_command = [
            sys.executable,
            str(training_script),
            "--text", "--image", "--graph",
            "--gpus", str(args.gpus),
            "--seed", str(seed),
            "--split-seed", "42",
            "--split-manifest-root", str(split_manifest_root.resolve()),
            "--train-tsv", str(args.train_tsv.resolve()),
            "--image-root", str(args.image_root.resolve()),
            "--text-model", args.text_model,
            "--vit-model", args.vit_model,
            "--save-path", str(checkpoint.resolve()),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--per-gpu-batch-size", str(args.per_gpu_batch_size),
            "--num-workers", str(args.num_workers),
            "--mixed-precision", args.mixed_precision,
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--neighbor-fanouts", *[str(value) for value in args.neighbor_fanouts],
            "--neighbor-workers", str(args.neighbor_workers),
            "--graph-feature-cache-dir", str(graph_feature_cache_dir.resolve()),
            "--last-checkpoint", str(last_checkpoint.resolve()),
            "--training-summary-json", str(training_summary.resolve()),
            "--training-metrics-jsonl", str((run_dir / "training_metrics.jsonl").resolve()),
        ]
        if last_checkpoint.exists() and not args.no_auto_resume:
            train_command.extend(["--resume-from", str(last_checkpoint.resolve())])
        if args.smoke_limit:
            train_command.extend(["--smoke-limit", str(args.smoke_limit)])
        best_checkpoint = Path(str(checkpoint) + ".best")
        if (
            not args.rerun_training
            and completed_training_artifacts(best_checkpoint, training_summary, seed)
        ):
            print(
                f"Reusing completed training artifacts for seed {seed}: "
                f"{best_checkpoint}",
                flush=True,
            )
            train_code = 0
        else:
            train_code = run_command(
                train_command,
                fakeddit_root,
                run_dir / "train.log",
                args.dry_run,
            )
        if train_code != 0:
            statuses.append({"seed": seed, "status": "training_failed", "returncode": train_code})
            continue
        if args.validation_only:
            status = "dry_run" if args.dry_run else ("complete" if result_path.exists() else "validation_output_missing")
            statuses.append({"seed": seed, "status": status, "returncode": train_code})
            if args.smoke_limit and status == "complete":
                for path in (checkpoint, Path(str(checkpoint) + ".best"), last_checkpoint):
                    path.unlink(missing_ok=True)
            continue
        eval_command = [
            sys.executable,
            str(evaluation_script),
            "--text", "--image", "--graph",
            "--model-path", str(checkpoint.resolve()) + ".best",
            "--test-manifest", str((split_manifest_root / "test.csv").resolve()),
            "--image-root", str(args.image_root.resolve()),
            "--text-model", args.text_model,
            "--vit-model", args.vit_model,
            "--seed", str(seed),
            "--gpus", str(args.gpus),
            "--output-json", str(result_path.resolve()),
            "--predictions-csv", str((run_dir / "test_predictions.csv").resolve()),
            "--mixed-precision", args.mixed_precision,
            "--batch-size", str(args.eval_batch_size),
            "--num-workers", str(args.eval_num_workers),
            "--neighbor-fanouts", *[str(value) for value in args.neighbor_fanouts],
            "--neighbor-workers", str(args.neighbor_workers),
            "--graph-feature-cache-dir", str(graph_feature_cache_dir.resolve()),
        ]
        if args.test_tsv is not None:
            eval_command.extend(["--test-tsv", str(args.test_tsv.resolve())])
        else:
            eval_command.extend(["--test-csv", str(test_csv.resolve())])
        if args.smoke_limit:
            eval_command.extend(["--smoke-limit", str(args.smoke_limit)])
        eval_code = run_command(eval_command, fakeddit_root, run_dir / "test.log", args.dry_run)
        status = "dry_run" if args.dry_run else ("complete" if eval_code == 0 else "evaluation_failed")
        statuses.append({"seed": seed, "status": status, "returncode": eval_code})
        if args.smoke_limit and status == "complete":
                for path in (checkpoint, best_checkpoint, last_checkpoint):
                    path.unlink(missing_ok=True)
    manifest = {"fixed_split_seed": 42, "runs": statuses}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "status.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    failures = [run for run in statuses if run["status"] not in {"complete", "skipped_complete", "dry_run"}]
    if failures:
        raise SystemExit(f"{len(failures)} DeFOMO runs failed; inspect status.json and per-seed logs")


if __name__ == "__main__":
    main()
