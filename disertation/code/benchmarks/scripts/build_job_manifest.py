#!/usr/bin/env python3
"""Build validation-tuning or five-seed production jobs for run_gpu_queue.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SEEDS = [13, 21, 42, 87, 101]
BASELINES = [
    "distilbert_text",
    "spotfake_faknow",
    "cafe_faknow",
]
FROZEN = {"clip_linear", "spotfake_faknow", "cafe_faknow"}
STABLE_OVERRIDES = {
    ("fakeddit", "spotfake_faknow"): {
        "head_learning_rate": 1e-4,
    },
}


def load_best(root: Path, dataset: str, model: str) -> dict[str, object]:
    path = root / dataset / model / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Run validation tuning first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not payload.get("best"):
        raise RuntimeError(f"No completed tuning trial: {path}")
    best = dict(payload["best"])
    best.update(STABLE_OVERRIDES.get((dataset, model), {}))
    return best


def job(job_id: str, command: list[str], cwd: Path, completion: Path) -> dict[str, object]:
    return {
        "id": job_id,
        "command": command,
        "cwd": str(cwd.resolve()),
        "completion_file": str(completion.resolve()),
    }


def tuning_jobs(args, repo_root: Path) -> list[dict[str, object]]:
    jobs = []
    python = sys.executable
    for dataset in ("fakeddit", "checked"):
        for model in BASELINES:
            completion = args.tuning_root / dataset / model / "best_hyperparameters.json"
            command = [
                python,
                str(repo_root / "benchmarks" / "scripts" / "tune_baseline.py"),
                "--dataset", dataset,
                "--model", model,
                "--prepared-root", str(args.prepared_root.resolve()),
                "--output-root", str(args.tuning_root.resolve()),
                "--feature-cache-root", str(args.feature_cache_root.resolve()),
                "--learning-rates", "2e-5", "5e-5",
                "--head-learning-rates", "3e-4", "1e-3",
                "--weight-decays", "0.01",
                "--batch-sizes", "16", "32",
                "--epochs", "3" if dataset == "fakeddit" else "10",
                "--num-workers", str(args.num_workers),
            ]
            jobs.append(job(f"tune_{dataset}_{model}", command, repo_root, completion))

    fakeddit_completion = args.tuning_root / "fakeddit" / "defomo" / "best_hyperparameters.json"
    fakeddit_command = [
        python,
        str(repo_root / "benchmarks" / "scripts" / "tune_defomo.py"),
        "--dataset", "fakeddit",
        "--train-tsv", str(args.fakeddit_train.resolve()),
        "--image-root", str(args.fakeddit_images.resolve()),
        "--split-manifest-root", str((repo_root / "benchmarks" / "manifests" / "fakeddit").resolve()),
        "--output-root", str(args.tuning_root.resolve()),
        "--learning-rates", "1e-5", "2e-5", "5e-5",
        "--weight-decays", "0.01",
        "--batch-sizes", "2", "4",
        "--epochs", "5",
        "--num-workers", str(args.num_workers),
    ]
    jobs.insert(0, job("tune_fakeddit_defomo", fakeddit_command, repo_root, fakeddit_completion))

    checked_completion = args.tuning_root / "checked" / "defomo" / "best_hyperparameters.json"
    checked_command = [
        python,
        str(repo_root / "benchmarks" / "scripts" / "tune_defomo.py"),
        "--dataset", "checked",
        "--output-root", str(args.tuning_root.resolve()),
        "--checked-train-csv", str(args.checked_train.resolve()),
        "--checked-val-csv", str(args.checked_val.resolve()),
        "--checked-test-csv", str(args.checked_test.resolve()),
        "--learning-rates", "1e-5", "2e-5", "5e-5",
        "--weight-decays", "0.01",
        "--batch-sizes", "2", "4",
        "--epochs", "15",
    ]
    jobs.insert(1, job("tune_checked_defomo", checked_command, repo_root, checked_completion))
    return jobs


def final_jobs(args, repo_root: Path) -> list[dict[str, object]]:
    jobs = []
    python = sys.executable
    fakeddit_best = load_best(args.tuning_root, "fakeddit", "defomo")
    for seed in SEEDS:
        completion = args.results_root / "fakeddit" / "defomo" / f"seed_{seed}" / "result.json"
        command = [
            python,
            str(repo_root / "benchmarks" / "scripts" / "run_defomo_fakeddit.py"),
            "--seeds", str(seed),
            "--gpus", "1",
            "--train-tsv", str(args.fakeddit_train.resolve()),
            "--image-root", str(args.fakeddit_images.resolve()),
            "--output-root", str((args.results_root / "fakeddit" / "defomo").resolve()),
            "--learning-rate", str(fakeddit_best["learning_rate"]),
            "--weight-decay", str(fakeddit_best["weight_decay"]),
            "--per-gpu-batch-size", str(fakeddit_best["batch_size"]),
            "--epochs", "10",
            "--num-workers", str(args.num_workers),
        ]
        if args.fakeddit_test:
            command.extend(["--test-tsv", str(args.fakeddit_test.resolve())])
        else:
            command.extend(
                [
                    "--test-csv",
                    str(
                        (
                            repo_root
                            / "benchmarks"
                            / "portable_splits"
                            / "fakeddit"
                            / "test.csv"
                        ).resolve()
                    ),
                ]
            )
        jobs.append(job(f"final_fakeddit_defomo_seed_{seed}", command, repo_root, completion))

    for dataset in ("fakeddit", "checked"):
        for model in BASELINES:
            best = load_best(args.tuning_root, dataset, model)
            for seed in SEEDS:
                completion = args.results_root / dataset / model / f"seed_{seed}" / "result.json"
                command = [
                    python,
                    str(repo_root / "benchmarks" / "src" / "train_baseline.py"),
                    "--dataset", dataset,
                    "--model", model,
                    "--seed", str(seed),
                    "--prepared-root", str(args.prepared_root.resolve()),
                    "--output-root", str(args.results_root.resolve()),
                    "--epochs", "3" if dataset == "fakeddit" else "20",
                    "--batch-size", str(best["batch_size"]),
                    "--learning-rate", str(best["learning_rate"]),
                    "--head-learning-rate", str(best["head_learning_rate"]),
                    "--weight-decay", str(best["weight_decay"]),
                    "--num-workers", str(args.num_workers),
                ]
                if model in FROZEN:
                    command.extend(
                        [
                            "--freeze-encoders",
                            "--feature-cache-root", str(args.feature_cache_root.resolve()),
                            "--require-feature-cache",
                        ]
                    )
                if model == "spotfake_faknow":
                    command.append("--no-mixed-precision")
                jobs.append(job(f"final_{dataset}_{model}_seed_{seed}", command, repo_root, completion))

    checked_best = load_best(args.tuning_root, "checked", "defomo")
    for seed in SEEDS:
        run_name = f"run_seed_{seed}"
        completion = args.results_root / "checked" / "defomo" / run_name / f"seed_{seed}" / "metrics.json"
        command = [
            python,
            "train.py",
            "--train-csv", str(args.checked_train.resolve()),
            "--val-csv", str(args.checked_val.resolve()),
            "--test-csv", str(args.checked_test.resolve()),
            "--output-root", str((args.results_root / "checked" / "defomo").resolve()),
            "--run-name", run_name,
            "--ablation", "text_image_graph",
            "--seeds", str(seed),
            "--epochs", "20",
            "--batch-size", str(checked_best["batch_size"]),
            "--eval-batch-size", "8",
            "--learning-rate", str(checked_best["learning_rate"]),
            "--weight-decay", str(checked_best["weight_decay"]),
            "--use_class_weights",
            "--use_threshold_tuning",
            "--fair_mode",
            "--strict_fair_audit",
            "--metadata_mode", "safe",
            "--graph_variant", "full_safe",
            "--use_response_graph",
            "--disable_user_graph",
        ]
        jobs.append(job(f"final_checked_defomo_seed_{seed}", command, repo_root / "checked", completion))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["tuning", "final"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--tuning-root", type=Path, default=Path("benchmarks/tuning"))
    parser.add_argument("--results-root", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--feature-cache-root", type=Path, default=Path("benchmarks/feature_cache/frozen"))
    parser.add_argument("--fakeddit-train", type=Path, required=True)
    parser.add_argument(
        "--fakeddit-test",
        type=Path,
        default=None,
        help="Optional original test.tsv; otherwise use the bundled portable locked test CSV.",
    )
    parser.add_argument("--fakeddit-images", type=Path, required=True)
    parser.add_argument("--checked-train", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/train.csv"))
    parser.add_argument("--checked-val", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/val.csv"))
    parser.add_argument("--checked-test", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/test.csv"))
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    jobs = tuning_jobs(args, repo_root) if args.phase == "tuning" else final_jobs(args, repo_root)
    payload = {"phase": args.phase, "fixed_seeds": SEEDS, "jobs": jobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
