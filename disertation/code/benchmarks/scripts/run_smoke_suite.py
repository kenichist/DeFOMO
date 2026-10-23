#!/usr/bin/env python3
"""Run one tiny end-to-end job for every baseline and both DeFOMO paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASELINES = (
    "distilbert_text",
    "bert_text",
    "roberta_text",
    "resnet50_image",
    "bert_resnet_concat",
    "debertnext_matched",
    "clip_linear",
    "spotfake_faknow",
    "cafe_faknow",
)


def execute(name: str, command: list[str], cwd: Path, log_root: Path, dry_run: bool) -> dict:
    print(f"[{name}] {' '.join(command)}", flush=True)
    if dry_run:
        return {"name": name, "status": "dry_run", "command": command}
    with (log_root / f"{name}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "name": name,
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--fakeddit-train-tsv", type=Path, required=True)
    parser.add_argument("--fakeddit-test-tsv", type=Path, default=None)
    parser.add_argument("--fakeddit-images", type=Path, required=True)
    parser.add_argument("--checked-train", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/train.csv"))
    parser.add_argument("--checked-val", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/val.csv"))
    parser.add_argument("--checked-test", type=Path, default=Path("checked/CHECKED/data_checked/processed/splits_group_content/test.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/smoke_results"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str], Path]] = []
    for dataset in ("fakeddit", "checked"):
        jobs.append(
            (
                f"baselines_{dataset}",
                [
                    sys.executable,
                    str(repo_root / "benchmarks/scripts/run_matrix.py"),
                    "--datasets", dataset,
                    "--models", *BASELINES,
                    "--seeds", "13",
                    "--prepared-root", str(args.prepared_root.resolve()),
                    "--output-root", str(output_root / "baselines"),
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--num-workers", "0",
                    "--smoke-limit", str(args.limit),
                    "--allow-uncached-frozen",
                    "--rerun-complete",
                ],
                repo_root,
            )
        )
    jobs.append(
        (
            "defomo_fakeddit",
            [
                sys.executable,
                str(repo_root / "benchmarks/scripts/run_defomo_fakeddit.py"),
                "--seeds", "13",
                "--train-tsv", str(args.fakeddit_train_tsv.resolve()),
                "--image-root", str(args.fakeddit_images.resolve()),
                "--output-root", str(output_root / "defomo_fakeddit"),
                "--epochs", "1",
                "--per-gpu-batch-size", "2",
                "--num-workers", "0",
                "--neighbor-workers", "0",
                "--smoke-limit", str(args.limit),
                "--rerun-complete",
            ],
            repo_root,
        )
    )
    if args.fakeddit_test_tsv:
        jobs[-1][1].extend(["--test-tsv", str(args.fakeddit_test_tsv.resolve())])
    else:
        jobs[-1][1].extend(
            [
                "--test-csv",
                str((repo_root / "benchmarks/portable_splits/fakeddit/test.csv").resolve()),
            ]
        )
    jobs.append(
        (
            "defomo_checked",
            [
                sys.executable,
                str(repo_root / "checked/train.py"),
                "--train-csv", str(args.checked_train.resolve()),
                "--val-csv", str(args.checked_val.resolve()),
                "--test-csv", str(args.checked_test.resolve()),
                "--output-root", str(output_root / "defomo_checked"),
                "--run-name", "smoke",
                "--ablation", "text_image_graph",
                "--seeds", "13",
                "--epochs", "1",
                "--batch-size", "2",
                "--eval-batch-size", "2",
                "--num-workers", "0",
                "--smoke-limit", str(args.limit),
                "--fair_mode",
                "--metadata_mode", "safe",
                "--graph_variant", "full_safe",
                "--use_response_graph",
                "--disable_user_graph",
            ],
            repo_root / "checked",
        )
    )

    statuses = [execute(name, command, cwd, output_root, args.dry_run) for name, command, cwd in jobs]
    manifest = {"limit": args.limit, "dry_run": args.dry_run, "jobs": statuses}
    (output_root / "smoke_status.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if any(item["status"] == "failed" for item in statuses):
        raise SystemExit("one or more smoke-test groups failed; inspect the named logs")


if __name__ == "__main__":
    main()
