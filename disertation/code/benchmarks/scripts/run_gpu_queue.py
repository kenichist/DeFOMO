#!/usr/bin/env python3
"""Run a resumable JSON job manifest with one independent process per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--status-file", type=Path, default=Path("benchmarks/queue_status.json"))
    parser.add_argument("--logs-root", type=Path, default=Path("benchmarks/queue_logs"))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.jobs_file.read_text(encoding="utf-8-sig"))
    jobs = payload["jobs"] if isinstance(payload, dict) else payload
    state = {job["id"]: {"status": "pending", "returncode": None, "gpu": None} for job in jobs}
    pending = []
    for job in jobs:
        completion = Path(job["completion_file"])
        if completion.exists() and not args.rerun_complete:
            state[job["id"]]["status"] = "skipped_complete"
        else:
            pending.append(job)
    running: dict[str, dict[str, object]] = {}
    available = list(args.gpus)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    def snapshot() -> None:
        counts = {}
        for entry in state.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        atomic_json(args.status_file, {"counts": counts, "jobs": state})

    snapshot()
    try:
        while pending or running:
            while pending and available:
                job = pending.pop(0)
                gpu = available.pop(0)
                log_path = args.logs_root / f"{job['id']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment.update({str(key): str(value) for key, value in job.get("env", {}).items()})
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    [str(part) for part in job["command"]],
                    cwd=job.get("cwd") or None,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[job["id"]] = {
                    "job": job,
                    "gpu": gpu,
                    "process": process,
                    "handle": handle,
                    "started": time.time(),
                }
                state[job["id"]].update({"status": "running", "gpu": gpu, "log": str(log_path.resolve())})
                print(f"START gpu={gpu} id={job['id']}", flush=True)
                snapshot()

            finished = []
            for job_id, item in running.items():
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].close()
                completion = Path(item["job"]["completion_file"])
                status = "complete" if returncode == 0 and completion.exists() else "failed"
                state[job_id].update(
                    {
                        "status": status,
                        "returncode": returncode,
                        "elapsed_seconds": time.time() - item["started"],
                    }
                )
                available.append(item["gpu"])
                available.sort(key=lambda value: args.gpus.index(value))
                finished.append(job_id)
                print(f"END gpu={item['gpu']} id={job_id} status={status} rc={returncode}", flush=True)
            for job_id in finished:
                del running[job_id]
            if finished:
                snapshot()
            if pending or running:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        for item in running.values():
            item["process"].terminate()
            item["handle"].close()
        for job_id in running:
            state[job_id]["status"] = "interrupted"
        snapshot()
        raise
    snapshot()
    failures = [job_id for job_id, entry in state.items() if entry["status"] == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect {args.status_file}")


if __name__ == "__main__":
    main()
