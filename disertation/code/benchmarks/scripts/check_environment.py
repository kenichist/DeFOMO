#!/usr/bin/env python3
"""Fail-fast environment check for the matched DeFOMO benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_ROWS = {
    "fakeddit": {"train": 642231, "val": 160558, "test": 84481},
    "checked": {"train": 1497, "val": 187, "test": 420},
}


def csv_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--offline-assets", type=Path, default=None)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    import torch
    import torch_geometric
    import torchvision
    import transformers
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader

    checks: dict[str, object] = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    failures: list[str] = []
    if not checks["cuda_available"]:
        failures.append("CUDA is unavailable")
    if checks["gpu_count"] < args.expected_gpus:
        failures.append(f"expected {args.expected_gpus} GPUs, found {checks['gpu_count']}")
    if torch.cuda.is_available() and not checks["bf16_supported"]:
        failures.append("CUDA is available but BF16 is unsupported")

    try:
        graph = HeteroData()
        graph["post"].x = torch.randn(3, 4)
        graph["user"].x = torch.randn(2, 4)
        edges = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
        graph["post", "authored_by", "user"].edge_index = edges
        graph["user", "authored", "post"].edge_index = edges.flip(0)
        loader = NeighborLoader(
            graph,
            input_nodes=("post", torch.tensor([0, 1])),
            num_neighbors={edge_type: [2, 2] for edge_type in graph.edge_types},
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        sampled = next(iter(loader))
        checks["pyg_neighbor_sampling"] = int(sampled["post"].batch_size) == 2
    except Exception as exc:
        checks["pyg_neighbor_sampling"] = False
        failures.append(f"PyG neighbor sampling failed: {exc!r}")

    split_rows: dict[str, dict[str, int | None]] = {}
    for dataset, expected in EXPECTED_ROWS.items():
        split_rows[dataset] = {}
        for split, expected_count in expected.items():
            path = args.prepared_root / dataset / f"{split}.csv"
            observed = csv_rows(path) if path.exists() else None
            split_rows[dataset][split] = observed
            if observed != expected_count:
                failures.append(f"{path}: expected {expected_count} rows, found {observed}")
    checks["prepared_rows"] = split_rows

    storage_path = args.prepared_root.resolve()
    free_bytes = shutil.disk_usage(storage_path).free
    checks["storage_path_checked"] = str(storage_path)
    checks["free_bytes"] = free_bytes
    if free_bytes < args.minimum_free_gb * 1024**3:
        failures.append(f"only {free_bytes / 1024**3:.1f} GiB free; require {args.minimum_free_gb:.1f} GiB")

    if args.offline_assets:
        manifest_path = args.offline_assets / "offline_assets_manifest.json"
        if not manifest_path.exists():
            failures.append(f"offline cache manifest missing: {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            checks["offline_assets_complete"] = bool(manifest.get("complete"))
            checks["offline_assets_bytes"] = manifest.get("total_bytes")
            if not manifest.get("complete"):
                failures.append(f"offline cache is incomplete: {manifest_path}")

    report = {"ready": not failures, "checks": checks, "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        base_version = torch.__version__.split("+")[0]
        cuda_tag = "cpu" if not torch.version.cuda else "cu" + torch.version.cuda.replace(".", "")
        print(
            "PyG wheel hint: python -m pip install pyg_lib -f "
            f"https://data.pyg.org/whl/torch-{base_version}+{cuda_tag}.html"
        )
        raise SystemExit("environment preflight failed")


if __name__ == "__main__":
    main()
