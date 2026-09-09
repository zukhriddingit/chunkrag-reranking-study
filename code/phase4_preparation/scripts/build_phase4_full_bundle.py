#!/usr/bin/env python3
"""Build a private, content-addressed Phase 4 Colab execution bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    prepared = args.prepared.resolve()
    bundle = args.bundle.resolve()
    working = workspace / "chunkrag-eaai-phase4-full-v1"
    import sys
    sys.path.insert(0, str(working / "src"))
    from chunkrag.eaai_phase4.full_execution import sha256_path, tree_inventory, write_immutable_json

    if bundle.exists():
        raise RuntimeError(f"Refusing existing bundle directory: {bundle}")
    bundle.mkdir(parents=True)
    shutil.copytree(working, bundle / "working_copy", ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc", ".DS_Store"))
    shutil.copytree(prepared, bundle / "frozen_execution")
    inventory = tree_inventory([bundle / "working_copy", bundle / "frozen_execution"], bundle)
    manifest = {
        "schema_version": 1,
        "bundle_id": "phase4_full_execution_bundle_v1",
        "private": True,
        "doc2dial_raw_data_included": False,
        "full_matrix_authorized": True,
        "hard_ceiling_allocated_A100_hours": 20.0,
        "inventory": inventory,
        "local_freeze_manifest_sha256": sha256_path(bundle / "frozen_execution/freeze/local_freeze_manifest.json"),
    }
    write_immutable_json(bundle / "bundle_manifest.json", manifest)
    print(json.dumps({
        "bundle": str(bundle),
        "manifest_sha256": sha256_path(bundle / "bundle_manifest.json"),
        "files": inventory["file_count"],
        "tree_sha256": inventory["tree_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
