#!/usr/bin/env python3
"""Create the local, append-only Phase 4 execution freeze and private bundle inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--working-copy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    working = args.working_copy.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(working / "src"))
    from chunkrag.eaai_phase4.full_execution import sha256_path, tree_inventory, write_immutable_json

    if output.exists() and any(output.rglob("*")):
        raise RuntimeError(f"New append-only output namespace is not empty: {output}")
    for directory in ("freeze", "private_inputs", "source", "qa", "analysis"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    base = workspace / "outputs/01a06a4b-73f7-7e23-a621-6e892e50f47f"
    preflight = base / "phase4_cpu_preflight_v1"
    smoke_report = base / "phase4_development_smoke_report_v1"
    smoke_audit = base / "phase4_development_smoke_v4_audit_bundle"
    phase3_freeze = base / "phase3_primary_metric_reconciliation_v1/private/verified_inputs/artifacts/eaai_phase3/techqa_evidence_reranker_v1/manifests/freeze.json"
    preservation_before = base / "phase4_development_smoke_v2/private/preservation_inventory_before.json"
    config_path = working / "configs/eaai_phase4/phase4_full_execution_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required = {
        "phase3_freeze": (phase3_freeze, config["phase3_boundary"]["freeze_file_sha256"]),
        "preflight_manifest": (preflight / "manifest.json", config["provenance"]["preflight_manifest_sha256"]),
        "successful_smoke_manifest": (smoke_audit / "results/smoke_manifest.json", config["provenance"]["successful_smoke_manifest_sha256"]),
        "successful_smoke_report_manifest": (smoke_report / "manifest.json", config["provenance"]["successful_smoke_report_manifest_sha256"]),
        "successful_smoke_runner": (working / "scripts/run_phase4_smoke.py", config["provenance"]["successful_smoke_runner_sha256"]),
        "B_training_groups": (preflight / "private/B_training_groups_label_blind_pair_prune_v1.jsonl", config["inputs"]["B_training_groups_sha256"]),
        "E_training_groups": (preflight / "private/E_training_groups_label_blind_pair_prune_v1.jsonl", config["inputs"]["E_training_groups_sha256"]),
        "B_validation_labels": (preflight / "private/B_validation_labels.jsonl", config["inputs"]["B_validation_labels_sha256"]),
        "E_validation_labels": (preflight / "private/E_validation_labels.jsonl", config["inputs"]["E_validation_labels_sha256"]),
        "B_refit_groups": (preflight / "private/B_refit_groups_label_blind_pair_prune_v1.jsonl", config["inputs"]["B_refit_groups_sha256"]),
        "E_refit_groups": (preflight / "private/E_refit_groups_label_blind_pair_prune_v1.jsonl", config["inputs"]["E_refit_groups_sha256"]),
        "doc2dial_selection_index": (preflight / "private/doc2dial_v1_0_1_one_case_per_document_index.jsonl", config["inputs"]["doc2dial_selection_index_sha256"]),
        "finalized_protocol": (smoke_report / "protocol/phase4_execution_ready_protocol_v2.json", "87cbc55f57526fe31af3931df4157a2216db146038376d42a1b66a1844373a6a"),
        "smoke_verification": (smoke_report / "analysis/smoke_verification.json", "d1a48f0a76f9bdc3db84794954964abbaedce1fa4fac3b26b90ed2b40f54b0fe"),
    }
    verified = {}
    for name, (path, expected) in required.items():
        observed = sha256_path(path)
        if observed != expected:
            raise RuntimeError(f"Freeze mismatch for {name}: expected {expected}, observed {observed}")
        verified[name] = {"source": str(path), "sha256": observed, "bytes": path.stat().st_size}

    freeze = json.loads(phase3_freeze.read_text(encoding="utf-8"))
    stored_identity = str(freeze.get("manifest_sha256", freeze.get("manifest_identity", "")))
    if stored_identity != config["phase3_boundary"]["freeze_identity"]:
        raise RuntimeError("Phase 3 freeze identity does not match the approved boundary")

    before = json.loads(preservation_before.read_text(encoding="utf-8"))
    missing = []
    changed = []
    for relative, identity in before["files"].items():
        path = workspace / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_path(path) != identity["sha256"]:
            changed.append(relative)
    if missing or changed:
        raise RuntimeError(f"Preservation gate failed: {len(missing)} missing, {len(changed)} changed")

    input_map = {
        "B_training_groups.jsonl": required["B_training_groups"][0],
        "E_training_groups.jsonl": required["E_training_groups"][0],
        "B_validation_labels.jsonl": required["B_validation_labels"][0],
        "E_validation_labels.jsonl": required["E_validation_labels"][0],
        "B_refit_groups.jsonl": required["B_refit_groups"][0],
        "E_refit_groups.jsonl": required["E_refit_groups"][0],
        "doc2dial_selection_index.jsonl": required["doc2dial_selection_index"][0],
    }
    for target_name, source in input_map.items():
        shutil.copy2(source, output / "private_inputs" / target_name)

    source_paths = [
        "configs/eaai_phase4/phase4_full_execution_v1.json",
        "configs/eaai_phase2/techqa_adaptive_v1.json",
        "src/chunkrag/__init__.py",
        "src/chunkrag/schemas.py",
        "src/chunkrag/text_utils.py",
        "src/chunkrag/chunking.py",
        "src/chunkrag/retrieval.py",
        "src/chunkrag/eaai_phase3/evidence.py",
        "src/chunkrag/eaai_phase3/generation_runtime.py",
        "src/chunkrag/eaai_phase4/__init__.py",
        "src/chunkrag/eaai_phase4/preprocessing.py",
        "src/chunkrag/eaai_phase4/training_contract.py",
        "src/chunkrag/eaai_phase4/dialogue_contract.py",
        "src/chunkrag/eaai_phase4/statistics.py",
        "src/chunkrag/eaai_phase4/full_execution.py",
        "scripts/run_phase4_smoke.py",
        "scripts/prepare_phase4_full.py",
        "scripts/run_phase4_full.py",
        "scripts/build_phase4_full_bundle.py",
        "tests/test_phase4_full_execution.py",
    ]
    for relative in source_paths:
        source = working / relative
        if not source.is_file():
            raise RuntimeError(f"Required execution source is absent: {source}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    shutil.copy2(required["finalized_protocol"][0], output / "freeze/phase4_execution_ready_protocol_v2.json")
    shutil.copy2(required["smoke_verification"][0], output / "freeze/smoke_verification.json")
    shutil.copy2(config_path, output / "freeze/phase4_full_execution_v1.json")
    shutil.copy2(preflight / "manifest.json", output / "freeze/preflight_manifest.json")
    shutil.copy2(smoke_audit / "results/smoke_manifest.json", output / "freeze/successful_smoke_manifest.json")

    source_inventory = tree_inventory([output / "source"], output)
    input_inventory = tree_inventory([output / "private_inputs"], output)
    preservation = {
        "files_rehashed": len(before["files"]),
        "missing": len(missing),
        "changed": len(changed),
        "tree_sha256": before["tree_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "execution_id": "phase4_full_execution_v1",
        "status": "local_freeze_prepared_colab_environment_and_doc2dial_gate_pending",
        "authorization": {
            "date": "2026-09-06",
            "complete_matrix_approved": True,
            "hard_ceiling_allocated_A100_hours": 20.0,
        },
        "verified_local_identities": verified,
        "phase3_freeze_identity": stored_identity,
        "phase3_preservation": preservation,
        "source_inventory": source_inventory,
        "private_input_inventory": input_inventory,
        "doc2dial_expected": {
            "zip_sha256": config["inputs"]["doc2dial_zip_sha256"],
            "document_json_sha256": config["inputs"]["doc2dial_document_json_sha256"],
            "validation_json_sha256": config["inputs"]["doc2dial_validation_json_sha256"],
            "selection_index_sha256": config["inputs"]["doc2dial_selection_index_sha256"],
            "membership": 273,
        },
        "successful_environment_expected": config["successful_environment"],
        "technical_attempts_and_repairs": config["provenance"]["attempts"],
        "output_namespace": "phase4_full_execution_v1",
        "private": True,
        "prohibitions": config["prohibitions"],
    }
    write_immutable_json(output / "freeze/local_freeze_manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": sha256_path(output / "freeze/local_freeze_manifest.json"),
        "source_tree_sha256": source_inventory["tree_sha256"],
        "private_input_tree_sha256": input_inventory["tree_sha256"],
        "phase3_files_reverified": len(before["files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
