#!/usr/bin/env python3
"""Execute the authorized, frozen Phase 4 H/U/B/E matrix on Colab A100.

The runner is stage-resumable and fail-closed.  It never uses Doc2Dial outcomes
for training, selection, prompt changes, stopping, or example selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_COPY = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKING_COPY / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from chunkrag.eaai_phase4.full_execution import (  # noqa: E402
    CHUNKERS,
    EXPECTED_CANDIDATES,
    EXPECTED_EVALUATION_DOCUMENTS,
    EXPECTED_GENERATIONS,
    EXPECTED_RERANKER_SCORES,
    EXPECTED_RETRIEVAL_CELLS,
    GRID_LEARNING_RATES,
    GRID_UPDATE_BUDGETS,
    METHODS,
    REFIT_UPDATE_BUDGETS,
    SEEDS,
    AllocationLedger,
    BudgetStop,
    ResumableJsonlStore,
    aggregate_validation_scores,
    answer_metrics,
    atomic_write_json,
    canonical_json,
    capture_rng_state,
    convert_preflight_group,
    deterministic_group_draws,
    evidence_grade,
    expected_generation_keys,
    expected_ranking_instances,
    family_key,
    frozen_primary_analysis,
    hashed_row,
    ndcg_at_k,
    primary_document_rows,
    read_jsonl,
    restore_rng_state,
    select_shared_configuration,
    sha256_bytes,
    sha256_json,
    sha256_path,
    tree_inventory,
    utc_now,
    write_immutable_json,
)
from run_phase4_smoke import (  # noqa: E402
    _call_peft_with_unused_incompatible_torchao_disabled,
    adapter_parameter_hash,
    add_lora,
    final_margin,
    gpu_memory,
    load_adapter_for_reload_check,
    load_causal_model,
    load_tokenizer,
    package_versions,
    parameter_hash,
    patch_margin_head,
    release_model,
    score_candidates,
)


CONFIG_PATH = WORKING_COPY / "configs/eaai_phase4/phase4_full_execution_v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
INSTRUCTION = CONFIG["reranker"]["instruction"]
RERANKER_MODEL = CONFIG["reranker"]["model"]
RERANKER_REVISION = CONFIG["reranker"]["revision"]
GENERATOR_MODEL = CONFIG["generator"]["model"]
GENERATOR_REVISION = CONFIG["generator"]["revision"]
MAX_RERANK_TOKENS = int(CONFIG["reranker"]["max_length"])
MAX_INPUT_TOKENS = int(CONFIG["generator"]["max_input_tokens"])
MAX_NEW_TOKENS = int(CONFIG["generator"]["max_new_tokens"])


class StageLog:
    def __init__(self, path: Path) -> None:
        self.store = ResumableJsonlStore(path, ["event_id"])

    def write(self, stage: str, event: str, **details: Any) -> None:
        now = utc_now()
        event_id = sha256_bytes(f"{now}|{stage}|{event}|{len(self.store.rows)}".encode("utf-8"))
        self.store.append({
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_utc": now,
            "stage": stage,
            "event": event,
            **details,
        })


def require_sha(path: Path, expected: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f"Hash mismatch for {path}: expected {expected}, observed {observed}")
    return observed


def require_a100(torch: Any) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    hardware = str(torch.cuda.get_device_name(0))
    if hardware != "NVIDIA A100-SXM4-40GB":
        raise RuntimeError(f"Expected NVIDIA A100-SXM4-40GB, found {hardware}")
    return hardware


def environment_identity(torch: Any) -> dict[str, Any]:
    versions = package_versions()
    expected = CONFIG["successful_environment"]
    observed_core = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "transformers": str(versions["transformers"]),
        "peft": str(versions["peft"]),
        "accelerate": str(versions["accelerate"]),
        "huggingface_hub": str(versions["huggingface-hub"]),
        "numpy": str(versions["numpy"]),
    }
    for name, value in observed_core.items():
        if str(expected[name]) != value:
            raise RuntimeError(f"Successful v4 environment mismatch for {name}: {value} != {expected[name]}")
    optional = {}
    for name in ["sentence-transformers", "faiss-cpu", "rank-bm25", "spacy", "langchain-text-splitters", "torchao"]:
        try:
            optional[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional[name] = None
    required_optional = {
        "sentence-transformers": "3.4.1",
        "faiss-cpu": "1.14.3",
        "rank-bm25": "0.2.2",
        "spacy": "3.8.14",
        "langchain-text-splitters": "0.3.11",
    }
    for name, version in required_optional.items():
        if optional[name] != version:
            raise RuntimeError(f"Pinned retrieval environment mismatch for {name}: {optional[name]} != {version}")
    return {
        "hardware": require_a100(torch),
        "gpu_total_gib": float(torch.cuda.get_device_properties(0).total_memory / 1024**3),
        "core_packages": observed_core,
        "retrieval_packages": optional,
    }


def snapshot_identity(path: Path) -> dict[str, Any]:
    files = {}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(path))
        files[relative] = {"bytes": item.stat().st_size, "sha256": sha256_path(item)}
    return {"path": str(path), "files": len(files), "tree_sha256": sha256_json(files), "members": files}


def ensure_snapshots(cache_dir: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    requested = {
        "reranker": (RERANKER_MODEL, RERANKER_REVISION),
        "generator": (GENERATOR_MODEL, GENERATOR_REVISION),
        "dense_encoder": (CONFIG["retrieval"]["embedding_model"], CONFIG["retrieval"]["embedding_model_revision"]),
        "chunking_tokenizer": (CONFIG["retrieval"]["chunking_tokenizer"], CONFIG["retrieval"]["chunking_tokenizer_revision"]),
    }
    identities = {}
    for name, (model, revision) in requested.items():
        path = Path(snapshot_download(repo_id=model, revision=revision, cache_dir=str(cache_dir)))
        identities[name] = {"model": model, "revision": revision, **snapshot_identity(path)}
    return identities


def extract_member_by_hash(archive: Path, expected: str, destination: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir() or not member.filename.endswith(".json"):
                continue
            digest = hashlib.sha256()
            with zipped.open(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                continue
            if destination.exists():
                require_sha(destination, expected)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            return {"member": member.filename, "bytes": destination.stat().st_size, "sha256": expected}
    raise RuntimeError(f"No archive JSON member has expected hash {expected}")


def ensure_doc2dial(output: Path) -> dict[str, Any]:
    source_dir = output / "private_evaluation/source"
    archive = source_dir / "doc2dial_v1.0.1.zip"
    if archive.exists():
        require_sha(archive, CONFIG["inputs"]["doc2dial_zip_sha256"])
    else:
        source_dir.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(CONFIG["inputs"]["doc2dial_zip_url"], temporary)
        require_sha(temporary, CONFIG["inputs"]["doc2dial_zip_sha256"])
        os.replace(temporary, archive)
    document = source_dir / "document.json"
    validation = source_dir / "validation.json"
    document_identity = extract_member_by_hash(archive, CONFIG["inputs"]["doc2dial_document_json_sha256"], document)
    validation_identity = extract_member_by_hash(archive, CONFIG["inputs"]["doc2dial_validation_json_sha256"], validation)
    return {
        "archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_path(archive)},
        "document": {"path": str(document), **document_identity},
        "validation": {"path": str(validation), **validation_identity},
    }


def span_text(document: Mapping[str, Any], span_id: str) -> str:
    spans = document.get("spans", {})
    if isinstance(spans, list):
        matches = [value for value in spans if str(value.get("id", value.get("sp_id", ""))) == str(span_id)]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot resolve span {span_id}")
        span = matches[0]
    else:
        span = spans.get(str(span_id))
        if span is None:
            matches = [value for value in spans.values() if str(value.get("id", value.get("sp_id", ""))) == str(span_id)]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot resolve span {span_id}")
            span = matches[0]
    text = str(span.get("text_sp", span.get("text", ""))).strip()
    if not text and "start_sp" in span and "end_sp" in span:
        body = str(document["doc_text"])
        start, end = int(span["start_sp"]), int(span["end_sp"])
        candidates = [body[start:end], body[start : end + 1]]
        text = max((value.strip() for value in candidates), key=len)
    if not text:
        raise RuntimeError(f"Empty grounding span {span_id}")
    return text


def materialize_doc2dial(output: Path, frozen_inputs: Path) -> dict[str, Any]:
    source_dir = output / "private_evaluation/source"
    docs = json.loads((source_dir / "document.json").read_text(encoding="utf-8"))["doc_data"]
    dialogues = json.loads((source_dir / "validation.json").read_text(encoding="utf-8"))["dial_data"]
    selection_path = frozen_inputs / "private_inputs/doc2dial_selection_index.jsonl"
    require_sha(selection_path, CONFIG["inputs"]["doc2dial_selection_index_sha256"])
    selection = read_jsonl(selection_path)
    if len(selection) != EXPECTED_EVALUATION_DOCUMENTS:
        raise RuntimeError("Fixed Doc2Dial index is not 273 rows")

    corpus_rows = []
    canonical_seen = set()
    raw_lookup = {}
    for domain in sorted(docs):
        for raw_id, document in sorted(docs[domain].items()):
            canonical_id = f"{domain}::{raw_id}"
            if canonical_id in canonical_seen:
                raise RuntimeError("Duplicate canonical Doc2Dial document ID")
            canonical_seen.add(canonical_id)
            raw_lookup[(domain, str(raw_id))] = document
            corpus_rows.append({
                "schema_version": 1,
                "document_id": canonical_id,
                "raw_document_id": str(raw_id),
                "domain": str(domain),
                "title": str(document["title"]),
                "text": str(document["doc_text"]),
            })
    if len(corpus_rows) != 488:
        raise RuntimeError(f"Expected 488 corpus documents, found {len(corpus_rows)}")

    cases = []
    for chosen in selection:
        domain = str(chosen["domain"])
        raw_doc_id = str(chosen["document_id"])
        document = raw_lookup[(domain, raw_doc_id)]
        source_dialogues = dialogues[domain][raw_doc_id]
        matching = [value for value in source_dialogues if str(value["dial_id"]) == str(chosen["dialogue_id"])]
        if len(matching) != 1:
            raise RuntimeError("Frozen dialogue ID did not resolve uniquely")
        turns = matching[0]["turns"]
        positions = [index for index, turn in enumerate(turns) if int(turn["turn_id"]) == int(chosen["history_end_turn_id"])]
        if len(positions) != 1:
            raise RuntimeError("Frozen history-end turn did not resolve uniquely")
        position = positions[0]
        if position + 1 >= len(turns):
            raise RuntimeError("Selected user turn has no target")
        current, target = turns[position], turns[position + 1]
        if str(current["role"]).lower() != "user" or str(target["role"]).lower() != "agent":
            raise RuntimeError("Frozen transition is not user-to-immediately-following-agent")
        if int(target["turn_id"]) != int(chosen["target_turn_id"]):
            raise RuntimeError("Frozen target turn changed")
        target_reference_ids = [str(value["sp_id"]) for value in target.get("references", [])]
        if target_reference_ids != [str(value) for value in chosen["reference_span_ids"]]:
            raise RuntimeError("Frozen reference-span identity changed")
        history = [{"role": str(turn["role"]), "text": str(turn["utterance"])} for turn in turns[: position + 1]]
        query = "\n".join(f"{turn['role'].upper()}: {turn['text']}" for turn in history)
        spans = [span_text(document, span_id) for span_id in target_reference_ids]
        canonical_id = f"{domain}::{raw_doc_id}"
        cases.append({
            "schema_version": 1,
            "document_id": canonical_id,
            "raw_document_id": raw_doc_id,
            "domain": domain,
            "document_title": str(document["title"]),
            "family_key": family_key(domain, str(document["title"])),
            "dialogue_id": str(chosen["dialogue_id"]),
            "history_end_turn_id": int(chosen["history_end_turn_id"]),
            "target_turn_id": int(chosen["target_turn_id"]),
            "dialogue_history": history,
            "retrieval_query": query,
            "target_agent_response": str(target["utterance"]),
            "grounding_span_ids": target_reference_ids,
            "grounding_span_texts": spans,
            "selection_hash": str(chosen["selection_hash"]),
            "inference_contract": {
                "included": ["dialogue_history_through_selected_user_turn", "retrieval_query", "retrieved_context_at_generation"],
                "excluded": CONFIG["evaluation"]["inference_exclusions"],
            },
        })
    cases.sort(key=lambda row: (row["domain"], row["document_id"]))
    domains = Counter(row["domain"] for row in cases)
    families = Counter(row["family_key"] for row in cases)
    if dict(sorted(domains.items())) != CONFIG["evaluation"]["domain_counts"]:
        raise RuntimeError(f"Doc2Dial domain membership changed: {domains}")
    if len(families) != CONFIG["evaluation"]["family_count"]:
        raise RuntimeError(f"Expected 235 families, found {len(families)}")
    corpus_path = output / "private_evaluation/corpus.jsonl"
    cases_path = output / "private_evaluation/cases.jsonl"
    if not corpus_path.exists():
        corpus_store = ResumableJsonlStore(corpus_path, ["document_id"])
        for row in corpus_rows:
            corpus_store.append(row)
    if not cases_path.exists():
        case_store = ResumableJsonlStore(cases_path, ["document_id"])
        for row in cases:
            case_store.append(row)
    require_rows = read_jsonl(corpus_path), read_jsonl(cases_path)
    if len(require_rows[0]) != 488 or len(require_rows[1]) != 273:
        raise RuntimeError("Materialized Doc2Dial row counts changed")
    return {
        "corpus": {"rows": 488, "sha256": sha256_path(corpus_path)},
        "cases": {"rows": 273, "sha256": sha256_path(cases_path)},
        "domains": dict(sorted(domains.items())),
        "families": len(families),
        "family_size_histogram": dict(sorted(Counter(families.values()).items())),
        "cross_domain_families": 0,
    }


def verify_local_freeze(bundle_root: Path) -> dict[str, Any]:
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    inventory = tree_inventory(
        [bundle_root / "working_copy", bundle_root / "frozen_execution"], bundle_root
    )
    # bundle_manifest.json was excluded when its own inventory was created.
    expected = manifest["inventory"]
    if inventory["file_count"] != expected["file_count"] or inventory["tree_sha256"] != expected["tree_sha256"]:
        raise RuntimeError("Uploaded bundle inventory differs from the local freeze")
    local_freeze = bundle_root / "frozen_execution/freeze/local_freeze_manifest.json"
    require_sha(local_freeze, manifest["local_freeze_manifest_sha256"])
    return {
        "bundle_manifest_sha256": sha256_path(bundle_manifest_path),
        "bundle_tree_sha256": inventory["tree_sha256"],
        "local_freeze_manifest_sha256": sha256_path(local_freeze),
    }


def freeze_execution(bundle_root: Path, output: Path, cache_dir: Path, torch: Any, log: StageLog) -> Path:
    marker = output / "freeze/execution_freeze_manifest.json"
    if marker.exists():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("status") != "frozen_authorized":
            raise RuntimeError("Existing execution freeze is not authorized/final")
        return marker
    if output.exists() and any(output.iterdir()):
        allowed = {"logs", "resource"}
        unexpected = {item.name for item in output.iterdir()} - allowed
        if unexpected:
            raise RuntimeError(f"Unfrozen output namespace is nonempty: {sorted(unexpected)}")
    output.mkdir(parents=True, exist_ok=True)
    log.write("freeze", "started")
    bundle_identity = verify_local_freeze(bundle_root)
    environment = environment_identity(torch)
    model_snapshots = ensure_snapshots(cache_dir)
    source = ensure_doc2dial(output)
    membership = materialize_doc2dial(output, bundle_root / "frozen_execution")
    frozen_config = bundle_root / "working_copy/configs/eaai_phase4/phase4_full_execution_v1.json"
    if sha256_path(frozen_config) != sha256_path(CONFIG_PATH):
        raise RuntimeError("Runtime source configuration differs from bundled source")
    repair_source = bundle_root / "working_copy/scripts/run_phase4_smoke.py"
    require_sha(repair_source, CONFIG["provenance"]["successful_smoke_runner_sha256"])
    manifest = {
        "schema_version": 1,
        "execution_id": "phase4_full_execution_v1",
        "status": "frozen_authorized",
        "frozen_at_utc": utc_now(),
        "authorization": {"full_matrix": True, "hard_ceiling_allocated_A100_hours": 20.0},
        "bundle": bundle_identity,
        "configuration": {"path": str(frozen_config), "sha256": sha256_path(frozen_config), "value": CONFIG},
        "environment": environment,
        "model_and_tokenizer_snapshots": model_snapshots,
        "doc2dial_source": source,
        "doc2dial_membership_and_conversion": membership,
        "inputs": tree_inventory([bundle_root / "frozen_execution/private_inputs"], bundle_root / "frozen_execution"),
        "source_tree": tree_inventory([bundle_root / "working_copy"], bundle_root),
        "technical_failures_preserved": CONFIG["provenance"]["attempts"],
        "technical_repairs": CONFIG["successful_environment"]["technical_repairs"],
        "phase3_boundary": CONFIG["phase3_boundary"],
        "outcome_access_at_freeze": "No H/U/B/E Doc2Dial ranking, generation, or comparative metric existed.",
    }
    write_immutable_json(marker, manifest)
    log.write("freeze", "completed", manifest_sha256=sha256_path(marker))
    return marker


def verify_execution_freeze(output: Path) -> dict[str, Any]:
    path = output / "freeze/execution_freeze_manifest.json"
    if not path.is_file():
        raise RuntimeError("Execution freeze is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "frozen_authorized":
        raise RuntimeError("Execution freeze status is not frozen_authorized")
    for name in ["B_training_groups", "E_training_groups", "B_validation_labels", "E_validation_labels", "B_refit_groups", "E_refit_groups"]:
        key = f"{name}_sha256"
        if CONFIG["inputs"].get(key) is None:
            continue
    return value


def load_groups(path: Path, method: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    converted = [convert_preflight_group(row, method) for row in rows]
    if any(row["policy"] != "label_blind_dedup_pair_prune_v1" for row in converted):
        raise RuntimeError("Unexpected preprocessing policy")
    return converted


def mutable_checkpoint_save(
    torch: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    path: Path,
    *,
    job: Mapping[str, Any],
    completed_updates: int,
    group_cursor: int,
    logs: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(torch),
        "completed_updates": int(completed_updates),
        "group_cursor": int(group_cursor),
        "job": dict(job),
        "logs": list(logs),
    }, temporary / "training_state.pt")
    identity = tree_inventory([temporary], temporary)
    atomic_write_json(temporary / "complete.json", {
        "schema_version": 1,
        "job": dict(job),
        "completed_updates": int(completed_updates),
        "group_cursor": int(group_cursor),
        "files_before_complete_marker": identity,
    })
    if path.exists():
        shutil.rmtree(path)
    os.replace(temporary, path)


def seal_terminal_adapter(model: Any, path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads((path / "terminal_manifest.json").read_text(encoding="utf-8"))
        inventory = tree_inventory([path / "adapter"], path)
        if inventory["tree_sha256"] != manifest["adapter_tree_sha256"]:
            raise RuntimeError(f"Sealed adapter changed: {path}")
        return manifest
    path.mkdir(parents=True)
    model.save_pretrained(path / "adapter", safe_serialization=True)
    inventory = tree_inventory([path / "adapter"], path)
    manifest = {
        "schema_version": 1,
        **dict(summary),
        "adapter_tree_sha256": inventory["tree_sha256"],
        "adapter_files": inventory["files"],
        "sealed_at_utc": utc_now(),
    }
    write_immutable_json(path / "terminal_manifest.json", manifest)
    return manifest


def load_training_model(torch: Any, cache_dir: Path, resume: Path | None, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    from peft import PeftModel

    random.seed(seed)
    import numpy as np

    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    base, load_metrics = load_causal_model(torch, RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    if resume is None:
        model, repair = add_lora(base)
    else:
        model, repair = _call_peft_with_unused_incompatible_torchao_disabled(
            base, lambda: PeftModel.from_pretrained(base, str(resume / "adapter"), is_trainable=True)
        )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    patch_margin_head(model, tokenizer)
    return model, tokenizer, {"load": load_metrics, "repair": repair}


def train_job(
    torch: Any,
    groups: Sequence[Mapping[str, Any]],
    *,
    method: str,
    learning_rate: float,
    total_updates: int,
    seed: int,
    cache_dir: Path,
    job_dir: Path,
    ledger: AllocationLedger,
    stage: str,
) -> dict[str, Any]:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from chunkrag.eaai_phase4.training_contract import collate_listwise_groups, torch_listwise_loss

    terminal = job_dir / "terminal"
    if terminal.exists():
        return json.loads((terminal / "terminal_manifest.json").read_text(encoding="utf-8"))
    resume = job_dir / "resume"
    model, tokenizer, load_info = load_training_model(torch, cache_dir, resume if resume.exists() else None, seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=float(learning_rate), weight_decay=0.01)
    warmup_steps = int(math.ceil(total_updates * 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
    job_identity = {"method": method, "learning_rate": learning_rate, "total_updates": total_updates, "seed": seed}
    completed_updates = 0
    group_cursor = 0
    logs: list[dict[str, Any]] = []
    if resume.exists():
        state = torch.load(resume / "training_state.pt", map_location="cuda", weights_only=False)
        if state["job"] != job_identity:
            raise RuntimeError("Resume checkpoint job identity changed")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(torch, state["rng"])
        completed_updates = int(state["completed_updates"])
        group_cursor = int(state["group_cursor"])
        logs = list(state["logs"])
    starting_completed_updates = completed_updates
    initial_hash = parameter_hash(torch, model)
    draws = deterministic_group_draws(len(groups), total_updates * 4, seed)
    processed_pairs = sum(int(row.get("pairs", 0)) for row in logs)
    processed_tokens_unpadded = sum(int(row.get("tokens_unpadded", 0)) for row in logs)
    processed_tokens_padded = sum(int(row.get("tokens_padded", 0)) for row in logs)
    questions = set(value for row in logs for value in row.get("question_ids", []))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    update_durations = []
    model.train()
    for update in range(completed_updates, total_updates):
        ledger.heartbeat(stage=stage)
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        update_log = {"update": update + 1, "microbatches": [], "pairs": 0, "tokens_unpadded": 0, "tokens_padded": 0, "question_ids": []}
        for accumulation in range(2):
            indices = draws[group_cursor : group_cursor + 2]
            if len(indices) != 2:
                raise RuntimeError("Deterministic sampler exhausted early")
            group_cursor += 2
            microgroups = [groups[index] for index in indices]
            batch = collate_listwise_groups(microgroups, tokenizer, instruction=INSTRUCTION, max_length=MAX_RERANK_TOKENS)
            labels = list(batch.pop("binary_labels"))
            group_sizes = list(batch.pop("group_sizes"))
            group_keys = list(batch.pop("group_keys"))
            token_counts = list(batch.pop("token_counts"))
            expected_labels = []
            for size in group_sizes:
                expected_labels.extend([1] + [0] * (int(size) - 1))
            if labels != expected_labels or len(group_keys) != 2:
                raise RuntimeError("Batching mixed groups or changed positive indexing")
            question_ids = [str(group["question_id"]) for group in microgroups]
            questions.update(question_ids)
            for key, value in batch.items():
                batch[key] = value.to("cuda")
            if [int(value) for value in batch["attention_mask"].sum(dim=1).cpu()] != token_counts:
                raise RuntimeError("Padding mask changed token counts")
            outputs = model(**batch, use_cache=False)
            scores = final_margin(outputs.logits, batch["attention_mask"])
            loss = torch_listwise_loss(scores, group_sizes, temperature=1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {method} training loss")
            (loss / 2.0).backward()
            update_log["microbatches"].append({"group_keys": group_keys, "group_sizes": group_sizes, "loss": float(loss.detach().float().cpu())})
            update_log["pairs"] += len(token_counts)
            update_log["tokens_unpadded"] += sum(token_counts)
            update_log["tokens_padded"] += int(batch["input_ids"].numel())
            update_log["question_ids"].extend(question_ids)
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
            raise RuntimeError(f"Nonfinite {method} gradient")
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - began
        update_durations.append(duration)
        update_log["seconds"] = duration
        logs.append(update_log)
        processed_pairs += update_log["pairs"]
        processed_tokens_unpadded += update_log["tokens_unpadded"]
        processed_tokens_padded += update_log["tokens_padded"]
        completed_updates = update + 1
        if completed_updates % int(CONFIG["reranker"]["checkpoint_every_updates"]) == 0 or completed_updates == total_updates:
            projected_next = (sum(update_durations[-20:]) / len(update_durations[-20:])) * min(25, total_updates - completed_updates) if update_durations else 0.0
            mutable_checkpoint_save(
                torch, model, optimizer, scheduler, resume,
                job=job_identity, completed_updates=completed_updates, group_cursor=group_cursor, logs=logs,
            )
            ledger.heartbeat(stage=stage, projected_remaining_seconds=projected_next)
    elapsed = time.perf_counter() - started
    final_hash = parameter_hash(torch, model)
    if completed_updates > starting_completed_updates and initial_hash == final_hash:
        raise RuntimeError("Trainable parameters did not change")
    summary = {
        "job": job_identity,
        "status": "complete",
        "completed_updates": completed_updates,
        "warmup_steps": warmup_steps,
        "group_draws": group_cursor,
        "group_repetitions": group_cursor / len(groups),
        "available_groups": len(groups),
        "unique_questions_exposed": len(questions),
        "processed_pairs": processed_pairs,
        "processed_tokens_unpadded": processed_tokens_unpadded,
        "processed_tokens_padded": processed_tokens_padded,
        "seconds_this_session": elapsed,
        "trainable_parameter_hash_at_session_start": initial_hash,
        "trainable_parameter_hash_final": final_hash,
        "adapter_parameter_hash": adapter_parameter_hash(torch, model),
        "memory": gpu_memory(torch),
        "environment_repair": load_info["repair"],
        "training_log_sha256": sha256_json(logs),
    }
    terminal_manifest = seal_terminal_adapter(model, terminal, summary)
    release_model(torch, model)
    return terminal_manifest


def grid_job_id(method: str, learning_rate: float, updates: int) -> str:
    return f"{method}_lr{learning_rate:.0e}_updates{updates}"


def run_grid(
    torch: Any,
    bundle_root: Path,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
) -> list[dict[str, Any]]:
    verify_execution_freeze(output)
    all_manifests = []
    for method in METHODS:
        input_path = bundle_root / f"frozen_execution/private_inputs/{method}_training_groups.jsonl"
        require_sha(input_path, CONFIG["inputs"][f"{method}_training_groups_sha256"])
        groups = load_groups(input_path, method)
        expected_groups = 2103 if method == "B" else 943
        if len(groups) != expected_groups:
            raise RuntimeError(f"{method} grid group count changed")
        for learning_rate in GRID_LEARNING_RATES:
            for updates in GRID_UPDATE_BUDGETS:
                job_id = grid_job_id(method, learning_rate, updates)
                log.write("grid_training", "job_started", job_id=job_id)
                manifest = train_job(
                    torch,
                    groups,
                    method=method,
                    learning_rate=learning_rate,
                    total_updates=updates,
                    seed=int(CONFIG["reranker"]["grid_seed"]),
                    cache_dir=cache_dir,
                    job_dir=output / "training_grid" / job_id,
                    ledger=ledger,
                    stage=f"grid_training/{job_id}",
                )
                all_manifests.append(manifest)
                log.write("grid_training", "job_completed", job_id=job_id, adapter_tree_sha256=manifest["adapter_tree_sha256"])
    if len(all_manifests) != 12:
        raise RuntimeError("Grid training is not the complete 12-job matrix")
    write_immutable_json(output / "training_grid/grid_training_manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "jobs": all_manifests,
        "job_count": len(all_manifests),
    })
    return all_manifests


def validation_rows(bundle_root: Path, method: str) -> list[dict[str, Any]]:
    path = bundle_root / f"frozen_execution/private_inputs/{method}_validation_labels.jsonl"
    require_sha(path, CONFIG["inputs"][f"{method}_validation_labels_sha256"])
    rows = read_jsonl(path)
    rows.sort(key=lambda row: (str(row["question_id"]), CHUNKERS.index(str(row["chunker"])), int(row["fused_rank"]), str(row["chunk_id"])))
    if len(rows) != 16000:
        raise RuntimeError(f"{method} validation labels are not 16,000 rows")
    return rows


def score_adapter_rows(
    torch: Any,
    adapter_path: Path,
    labels: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Path,
    system: str,
    output_path: Path,
    ledger: AllocationLedger,
    stage: str,
) -> tuple[list[float], dict[str, Any]]:
    if output_path.exists():
        existing = read_jsonl(output_path)
        if len(existing) != len(labels):
            raise RuntimeError(f"Incomplete immutable scoring output: {output_path}")
        expected_keys = {
            (str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"]))
            for row in labels
        }
        observed_keys = {
            (str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"]))
            for row in existing
        }
        if observed_keys != expected_keys:
            raise RuntimeError(f"Immutable scoring keys changed: {output_path}")
        score_map = {
            (str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"])): float(row["score_margin"])
            for row in existing
        }
        ordered = [score_map[(str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"]))] for row in labels]
        return ordered, {"resumed_complete": True}
    model, tokenizer, load_metrics = load_adapter_for_reload_check(torch, adapter_path, cache_dir)
    candidates = [
        {
            "question_id": str(row["question_id"]),
            "question": str(row["question"]),
            "chunker": str(row["chunker"]),
            "chunk_id": str(row["chunk_id"]),
            "fused_rank": int(row["fused_rank"]),
            "text": str(row["text"]),
        }
        for row in labels
    ]
    scored, metrics = score_candidates(torch, model, tokenizer, candidates, batch_size=32)
    store = ResumableJsonlStore(output_path, ["question_id", "chunker", "chunk_id"])
    for label, score in zip(labels, scored, strict=True):
        store.append({
            "schema_version": 1,
            "system": system,
            "question_id": str(label["question_id"]),
            "chunker": str(label["chunker"]),
            "chunk_id": str(label["chunk_id"]),
            "fused_rank": int(label["fused_rank"]),
            "grade": int(label["grade"]),
            "score_margin": float(score["score_margin"]),
        })
    if len(store.rows) != len(labels):
        raise RuntimeError("Validation scoring output is incomplete")
    release_model(torch, model)
    ledger.heartbeat(stage=stage)
    return [float(row["score_margin"]) for row in scored], {"model_loading": load_metrics, "scoring": metrics}


def run_grid_validation(
    torch: Any,
    bundle_root: Path,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
) -> list[dict[str, Any]]:
    verify_execution_freeze(output)
    result_rows = []
    for method in METHODS:
        labels = validation_rows(bundle_root, method)
        for learning_rate in GRID_LEARNING_RATES:
            for updates in GRID_UPDATE_BUDGETS:
                job_id = grid_job_id(method, learning_rate, updates)
                terminal = output / "training_grid" / job_id / "terminal"
                terminal_manifest = json.loads((terminal / "terminal_manifest.json").read_text(encoding="utf-8"))
                adapter_inventory = tree_inventory([terminal / "adapter"], terminal)
                if adapter_inventory["tree_sha256"] != terminal_manifest["adapter_tree_sha256"]:
                    raise RuntimeError(f"Grid adapter identity failed for {job_id}")
                log.write("grid_validation", "job_started", job_id=job_id)
                scores, runtime = score_adapter_rows(
                    torch,
                    terminal / "adapter",
                    labels,
                    cache_dir=cache_dir,
                    system=job_id,
                    output_path=output / "validation" / f"{job_id}_scores.jsonl",
                    ledger=ledger,
                    stage=f"grid_validation/{job_id}",
                )
                aggregate = aggregate_validation_scores(labels, scores)
                result = {
                    "schema_version": 1,
                    "method": method,
                    "learning_rate": learning_rate,
                    "optimizer_updates": updates,
                    "seed": int(CONFIG["reranker"]["grid_seed"]),
                    "validation_question_mean_common_evidence_ndcg_at_4": aggregate["question_mean_common_evidence_ndcg_at_4"],
                    "validation_questions": aggregate["question_count"],
                    "validation_cells": aggregate["cell_count"],
                    "candidate_scores": len(scores),
                    "score_file_sha256": sha256_path(output / "validation" / f"{job_id}_scores.jsonl"),
                    "adapter_tree_sha256": terminal_manifest["adapter_tree_sha256"],
                    "runtime": runtime,
                }
                write_immutable_json(output / "validation" / f"{job_id}_summary.json", result)
                result_rows.append(result)
                log.write("grid_validation", "job_completed", job_id=job_id, candidate_scores=len(scores))
    if len(result_rows) != 12 or sum(row["candidate_scores"] for row in result_rows) != 192000:
        raise RuntimeError("Grid validation matrix is incomplete")
    write_immutable_json(output / "validation/grid_validation_manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "rows": result_rows,
        "candidate_scores": 192000,
    })
    return result_rows


def seal_selection(output: Path, validation: Sequence[Mapping[str, Any]], log: StageLog) -> dict[str, Any]:
    selection = select_shared_configuration(validation)
    manifest = {
        "schema_version": 1,
        "status": "sealed_before_doc2dial_outcomes",
        "sealed_at_utc": utc_now(),
        "rule": CONFIG["selection"],
        "grid_rows": list(validation),
        **selection,
        "doc2dial_ranking_or_generation_exists_at_selection": any((output / name).exists() for name in ["evaluation", "analysis"]),
    }
    if manifest["doc2dial_ranking_or_generation_exists_at_selection"]:
        raise RuntimeError("Doc2Dial outcomes exist before configuration selection")
    path = output / "selection/shared_configuration_manifest.json"
    write_immutable_json(path, manifest)
    log.write("selection", "sealed", manifest_sha256=sha256_path(path), selected=manifest["selected"])
    return manifest


def run_refits(
    torch: Any,
    bundle_root: Path,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = selection["selected"]
    learning_rate = float(selected["learning_rate"])
    updates = REFIT_UPDATE_BUDGETS[int(selected["optimizer_updates"])]
    manifests = []
    for method in METHODS:
        input_path = bundle_root / f"frozen_execution/private_inputs/{method}_refit_groups.jsonl"
        require_sha(input_path, CONFIG["inputs"][f"{method}_refit_groups_sha256"])
        groups = load_groups(input_path, method)
        expected = 4019 if method == "B" else 1799
        if len(groups) != expected:
            raise RuntimeError(f"{method} refit group count changed")
        for seed in SEEDS:
            job_id = f"{method}_seed{seed}_lr{learning_rate:.0e}_updates{updates}"
            log.write("final_refit", "job_started", job_id=job_id)
            terminal = train_job(
                torch,
                groups,
                method=method,
                learning_rate=learning_rate,
                total_updates=updates,
                seed=seed,
                cache_dir=cache_dir,
                job_dir=output / "refits" / job_id,
                ledger=ledger,
                stage=f"final_refit/{job_id}",
            )
            manifests.append(terminal)
            log.write("final_refit", "job_completed", job_id=job_id, adapter_tree_sha256=terminal["adapter_tree_sha256"])
    if len(manifests) != 6:
        raise RuntimeError("Final refit matrix is not six adapters")
    return manifests


def seal_selection_refits(output: Path, selection: Mapping[str, Any], refits: Sequence[Mapping[str, Any]], log: StageLog) -> Path:
    adapters = {}
    for row in refits:
        job = row["job"]
        instance = f"{job['method']}_{job['seed']}"
        matches = sorted((output / "refits").glob(f"{job['method']}_seed{job['seed']}_*/terminal"))
        if len(matches) != 1:
            raise RuntimeError(f"Cannot resolve one terminal adapter for {instance}")
        terminal = matches[0]
        inventory = tree_inventory([terminal / "adapter"], terminal)
        if inventory["tree_sha256"] != row["adapter_tree_sha256"]:
            raise RuntimeError(f"Refit adapter identity mismatch for {instance}")
        adapters[instance] = {"path": str(terminal / "adapter"), "tree_sha256": inventory["tree_sha256"], "files": inventory["files"]}
    manifest = {
        "schema_version": 1,
        "status": "sealed_and_verified_before_evaluation",
        "selection_manifest_sha256": sha256_path(output / "selection/shared_configuration_manifest.json"),
        "selected": selection["selected"],
        "adapters": adapters,
        "adapter_count": len(adapters),
        "sealed_at_utc": utc_now(),
    }
    path = output / "selection/selection_refit_gate_manifest.json"
    write_immutable_json(path, manifest)
    log.write("selection_refit_gate", "sealed", manifest_sha256=sha256_path(path), adapters=len(adapters))
    return path


def verify_selection_refit_gate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "sealed_and_verified_before_evaluation" or int(value.get("adapter_count", 0)) != 6:
        raise RuntimeError("Selection/refit integrity gate is not complete")
    for instance, identity in value["adapters"].items():
        adapter = Path(identity["path"])
        observed = tree_inventory([adapter], adapter.parent)
        if observed["tree_sha256"] != identity["tree_sha256"]:
            raise RuntimeError(f"Sealed adapter changed before evaluation: {instance}")
    return value


def serialize_retrieval(rows: Sequence[tuple[Any, float]], include_text: bool) -> list[dict[str, Any]]:
    output = []
    for rank, (chunk, score) in enumerate(rows, start=1):
        row = {
            "rank": rank,
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.doc_id),
            "title": str(chunk.title),
            "token_count": int(chunk.token_count),
            "score": float(score),
        }
        if include_text:
            row["text"] = str(chunk.text)
        output.append(row)
    return output


def candidate_pool_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(
        (
            int(row["fused_rank"]),
            str(row["chunk_id"]),
            str(row["document_id"]),
            sha256_bytes(str(row["text"]).encode("utf-8")),
        )
        for row in candidates
    )
    return sha256_json(identities)


def build_common_retrieval(
    torch: Any,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
) -> Path:
    verify_execution_freeze(output)
    cases = read_jsonl(output / "private_evaluation/cases.jsonl")
    corpus_rows = read_jsonl(output / "private_evaluation/corpus.jsonl")
    if len(cases) != 273 or len(corpus_rows) != 488:
        raise RuntimeError("Frozen Doc2Dial materialization changed")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from chunkrag.chunking import build_chunks
    from chunkrag.retrieval import BM25Retriever, DenseRetriever, mean_reciprocal_rank_fusion
    from chunkrag.schemas import Document

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["retrieval"]["chunking_tokenizer"],
        revision=CONFIG["retrieval"]["chunking_tokenizer_revision"],
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    tokenizer.model_max_length = 1_000_000
    encoder = SentenceTransformer(
        CONFIG["retrieval"]["embedding_model"],
        revision=CONFIG["retrieval"]["embedding_model_revision"],
        device="cuda",
        cache_folder=str(cache_dir),
    )
    documents = [
        Document(doc_id=str(row["document_id"]), title=str(row["title"]), text=str(row["text"]), dataset="doc2dial_v1.0.1")
        for row in corpus_rows
    ]
    case_index = {str(row["document_id"]): row for row in cases}
    cell_store = ResumableJsonlStore(output / "evaluation/common_candidate_cells.jsonl", ["document_id", "chunker"])
    candidate_store = ResumableJsonlStore(output / "evaluation/common_candidates.jsonl", ["document_id", "chunker", "chunk_id"])
    retrieval_runtime = []
    for chunker_spec in CONFIG["retrieval"]["chunkers"]:
        chunker = str(chunker_spec["name"])
        ledger.heartbeat(stage=f"evaluation_retrieval/{chunker}")
        chunk_started = time.perf_counter()
        chunks = build_chunks(documents, chunker_spec, tokenizer, None)
        dense = DenseRetriever(
            encoder=encoder,
            encoder_identifier=f"{CONFIG['retrieval']['embedding_model']}@{CONFIG['retrieval']['embedding_model_revision']}",
            device="cuda",
            batch_size=int(CONFIG["retrieval"]["embedding_batch_size"]),
            cache_dir=output / "evaluation/embedding_cache",
            cache_namespace=f"doc2dial/{chunker}",
            query_prefix=CONFIG["retrieval"]["query_prefix"],
        )
        dense.build(chunks)
        bm25 = BM25Retriever()
        bm25.build(chunks)
        for case in cases:
            document_id = str(case["document_id"])
            key = (document_id, chunker)
            if key in cell_store.rows:
                existing = [row for row_key, row in candidate_store.rows.items() if row_key[:2] == key]
                if len(existing) != 20:
                    raise RuntimeError(f"Resumed retrieval cell lacks 20 candidates: {key}")
                continue
            query = str(case["retrieval_query"])
            began = time.perf_counter()
            dense_rows = dense.retrieve(query, 20)
            dense_seconds = time.perf_counter() - began
            began = time.perf_counter()
            bm25_rows = bm25.retrieve(query, 20)
            bm25_seconds = time.perf_counter() - began
            began = time.perf_counter()
            fused = mean_reciprocal_rank_fusion(
                [dense_rows, bm25_rows],
                [float(CONFIG["retrieval"]["dense_weight"]), float(CONFIG["retrieval"]["bm25_weight"])],
                float(CONFIG["retrieval"]["rrf_k"]),
            )[:20]
            fusion_seconds = time.perf_counter() - began
            if len(fused) != 20:
                raise RuntimeError(f"Incomplete common pool at {key}")
            rows = []
            for rank, (chunk, score) in enumerate(fused, start=1):
                grade = evidence_grade(
                    str(chunk.text), str(chunk.doc_id), document_id, [str(value) for value in case["grounding_span_texts"]]
                )
                rows.append({
                    "schema_version": 1,
                    "document_id": document_id,
                    "domain": str(case["domain"]),
                    "family_key": str(case["family_key"]),
                    "question": query,
                    "chunker": chunker,
                    "chunk_id": str(chunk.chunk_id),
                    "candidate_document_id": str(chunk.doc_id),
                    "candidate_title": str(chunk.title),
                    "text": str(chunk.text),
                    "token_count": int(chunk.token_count),
                    "fused_rank": rank,
                    "fused_score": float(score),
                    "evidence_grade": grade,
                })
            pool_hash = candidate_pool_hash([
                {**row, "document_id": row["candidate_document_id"]} for row in rows
            ])
            for row in rows:
                candidate_store.append({**row, "common_candidate_pool_sha256": pool_hash})
            cell_store.append({
                "schema_version": 1,
                "document_id": document_id,
                "domain": str(case["domain"]),
                "family_key": str(case["family_key"]),
                "chunker": chunker,
                "question": query,
                "common_candidate_pool_sha256": pool_hash,
                "dense_candidates": serialize_retrieval(dense_rows, False),
                "bm25_candidates": serialize_retrieval(bm25_rows, False),
                "candidate_count": 20,
                "timing_seconds": {
                    "dense": dense_seconds,
                    "bm25": bm25_seconds,
                    "fusion": fusion_seconds,
                    "total": dense_seconds + bm25_seconds + fusion_seconds,
                    "model_loading_excluded": True,
                },
            })
            ledger.heartbeat(stage=f"evaluation_retrieval/{chunker}")
        retrieval_runtime.append({
            "chunker": chunker,
            "chunks": len(chunks),
            "seconds_including_chunking_and_index": time.perf_counter() - chunk_started,
            "memory": gpu_memory(torch),
        })
        del dense, bm25, chunks
        gc.collect()
        torch.cuda.empty_cache()
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if len(cell_store.rows) != EXPECTED_RETRIEVAL_CELLS or len(candidate_store.rows) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Common retrieval incomplete: {len(cell_store.rows)} cells/{len(candidate_store.rows)} candidates")
    path = output / "evaluation/common_retrieval_manifest.json"
    write_immutable_json(path, {
        "schema_version": 1,
        "status": "complete",
        "cells": len(cell_store.rows),
        "candidates": len(candidate_store.rows),
        "cells_sha256": sha256_path(cell_store.path),
        "candidates_sha256": sha256_path(candidate_store.path),
        "runtime": retrieval_runtime,
    })
    log.write("evaluation_retrieval", "completed", cells=len(cell_store.rows), candidates=len(candidate_store.rows))
    return path


def score_evaluation_system(
    torch: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    system_instance: str,
    adapter: Path | None,
    cache_dir: Path,
    output: Path,
    ledger: AllocationLedger,
) -> dict[str, Any]:
    score_path = output / "evaluation/reranker_scores" / f"{system_instance}.jsonl"
    if score_path.exists():
        rows = read_jsonl(score_path)
        if len(rows) != EXPECTED_CANDIDATES:
            raise RuntimeError(f"Incomplete score file for {system_instance}")
        expected = {
            (system_instance, str(row["document_id"]), str(row["chunker"]), str(row["chunk_id"]), str(row["common_candidate_pool_sha256"]))
            for row in candidates
        }
        observed = {
            (str(row["system_instance"]), str(row["document_id"]), str(row["chunker"]), str(row["chunk_id"]), str(row["common_candidate_pool_sha256"]))
            for row in rows
        }
        if observed != expected:
            raise RuntimeError(f"Resumed score keys/pool identities changed for {system_instance}")
        return {"system_instance": system_instance, "rows": len(rows), "sha256": sha256_path(score_path), "resumed": True}
    if adapter is None:
        tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
        model, load_metrics = load_causal_model(torch, RERANKER_MODEL, RERANKER_REVISION, cache_dir)
        patch_margin_head(model, tokenizer)
        model.eval()
    else:
        model, tokenizer, load_metrics = load_adapter_for_reload_check(torch, adapter, cache_dir)
    scorer_input = [
        {
            "question_id": str(row["document_id"]),
            "question": str(row["question"]),
            "chunker": str(row["chunker"]),
            "chunk_id": str(row["chunk_id"]),
            "fused_rank": int(row["fused_rank"]),
            "text": str(row["text"]),
        }
        for row in candidates
    ]
    scores, metrics = score_candidates(torch, model, tokenizer, scorer_input, batch_size=32)
    store = ResumableJsonlStore(score_path, ["system_instance", "document_id", "chunker", "chunk_id"])
    for source, score in zip(candidates, scores, strict=True):
        store.append({
            "schema_version": 1,
            "system_instance": system_instance,
            "document_id": str(source["document_id"]),
            "chunker": str(source["chunker"]),
            "chunk_id": str(source["chunk_id"]),
            "fused_rank": int(source["fused_rank"]),
            "score_margin": float(score["score_margin"]),
            "common_candidate_pool_sha256": str(source["common_candidate_pool_sha256"]),
        })
    release_model(torch, model)
    ledger.heartbeat(stage=f"evaluation_reranking/{system_instance}")
    return {
        "system_instance": system_instance,
        "rows": len(store.rows),
        "sha256": sha256_path(score_path),
        "model_loading": load_metrics,
        "scoring": metrics,
    }


def build_evaluation_rankings(
    torch: Any,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
    gate: Mapping[str, Any],
) -> Path:
    verify_selection_refit_gate(output / "selection/selection_refit_gate_manifest.json")
    candidates = read_jsonl(output / "evaluation/common_candidates.jsonl")
    candidates.sort(key=lambda row: (str(row["document_id"]), CHUNKERS.index(str(row["chunker"])), int(row["fused_rank"])))
    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError("Common candidate rows are incomplete")
    runtime = []
    runtime.append(score_evaluation_system(
        torch, candidates, system_instance="U", adapter=None, cache_dir=cache_dir, output=output, ledger=ledger
    ))
    for method in METHODS:
        for seed in SEEDS:
            instance = f"{method}_{seed}"
            adapter = Path(gate["adapters"][instance]["path"])
            runtime.append(score_evaluation_system(
                torch, candidates, system_instance=instance, adapter=adapter, cache_dir=cache_dir, output=output, ledger=ledger
            ))
    if sum(int(row["rows"]) for row in runtime) != EXPECTED_RERANKER_SCORES:
        raise RuntimeError("Evaluation reranker score matrix is incomplete")
    score_maps = {}
    for instance in ["U", *(f"{method}_{seed}" for method in METHODS for seed in SEEDS)]:
        rows = read_jsonl(output / "evaluation/reranker_scores" / f"{instance}.jsonl")
        score_maps[instance] = {
            (str(row["document_id"]), str(row["chunker"]), str(row["chunk_id"])): float(row["score_margin"])
            for row in rows
        }
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cell[(str(row["document_id"]), str(row["chunker"]))].append(dict(row))
    ranking_store = ResumableJsonlStore(output / "evaluation/rankings.jsonl", ["system_instance", "document_id", "chunker"])
    for document_id, chunker in sorted(by_cell, key=lambda key: (key[0], CHUNKERS.index(key[1]))):
        pool = by_cell[(document_id, chunker)]
        hashes = {str(row["common_candidate_pool_sha256"]) for row in pool}
        if len(pool) != 20 or len(hashes) != 1:
            raise RuntimeError(f"Invalid common pool at {document_id}/{chunker}")
        for instance in expected_ranking_instances():
            if instance == "H":
                ranked = sorted(pool, key=lambda row: (int(row["fused_rank"]), str(row["chunk_id"])))
            else:
                mapping = score_maps[instance]
                ranked = sorted(pool, key=lambda row: (-mapping[(document_id, chunker, str(row["chunk_id"]))], int(row["fused_rank"]), str(row["chunk_id"])))
            grades = [int(row["evidence_grade"]) for row in ranked]
            top4 = [
                {
                    "rank": rank,
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["candidate_document_id"]),
                    "text": str(row["text"]),
                    "evidence_grade": int(row["evidence_grade"]),
                    "source_fused_rank": int(row["fused_rank"]),
                }
                for rank, row in enumerate(ranked[:4], start=1)
            ]
            ranking_store.append({
                "schema_version": 1,
                "system_instance": instance,
                "system": instance.split("_", 1)[0],
                "seed": int(instance.split("_", 1)[1]) if "_" in instance else None,
                "document_id": document_id,
                "domain": str(pool[0]["domain"]),
                "family_key": str(pool[0]["family_key"]),
                "question": str(pool[0]["question"]),
                "chunker": chunker,
                "common_candidate_pool_sha256": next(iter(hashes)),
                "candidate_count": 20,
                "corrected_common_pool_evidence_ndcg_at_4": ndcg_at_k(grades, 4),
                "answer_visibility_at_4": float(any(value == 2 for value in grades[:4])),
                "gold_document_coverage_at_4": float(any(row["candidate_document_id"] == document_id for row in ranked[:4])),
                "maximum_grade_at_4": max(grades[:4]),
                "top_k": top4,
            })
    if len(ranking_store.rows) != EXPECTED_GENERATIONS:
        raise RuntimeError(f"Ranking matrix incomplete: {len(ranking_store.rows)}")
    path = output / "evaluation/reranking_manifest.json"
    write_immutable_json(path, {
        "schema_version": 1,
        "status": "complete",
        "reranker_instances": 7,
        "candidate_scores": sum(int(row["rows"]) for row in runtime),
        "rankings_all_instances_including_H": len(ranking_store.rows),
        "ranking_sha256": sha256_path(ranking_store.path),
        "runtime": runtime,
    })
    log.write("evaluation_reranking", "completed", candidate_scores=EXPECTED_RERANKER_SCORES, rankings=len(ranking_store.rows))
    return path


def generate_evaluation(
    torch: Any,
    output: Path,
    cache_dir: Path,
    ledger: AllocationLedger,
    log: StageLog,
) -> Path:
    from chunkrag.eaai_phase3.generation_runtime import _normalize_complete_response, pack_generation_messages

    rankings = read_jsonl(output / "evaluation/rankings.jsonl")
    if len(rankings) != EXPECTED_GENERATIONS:
        raise RuntimeError("Cannot generate from an incomplete ranking matrix")
    cases = {str(row["document_id"]): row for row in read_jsonl(output / "private_evaluation/cases.jsonl")}
    expected_keys = expected_generation_keys(sorted(cases))
    observed_ranking_keys = {(str(row["system_instance"]), str(row["document_id"]), str(row["chunker"])) for row in rankings}
    if observed_ranking_keys != expected_keys:
        raise RuntimeError("Ranking keys do not match the exact frozen generation matrix")
    store = ResumableJsonlStore(output / "evaluation/generations.jsonl", ["system_instance", "document_id", "chunker"])
    if not set(store.rows).issubset(expected_keys):
        raise RuntimeError("Generation checkpoint contains an unprescribed key")
    pending = [row for row in rankings if (str(row["system_instance"]), str(row["document_id"]), str(row["chunker"])) not in store.rows]
    if not pending:
        if len(store.rows) != EXPECTED_GENERATIONS:
            raise RuntimeError("Generation checkpoint is incomplete but no pending key was found")
        path = output / "evaluation/generation_manifest.json"
        write_immutable_json(path, {
            "schema_version": 1,
            "status": "complete",
            "records": len(store.rows),
            "unique_keys": len(set(store.rows)),
            "generation_sha256": sha256_path(store.path),
            "model_loading": {"resumed_complete": True},
            "runtime_this_session": {"batches": 0, "records": 0, "seconds": 0.0},
        })
        return path
    ledger.heartbeat(
        stage="evaluation_generation_preload",
        projected_remaining_seconds=(len(pending) / 0.6025931908436136) * 1.25 + 120.0,
    )
    tokenizer = load_tokenizer(GENERATOR_MODEL, GENERATOR_REVISION, cache_dir)
    model, load_metrics = load_causal_model(torch, GENERATOR_MODEL, GENERATOR_REVISION, cache_dir)
    model.eval()
    batch_size = int(CONFIG["generator"]["batch_size"])
    batch_metrics = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(pending), batch_size):
        block = pending[start : start + batch_size]
        if batch_metrics:
            recent = batch_metrics[-min(20, len(batch_metrics)) :]
            seconds_per_record = sum(row["seconds"] for row in recent) / sum(row["records"] for row in recent)
            projected_next = seconds_per_record * (len(pending) - start) * 1.10 + 60.0
        else:
            projected_next = ((len(pending) - start) / 0.6025931908436136) * 1.25 + 120.0
        ledger.heartbeat(stage="evaluation_generation", projected_remaining_seconds=projected_next)
        packed_rows = []
        texts = []
        for ranking in block:
            context = "\n\n".join(f"[{index}] {item['text']}" for index, item in enumerate(ranking["top_k"], start=1))
            packed = pack_generation_messages(
                tokenizer,
                question=str(ranking["question"]),
                context=context,
                answer_style="complete",
                max_input_tokens=MAX_INPUT_TOKENS,
            )
            try:
                rendered = tokenizer.apply_chat_template(
                    packed["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    packed["messages"], tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            packed_rows.append((ranking, packed, context))
            texts.append(rendered)
        batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        if int(batch["input_ids"].shape[1]) > MAX_INPUT_TOKENS:
            raise RuntimeError("Packed generation input exceeds the frozen 1,536-token budget")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        input_width = int(batch["input_ids"].shape[1])
        generated_ids = generated[:, input_width:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        token_counts = []
        for ids in generated_ids:
            values = ids.tolist()
            count = len(values)
            if tokenizer.eos_token_id in values:
                count = values.index(tokenizer.eos_token_id) + 1
            token_counts.append(count)
        for (ranking, packed, original_context), raw, generated_count in zip(packed_rows, decoded, token_counts, strict=True):
            normalized = _normalize_complete_response(raw)
            if not normalized:
                raise RuntimeError(f"Empty deterministic generation for {ranking['system_instance']}/{ranking['document_id']}/{ranking['chunker']}")
            case = cases[str(ranking["document_id"])]
            if str(ranking["question"]) != str(case["retrieval_query"]):
                raise RuntimeError("Generation question differs from the frozen inference query")
            if original_context != "\n\n".join(f"[{index}] {item['text']}" for index, item in enumerate(ranking["top_k"], start=1)):
                raise RuntimeError("Generation context changed after ranking")
            store.append({
                "schema_version": 1,
                "system_instance": str(ranking["system_instance"]),
                "system": str(ranking["system"]),
                "seed": ranking.get("seed"),
                "document_id": str(ranking["document_id"]),
                "domain": str(ranking["domain"]),
                "family_key": str(ranking["family_key"]),
                "chunker": str(ranking["chunker"]),
                "raw_output": str(raw),
                "normalized_output": normalized,
                "packed_context": str(packed["packed_context"]),
                "full_prompt_tokens": int(packed["full_prompt_tokens"]),
                "used_prompt_tokens": int(packed["used_prompt_tokens"]),
                "context_truncated": bool(packed["context_truncated"]),
                "generated_tokens": int(generated_count),
                "generation_length_capped": bool(generated_count >= MAX_NEW_TOKENS),
                "thinking_enabled": False,
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "common_candidate_pool_sha256": str(ranking["common_candidate_pool_sha256"]),
                "top_k_sha256": sha256_json(ranking["top_k"]),
            })
        batch_metrics.append({
            "batch": len(batch_metrics) + 1,
            "records": len(block),
            "seconds": elapsed,
            "generated_tokens": sum(token_counts),
            "completed_records": len(store.rows),
            "timestamp_utc": utc_now(),
        })
        if len(batch_metrics) % 25 == 0:
            atomic_write_json(output / "evaluation/generation_progress.json", {
                "schema_version": 1,
                "completed": len(store.rows),
                "expected": EXPECTED_GENERATIONS,
                "last_batch": batch_metrics[-1],
                "generation_jsonl_sha256": sha256_path(store.path),
            })
            print(f"GENERATION_PROGRESS {len(store.rows)}/{EXPECTED_GENERATIONS}", flush=True)
    release_model(torch, model)
    if len(store.rows) != EXPECTED_GENERATIONS or set(store.rows) != expected_keys:
        raise RuntimeError(f"Generation incomplete: {len(store.rows)}/{EXPECTED_GENERATIONS}")
    total_seconds = sum(row["seconds"] for row in batch_metrics)
    total_tokens = sum(row["generated_tokens"] for row in batch_metrics)
    path = output / "evaluation/generation_manifest.json"
    write_immutable_json(path, {
        "schema_version": 1,
        "status": "complete",
        "records": len(store.rows),
        "unique_keys": len(set(store.rows)),
        "generation_sha256": sha256_path(store.path),
        "model_loading": load_metrics,
        "runtime_this_session": {
            "batches": len(batch_metrics),
            "records": sum(row["records"] for row in batch_metrics),
            "seconds": total_seconds,
            "records_per_second": sum(row["records"] for row in batch_metrics) / total_seconds,
            "generated_tokens": total_tokens,
            "generated_tokens_per_second": total_tokens / total_seconds,
            "memory": gpu_memory(torch),
        },
    })
    log.write("evaluation_generation", "completed", records=len(store.rows), sha256=sha256_path(store.path))
    return path


def compute_final_analysis(output: Path, log: StageLog) -> Path:
    generations = read_jsonl(output / "evaluation/generations.jsonl")
    rankings = read_jsonl(output / "evaluation/rankings.jsonl")
    candidates = read_jsonl(output / "evaluation/common_candidates.jsonl")
    scores = [row for path in sorted((output / "evaluation/reranker_scores").glob("*.jsonl")) for row in read_jsonl(path)]
    if len(generations) != EXPECTED_GENERATIONS or len(rankings) != EXPECTED_GENERATIONS:
        raise RuntimeError("Final analysis blocked by incomplete generation/ranking rows")
    if len(candidates) != EXPECTED_CANDIDATES or len(scores) != EXPECTED_RERANKER_SCORES:
        raise RuntimeError("Final analysis blocked by incomplete candidate-score rows")
    cases = {str(row["document_id"]): row for row in read_jsonl(output / "private_evaluation/cases.jsonl")}
    ranking_index = {(str(row["system_instance"]), str(row["document_id"]), str(row["chunker"])): row for row in rankings}
    metric_store = ResumableJsonlStore(output / "analysis/private_generation_metrics.jsonl", ["system_instance", "document_id", "chunker"])
    for generation in generations:
        key = (str(generation["system_instance"]), str(generation["document_id"]), str(generation["chunker"]))
        case = cases[key[1]]
        ranking = ranking_index[key]
        response = answer_metrics(str(generation["normalized_output"]), [str(case["target_agent_response"])])
        grounding = answer_metrics(str(generation["normalized_output"]), [str(value) for value in case["grounding_span_texts"]])
        metric_store.append({
            "schema_version": 1,
            "system_instance": key[0],
            "system": str(generation["system"]),
            "seed": generation.get("seed"),
            "document_id": key[1],
            "domain": str(case["domain"]),
            "family_key": str(case["family_key"]),
            "chunker": key[2],
            "agent_response_f1": response["f1"],
            "agent_response_exact_match": response["exact_match"],
            "grounding_span_f1": grounding["f1"],
            "corrected_common_pool_evidence_ndcg_at_4": float(ranking["corrected_common_pool_evidence_ndcg_at_4"]),
            "answer_visibility_at_4": float(ranking["answer_visibility_at_4"]),
            "gold_document_coverage_at_4": float(ranking["gold_document_coverage_at_4"]),
            "maximum_grade_at_4": float(ranking["maximum_grade_at_4"]),
            "generated_tokens": int(generation["generated_tokens"]),
            "used_prompt_tokens": int(generation["used_prompt_tokens"]),
            "context_truncated": bool(generation["context_truncated"]),
            "generation_length_capped": bool(generation["generation_length_capped"]),
        })
    metric_rows = list(metric_store.rows.values())
    documents = primary_document_rows(metric_rows)
    primary = frozen_primary_analysis(documents)
    document_store = ResumableJsonlStore(output / "analysis/private_primary_document_differences.jsonl", ["document_id"])
    for row in documents:
        document_store.append(row)
    write_immutable_json(output / "analysis/primary_result.json", primary)

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_instance[str(row["system_instance"])].append(row)
    instance_aggregates = {}
    for instance, rows in sorted(by_instance.items()):
        instance_aggregates[instance] = {
            "records": len(rows),
            "documents": len({row["document_id"] for row in rows}),
            "agent_response_f1": sum(float(row["agent_response_f1"]) for row in rows) / len(rows),
            "agent_response_exact_match": sum(float(row["agent_response_exact_match"]) for row in rows) / len(rows),
            "grounding_span_f1": sum(float(row["grounding_span_f1"]) for row in rows) / len(rows),
            "corrected_common_pool_evidence_ndcg_at_4": sum(float(row["corrected_common_pool_evidence_ndcg_at_4"]) for row in rows) / len(rows),
            "answer_visibility_at_4": sum(float(row["answer_visibility_at_4"]) for row in rows) / len(rows),
            "gold_document_coverage_at_4": sum(float(row["gold_document_coverage_at_4"]) for row in rows) / len(rows),
        }
    system_aggregates = {}
    for system in ["H", "U", "B", "E"]:
        rows = [row for row in metric_rows if str(row["system"]) == system]
        system_aggregates[system] = {
            "records": len(rows),
            "agent_response_f1": sum(float(row["agent_response_f1"]) for row in rows) / len(rows),
            "agent_response_exact_match": sum(float(row["agent_response_exact_match"]) for row in rows) / len(rows),
            "grounding_span_f1": sum(float(row["grounding_span_f1"]) for row in rows) / len(rows),
            "corrected_common_pool_evidence_ndcg_at_4": sum(float(row["corrected_common_pool_evidence_ndcg_at_4"]) for row in rows) / len(rows),
        }
    aggregates = {"schema_version": 1, "status": "complete", "instance_aggregates": instance_aggregates, "system_aggregates": system_aggregates, "primary": primary}
    write_immutable_json(output / "analysis/private_aggregates.json", aggregates)
    write_immutable_json(output / "analysis/public_ready_aggregates_not_published.json", {
        "schema_version": 1,
        "publication_status": "prepared_not_published",
        "dataset": "IBM Doc2Dial v1.0.1",
        "sample_size": 273,
        "systems": system_aggregates,
        "primary": primary,
        "privacy": "Aggregates only; no dialogue, response, document, span, or stable case identifier.",
    })
    log.write("frozen_analysis", "completed", primary_sha256=sha256_path(output / "analysis/primary_result.json"))
    return output / "analysis/primary_result.json"


def preservation_recheck_from_bundle(bundle_root: Path) -> dict[str, Any]:
    local_freeze = json.loads((bundle_root / "frozen_execution/freeze/local_freeze_manifest.json").read_text(encoding="utf-8"))
    inventory = tree_inventory([bundle_root / "frozen_execution/private_inputs"], bundle_root / "frozen_execution")
    expected = local_freeze["private_input_inventory"]
    return {
        "phase3_local_pre_execution": local_freeze["phase3_preservation"],
        "frozen_private_input_files": inventory["file_count"],
        "frozen_private_input_tree_sha256": inventory["tree_sha256"],
        "frozen_private_inputs_unchanged": inventory["tree_sha256"] == expected["tree_sha256"],
    }


def final_audit(bundle_root: Path, output: Path, ledger_value: Mapping[str, Any]) -> Path:
    matrix = {
        "grid_training_jobs": len(list((output / "training_grid").glob("*/terminal/terminal_manifest.json"))),
        "grid_validation_jobs": len(list((output / "validation").glob("*_summary.json"))),
        "grid_validation_candidate_scores": sum(len(read_jsonl(path)) for path in (output / "validation").glob("*_scores.jsonl")),
        "final_refit_jobs": len(list((output / "refits").glob("*/terminal/terminal_manifest.json"))),
        "evaluation_retrieval_cells": len(read_jsonl(output / "evaluation/common_candidate_cells.jsonl")),
        "evaluation_candidates": len(read_jsonl(output / "evaluation/common_candidates.jsonl")),
        "evaluation_reranker_scores": sum(len(read_jsonl(path)) for path in (output / "evaluation/reranker_scores").glob("*.jsonl")),
        "evaluation_rankings": len(read_jsonl(output / "evaluation/rankings.jsonl")),
        "evaluation_generations": len(read_jsonl(output / "evaluation/generations.jsonl")),
        "primary_document_differences": len(read_jsonl(output / "analysis/private_primary_document_differences.jsonl")),
    }
    expected = {
        "grid_training_jobs": 12,
        "grid_validation_jobs": 12,
        "grid_validation_candidate_scores": 192000,
        "final_refit_jobs": 6,
        "evaluation_retrieval_cells": 1092,
        "evaluation_candidates": 21840,
        "evaluation_reranker_scores": 152880,
        "evaluation_rankings": 8736,
        "evaluation_generations": 8736,
        "primary_document_differences": 273,
    }
    if matrix != expected:
        raise RuntimeError(f"Final matrix audit failed: {matrix} != {expected}")
    preservation = preservation_recheck_from_bundle(bundle_root)
    if not preservation["frozen_private_inputs_unchanged"]:
        raise RuntimeError("Frozen private inputs changed during execution")
    artifact_paths = [
        output / "freeze/execution_freeze_manifest.json",
        output / "training_grid/grid_training_manifest.json",
        output / "validation/grid_validation_manifest.json",
        output / "selection/shared_configuration_manifest.json",
        output / "selection/selection_refit_gate_manifest.json",
        output / "evaluation/common_retrieval_manifest.json",
        output / "evaluation/reranking_manifest.json",
        output / "evaluation/generation_manifest.json",
        output / "analysis/primary_result.json",
        output / "analysis/private_aggregates.json",
        output / "analysis/public_ready_aggregates_not_published.json",
    ]
    artifacts = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_path(path)} for path in artifact_paths}
    claims = [
        {"claim": "The fixed Phase 4 matrix completed exactly.", "classification": "verified", "artifacts": ["complete_matrix_audit"]},
        {"claim": "Doc2Dial outcomes were not used for configuration selection.", "classification": "verified", "artifacts": ["selection/shared_configuration_manifest.json", "selection/selection_refit_gate_manifest.json"]},
        {"claim": "The primary E-minus-B result uses the frozen document/family procedures.", "classification": "verified", "artifacts": ["analysis/primary_result.json", "analysis/private_primary_document_differences.jsonl"]},
        {"claim": "Foundation-model pretraining did not include Doc2Dial.", "classification": "unresolved", "artifacts": [], "reason": "Pretraining exposure is unknown."},
        {"claim": "Phase 4 replaces Phase 3.", "classification": "contradicted", "artifacts": ["freeze/execution_freeze_manifest.json"]},
    ]
    manifest = {
        "schema_version": 1,
        "execution_id": "phase4_full_execution_v1",
        "status": "COMPLETE",
        "completed_at_utc": utc_now(),
        "matrix": matrix,
        "expected_matrix": expected,
        "matrix_exact": True,
        "preservation": preservation,
        "resource_ledger": dict(ledger_value),
        "artifact_identities": artifacts,
        "claim_to_artifact_ledger": claims,
        "phase3_interpretation": "Separate prior study; Phase 4 does not replace or reinterpret Phase 3.",
        "publication": "Public-ready aggregates prepared but not published.",
        "actions_not_taken": ["No commit", "No push", "No sharing change", "No publication", "No fabricated human ratings", "No manuscript submission"],
    }
    path = output / "final_manifest.json"
    write_immutable_json(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output = args.output.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    import torch

    hardware = require_a100(torch)
    log = StageLog(output / "logs/stage_ledger.jsonl")
    ledger = AllocationLedger(output / "resource/allocation_ledger.json", ceiling_hours=20.0)
    ledger.begin(stage="freeze", hardware=hardware)
    current_stage = "freeze"
    try:
        print("PHASE4_STAGE freeze", flush=True)
        freeze_execution(bundle_root, output, cache_dir, torch, log)
        ledger.heartbeat(stage="freeze_complete", projected_remaining_seconds=17.8 * 3600.0)

        current_stage = "grid_training"
        print("PHASE4_STAGE grid_training", flush=True)
        ledger.heartbeat(stage="grid_training_precheck", projected_remaining_seconds=17.5 * 3600.0)
        run_grid(torch, bundle_root, output, cache_dir, ledger, log)

        current_stage = "grid_validation"
        print("PHASE4_STAGE grid_validation", flush=True)
        ledger.heartbeat(stage="grid_validation_precheck", projected_remaining_seconds=12.8 * 3600.0)
        validation = run_grid_validation(torch, bundle_root, output, cache_dir, ledger, log)

        current_stage = "selection"
        print("PHASE4_STAGE selection", flush=True)
        selection_path = output / "selection/shared_configuration_manifest.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else seal_selection(output, validation, log)

        current_stage = "final_refit"
        print("PHASE4_STAGE final_refit", flush=True)
        selected_refit_hours = {227: 1.79630784092019, 454: 3.59261568184038, 681: 5.388923522760571}[int(selection["selected"]["optimizer_updates"])]
        ledger.heartbeat(stage="final_refit_precheck", projected_remaining_seconds=(selected_refit_hours + 4.50) * 1.20 * 3600.0)
        refits = run_refits(torch, bundle_root, output, cache_dir, ledger, log, selection)

        current_stage = "selection_refit_gate"
        print("PHASE4_STAGE selection_refit_gate", flush=True)
        gate_path = output / "selection/selection_refit_gate_manifest.json"
        if gate_path.exists():
            gate = verify_selection_refit_gate(gate_path)
        else:
            seal_selection_refits(output, selection, refits, log)
            gate = verify_selection_refit_gate(gate_path)

        current_stage = "evaluation_retrieval"
        print("PHASE4_STAGE evaluation_retrieval", flush=True)
        ledger.heartbeat(stage="evaluation_retrieval_precheck", projected_remaining_seconds=5.55 * 3600.0)
        build_common_retrieval(torch, output, cache_dir, ledger, log)

        current_stage = "evaluation_reranking"
        print("PHASE4_STAGE evaluation_reranking", flush=True)
        ledger.heartbeat(stage="evaluation_reranking_precheck", projected_remaining_seconds=5.50 * 3600.0)
        build_evaluation_rankings(torch, output, cache_dir, ledger, log, gate)

        current_stage = "evaluation_generation"
        print("PHASE4_STAGE evaluation_generation", flush=True)
        generate_evaluation(torch, output, cache_dir, ledger, log)

        current_stage = "cpu_analysis"
        print("PHASE4_STAGE cpu_analysis", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        compute_final_analysis(output, log)
        ledger_value = ledger.end(status="complete", stage=current_stage)
        final_path = final_audit(bundle_root, output, ledger_value)
        print(json.dumps({
            "status": "COMPLETE",
            "final_manifest": str(final_path),
            "final_manifest_sha256": sha256_path(final_path),
            "allocated_A100_hours": ledger_value["cumulative_allocated_seconds"] / 3600.0,
        }, indent=2), flush=True)
    except BudgetStop as error:
        ledger_value = ledger.end(status="budget_stop", stage=current_stage)
        atomic_write_json(output / "BUDGET_STOP.json", {
            "schema_version": 1,
            "status": "BUDGET_STOP",
            "stage": current_stage,
            "reason": str(error),
            "resource_ledger": ledger_value,
            "resume_policy": "Resume only after new authorization; do not alter or shrink the frozen matrix.",
        })
        print(f"PHASE4_BUDGET_STOP {error}", flush=True)
        raise
    except Exception as error:
        ledger_value = ledger.end(status="blocked", stage=current_stage)
        atomic_write_json(output / "BLOCKED.json", {
            "schema_version": 1,
            "status": "BLOCKED",
            "stage": current_stage,
            "error_type": type(error).__name__,
            "reason": str(error),
            "resource_ledger": ledger_value,
            "scientific_choices_changed": False,
        })
        print(f"PHASE4_BLOCKED {current_stage} {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
