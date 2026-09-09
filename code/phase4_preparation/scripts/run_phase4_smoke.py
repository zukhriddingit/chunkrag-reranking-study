#!/usr/bin/env python3
"""Run the approved eight-question Phase 4 development-only GPU smoke.

This runner refuses non-A100 hardware, verifies the sealed smoke inputs before
model loading, performs exactly two optimizer updates for B and E, scores the
same 640 candidate pairs with U/B/E, and generates exactly 128 H/U/B/E outputs.
It never reads Doc2Dial cases or references.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
GENERATOR_MODEL = "Qwen/Qwen3.5-9B"
GENERATOR_REVISION = "1eb6d84878284892ecfd4c3fb760ffbf34548234"
INSTRUCTION = "Given a technical-support question, rank passages by whether they contain sufficient evidence to answer it."
SEED = 20260904
MAX_RERANK_TOKENS = 512
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 512
CHUNKERS = ["fixed_128", "fixed_254", "recursive_254", "sentence_254"]
EXPECTED_FULL_INPUTS = {
    "B": "c11a5bb4541c68b3c466f12fadac3ddfde1c85fa8a2c9e28537a3a908fc45e13",
    "E": "58a67253104ded82d5ecaa2ffae12f086102369a9296e01be8243579ad5c7144",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                stored = row.get("row_sha256")
                if stored is not None:
                    observed = sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
                    if observed != stored:
                        raise RuntimeError(f"Row hash mismatch in {path}")
                rows.append(row)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ResumableJsonlStore:
    def __init__(self, path: Path, key_fields: Sequence[str]) -> None:
        self.path = path
        self.key_fields = tuple(key_fields)
        self.rows: dict[tuple[str, ...], dict[str, Any]] = {}
        if path.exists():
            for row in read_jsonl(path):
                key = self.key(row)
                if key in self.rows:
                    raise RuntimeError(f"Duplicate stored key {key} in {path}")
                self.rows[key] = row

    def key(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in self.key_fields)

    def append(self, row: Mapping[str, Any]) -> str:
        materialized = dict(row)
        materialized.pop("row_sha256", None)
        materialized["row_sha256"] = sha256_json(materialized)
        key = self.key(materialized)
        existing = self.rows.get(key)
        if existing is not None:
            if canonical_json(existing) != canonical_json(materialized):
                raise RuntimeError(f"Conflicting duplicate key rejected: {key}")
            return "skipped_identical"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(materialized) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = materialized
        return "appended"


def gpu_memory(torch: Any) -> dict[str, float]:
    return {
        "allocated_gib": float(torch.cuda.memory_allocated() / 1024**3),
        "reserved_gib": float(torch.cuda.memory_reserved() / 1024**3),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 1024**3),
    }


def release_model(torch: Any, *objects: Any) -> None:
    for obj in objects:
        if hasattr(obj, "to"):
            try:
                obj.to("cpu")
            except Exception:
                pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def package_versions() -> dict[str, str | None]:
    names = ["torch", "transformers", "peft", "accelerate", "huggingface-hub", "numpy"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_kwargs(torch: Any, revision: str, cache_dir: Path) -> dict[str, Any]:
    return {
        "revision": revision,
        "cache_dir": str(cache_dir),
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }


def load_tokenizer(model_name: str, revision: str, cache_dir: Path, *, padding_side: str = "left") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
        padding_side=padding_side,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(torch: Any, model_name: str, revision: str, cache_dir: Path) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs(torch, revision, cache_dir))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return model, {"seconds": elapsed, "memory": gpu_memory(torch)}


def patch_margin_head(model: Any, tokenizer: Any) -> tuple[int, int]:
    """Apply the exact ms-swift generative-reranker yes-minus-no head."""

    import torch.nn.functional as functional

    yes_token_id = int(tokenizer.convert_tokens_to_ids("yes"))
    no_token_id = int(tokenizer.convert_tokens_to_ids("no"))
    if yes_token_id == tokenizer.unk_token_id or no_token_id == tokenizer.unk_token_id:
        raise RuntimeError("Pinned tokenizer does not expose the yes/no reranker tokens")
    head = model.get_output_embeddings()
    if getattr(head, "_phase4_margin_patched", False):
        return yes_token_id, no_token_id

    def margin_forward(module: Any, hidden_states: Any) -> Any:
        weight = module.weight[[yes_token_id, no_token_id]]
        logits = functional.linear(hidden_states, weight)
        return logits[..., 0:1] - logits[..., 1:2]

    head.forward = types.MethodType(margin_forward, head)
    head._phase4_margin_patched = True
    return yes_token_id, no_token_id


def final_margin(logits: Any, attention_mask: Any) -> Any:
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_positions = (positions * attention_mask.long()).max(dim=1).values
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, last_positions, 0]


def encode_rerank_pairs(tokenizer: Any, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    from chunkrag.eaai_phase4.preprocessing import QWEN_PREFIX, QWEN_SUFFIX

    prefix_tokens = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    available = MAX_RERANK_TOKENS - len(prefix_tokens) - len(suffix_tokens)
    bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {document}" for question, document in pairs]
    encoded = tokenizer(
        bodies,
        padding=False,
        truncation=True,
        max_length=available,
        add_special_tokens=False,
    )
    features = [
        {
            "input_ids": [*prefix_tokens, *ids, *suffix_tokens],
            "attention_mask": [1] * (len(prefix_tokens) + len(ids) + len(suffix_tokens)),
        }
        for ids in encoded["input_ids"]
    ]
    return tokenizer.pad(features, padding=True, max_length=MAX_RERANK_TOKENS, return_tensors="pt")


def score_candidates(
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    batch_timings = []
    for start in range(0, len(candidates), batch_size):
        block = candidates[start : start + batch_size]
        pairs = [(str(row["question"]), str(row["text"])) for row in block]
        batch = encode_rerank_pairs(tokenizer, pairs)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**batch, use_cache=False)
            margins = final_margin(outputs.logits, batch["attention_mask"])
            probabilities = torch.sigmoid(margins)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        batch_timings.append({"pairs": len(block), "seconds": elapsed})
        for source, margin, probability in zip(block, margins.detach().float().cpu(), probabilities.detach().float().cpu()):
            rows.append({
                "schema_version": 1,
                "question_id": str(source["question_id"]),
                "chunker": str(source["chunker"]),
                "chunk_id": str(source["chunk_id"]),
                "fused_rank": int(source["fused_rank"]),
                "score_margin": float(margin),
                "score_probability": float(probability),
            })
    total_seconds = sum(item["seconds"] for item in batch_timings)
    steady = batch_timings[1:] if len(batch_timings) > 1 else batch_timings
    steady_seconds = sum(item["seconds"] for item in steady)
    steady_pairs = sum(item["pairs"] for item in steady)
    metrics = {
        "pairs": len(candidates),
        "batch_size": batch_size,
        "execution_seconds_all_batches": total_seconds,
        "throughput_pairs_per_second_all_batches": len(candidates) / total_seconds,
        "first_batch_seconds": batch_timings[0]["seconds"],
        "steady_state_excluding_first_batch_pairs": steady_pairs,
        "steady_state_excluding_first_batch_seconds": steady_seconds,
        "steady_state_pairs_per_second": steady_pairs / steady_seconds,
        "memory": gpu_memory(torch),
    }
    return rows, metrics


def parameter_hash(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint16)
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        count += parameter.numel()
    if count == 0:
        raise RuntimeError("No trainable parameters")
    return f"{count}:{digest.hexdigest()}"


def adapter_parameter_hash(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        tensor = parameter.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint16)
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        count += parameter.numel()
    if count == 0:
        raise RuntimeError("No LoRA parameters found for save/reload verification")
    return f"{count}:{digest.hexdigest()}"


def checkpoint_save(
    torch: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    path: Path,
    *,
    method: str,
    update: int,
    order: Sequence[str],
) -> None:
    if path.exists():
        marker = path / "complete.json"
        if not marker.is_file():
            raise RuntimeError(f"Incomplete checkpoint cannot be overwritten: {path}")
        return
    path.mkdir(parents=True)
    model.save_pretrained(path / "adapter", safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path / "optimizer_scheduler.pt")
    state = {"method": method, "completed_update": update, "question_order": list(order)}
    atomic_write_json(path / "state.json", state)
    file_hashes = {
        str(item.relative_to(path)): sha256_path(item)
        for item in sorted(child for child in path.rglob("*") if child.is_file())
    }
    atomic_write_json(path / "complete.json", {"state": state, "files": file_hashes})


def _call_peft_with_unused_incompatible_torchao_disabled(
    model: Any, factory: Any
) -> tuple[Any, dict[str, Any]]:
    """Bypass PEFT's optional TorchAO dispatcher for an unquantized BF16 model.

    Colab currently preinstalls torchao 0.10.0 while PEFT 0.20.0 accepts only
    torchao >=0.16.  PEFT's dispatcher checks that optional dependency before
    checking whether a module is TorchAO-quantized.  Our pinned model is plain
    BF16 and must never use that dispatcher, so the availability callback is
    locally replaced only for the duration of PEFT adapter construction.
    """

    dispatch_module = __import__("peft.tuners.lora.torchao", fromlist=["is_torchao_available"])
    original = dispatch_module.is_torchao_available
    repair = {
        "id": "exclude_unused_incompatible_torchao_v1",
        "applied": False,
        "scientific_change": False,
        "base_model_quantized": bool(getattr(model, "hf_quantizer", None) is not None),
    }
    if repair["base_model_quantized"] or any(
        type(module).__module__.startswith("torchao") for module in model.modules()
    ):
        raise RuntimeError("Refusing the TorchAO compatibility repair for a quantized model")
    try:
        original()
    except ImportError as error:
        message = str(error)
        if "incompatible version of torchao" not in message:
            raise
        repair.update({
            "applied": True,
            "installed_torchao": importlib.metadata.version("torchao"),
            "reason": message,
        })
        dispatch_module.is_torchao_available = lambda: False
    try:
        adapted = factory()
    finally:
        dispatch_module.is_torchao_available = original
    if any(type(module).__module__.startswith("torchao") for module in adapted.modules()):
        raise RuntimeError("TorchAO module unexpectedly entered the smoke model")
    return adapted, repair


def add_lora(model: Any) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    return _call_peft_with_unused_incompatible_torchao_disabled(
        model, lambda: get_peft_model(model, config)
    )


def train_smoke_adapter(
    torch: Any,
    groups: Sequence[Mapping[str, Any]],
    *,
    method: str,
    cache_dir: Path,
    output: Path,
) -> tuple[Path, dict[str, Any], Any, Any]:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from chunkrag.eaai_phase4.training_contract import collate_listwise_groups, torch_listwise_loss

    tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    base, load_metrics = load_causal_model(torch, RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    model, peft_environment_repair = add_lora(base)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    patch_margin_head(model, tokenizer)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=0.0001, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=2)
    initial_hash = parameter_hash(torch, model)

    order = sorted(range(len(groups)), key=lambda index: str(groups[index]["question_id"]))
    random.Random(SEED).shuffle(order)
    ordered = [groups[index] for index in order]
    question_order = [str(group["question_id"]) for group in ordered]
    logs = []
    unpadded_tokens = 0
    padded_tokens = 0
    processed_pairs = 0
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for microbatch_index in range(4):
        microgroups = ordered[microbatch_index * 2 : (microbatch_index + 1) * 2]
        batch = collate_listwise_groups(
            microgroups,
            tokenizer,
            instruction=INSTRUCTION,
            max_length=MAX_RERANK_TOKENS,
        )
        labels = list(batch.pop("binary_labels"))
        group_sizes = list(batch.pop("group_sizes"))
        group_keys = list(batch.pop("group_keys"))
        token_counts = list(batch.pop("token_counts"))
        expected_labels = []
        for size in group_sizes:
            expected_labels.extend([1] + [0] * (int(size) - 1))
        if labels != expected_labels:
            raise RuntimeError("Batch flattened groups were mixed or positive indexing changed")
        for key, value in batch.items():
            batch[key] = value.to("cuda")
        if [int(value) for value in batch["attention_mask"].sum(dim=1).cpu()] != token_counts:
            raise RuntimeError("Padding/masking changed a pair token count")
        outputs = model(**batch, use_cache=False)
        scores = final_margin(outputs.logits, batch["attention_mask"])
        loss = torch_listwise_loss(scores, group_sizes, temperature=1.0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {method} loss")
        (loss / 2.0).backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if not gradients_finite:
            raise RuntimeError(f"Non-finite {method} gradient")
        unpadded_tokens += sum(token_counts)
        padded_tokens += int(batch["input_ids"].numel())
        processed_pairs += len(token_counts)
        entry = {
            "microbatch": microbatch_index + 1,
            "group_keys": group_keys,
            "group_sizes": group_sizes,
            "loss": float(loss.detach().float().cpu()),
            "pairs": len(token_counts),
            "tokens_unpadded": sum(token_counts),
            "tokens_padded": int(batch["input_ids"].numel()),
        }
        if (microbatch_index + 1) % 2 == 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update = (microbatch_index + 1) // 2
            checkpoint_save(
                torch,
                model,
                optimizer,
                scheduler,
                output / "checkpoints" / method / f"update-{update}",
                method=method,
                update=update,
                order=question_order,
            )
            marker = json.loads((output / "checkpoints" / method / f"update-{update}/complete.json").read_text())
            if marker["state"]["completed_update"] != update:
                raise RuntimeError("Checkpoint resumption marker failed round trip")
            restored = torch.load(
                output / "checkpoints" / method / f"update-{update}/optimizer_scheduler.pt",
                map_location="cuda",
                weights_only=False,
            )
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            entry["optimizer_update"] = update
        logs.append(entry)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_hash = parameter_hash(torch, model)
    if initial_hash == final_hash:
        raise RuntimeError(f"{method} trainable parameters did not change")
    final_adapter = output / "checkpoints" / method / "update-2/adapter"
    metrics = {
        "method": method,
        "cold_base_load": load_metrics,
        "optimizer_updates": 2,
        "group_draws": 8,
        "unique_questions_exposed": len(set(question_order)),
        "group_repetitions": 1.0,
        "question_order": question_order,
        "processed_pairs": processed_pairs,
        "processed_tokens_unpadded": unpadded_tokens,
        "processed_tokens_padded": padded_tokens,
        "training_seconds": elapsed,
        "pairs_per_second": processed_pairs / elapsed,
        "unpadded_tokens_per_second": unpadded_tokens / elapsed,
        "trainable_parameter_hash_before": initial_hash,
        "trainable_parameter_hash_after": final_hash,
        "trainable_parameters_changed": True,
        "losses": logs,
        "memory": gpu_memory(torch),
        "checkpoint_resumption_markers_verified": 2,
        "peft_environment_repair": peft_environment_repair,
    }
    return final_adapter, metrics, model, tokenizer


def load_adapter_for_reload_check(
    torch: Any, adapter: Path, cache_dir: Path
) -> tuple[Any, Any, dict[str, Any]]:
    from peft import PeftModel

    tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    base, load_metrics = load_causal_model(torch, RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    started = time.perf_counter()
    model, peft_environment_repair = _call_peft_with_unused_incompatible_torchao_disabled(
        base, lambda: PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    )
    patch_margin_head(model, tokenizer)
    model.eval()
    torch.cuda.synchronize()
    load_metrics["adapter_seconds"] = time.perf_counter() - started
    load_metrics["adapter_memory"] = gpu_memory(torch)
    load_metrics["peft_environment_repair"] = peft_environment_repair
    return model, tokenizer, load_metrics


def probe_scores(torch: Any, model: Any, tokenizer: Any, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    rows, _ = score_candidates(torch, model, tokenizer, candidates[:16], batch_size=16)
    return [float(row["score_margin"]) for row in rows]


def compute_rankings(
    candidates: Sequence[Mapping[str, Any]],
    scores_by_system: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    from chunkrag.eaai_phase3.evidence import ndcg_at_k
    from chunkrag.eaai_phase4.dialogue_contract import candidate_pool_hash, prompt_contract_hash

    sources: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_by_key = {}
    for row in candidates:
        key = (str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"]))
        source_by_key[key] = dict(row)
        sources[(key[0], key[1])].append(dict(row))
    score_maps = {
        system: {
            (str(row["question_id"]), str(row["chunker"]), str(row["chunk_id"])): float(row["score_margin"])
            for row in score_rows
        }
        for system, score_rows in scores_by_system.items()
    }
    rows = []
    for qid, chunker in sorted(sources, key=lambda key: (key[0], CHUNKERS.index(key[1]))):
        pool = sources[(qid, chunker)]
        if len(pool) != 20:
            raise RuntimeError("A ranking does not have the common pool of 20")
        for system in ["H", "U", "B", "E"]:
            if system == "H":
                ranked = sorted(pool, key=lambda row: (int(row["fused_rank"]), str(row["chunk_id"])))
            else:
                mapping = score_maps[system]
                ranked = sorted(
                    pool,
                    key=lambda row: (
                        -mapping[(qid, chunker, str(row["chunk_id"]))],
                        int(row["fused_rank"]),
                        str(row["chunk_id"]),
                    ),
                )
            grades = [int(row["evidence_grade"]) for row in ranked]
            rows.append({
                "schema_version": 1,
                "system": system,
                "question_id": qid,
                "question": str(pool[0]["question"]),
                "chunker": chunker,
                "common_candidate_pool_sha256": candidate_pool_hash(pool),
                "candidate_count": len(ranked),
                "corrected_common_pool_evidence_ndcg_at_4": ndcg_at_k(grades, k=4),
                "zero_idcg_convention": 0.0,
                "top_k": [
                    {
                        "rank": index,
                        "chunk_id": str(item["chunk_id"]),
                        "document_id": str(item["document_id"]),
                        "text": str(item["text"]),
                        "evidence_grade": int(item["evidence_grade"]),
                    }
                    for index, item in enumerate(ranked[:4], start=1)
                ],
                "prompt_contract_sha256": prompt_contract_hash(str(pool[0]["question"])),
            })
    if len(rows) != 128:
        raise RuntimeError(f"Expected 128 ranking cells, found {len(rows)}")
    for key in {(row["question_id"], row["chunker"]) for row in rows}:
        hashes = {
            row["common_candidate_pool_sha256"]
            for row in rows
            if (row["question_id"], row["chunker"]) == key
        }
        prompts = {
            row["prompt_contract_sha256"]
            for row in rows
            if (row["question_id"], row["chunker"]) == key
        }
        if len(hashes) != 1 or len(prompts) != 1:
            raise RuntimeError("Common inference candidates or prompt skeleton differ by system")
    return rows


def write_immutable_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable output {path}")
    store = ResumableJsonlStore(path, ["system", "question_id", "chunker"])
    for row in rows:
        store.append(row)


def generate_outputs(
    torch: Any,
    rankings: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Path,
    output: Path,
    batch_size: int = 4,
) -> dict[str, Any]:
    from chunkrag.eaai_phase3.generation_runtime import (
        _normalize_complete_response,
        pack_generation_messages,
    )

    tokenizer = load_tokenizer(GENERATOR_MODEL, GENERATOR_REVISION, cache_dir)
    model, load_metrics = load_causal_model(torch, GENERATOR_MODEL, GENERATOR_REVISION, cache_dir)
    model.eval()
    store = ResumableJsonlStore(output / "generation_rows.jsonl", ["system", "question_id", "chunker"])
    pending = [row for row in rankings if (str(row["system"]), str(row["question_id"]), str(row["chunker"])) not in store.rows]
    batch_metrics = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(pending), batch_size):
        block = pending[start : start + batch_size]
        packed_rows = []
        texts = []
        for row in block:
            context = "\n\n".join(f"[{index}] {item['text']}" for index, item in enumerate(row["top_k"], start=1))
            packed = pack_generation_messages(
                tokenizer,
                question=str(row["question"]),
                context=context,
                answer_style="complete",
                max_input_tokens=MAX_INPUT_TOKENS,
            )
            try:
                text = tokenizer.apply_chat_template(
                    packed["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    packed["messages"], tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            packed_rows.append((row, packed))
            texts.append(text)
        batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        if int(batch["input_ids"].shape[1]) > MAX_INPUT_TOKENS:
            raise RuntimeError("A packed generator input exceeds 1,536 tokens")
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
        batch_generated_tokens = []
        for ids in generated_ids:
            values = ids.tolist()
            count = len(values)
            if tokenizer.eos_token_id in values:
                count = values.index(tokenizer.eos_token_id) + 1
            batch_generated_tokens.append(count)
        batch_metrics.append({
            "batch_index": len(batch_metrics) + 1,
            "records": len(block),
            "seconds": elapsed,
            "generated_tokens": sum(batch_generated_tokens),
        })
        for (row, packed), raw, generated_count in zip(packed_rows, decoded, batch_generated_tokens):
            store.append({
                "schema_version": 1,
                "system": str(row["system"]),
                "question_id": str(row["question_id"]),
                "chunker": str(row["chunker"]),
                "raw_output": raw,
                "normalized_output": _normalize_complete_response(raw),
                "packed_context": packed["packed_context"],
                "full_prompt_tokens": int(packed["full_prompt_tokens"]),
                "used_prompt_tokens": int(packed["used_prompt_tokens"]),
                "context_truncated": bool(packed["context_truncated"]),
                "generated_tokens": int(generated_count),
                "thinking_enabled": False,
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "prompt_contract_sha256": str(row["prompt_contract_sha256"]),
                "common_candidate_pool_sha256": str(row["common_candidate_pool_sha256"]),
            })
    if len(store.rows) != 128:
        raise RuntimeError(f"Generation incomplete: {len(store.rows)}/128")
    total_seconds = sum(item["seconds"] for item in batch_metrics)
    total_tokens = sum(item["generated_tokens"] for item in batch_metrics)
    steady = batch_metrics[1:] if len(batch_metrics) > 1 else batch_metrics
    steady_seconds = sum(item["seconds"] for item in steady)
    steady_records = sum(item["records"] for item in steady)
    steady_tokens = sum(item["generated_tokens"] for item in steady)

    first = next(iter(store.rows.values()))
    if store.append(first) != "skipped_identical":
        raise RuntimeError("Identical generation checkpoint was not idempotent")
    conflict = dict(first)
    conflict.pop("row_sha256", None)
    conflict["raw_output"] = str(conflict["raw_output"]) + " conflict"
    duplicate_rejected = False
    try:
        store.append(conflict)
    except RuntimeError:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise RuntimeError("Conflicting duplicate generation record was accepted")

    metrics = {
        "cold_model_load": load_metrics,
        "records": len(store.rows),
        "execution_seconds_all_batches": total_seconds,
        "records_per_second_all_batches": len(store.rows) / total_seconds,
        "generated_tokens": total_tokens,
        "generated_tokens_per_second_all_batches": total_tokens / total_seconds,
        "first_batch_seconds": batch_metrics[0]["seconds"] if batch_metrics else None,
        "steady_state_excluding_first_batch_records": steady_records,
        "steady_state_excluding_first_batch_seconds": steady_seconds,
        "steady_state_records_per_second": steady_records / steady_seconds,
        "steady_state_generated_tokens_per_second": steady_tokens / steady_seconds,
        "memory": gpu_memory(torch),
        "resume_identical_skip": True,
        "conflicting_duplicate_rejected": True,
    }
    release_model(torch, model, tokenizer)
    return metrics


def tokenizer_identity(cache_dir: Path) -> dict[str, Any]:
    candidates = [
        cache_dir / "hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots" / RERANKER_REVISION,
        cache_dir / "models--Qwen--Qwen3-Reranker-0.6B/snapshots" / RERANKER_REVISION,
    ]
    snapshot = next((path for path in candidates if path.is_dir()), candidates[0])
    files = {}
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        path = snapshot / name
        if path.is_file():
            files[name] = sha256_path(path)
    if not files:
        raise RuntimeError(f"Pinned reranker tokenizer snapshot not found at {snapshot}")
    return {"model": RERANKER_MODEL, "revision": RERANKER_REVISION, "files": files}


def prechecks(bundle: Path, cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from chunkrag.eaai_phase4.preprocessing import (
        _label_blind_representative_key,
        dedup_identity,
        training_pair_token_count,
    )
    from chunkrag.eaai_phase4.training_contract import hand_calculated_listwise_loss

    manifest_path = bundle / "smoke_artifacts/presmoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full_hashes = {
        method: sha256_path(bundle / f"full_inputs/{method}_training_groups_label_blind_pair_prune_v1.jsonl")
        for method in ["B", "E"]
    }
    if full_hashes != EXPECTED_FULL_INPUTS:
        raise RuntimeError(f"Full B/E training-input identity mismatch: {full_hashes}")
    B_groups = read_jsonl(bundle / "smoke_artifacts/private/smoke_groups_B.jsonl")
    E_groups = read_jsonl(bundle / "smoke_artifacts/private/smoke_groups_E.jsonl")
    candidates = read_jsonl(bundle / "smoke_artifacts/private/smoke_candidates.jsonl")
    if len(B_groups) != 8 or len(E_groups) != 8 or len(candidates) != 640:
        raise RuntimeError("Smoke input row counts are not 8 B, 8 E, and 640 candidates")
    selection_path = bundle / "smoke_artifacts/private/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_question_ids = {str(value) for value in selection["selected_question_ids"]}
    observed_question_ids = {
        str(row["question_id"]) for row in [*B_groups, *E_groups, *candidates]
    }
    # The frozen Phase 4 development pool joins the original TechQA training
    # and validation partitions.  Their stable IDs retain TRAIN_Q/DEV_Q
    # prefixes, so prefix inspection is not a valid split-membership test.
    # Membership is instead checked against the prepared selection, whose
    # source rows are content-addressed in the verified pre-smoke bundle.
    if selection.get("data_scope") != "existing TechQA development questions only":
        raise RuntimeError("Smoke selection does not declare the approved development-only scope")
    if selection.get("doc2dial_cases_or_references_included") is not False:
        raise RuntimeError("Smoke selection contains or ambiguously declares Doc2Dial content")
    if len(selected_question_ids) != 8 or observed_question_ids != selected_question_ids:
        raise RuntimeError(
            "Smoke rows do not exactly match the eight content-addressed development selections"
        )
    if set(row["question_id"] for row in B_groups) != set(row["question_id"] for row in E_groups):
        raise RuntimeError("B/E smoke questions differ")

    # Static and dynamic evidence that representative selection is label blind.
    import inspect
    source = inspect.getsource(_label_blind_representative_key)
    if 'row.get("grade"' in source or 'row["grade"]' in source or 'row.get("label"' in source or 'row["label"]' in source:
        raise RuntimeError("Deduplication representative key refers to a label or grade")
    synthetic = [
        {"qid": "synthetic", "chunk_id": "later", "doc_id": "d", "chunker": "x", "fused_rank": 2, "text": "The A!", "grade": 2},
        {"qid": "synthetic", "chunk_id": "first", "doc_id": "d", "chunker": "x", "fused_rank": 1, "text": "a", "grade": 0},
    ]
    if dedup_identity(synthetic) != dedup_identity([{**row, "grade": 2 - row["grade"]} for row in synthetic]):
        raise RuntimeError("Deduplication changed under synthetic label permutation")

    tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
    pair_checks = {}
    for method, groups in [("B", B_groups), ("E", E_groups)]:
        method_counts = []
        for group in groups:
            pairs = [group["positive"], *group["negatives"]]
            observed = [
                training_pair_token_count(tokenizer, group["question"], pair["text"], INSTRUCTION)
                for pair in pairs
            ]
            expected = [int(value) for value in group["preflight_pair_token_counts"]]
            if observed != expected or any(value > MAX_RERANK_TOKENS for value in observed):
                raise RuntimeError(f"Pinned-tokenizer pair-length mismatch for {method}/{group['group_id']}")
            method_counts.extend(observed)
        pair_checks[method] = {
            "groups": len(groups),
            "pairs": len(method_counts),
            "tokens": sum(method_counts),
            "min_pair_tokens": min(method_counts),
            "max_pair_tokens": max(method_counts),
        }
    hand_loss = hand_calculated_listwise_loss([2.0, 0.0, 0.0, 1.0, -1.0], [2, 3])
    if not math.isclose(hand_loss, 0.7672669877436765, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Unexpected hand-calculated listwise loss {hand_loss}")
    identity = {
        "presmoke_manifest_sha256": sha256_path(manifest_path),
        "full_training_inputs": full_hashes,
        "tokenizer": tokenizer_identity(cache_dir),
        "pair_checks": pair_checks,
        "label_blind_static_check": True,
        "label_permutation_invariance_check": True,
        "hand_calculated_listwise_loss": hand_loss,
        "doc2dial_cases_or_references_read": False,
        "selection_declared_doc2dial_content": manifest["privacy"]["doc2dial_content_in_bundle"],
    }
    return B_groups, E_groups, candidates, identity


def inventory(path: Path) -> dict[str, Any]:
    files = {}
    for item in sorted(child for child in path.rglob("*") if child.is_file() and child.name != "smoke_manifest.json"):
        files[str(item.relative_to(path))] = {"bytes": item.stat().st_size, "sha256": sha256_path(item)}
    return {"file_count": len(files), "tree_sha256": sha256_json(files), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    cache_dir = args.cache_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle / "working_copy/src"))

    import torch

    status_path = output / "smoke_status.json"
    hardware = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_memory_gib": float(torch.cuda.get_device_properties(0).total_memory / 1024**3) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    try:
        if not hardware["cuda_available"] or "A100" not in str(hardware["device_name"]).upper():
            raise RuntimeError(f"BLOCKED: approved smoke requires an A100, observed {hardware['device_name']}")
        B_groups, E_groups, candidates, identity = prechecks(bundle, cache_dir)
        atomic_write_json(output / "input_identity.json", identity)

        # U: exactly the authorized 640 development candidate pairs.
        U_tokenizer = load_tokenizer(RERANKER_MODEL, RERANKER_REVISION, cache_dir)
        U_model, U_load = load_causal_model(torch, RERANKER_MODEL, RERANKER_REVISION, cache_dir)
        patch_margin_head(U_model, U_tokenizer)
        U_scores, U_metrics = score_candidates(torch, U_model, U_tokenizer, candidates)
        U_metrics["cold_base_load"] = U_load
        release_model(torch, U_model, U_tokenizer)
        del U_model, U_tokenizer

        scores_by_system = {"U": U_scores}
        training_metrics = {}
        reranking_metrics = {"U": U_metrics}
        reload_checks = {}
        for method, groups in [("B", B_groups), ("E", E_groups)]:
            adapter, metrics, trained_model, trained_tokenizer = train_smoke_adapter(
                torch, groups, method=method, cache_dir=cache_dir, output=output
            )
            before_hash = adapter_parameter_hash(torch, trained_model)
            release_model(torch, trained_model, trained_tokenizer)
            del trained_model, trained_tokenizer
            reloaded_model, reloaded_tokenizer, reload_metrics = load_adapter_for_reload_check(
                torch, adapter, cache_dir
            )
            after_hash = adapter_parameter_hash(torch, reloaded_model)
            if before_hash != after_hash:
                raise RuntimeError(f"{method} adapter parameters changed across save/reload")
            method_scores, method_rerank = score_candidates(
                torch, reloaded_model, reloaded_tokenizer, candidates
            )
            release_model(torch, reloaded_model, reloaded_tokenizer)
            del reloaded_model, reloaded_tokenizer
            training_metrics[method] = metrics
            reranking_metrics[method] = method_rerank
            scores_by_system[method] = method_scores
            reload_checks[method] = {
                "adapter_parameter_hash_before_save": before_hash,
                "adapter_parameter_hash_after_reload": after_hash,
                "pass": True,
                "reload": reload_metrics,
            }

        rankings = compute_rankings(candidates, scores_by_system)
        write_immutable_jsonl(output / "rankings.jsonl", rankings)
        generation_metrics = generate_outputs(
            torch, rankings, cache_dir=cache_dir, output=output, batch_size=4
        )

        results = {
            "schema_version": 1,
            "smoke_id": "phase4_development_smoke_v4",
            "status": "PASS",
            "scope": "eight exposed TechQA development questions; no Doc2Dial cases or references",
            "hardware": hardware,
            "input_identity": identity,
            "training": training_metrics,
            "reranking": reranking_metrics,
            "reload_checks": reload_checks,
            "generation": generation_metrics,
            "row_counts": {
                "candidate_pairs_per_U_B_E": 640,
                "rankings": len(rankings),
                "generations": len(read_jsonl(output / "generation_rows.jsonl")),
            },
            "scientific_interpretation": "Operational smoke only; no answer-quality scores were computed and no model-selection evidence is created.",
        }
        atomic_write_json(output / "smoke_results.json", results)
        atomic_write_json(status_path, {"status": "PASS", "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        file_inventory = inventory(output)
        manifest = {
            "schema_version": 1,
            "smoke_id": "phase4_development_smoke_v4",
            "technical_repairs": [
                {
                    "id": "split_membership_assertion_v1",
                    "scientific_change": False,
                    "description": "Replaced an invalid DEV_Q prefix assertion with exact membership against the content-addressed eight-question development selection; approved inputs and budgets are unchanged.",
                }
                ,
                {
                    "id": "exclude_unused_incompatible_torchao_v1",
                    "scientific_change": False,
                    "description": "Locally disables PEFT's optional TorchAO dispatcher while constructing adapters for the explicitly unquantized BF16 model because Colab's unused torchao 0.10.0 is below PEFT's supported version; the original callback is restored immediately.",
                }
            ],
            "status": "PASS",
            "input_identity_sha256": sha256_path(output / "input_identity.json"),
            "results_sha256": sha256_path(output / "smoke_results.json"),
            "files_excluding_manifest": file_inventory,
            "actions_not_taken": ["No Doc2Dial case or reference was read.", "No full grid or refit was launched.", "No evaluation retrieval, reranking, or generation was launched.", "No publish, sharing, commit, or push action was taken."],
        }
        atomic_write_json(output / "smoke_manifest.json", manifest)
        print(json.dumps({"status": "PASS", "output": str(output), "manifest_sha256": sha256_path(output / "smoke_manifest.json")}, indent=2))
    except Exception as error:
        failure = {
            "status": "BLOCKED" if str(error).startswith("BLOCKED:") else "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "hardware": hardware,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(output / "smoke_failure.json", failure)
        atomic_write_json(status_path, {"status": failure["status"], "error": str(error)})
        raise


if __name__ == "__main__":
    main()
