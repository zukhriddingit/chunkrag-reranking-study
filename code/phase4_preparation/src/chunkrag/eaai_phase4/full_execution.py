"""Manifest-driven contracts for the frozen Phase 4 full execution.

This module intentionally keeps scientific decisions as data in the frozen
configuration.  The GPU runner calls these helpers, while the CPU tests exercise
the same exact-key, metric, clustering, and budget logic.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CHUNKERS = ("fixed_128", "fixed_254", "recursive_254", "sentence_254")
METHODS = ("B", "E")
SEEDS = (20260904, 20260905, 20260906)
GRID_LEARNING_RATES = (0.00005, 0.0001)
GRID_UPDATE_BUDGETS = (227, 454, 681)
REFIT_UPDATE_BUDGETS = {227: 431, 454: 862, 681: 1293}
EXPECTED_EVALUATION_DOCUMENTS = 273
EXPECTED_RETRIEVAL_CELLS = 1092
EXPECTED_CANDIDATES = 21840
EXPECTED_RERANKER_SCORES = 152880
EXPECTED_GENERATIONS = 8736
HARD_CEILING_HOURS = 20.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path, *, verify_hashes: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stored = row.get("row_sha256")
            if verify_hashes and stored is not None:
                observed = sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
                if observed != stored:
                    raise RuntimeError(f"Row hash mismatch in {path}:{line_number}")
            rows.append(row)
    return rows


def hashed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(row)
    materialized.pop("row_sha256", None)
    materialized["row_sha256"] = sha256_json(materialized)
    return materialized


def write_immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(observed) != canonical_json(value):
            raise RuntimeError(f"Conflicting immutable JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ResumableJsonlStore:
    """Append-only hashed JSONL with exact-key idempotence and conflict rejection."""

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
        materialized = hashed_row(row)
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


def tree_inventory(roots: Sequence[Path], base: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = str(path.relative_to(base))
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
    return {
        "file_count": len(files),
        "tree_sha256": sha256_json(files),
        "files": files,
    }


class AllocationLedger:
    """Persist cumulative allocated-GPU time, conservatively recovering crashes."""

    def __init__(self, path: Path, *, ceiling_hours: float = HARD_CEILING_HOURS) -> None:
        self.path = path
        self.ceiling_seconds = float(ceiling_hours) * 3600.0
        self.session_id: str | None = None

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "unit": "allocated_A100_seconds",
                "hard_ceiling_seconds": self.ceiling_seconds,
                "closed_seconds": 0.0,
                "sessions": [],
            }
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if float(value["hard_ceiling_seconds"]) != self.ceiling_seconds:
            raise RuntimeError("GPU ceiling changed across resumptions")
        return value

    def begin(self, *, stage: str, hardware: str) -> dict[str, Any]:
        if hardware != "NVIDIA A100-SXM4-40GB":
            raise RuntimeError(f"Refusing non-frozen GPU hardware: {hardware}")
        ledger = self._load()
        for session in ledger["sessions"]:
            if session.get("status") == "active":
                conservative_end = parse_utc(session["last_heartbeat_utc"]) + 60.0
                duration = max(0.0, conservative_end - parse_utc(session["started_utc"]))
                ledger["closed_seconds"] += duration
                session.update({
                    "status": "recovered_after_ungraceful_stop",
                    "ended_utc": datetime.fromtimestamp(conservative_end, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "allocated_seconds": duration,
                    "recovery_margin_seconds": 60.0,
                })
        now = utc_now()
        self.session_id = sha256_bytes(f"{now}|{stage}|{len(ledger['sessions'])}".encode("utf-8"))[:20]
        ledger["sessions"].append({
            "session_id": self.session_id,
            "stage_at_start": stage,
            "hardware": hardware,
            "started_utc": now,
            "last_heartbeat_utc": now,
            "status": "active",
        })
        atomic_write_json(self.path, ledger)
        return ledger

    def heartbeat(self, *, stage: str, projected_remaining_seconds: float = 0.0) -> float:
        if self.session_id is None:
            raise RuntimeError("Allocation session has not begun")
        ledger = self._load()
        session = next(row for row in ledger["sessions"] if row["session_id"] == self.session_id)
        now = utc_now()
        session["last_heartbeat_utc"] = now
        session["stage"] = stage
        current = max(0.0, parse_utc(now) - parse_utc(session["started_utc"]))
        cumulative = float(ledger["closed_seconds"]) + current
        session["current_allocated_seconds"] = current
        ledger["cumulative_allocated_seconds"] = cumulative
        ledger["remaining_seconds"] = max(0.0, self.ceiling_seconds - cumulative)
        atomic_write_json(self.path, ledger)
        if cumulative + float(projected_remaining_seconds) > self.ceiling_seconds:
            raise BudgetStop(
                f"Projected allocation {cumulative + projected_remaining_seconds:.1f}s exceeds "
                f"the frozen {self.ceiling_seconds:.1f}s ceiling"
            )
        return cumulative

    def end(self, *, status: str, stage: str) -> dict[str, Any]:
        if self.session_id is None:
            return self._load()
        ledger = self._load()
        session = next(row for row in ledger["sessions"] if row["session_id"] == self.session_id)
        now = utc_now()
        duration = max(0.0, parse_utc(now) - parse_utc(session["started_utc"]))
        session.update({"status": status, "stage": stage, "ended_utc": now, "last_heartbeat_utc": now, "allocated_seconds": duration})
        ledger["closed_seconds"] = float(ledger["closed_seconds"]) + duration
        ledger["cumulative_allocated_seconds"] = float(ledger["closed_seconds"])
        ledger["remaining_seconds"] = max(0.0, self.ceiling_seconds - float(ledger["closed_seconds"]))
        atomic_write_json(self.path, ledger)
        self.session_id = None
        return ledger


class BudgetStop(RuntimeError):
    pass


def deterministic_group_draws(group_count: int, total_draws: int, seed: int) -> list[int]:
    if group_count <= 0 or total_draws < 0:
        raise ValueError("Invalid group/draw count")
    output: list[int] = []
    epoch = 0
    while len(output) < total_draws:
        indices = list(range(group_count))
        random.Random(int(seed) + 1_000_003 * epoch).shuffle(indices)
        output.extend(indices[: total_draws - len(output)])
        epoch += 1
    return output


def capture_rng_state(torch: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii"),
        "numpy": base64.b64encode(pickle.dumps(np.random.get_state())).decode("ascii"),
        "torch_cpu": base64.b64encode(torch.get_rng_state().cpu().numpy().tobytes()).decode("ascii"),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = [
            base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")
            for state in torch.cuda.get_rng_state_all()
        ]
    return value


def restore_rng_state(torch: Any, value: Mapping[str, Any]) -> None:
    random.setstate(pickle.loads(base64.b64decode(value["python"])))
    np.random.set_state(pickle.loads(base64.b64decode(value["numpy"])))
    cpu = np.frombuffer(base64.b64decode(value["torch_cpu"]), dtype=np.uint8).copy()
    torch.set_rng_state(torch.from_numpy(cpu))
    if torch.cuda.is_available() and value.get("torch_cuda"):
        states = []
        for encoded in value["torch_cuda"]:
            array = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).copy()
            states.append(torch.from_numpy(array))
        torch.cuda.set_rng_state_all(states)


def convert_preflight_group(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    question = str(row["messages"][1]["content"])
    instruction = str(row["messages"][0]["content"])
    positive_text = str(row["positive_messages"][0][0]["content"])
    negatives = [
        {"chunk_id": str(chunk_id), "text": str(messages[0]["content"]), "binary_label": 0}
        for chunk_id, messages in zip(row["negative_chunk_ids"], row["negative_messages"], strict=True)
    ]
    return {
        "schema_version": 1,
        "method": method,
        "policy": "label_blind_dedup_pair_prune_v1",
        "group_id": sha256_bytes(f"{method}|{row['row_sha256']}".encode("utf-8")),
        "question_id": str(row["question_id"]),
        "question": question,
        "instruction": instruction,
        "positive": {
            "chunk_id": str(row["positive_chunk_id"]),
            "text": positive_text,
            "binary_label": 1,
            "source_grade": int(row["positive_grade"]),
        },
        "negatives": negatives,
        "preflight_pair_token_counts": [int(value) for value in row["pair_token_counts"]],
        "preflight_source_row_sha256": str(row["row_sha256"]),
    }


def ndcg_at_k(grades: Sequence[int | float], k: int = 4) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    values = [float(value) for value in grades]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Grades must be finite and non-negative")

    def dcg(items: Sequence[float]) -> float:
        return sum((2.0**value - 1.0) / math.log2(index + 2.0) for index, value in enumerate(items))

    observed = values[:k]
    ideal = sorted(values, reverse=True)[:k]
    denominator = dcg(ideal)
    return 0.0 if denominator == 0.0 else dcg(observed) / denominator


def aggregate_validation_scores(
    labels: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> dict[str, Any]:
    if len(labels) != len(scores):
        raise ValueError("Labels and scores differ in length")
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for label, score in zip(labels, scores, strict=True):
        row = dict(label)
        row["score"] = float(score)
        by_cell[(str(row["question_id"]), str(row["chunker"]))].append(row)
    if len(by_cell) != 800:
        raise RuntimeError(f"Expected 800 validation cells, found {len(by_cell)}")
    cell_values: dict[tuple[str, str], float] = {}
    for key, rows in by_cell.items():
        if len(rows) != 20:
            raise RuntimeError(f"Validation pool is not 20 candidates at {key}")
        ranked = sorted(rows, key=lambda row: (-float(row["score"]), int(row["fused_rank"]), str(row["chunk_id"])))
        cell_values[key] = ndcg_at_k([int(row["grade"]) for row in ranked], 4)
    by_question: dict[str, list[float]] = defaultdict(list)
    for (question_id, _chunker), value in cell_values.items():
        by_question[question_id].append(value)
    if len(by_question) != 200 or any(len(values) != 4 for values in by_question.values()):
        raise RuntimeError("Validation aggregation is not 200 questions x four chunkers")
    question_values = {question: float(np.mean(values)) for question, values in by_question.items()}
    return {
        "question_count": len(question_values),
        "cell_count": len(cell_values),
        "question_mean_common_evidence_ndcg_at_4": float(np.mean(list(question_values.values()))),
        "question_values": question_values,
    }


def select_shared_configuration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {(str(row["method"]), float(row["learning_rate"]), int(row["optimizer_updates"])): row for row in rows}
    candidates = []
    for learning_rate in GRID_LEARNING_RATES:
        for updates in GRID_UPDATE_BUDGETS:
            b = indexed[("B", learning_rate, updates)]
            e = indexed[("E", learning_rate, updates)]
            shared = (float(b["validation_question_mean_common_evidence_ndcg_at_4"]) + float(e["validation_question_mean_common_evidence_ndcg_at_4"])) / 2.0
            candidates.append({
                "learning_rate": learning_rate,
                "optimizer_updates": updates,
                "B": float(b["validation_question_mean_common_evidence_ndcg_at_4"]),
                "E": float(e["validation_question_mean_common_evidence_ndcg_at_4"]),
                "shared_mean": shared,
            })
    selected = min(candidates, key=lambda row: (-row["shared_mean"], row["optimizer_updates"], row["learning_rate"]))
    return {"selected": selected, "all_shared_cells": candidates}


def normalize_text(text: str) -> str:
    import string

    value = str(text).lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def evidence_grade(chunk_text: str, chunk_document_id: str, gold_document_id: str, spans: Sequence[str]) -> int:
    normalized_chunk = normalize_text(chunk_text)
    if any(normalize_text(span) and normalize_text(span) in normalized_chunk for span in spans):
        return 2
    if str(chunk_document_id) == str(gold_document_id):
        return 1
    return 0


def family_key(domain: str, title: str) -> str:
    normalized = re.sub(r"\s*#\d+$", "", str(title).strip().lower())
    normalized = " ".join(normalized.split())
    return f"{str(domain).strip().lower()}::{normalized}"


def expected_ranking_instances() -> tuple[str, ...]:
    return ("H", "U", *(f"B_{seed}" for seed in SEEDS), *(f"E_{seed}" for seed in SEEDS))


def expected_generation_keys(document_ids: Sequence[str]) -> set[tuple[str, str, str]]:
    return {
        (instance, str(document_id), chunker)
        for instance in expected_ranking_instances()
        for document_id in document_ids
        for chunker in CHUNKERS
    }


def answer_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    from chunkrag.text_utils import best_exact_match, best_f1

    values = [str(value) for value in references]
    return {
        "exact_match": float(best_exact_match(str(prediction), values)),
        "f1": float(best_f1(str(prediction), values)),
    }


def primary_document_rows(generations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["system_instance"]), str(row["document_id"]), str(row["chunker"])): row for row in generations}
    document_meta: dict[str, tuple[str, str]] = {}
    for row in generations:
        document_meta[str(row["document_id"])] = (str(row["domain"]), str(row["family_key"]))
    if len(document_meta) != EXPECTED_EVALUATION_DOCUMENTS:
        raise RuntimeError("Primary analysis does not contain 273 documents")
    output = []
    for document_id in sorted(document_meta):
        paired = []
        seed_values = {}
        for seed in SEEDS:
            values = []
            for chunker in CHUNKERS:
                e = by_key[(f"E_{seed}", document_id, chunker)]
                b = by_key[(f"B_{seed}", document_id, chunker)]
                values.append(float(e["agent_response_f1"]) - float(b["agent_response_f1"]))
            seed_values[str(seed)] = float(np.mean(values))
            paired.extend(values)
        domain, family = document_meta[document_id]
        output.append({
            "schema_version": 1,
            "document_id": document_id,
            "domain": domain,
            "family_key": family,
            "difference": float(np.mean(paired)),
            "seed_differences": seed_values,
        })
    return output


def frozen_primary_analysis(document_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from chunkrag.eaai_phase4.statistics import (
        family_bootstrap_statistics,
        family_sign_flip_pvalue,
        percentile_interval,
    )

    values = [float(row["difference"]) for row in document_rows]
    families = [str(row["family_key"]) for row in document_rows]
    domains = [str(row["domain"]) for row in document_rows]
    estimate = float(np.mean(values))
    pvalue = family_sign_flip_pvalue(values, families, draws=100000, seed=20260923)
    bootstrap = family_bootstrap_statistics(values, families, domains, draws=20000, seed=20260922)
    interval = percentile_interval(bootstrap, 0.95)
    domain_effects = {
        domain: float(np.mean([value for value, observed in zip(values, domains, strict=True) if observed == domain]))
        for domain in sorted(set(domains))
    }
    seed_effects = {
        str(seed): float(np.mean([float(row["seed_differences"][str(seed)]) for row in document_rows]))
        for seed in SEEDS
    }
    seed_array = np.asarray(list(seed_effects.values()), dtype=np.float64)
    return {
        "schema_version": 1,
        "estimand": "document-weighted E-minus-B agent-response F1 averaged over four chunkers and three paired seeds",
        "documents": len(values),
        "families": len(set(families)),
        "mean_difference": estimate,
        "family_sign_flip_two_sided_p": pvalue,
        "family_cluster_bootstrap_95pct": {"low": interval[0], "high": interval[1]},
        "decision_rule_passed": bool(estimate >= 0.010 and pvalue < 0.05),
        "interpretation_warning": "The decision rule does not prove that the population improvement is at least 0.010.",
        "domain_effects_secondary": domain_effects,
        "equal_domain_mean_secondary": float(np.mean(list(domain_effects.values()))),
        "seed_effects_secondary": seed_effects,
        "seed_sd_secondary": float(np.std(seed_array, ddof=1)),
        "seed_range_secondary": [float(np.min(seed_array)), float(np.max(seed_array))],
        "leave_one_seed_out_secondary": {
            omitted: float(np.mean([value for key, value in seed_effects.items() if key != omitted]))
            for omitted in seed_effects
        },
    }

