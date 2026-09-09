from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from chunkrag.eaai_phase2.io import (
    add_row_hash,
    canonical_json_bytes,
    iter_jsonl,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_row_hash,
    write_immutable_json,
    write_immutable_jsonl,
)
from chunkrag.eaai_phase3.constants import (
    CHUNKERS,
    PARTITION_SHA256,
    PRIMARY_RERANKER_SEED,
    RERANKER_SEEDS,
    RESERVE_SIZE,
)
from chunkrag.eaai_phase3.evidence import grade_candidate, ndcg_at_k
from chunkrag.eaai_phase3.protocol import verify_freeze_manifest


CORRECTION_ID = "reserve_ndcg_common_pool_v1"
EXPECTED_SYSTEMS = (
    "comparator",
    *(f"evidence_seed_{seed}" for seed in RERANKER_SEEDS),
)
PRIVATE_FIELDS = {
    "question_id",
    "question",
    "reference_answers",
    "relevant_document_ids",
    "text",
    "context",
    "packed_context",
    "raw_output",
    "normalized_output",
    "top_k",
    "candidate_ranking",
}


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return normalized


def _require_git_commit(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected release commit must be a full 40-character Git object ID")
    return normalized


def _check_file_hash(path: Path, expected: str, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {name}: {path}")
    expected = _require_sha256(expected, name=f"expected {name} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_release_scientific_code(
    *,
    release_archive: Path,
    freeze: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    """Compare frozen executed scientific files to bytes in the release archive."""
    records = freeze.get("scientific_code")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Freeze does not enumerate scientific code")
    matches: list[dict[str, Any]] = []
    with tarfile.open(release_archive, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for record in records:
            if not isinstance(record, dict) or record.get("type", "file") != "file":
                raise RuntimeError("Scientific-code freeze record is not a file")
            frozen_path = Path(str(record["path"])).resolve()
            try:
                relative = frozen_path.relative_to(workspace_root.resolve()).as_posix()
            except ValueError as error:
                raise RuntimeError(f"Frozen scientific path is outside the workspace: {frozen_path}") from error
            candidates = [
                member
                for member in members
                if member.name == relative or member.name.endswith(f"/{relative}")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Release archive has {len(candidates)} matches for frozen scientific file {relative}"
                )
            handle = archive.extractfile(candidates[0])
            if handle is None:
                raise RuntimeError(f"Cannot read release archive member: {candidates[0].name}")
            payload = handle.read()
            digest = sha256_bytes(payload)
            if len(payload) != int(record["bytes"]) or digest != str(record["sha256"]):
                raise RuntimeError(f"Release archive differs from frozen executed code: {relative}")
            matches.append(
                {
                    "relative_path": relative,
                    "release_member": candidates[0].name,
                    "bytes": len(payload),
                    "sha256": digest,
                }
            )
    return {
        "frozen_scientific_files_compared": len(matches),
        "exact_matches": len(matches),
        "mismatches": 0,
        "files": matches,
    }


def _directory_record(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    members = [
        {
            "path": str(member.relative_to(path)),
            "bytes": member.stat().st_size,
            "sha256": sha256_file(member),
        }
        for member in sorted(path.rglob("*"))
        if member.is_file()
    ]
    if not members:
        raise RuntimeError(f"Required directory is empty: {path}")
    return {
        "path": str(path.resolve()),
        "file_count": len(members),
        "bytes": sum(int(member["bytes"]) for member in members),
        "sha256": sha256_bytes(canonical_json_bytes(members)),
    }


def _stable_join_key(question_id: str, chunker: str, system: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"question_id": question_id, "chunker": chunker, "system": system}
        )
    )


def _validate_public(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        overlap = PRIVATE_FIELDS & set(value)
        if overlap:
            raise RuntimeError(f"Public correction contains protected fields at {location}: {sorted(overlap)}")
        for key, child in value.items():
            _validate_public(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public(child, location=f"{location}[{index}]")


def metric_self_tests() -> dict[str, Any]:
    regression = [1, 1, 1, 1, 2, *([0] * 15)]
    cases = {
        "specified_legacy_top_four": (ndcg_at_k(regression[:4], k=4), 1.0),
        "specified_corrected_common_pool": (
            ndcg_at_k(regression, k=4),
            0.5615579549479297,
        ),
        "ideal_ranking": (ndcg_at_k([2, 1, 1, 0, 0], k=4), 1.0),
        "zero_idcg_convention": (ndcg_at_k([0] * 20, k=4), 0.0),
        "pool_shorter_than_k": (ndcg_at_k([2, 1], k=4), 1.0),
    }
    from chunkrag.eaai_phase3.analysis import select_reranker_hyperparameters

    validation_rows = [
        {
            "learning_rate": 0.00005,
            "epochs": 1,
            "question_id": "synthetic-question",
            "chunker": "fixed_128",
            "chunk_id": f"candidate-{index:02d}",
            "score": float(len(regression) - index),
            "grade": grade,
        }
        for index, grade in enumerate(regression)
    ]
    validation_value = select_reranker_hyperparameters(
        validation_rows,
        learning_rates=(0.00005,),
        epochs=(1,),
        expected_question_ids={"synthetic-question"},
        chunkers=("fixed_128",),
        k=4,
    )["selected_metric"]
    cases["validation_implementation_common_pool_consistency"] = (
        float(validation_value),
        ndcg_at_k(regression, k=4),
    )
    results: dict[str, Any] = {}
    for name, (observed, expected) in cases.items():
        passed = math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        results[name] = {"observed": observed, "expected": expected, "passed": passed}
        if not passed:
            raise RuntimeError(f"Built-in NDCG regression failed: {name}")
    return {
        "status": "passed",
        "gain": "2^grade - 1",
        "discount": "log2(rank + 1), with one-based rank",
        "k": 4,
        "zero_idcg_convention": 0.0,
        "cases": results,
    }


def _validate_ranking(
    row: Mapping[str, Any],
    *,
    expected_pool_size: int,
    final_top_k: int,
) -> tuple[list[dict[str, Any]], list[int], str]:
    candidates = row.get("candidate_ranking")
    top_k = row.get("top_k")
    if not isinstance(candidates, list) or len(candidates) != expected_pool_size:
        raise RuntimeError(
            f"Required full candidate pool is missing or incomplete for "
            f"{row.get('question_id')}/{row.get('chunker')}/{row.get('system')}: "
            f"{len(candidates) if isinstance(candidates, list) else 'not-a-list'} != {expected_pool_size}"
        )
    if not isinstance(top_k, list) or len(top_k) != final_top_k:
        raise RuntimeError("Stored top_k is missing or has the wrong length")
    normalized = [dict(candidate) for candidate in candidates]
    if top_k != candidates[:final_top_k]:
        raise RuntimeError("Stored top_k does not equal the first four full-ranking candidates")
    ranks = [int(candidate.get("rank", -1)) for candidate in normalized]
    if ranks != list(range(1, expected_pool_size + 1)):
        raise RuntimeError("Full candidate ranking does not have exact one-based consecutive ranks")
    chunk_ids = [str(candidate.get("chunk_id", "")) for candidate in normalized]
    if any(not value for value in chunk_ids) or len(set(chunk_ids)) != expected_pool_size:
        raise RuntimeError("Full candidate ranking has missing or duplicate chunk IDs")
    reference_answers = [str(value) for value in row.get("reference_answers", [])]
    relevant_document_ids = [str(value) for value in row.get("relevant_document_ids", [])]
    grades: list[int] = []
    pool_identity: list[dict[str, Any]] = []
    for candidate in normalized:
        if "text" not in candidate or "document_id" not in candidate:
            raise RuntimeError("Full candidate ranking lacks text or document_id required for frozen grading")
        text = str(candidate["text"])
        document_id = str(candidate["document_id"])
        grade = grade_candidate(
            text=text,
            document_id=document_id,
            reference_answers=reference_answers,
            relevant_document_ids=relevant_document_ids,
        )
        grades.append(grade)
        pool_identity.append(
            {
                "chunk_id": str(candidate["chunk_id"]),
                "document_id": document_id,
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "grade": grade,
            }
        )
    pool_identity.sort(key=lambda value: value["chunk_id"])
    return normalized, grades, sha256_bytes(canonical_json_bytes(pool_identity))


def _verify_legacy_retrieval_metrics(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    grades: Sequence[int],
    *,
    final_top_k: int,
) -> float:
    """Reproduce every stored retrieval metric and return corrected NDCG@k."""
    stored = row.get("retrieval_metrics")
    expected_fields = {
        "grade_ndcg_at_4",
        "answer_visibility_at_4",
        "gold_document_coverage_at_4",
        "maximum_grade_at_4",
    }
    if not isinstance(stored, dict) or set(stored) != expected_fields:
        raise RuntimeError("Stored reserve retrieval metrics have an unexpected schema")
    relevant = {str(value) for value in row.get("relevant_document_ids", [])}
    top_candidates = candidates[:final_top_k]
    top_grades = list(grades[:final_top_k])
    top_documents = {str(candidate["document_id"]) for candidate in top_candidates}
    expected = {
        "grade_ndcg_at_4": ndcg_at_k(top_grades, k=final_top_k),
        "answer_visibility_at_4": float(any(grade == 2 for grade in top_grades)),
        "gold_document_coverage_at_4": (
            len(top_documents & relevant) / len(relevant) if relevant else 0.0
        ),
        "maximum_grade_at_4": float(max(top_grades, default=0)),
    }
    for field, expected_value in expected.items():
        try:
            stored_value = float(stored[field])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Stored reserve metric is invalid: {field}") from error
        if not math.isclose(stored_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Stored reserve metric is not reproducible for "
                f"{row.get('question_id')}/{row.get('chunker')}/{row.get('system')}: "
                f"{field}={stored_value} != {expected_value}"
            )
    return ndcg_at_k(grades, k=final_top_k)


def build_per_record_corrections(
    retrieval_rows: Iterable[dict[str, Any]],
    *,
    expected_question_count: int = RESERVE_SIZE,
    chunkers: Sequence[str] = CHUNKERS,
    systems: Sequence[str] = EXPECTED_SYSTEMS,
    expected_pool_size: int = 20,
    final_top_k: int = 4,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    rows = [dict(row) for row in retrieval_rows]
    expected_rows = expected_question_count * len(chunkers) * len(systems)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Reserve retrieval matrix has {len(rows)} rows; expected {expected_rows}")

    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    question_ids: set[str] = set()
    common_pools: dict[tuple[str, str], str] = {}
    common_metadata: dict[tuple[str, str], bytes] = {}
    corrections: list[dict[str, Any]] = []
    for row in rows:
        validate_row_hash(row)
        if row.get("study_stage") != "eaai_phase3_reserve_ranking":
            raise RuntimeError("Unexpected study stage in reserve retrieval")
        question_id = str(row.get("question_id", ""))
        chunker = str(row.get("chunker", ""))
        system = str(row.get("system", ""))
        key = (question_id, chunker, system)
        if not question_id or chunker not in chunkers or system not in systems or key in index:
            raise RuntimeError(f"Invalid or duplicate reserve retrieval key: {key}")
        index[key] = row
        question_ids.add(question_id)
        candidates, grades, pool_sha256 = _validate_ranking(
            row,
            expected_pool_size=expected_pool_size,
            final_top_k=final_top_k,
        )
        cell = (question_id, chunker)
        metadata = canonical_json_bytes(
            {
                "question": str(row.get("question", "")),
                "reference_answers": [str(value) for value in row.get("reference_answers", [])],
                "relevant_document_ids": [
                    str(value) for value in row.get("relevant_document_ids", [])
                ],
                "partition_sha256": str(row.get("partition_sha256", "")),
                "config_sha256": str(row.get("config_sha256", "")),
            }
        )
        if cell in common_pools and common_pools[cell] != pool_sha256:
            raise RuntimeError(f"Systems do not use a common candidate pool for {cell}")
        if cell in common_metadata and common_metadata[cell] != metadata:
            raise RuntimeError(f"Systems disagree on frozen question/grading metadata for {cell}")
        common_pools[cell] = pool_sha256
        common_metadata[cell] = metadata

        old_value = float(row["retrieval_metrics"]["grade_ndcg_at_4"])
        corrected_value = _verify_legacy_retrieval_metrics(
            row,
            candidates,
            grades,
            final_top_k=final_top_k,
        )
        grade_counts = Counter(grades)
        payload = {
            "schema_version": 1,
            "study_stage": "eaai_phase3_append_only_metric_correction",
            "correction_id": CORRECTION_ID,
            "stable_join_key": _stable_join_key(question_id, chunker, system),
            "question_id": question_id,
            "chunker": chunker,
            "system": system,
            "source_retrieval_row_sha256": str(row["row_sha256"]),
            "common_candidate_pool_sha256": pool_sha256,
            "candidate_pool_size": len(candidates),
            "top_four_grades": grades[:final_top_k],
            "full_pool_grade_counts": {
                "grade_0": grade_counts.get(0, 0),
                "grade_1": grade_counts.get(1, 0),
                "grade_2": grade_counts.get(2, 0),
            },
            "legacy_grade_ndcg_at_4": old_value,
            "corrected_grade_ndcg_at_4": corrected_value,
            "delta_grade_ndcg_at_4": corrected_value - old_value,
        }
        corrections.append(add_row_hash(payload))

    if len(question_ids) != expected_question_count:
        raise RuntimeError(
            f"Reserve retrieval contains {len(question_ids)} unique questions; expected {expected_question_count}"
        )
    expected_keys = {
        (question_id, str(chunker), str(system))
        for question_id in question_ids
        for chunker in chunkers
        for system in systems
    }
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))[:5]
        extra = sorted(set(index) - expected_keys)[:5]
        raise RuntimeError(f"Reserve retrieval matrix mismatch: missing={missing}, extra={extra}")
    corrections.sort(key=lambda row: (str(row["system"]), str(row["chunker"]), str(row["question_id"])))
    return corrections, index


def verify_generation_copies(
    generation_rows: Iterable[dict[str, Any]],
    retrieval_index: Mapping[tuple[str, str, str], dict[str, Any]],
    *,
    expected_question_count: int = RESERVE_SIZE,
    chunkers: Sequence[str] = CHUNKERS,
    primary_seed: int = PRIMARY_RERANKER_SEED,
) -> dict[str, Any]:
    rows = [dict(row) for row in generation_rows]
    expected_count = expected_question_count * len(chunkers) * 2
    if len(rows) != expected_count:
        raise RuntimeError(f"Reserve generation matrix has {len(rows)} rows; expected {expected_count}")
    seen: set[tuple[str, str, str]] = set()
    row_hash_by_retrieval_key: dict[tuple[str, str, str], str] = {}
    for row in rows:
        validate_row_hash(row)
        if row.get("study_stage") != "eaai_phase3_reserve_generation":
            raise RuntimeError("Unexpected study stage in reserve generation")
        question_id = str(row.get("question_id", ""))
        chunker = str(row.get("chunker", ""))
        condition = str(row.get("condition", ""))
        generation_key = (question_id, chunker, condition)
        if generation_key in seen:
            raise RuntimeError(f"Duplicate generation key: {generation_key}")
        seen.add(generation_key)
        expected_system = "comparator" if condition == "comparator" else f"evidence_seed_{primary_seed}"
        if condition not in {"comparator", "evidence_aware"} or row.get("retrieval_system") != expected_system:
            raise RuntimeError(f"Generation-to-retrieval system mismatch for {generation_key}")
        retrieval_key = (question_id, chunker, expected_system)
        if retrieval_key not in retrieval_index:
            raise RuntimeError(f"Generation row lacks a matching retrieval row: {retrieval_key}")
        retrieval = retrieval_index[retrieval_key]
        if row.get("top_k") != retrieval.get("top_k"):
            raise RuntimeError(f"Generation top_k differs from frozen retrieval top_k: {generation_key}")
        if row.get("retrieval_metrics") != retrieval.get("retrieval_metrics"):
            raise RuntimeError(f"Generation copied retrieval metrics differ from source: {generation_key}")
        row_hash_by_retrieval_key[retrieval_key] = str(row["row_sha256"])
    question_ids = sorted(
        {
            question_id
            for question_id, chunker, system in retrieval_index
            if system == "comparator" and chunker in chunkers
        }
    )
    expected_keys = {
        (question_id, str(chunker), condition)
        for question_id in question_ids
        for chunker in chunkers
        for condition in ("comparator", "evidence_aware")
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)[:5]
        extra = sorted(seen - expected_keys)[:5]
        raise RuntimeError(f"Reserve generation matrix mismatch: missing={missing}, extra={extra}")
    return {
        "row_count": len(rows),
        "verified_copy_count": len(rows),
        "row_hash_by_retrieval_key": row_hash_by_retrieval_key,
    }


def attach_generation_hashes(
    corrections: Iterable[dict[str, Any]],
    row_hash_by_retrieval_key: Mapping[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for correction in corrections:
        validate_row_hash(correction)
        base = dict(correction)
        base.pop("row_sha256")
        key = (str(base["question_id"]), str(base["chunker"]), str(base["system"]))
        base["copied_generation_row_sha256"] = row_hash_by_retrieval_key.get(key)
        result.append(add_row_hash(base))
    return result


def aggregate_corrections(
    corrections: Iterable[dict[str, Any]],
    *,
    primary_seed: int = PRIMARY_RERANKER_SEED,
) -> dict[str, Any]:
    rows = [dict(row) for row in corrections]
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_system_chunker: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row["system"])].append(row)
        by_system_chunker[(str(row["system"]), str(row["chunker"]))].append(row)

    def summary(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        old = mean(float(value["legacy_grade_ndcg_at_4"]) for value in values)
        new = mean(float(value["corrected_grade_ndcg_at_4"]) for value in values)
        return {"record_count": len(values), "legacy_mean": old, "corrected_mean": new, "delta": new - old}

    system_summary = {system: summary(values) for system, values in sorted(by_system.items())}
    chunker_summary = {
        system: {
            chunker: summary(by_system_chunker[(system, chunker)])
            for chunker in sorted({key[1] for key in by_system_chunker if key[0] == system})
        }
        for system in sorted(by_system)
    }
    old_order = sorted(system_summary, key=lambda system: (-system_summary[system]["legacy_mean"], system))
    new_order = sorted(system_summary, key=lambda system: (-system_summary[system]["corrected_mean"], system))
    primary_system = f"evidence_seed_{primary_seed}"
    condition_summary = {
        "comparator": dict(system_summary["comparator"]),
        "evidence_aware": dict(system_summary[primary_system]),
    }
    result = {
        "schema_version": 1,
        "correction_id": CORRECTION_ID,
        "classification": "append_only_measurement_correction",
        "metric": "grade_ndcg_at_4",
        "aggregation": "arithmetic_mean_over_original_question_chunker_rows",
        "by_system": system_summary,
        "by_system_and_chunker": chunker_summary,
        "primary_conditions": condition_summary,
        "system_ordering": {
            "legacy_descending": old_order,
            "corrected_descending": new_order,
            "changed": old_order != new_order,
        },
        "seed_robustness": {
            "legacy_descending": [value for value in old_order if value.startswith("evidence_seed_")],
            "corrected_descending": [value for value in new_order if value.startswith("evidence_seed_")],
        },
        "scope_note": "No candidates, rankings, contexts, checkpoints, comparator selection, generation, F1, or human ratings were changed.",
    }
    result["seed_robustness"]["changed"] = (
        result["seed_robustness"]["legacy_descending"]
        != result["seed_robustness"]["corrected_descending"]
    )
    _validate_public(result)
    return result


def _load_and_verify_row_copies(
    *,
    combined_rows: Sequence[dict[str, Any]],
    row_directory: Path,
    expected_count: int,
    key_fields: Sequence[str],
) -> dict[str, Any]:
    files = sorted(row_directory.rglob("*.json"))
    if len(files) != expected_count:
        raise RuntimeError(f"{row_directory} has {len(files)} row files; expected {expected_count}")
    combined = {
        tuple(str(row[field]) for field in key_fields): row for row in combined_rows
    }
    if len(combined) != expected_count:
        raise RuntimeError("Combined JSONL contains duplicate stable keys")
    for path in files:
        row = dict(read_json(path))
        validate_row_hash(row)
        key = tuple(str(row[field]) for field in key_fields)
        if key not in combined or row != combined[key]:
            raise RuntimeError(f"Individual row does not exactly match combined JSONL: {path}")
    return _directory_record(row_directory)


def _verify_system_ranking_files(
    combined_rows: Sequence[dict[str, Any]],
    ranking_directory: Path,
    *,
    systems: Sequence[str],
    expected_rows_per_system: int,
) -> dict[str, Any]:
    combined = {
        (str(row["question_id"]), str(row["chunker"]), str(row["system"])): row
        for row in combined_rows
    }
    records: dict[str, Any] = {}
    seen: set[tuple[str, str, str]] = set()
    for system in systems:
        path = ranking_directory / f"{system}.jsonl"
        system_rows = [dict(row) for row in iter_jsonl(path)]
        if len(system_rows) != expected_rows_per_system:
            raise RuntimeError(f"{path} has {len(system_rows)} rows; expected {expected_rows_per_system}")
        for row in system_rows:
            validate_row_hash(row)
            key = (str(row["question_id"]), str(row["chunker"]), str(row["system"]))
            if key in seen or key not in combined or combined[key] != row:
                raise RuntimeError(f"System ranking JSONL disagrees with combined retrieval: {key}")
            seen.add(key)
        records[system] = _file_record(path)
    if seen != set(combined):
        raise RuntimeError("Per-system ranking JSONLs do not cover the combined retrieval matrix")
    return records


def _snapshot_inputs(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _file_record(path) for name, path in sorted(paths.items())}


def _unchanged(before: Mapping[str, dict[str, Any]], after: Mapping[str, dict[str, Any]]) -> bool:
    return all(after[name]["sha256"] == record["sha256"] for name, record in before.items())


def _verify_human_artifacts(paths: Mapping[str, Path]) -> dict[str, Any]:
    package_rows = [dict(row) for row in iter_jsonl(paths["blinded_package"])]
    mapping_rows = [dict(row) for row in iter_jsonl(paths["private_mapping"])]
    template_rows = [dict(row) for row in iter_jsonl(paths["ratings_template"])]
    if len(package_rows) != 100 or len(mapping_rows) != 100 or len(template_rows) != 400:
        raise RuntimeError("Human package, mapping, or ratings-template row count is incomplete")
    for row in [*package_rows, *mapping_rows]:
        validate_row_hash(row)
    forbidden_dependency_fields = {"retrieval_metrics", "grade_ndcg_at_4", "candidate_ranking"}
    for row in [*package_rows, *mapping_rows, *template_rows]:
        if forbidden_dependency_fields & set(row):
            raise RuntimeError("Human artifact unexpectedly embeds an NDCG/ranking dependency")
    return {
        "blinded_cases": len(package_rows),
        "private_mappings": len(mapping_rows),
        "rating_rows": len(template_rows),
        "ndcg_or_ranking_fields_present": False,
    }


def _verify_stage_manifests(paths: Mapping[str, Path], freeze_sha256: str) -> dict[str, Any]:
    retrieval = read_json(paths["reserve_retrieval_manifest"])
    generation = read_json(paths["reserve_generation_manifest"])
    human = read_json(paths["human_package_manifest"])
    checks = {
        "reserve_retrieval": (
            retrieval.get("status") == "complete"
            and int(retrieval.get("row_count", -1)) == RESERVE_SIZE * len(CHUNKERS) * len(EXPECTED_SYSTEMS)
            and retrieval.get("output_sha256") == sha256_file(paths["reserve_retrieval"])
            and retrieval.get("freeze_manifest_sha256") == freeze_sha256
        ),
        "reserve_generation": (
            generation.get("status") == "complete"
            and int(generation.get("row_count", -1)) == RESERVE_SIZE * len(CHUNKERS) * 2
            and generation.get("output_sha256") == sha256_file(paths["reserve_generation"])
            and generation.get("retrieval_sha256") == sha256_file(paths["reserve_retrieval"])
            and generation.get("freeze_manifest_sha256") == freeze_sha256
        ),
        "human_package": (
            human.get("status") == "complete"
            and int(human.get("cases", -1)) == 100
            and human.get("package_sha256") == sha256_file(paths["blinded_package"])
            and human.get("private_mapping_sha256") == sha256_file(paths["private_mapping"])
            and human.get("ratings_template_sha256") == sha256_file(paths["ratings_template"])
            and human.get("freeze_manifest_sha256") == freeze_sha256
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Original stage manifest verification failed: {failed}")
    release = read_json(paths["release_manifest"])
    if (
        release.get("analysis", {}).get("sha256") != sha256_file(paths["public_analysis"])
        or release.get("freeze_manifest_sha256") != freeze_sha256
    ):
        raise RuntimeError("Original release manifest does not bind the sealed analysis/freeze")
    return {name: {"passed": passed} for name, passed in checks.items()}


def _scan_report_references(
    workspace_root: Path,
    *,
    public_analysis_path: Path,
) -> dict[str, Any]:
    """Locate public-facing NDCG references without copying manuscript content."""
    pattern = re.compile(r"grade_ndcg_at_4|(?:grade[- ]?)?ndcg\s*@?\s*4", re.IGNORECASE)
    roots = [workspace_root / name for name in ("paper", "reports", "generated")]
    candidates: set[Path] = {public_analysis_path}
    suffixes = {".json", ".md", ".tex", ".txt"}
    for root in roots:
        if root.is_dir():
            candidates.update(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size <= 5_000_000
            )
    matches: list[dict[str, Any]] = []
    for path in sorted(candidates):
        text = path.read_text(encoding="utf-8", errors="replace")
        line_numbers = [
            index
            for index, line in enumerate(text.splitlines(), start=1)
            if pattern.search(line)
        ]
        if not line_numbers:
            continue
        relative = str(path.resolve().relative_to(workspace_root.resolve()))
        if path.resolve() == public_analysis_path.resolve():
            classification = "affected_numeric_summary"
        elif relative == "reports/eaai_phase3_protocol.md":
            classification = "verified_unaffected_validation_definition"
        else:
            classification = "requires_manual_claim_or_table_update"
        matches.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "line_numbers": line_numbers,
                "classification": classification,
            }
        )
    return {
        "roots_scanned": [str(root.relative_to(workspace_root)) for root in roots],
        "eligible_file_count": len(candidates),
        "matches": matches,
        "unresolved_claim_or_table_matches": [
            match["path"]
            for match in matches
            if match["classification"] == "requires_manual_claim_or_table_update"
        ],
    }


def _build_report(
    *,
    aggregates: Mapping[str, Any],
    input_records: Mapping[str, dict[str, Any]],
    output_records: Mapping[str, dict[str, Any]],
    test_results: Mapping[str, Any],
    retrieval_row_count: int,
    generation_row_count: int,
    originals_unchanged: bool,
    unresolved: Sequence[str],
    manifest_path: Path,
) -> str:
    lines = [
        "# EAAI Phase 3 Reserve NDCG Measurement Correction v1",
        "",
        "## Outcome",
        "",
        "This is an append-only measurement correction. It does not rerun or replace the original experiment.",
        f"The freeze gate and immutable-input checks passed: **{str(originals_unchanged).lower()}**.",
        f"Corrected retrieval records: **{retrieval_row_count}**; verified copied generation rows: **{generation_row_count}**.",
        "The original primary token-F1 result, generation, contexts, rankings, model selection, and checkpoints remain unchanged.",
        "",
        "## Metric definition",
        "",
        "- DCG@4 uses the original returned top four grades.",
        "- IDCG@4 uses the best four grades from the complete original 20-candidate common pool.",
        "- Gain is `2^grade - 1`; discount is `log2(rank + 1)`; zero IDCG returns 0.0.",
        "",
        "## Regression tests",
        "",
    ]
    for name, result in test_results["cases"].items():
        lines.append(
            f"- {name}: observed `{result['observed']:.12f}`, expected `{result['expected']:.12f}` — "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
    lines.extend(["", "## Old versus corrected NDCG@4", "", "| System | Records | Legacy mean | Corrected mean | Delta |", "|---|---:|---:|---:|---:|"])
    for system, values in aggregates["by_system"].items():
        lines.append(
            f"| {system} | {values['record_count']} | {values['legacy_mean']:.10f} | "
            f"{values['corrected_mean']:.10f} | {values['delta']:+.10f} |"
        )
    ordering = aggregates["system_ordering"]
    seeds = aggregates["seed_robustness"]
    lines.extend(
        [
            "",
            "## Impact boundary",
            "",
            f"- System ordering changed: **{str(ordering['changed']).lower()}**.",
            f"- Retrieval-seed ordering changed: **{str(seeds['changed']).lower()}**.",
            "- The stored NDCG fields in original ranking and generation rows were not overwritten; the new values live only in the correction namespace.",
            "- No model, checkpoint, comparator, candidate, rank, context, answer, F1 value, human package, or human rating was changed or reselected.",
            "",
            "## Hash verification",
            "",
        ]
    )
    for name, record in input_records.items():
        lines.append(f"- `{name}`: `{record['sha256']}`")
    lines.extend(["", "## New artifacts", ""])
    for name, record in output_records.items():
        lines.append(f"- `{name}`: `{record['path']}` — `{record['sha256']}`")
    lines.append(f"- `manifest`: `{manifest_path.resolve()}` — identity and file hashes are embedded/reported by the command")
    lines.extend(["", "## Unresolved issues", ""])
    if unresolved:
        lines.extend(f"- {issue}" for issue in unresolved)
    else:
        lines.append("- None within the approved correction scope.")
    return "\n".join(lines) + "\n"


def run_correction(
    *,
    workspace_root: Path,
    source_root: Path,
    output_root: Path,
    release_archive: Path,
    expected_release_archive_sha256: str,
    expected_release_commit: str,
    expected_freeze_manifest_sha256: str,
    expected_retrieval_sha256: str,
    expected_generation_sha256: str,
    expected_private_analysis_sha256: str,
    expected_public_analysis_sha256: str,
    expected_release_manifest_sha256: str,
    expected_blinded_package_sha256: str,
    expected_private_mapping_sha256: str,
    expected_ratings_template_sha256: str,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    expected_release_commit = _require_git_commit(expected_release_commit)

    run_id = "techqa_evidence_reranker_v1"
    run_artifacts = workspace_root / "artifacts" / "eaai_phase3" / run_id
    run_results = workspace_root / "results" / "eaai_phase3" / run_id
    protected_roots = (workspace_root, run_artifacts.resolve(), run_results.resolve())
    if output_root == workspace_root or any(root == output_root or root in output_root.parents for root in protected_roots[1:]):
        raise ValueError("Correction output must be a distinct versioned namespace outside original artifacts/results")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing correction namespace: {output_root}")
    paths = {
        "freeze": run_artifacts / "manifests" / "freeze.json",
        "reserve_retrieval": run_artifacts / "reserve" / "retrieval.jsonl",
        "reserve_generation": run_artifacts / "reserve" / "generation.jsonl",
        "private_analysis": run_artifacts / "analysis" / "full_analysis.json",
        "public_analysis": run_results / "analysis_summary.json",
        "release_manifest": run_results / "release_manifest.json",
        "blinded_package": run_artifacts / "human" / "blinded_package.jsonl",
        "private_mapping": run_artifacts / "human" / "private_mapping.jsonl",
        "ratings_template": run_artifacts / "human" / "ratings_template.jsonl",
        "reserve_retrieval_manifest": run_artifacts / "manifests" / "reserve_retrieval.json",
        "reserve_generation_manifest": run_artifacts / "manifests" / "reserve_generation.json",
        "human_package_manifest": run_artifacts / "manifests" / "human_package.json",
    }

    # Fail closed before opening any protected reserve row content.
    freeze = verify_freeze_manifest(paths["freeze"])
    if freeze.get("manifest_sha256") != _require_sha256(
        expected_freeze_manifest_sha256,
        name="expected freeze manifest SHA-256",
    ):
        raise RuntimeError("Verified freeze identity is not the approved Phase 3 freeze")
    if freeze.get("partition_sha256") != PARTITION_SHA256:
        raise RuntimeError("Verified freeze does not identify the original reserve partition")
    if freeze.get("selected_comparator") != "hybrid_rrf":
        raise RuntimeError("Verified freeze comparator differs from the original selection")
    if tuple(sorted(int(seed) for seed in freeze.get("checkpoints", {}))) != tuple(RERANKER_SEEDS):
        raise RuntimeError("Verified freeze checkpoint seed set differs from the original selection")

    expected_hashes = {
        "reserve_retrieval": expected_retrieval_sha256,
        "reserve_generation": expected_generation_sha256,
        "private_analysis": expected_private_analysis_sha256,
        "public_analysis": expected_public_analysis_sha256,
        "release_manifest": expected_release_manifest_sha256,
        "blinded_package": expected_blinded_package_sha256,
        "private_mapping": expected_private_mapping_sha256,
        "ratings_template": expected_ratings_template_sha256,
    }
    release_record = _check_file_hash(
        release_archive,
        expected_release_archive_sha256,
        name="35d21a8 release archive",
    )
    release_comparison = verify_release_scientific_code(
        release_archive=release_archive,
        freeze=freeze,
        workspace_root=workspace_root,
    )
    input_records = {"release_archive_35d21a8": release_record, "freeze": _file_record(paths["freeze"])}
    for name, expected in expected_hashes.items():
        input_records[name] = _check_file_hash(paths[name], expected, name=name)
    for name in (
        "reserve_retrieval_manifest",
        "reserve_generation_manifest",
        "human_package_manifest",
    ):
        input_records[name] = _file_record(paths[name])
    stage_manifest_checks = _verify_stage_manifests(paths, str(freeze["manifest_sha256"]))
    human_artifact_checks = _verify_human_artifacts(paths)
    before = _snapshot_inputs(paths)

    retrieval_rows = [dict(row) for row in iter_jsonl(paths["reserve_retrieval"])]
    corrections, retrieval_index = build_per_record_corrections(retrieval_rows)
    ranking_files = _verify_system_ranking_files(
        retrieval_rows,
        run_artifacts / "reserve" / "rankings",
        systems=EXPECTED_SYSTEMS,
        expected_rows_per_system=RESERVE_SIZE * len(CHUNKERS),
    )
    ranking_rows_record = _load_and_verify_row_copies(
        combined_rows=retrieval_rows,
        row_directory=run_artifacts / "reserve" / "ranking_rows",
        expected_count=RESERVE_SIZE * len(CHUNKERS) * len(EXPECTED_SYSTEMS),
        key_fields=("question_id", "chunker", "system"),
    )

    generation_rows = [dict(row) for row in iter_jsonl(paths["reserve_generation"])]
    generation_verification = verify_generation_copies(generation_rows, retrieval_index)
    corrections = attach_generation_hashes(
        corrections,
        generation_verification.pop("row_hash_by_retrieval_key"),
    )
    generation_rows_record = _load_and_verify_row_copies(
        combined_rows=generation_rows,
        row_directory=run_artifacts / "reserve" / "generation_rows",
        expected_count=RESERVE_SIZE * len(CHUNKERS) * 2,
        key_fields=("question_id", "chunker", "condition"),
    )
    aggregates = aggregate_corrections(corrections)
    tests = metric_self_tests()

    private_analysis = read_json(paths["private_analysis"])
    public_analysis = read_json(paths["public_analysis"])
    old_robustness = private_analysis.get("retrieval_seed_robustness")
    if not isinstance(old_robustness, dict):
        raise RuntimeError("Original analysis lacks retrieval_seed_robustness")
    for system, values in aggregates["by_system"].items():
        stored = float(old_robustness[system]["grade_ndcg_at_4"])
        if not math.isclose(stored, float(values["legacy_mean"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Original analysis NDCG aggregate is not reproducible for {system}")
    public_robustness = public_analysis.get("retrieval_seed_robustness")
    if not isinstance(public_robustness, dict):
        raise RuntimeError("Public original analysis lacks retrieval_seed_robustness")
    for system, values in aggregates["by_system"].items():
        stored = float(public_robustness[system]["grade_ndcg_at_4"])
        if not math.isclose(stored, float(values["legacy_mean"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Public original NDCG aggregate is not reproducible for {system}")

    # Compare public/private primary after removing the intentionally private question-level rows.
    private_primary_public = dict(private_analysis.get("primary", {}))
    private_primary_public.pop("question_level", None)
    if private_primary_public != public_analysis.get("primary"):
        raise RuntimeError("Public and private original primary analyses disagree")

    primary = dict(public_analysis.get("primary", {}))
    primary_snapshot = {
        "endpoint": primary.get("endpoint"),
        "condition_means": primary.get("condition_means"),
        "estimate": primary.get("estimate"),
        "confirmatory_success": primary.get("confirmatory_success"),
    }
    report_reference_scan = _scan_report_references(
        workspace_root,
        public_analysis_path=paths["public_analysis"],
    )
    unresolved = [
        f"Unclassified public NDCG reference requires manual review: {path}"
        for path in report_reference_scan["unresolved_claim_or_table_matches"]
    ]
    ledger = {
        "schema_version": 1,
        "correction_id": CORRECTION_ID,
        "affected": [
            {
                "artifact_class": "combined_reserve_ranking_rows",
                "path": str(paths["reserve_retrieval"].resolve()),
                "record_count": len(corrections),
                "field": "retrieval_metrics.grade_ndcg_at_4",
                "action": "original preserved; corrected value appended by stable join key",
            },
            {
                "artifact_class": "per_system_reserve_ranking_files",
                "paths": [record["path"] for record in ranking_files.values()],
                "record_count": len(corrections),
                "field": "retrieval_metrics.grade_ndcg_at_4",
                "action": "original preserved; exact copies verified against combined retrieval",
            },
            {
                "artifact_class": "individual_reserve_ranking_rows",
                "path": ranking_rows_record["path"],
                "record_count": len(corrections),
                "field": "retrieval_metrics.grade_ndcg_at_4",
                "action": "original preserved; exact copies verified against combined retrieval",
            },
            {
                "artifact_class": "combined_reserve_generation_rows",
                "path": str(paths["reserve_generation"].resolve()),
                "record_count": generation_verification["verified_copy_count"],
                "field": "retrieval_metrics.grade_ndcg_at_4 copied from primary/comparator ranking",
                "action": "original preserved; dependency verified; no regeneration",
            },
            {
                "artifact_class": "individual_reserve_generation_rows",
                "path": generation_rows_record["path"],
                "record_count": generation_verification["verified_copy_count"],
                "field": "retrieval_metrics.grade_ndcg_at_4 copied from primary/comparator ranking",
                "action": "original preserved; exact copies verified; no regeneration",
            },
            {
                "artifact_class": "private_analysis",
                "path": str(paths["private_analysis"].resolve()),
                "record_count": len(aggregates["by_system"]),
                "field": "retrieval_seed_robustness.*.grade_ndcg_at_4",
                "action": "original preserved; corrected summaries emitted in public aggregate addendum",
            },
            {
                "artifact_class": "public_analysis",
                "path": str(paths["public_analysis"].resolve()),
                "record_count": len(aggregates["by_system"]),
                "field": "retrieval_seed_robustness.*.grade_ndcg_at_4",
                "action": "corrected summaries emitted in public aggregate addendum",
            },
            {
                "artifact_class": "release_manifest_analysis_binding",
                "path": str(paths["release_manifest"].resolve()),
                "record_count": 1,
                "field": "analysis.sha256",
                "action": "original release preserved; future release must bind this correction addendum",
            },
        ],
        "verified_unaffected": [
            "candidate membership and ordering",
            "top-four contexts",
            "answer_visibility_at_4",
            "gold_document_coverage_at_4",
            "maximum_grade_at_4",
            "retrieval and generation timings",
            "generated answers, exact match, and token F1",
            "primary paired F1 test and confirmatory decision",
            "comparator selection and hyperparameter/checkpoint selection",
            "three checkpoint directories and model revisions",
            "human package, mapping, and ratings template",
        ],
        "report_claim_and_table_scan": report_reference_scan,
        "original_primary_f1_snapshot": primary_snapshot,
        "ordering_impact": aggregates["system_ordering"],
        "seed_robustness_impact": aggregates["seed_robustness"],
    }
    _validate_public(ledger)

    after = _snapshot_inputs(paths)
    after_ranking_rows_record = _directory_record(run_artifacts / "reserve" / "ranking_rows")
    after_generation_rows_record = _directory_record(run_artifacts / "reserve" / "generation_rows")
    after_ranking_files = {
        system: _file_record(run_artifacts / "reserve" / "rankings" / f"{system}.jsonl")
        for system in EXPECTED_SYSTEMS
    }
    post_freeze = verify_freeze_manifest(paths["freeze"])
    originals_unchanged = (
        _unchanged(before, after)
        and after_ranking_rows_record["sha256"] == ranking_rows_record["sha256"]
        and after_generation_rows_record["sha256"] == generation_rows_record["sha256"]
        and all(
            after_ranking_files[system]["sha256"] == ranking_files[system]["sha256"]
            for system in EXPECTED_SYSTEMS
        )
        and post_freeze["manifest_sha256"] == freeze["manifest_sha256"]
    )
    if not originals_unchanged:
        raise RuntimeError("An original experiment input changed during correction processing")

    source_paths = {
        "evidence_metric": source_root / "src" / "chunkrag" / "eaai_phase3" / "evidence.py",
        "corrected_reserve_workflow": source_root / "src" / "chunkrag" / "eaai_phase3" / "reserve_workflows.py",
        "correction_module": source_root / "src" / "chunkrag" / "eaai_phase3" / "ndcg_correction.py",
        "correction_cli": source_root / "scripts" / "correct_eaai_phase3_reserve_ndcg.py",
        "phase3_regression_tests": source_root / "tests" / "test_eaai_phase3.py",
        "correction_tests": source_root / "tests" / "test_eaai_phase3_ndcg_correction.py",
        "correction_design": source_root / "docs" / "superpowers" / "specs" / "2026-09-05-eaai-phase3-ndcg-correction-v1-design.md",
        "correction_plan": source_root / "docs" / "superpowers" / "plans" / "2026-09-05-eaai-phase3-ndcg-correction-v1.md",
        "correction_patch": source_root / "patches" / "reserve_ndcg_common_pool_v1.patch",
    }
    source_records = {name: _file_record(path) for name, path in source_paths.items()}

    output_root.mkdir(parents=True)
    private_path = output_root / "private" / "per_record_ndcg.jsonl"
    aggregates_path = output_root / "public" / "corrected_aggregates.json"
    ledger_path = output_root / "public" / "affected_output_ledger.json"
    write_immutable_jsonl(private_path, corrections)
    write_immutable_json(aggregates_path, aggregates)
    write_immutable_json(ledger_path, ledger)
    preliminary_output_records = {
        "private_per_record_ndcg": _file_record(private_path),
        "public_corrected_aggregates": _file_record(aggregates_path),
        "public_affected_output_ledger": _file_record(ledger_path),
    }
    report_path = output_root / "correction_report.md"
    report = _build_report(
        aggregates=aggregates,
        input_records=input_records,
        output_records=preliminary_output_records,
        test_results=tests,
        retrieval_row_count=len(corrections),
        generation_row_count=generation_verification["row_count"],
        originals_unchanged=originals_unchanged,
        unresolved=unresolved,
        manifest_path=output_root / "manifest.json",
    )
    from chunkrag.eaai_phase2.io import write_immutable_bytes

    write_immutable_bytes(report_path, report.encode("utf-8"))
    output_records = {**preliminary_output_records, "correction_report": _file_record(report_path)}
    manifest_payload: dict[str, Any] = {
        "schema_version": 1,
        "correction_id": CORRECTION_ID,
        "classification": "append_only_measurement_correction",
        "original_release": {
            "commit": expected_release_commit,
            "archive": release_record,
            "frozen_scientific_code_comparison": release_comparison,
            "freeze_manifest_sha256": freeze["manifest_sha256"],
            "partition_sha256": freeze["partition_sha256"],
            "selected_comparator": freeze["selected_comparator"],
            "checkpoint_seeds": sorted(int(seed) for seed in freeze["checkpoints"]),
            "model_revisions": freeze["model_revisions"],
        },
        "metric_definitions": {
            "legacy": "DCG@4 and IDCG@4 both derived from the returned top four grades",
            "corrected": "DCG@4 from returned top four; IDCG@4 from full original common candidate pool",
            "gain": "2^grade - 1",
            "discount": "log2(rank + 1), with one-based rank",
            "zero_idcg_convention": 0.0,
        },
        "record_counts": {
            "reserve_rankings": len(corrections),
            "reserve_generation_copies_verified": generation_verification["verified_copy_count"],
            "systems": len(aggregates["by_system"]),
            "candidate_pool_size": 20,
            "top_k": 4,
        },
        "input_files": input_records,
        "input_directories": {
            "ranking_rows": ranking_rows_record,
            "generation_rows": generation_rows_record,
        },
        "system_ranking_files": ranking_files,
        "correction_source": source_records,
        "tests": tests,
        "stage_manifest_checks": stage_manifest_checks,
        "human_artifact_checks": human_artifact_checks,
        "outputs": output_records,
        "impact_boundary": {
            "originals_unchanged": originals_unchanged,
            "changed_fields": ["corrected_grade_ndcg_at_4", "dependent corrected NDCG summaries"],
            "primary_f1_reinterpreted": False,
            "models_reselected": False,
            "experiments_rerun": False,
        },
        "unresolved_issues": unresolved,
    }
    manifest_payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest_payload))
    manifest_path = output_root / "manifest.json"
    write_immutable_json(manifest_path, manifest_payload)
    return {
        "status": "complete",
        "correction_id": CORRECTION_ID,
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest_payload["manifest_sha256"],
        "report": str(report_path),
        "aggregates": aggregates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append-only correction of Phase 3 reserve NDCG@4")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--expected-release-archive-sha256", required=True)
    parser.add_argument("--expected-release-commit", required=True)
    parser.add_argument("--expected-freeze-manifest-sha256", required=True)
    parser.add_argument("--expected-retrieval-sha256", required=True)
    parser.add_argument("--expected-generation-sha256", required=True)
    parser.add_argument("--expected-private-analysis-sha256", required=True)
    parser.add_argument("--expected-public-analysis-sha256", required=True)
    parser.add_argument("--expected-release-manifest-sha256", required=True)
    parser.add_argument("--expected-blinded-package-sha256", required=True)
    parser.add_argument("--expected-private-mapping-sha256", required=True)
    parser.add_argument("--expected-ratings-template-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_correction(**vars(arguments))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
