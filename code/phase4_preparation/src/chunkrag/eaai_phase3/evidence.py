from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

from chunkrag.eaai_phase3.constants import CHUNKERS
from chunkrag.eaai_phase2.io import add_row_hash
from chunkrag.eaai_phase3.protocol import assert_development_only_ids
from chunkrag.text_utils import contains_normalized_answer, normalize_answer


def grade_candidate(
    *,
    text: str,
    document_id: str,
    reference_answers: Sequence[str],
    relevant_document_ids: Sequence[str],
) -> int:
    if contains_normalized_answer(text, list(reference_answers)):
        return 2
    if document_id in set(relevant_document_ids):
        return 1
    return 0


def label_retrieval_rows(
    retrieval_rows: Iterable[dict[str, Any]],
    chunk_text_by_id: dict[str, str],
    *,
    allowed_question_ids: set[str],
) -> list[dict[str, Any]]:
    """Construct the private three-grade evidence table from exposed retrieval rows."""
    source_rows = [dict(row) for row in retrieval_rows]
    assert_development_only_ids(
        (str(row["question_id"]) for row in source_rows),
        allowed_question_ids,
    )
    labeled: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sorted(
        source_rows,
        key=lambda row: (
            str(row["question_id"]),
            CHUNKERS.index(str(row["chunker"])),
        ),
    ):
        question_id = str(source["question_id"])
        chunker = str(source["chunker"])
        candidates = source.get("fused_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"No fused candidates for {question_id}/{chunker}")
        for candidate in sorted(
            candidates,
            key=lambda row: (int(row["rank"]), str(row["chunk_id"])),
        ):
            chunk_id = str(candidate["chunk_id"])
            key = (question_id, chunker, chunk_id)
            if key in seen:
                raise ValueError(f"Duplicate candidate in exposed retrieval rows: {key}")
            seen.add(key)
            try:
                text = str(chunk_text_by_id[chunk_id])
            except KeyError as error:
                raise KeyError(f"Missing reconstructed text for candidate {chunk_id}") from error
            payload = {
                "schema_version": 1,
                "study_stage": "eaai_phase3_evidence_label",
                "source_split": source.get("split"),
                "question_id": question_id,
                "question": str(source["question"]),
                "chunker": chunker,
                "chunk_id": chunk_id,
                "document_id": str(candidate["document_id"]),
                "text": text,
                "fused_rank": int(candidate["rank"]),
                "fused_score": float(candidate["score"]),
                "reference_answers": [str(value) for value in source["reference_answers"]],
                "relevant_document_ids": [
                    str(value) for value in source["relevant_document_ids"]
                ],
                "grade": grade_candidate(
                    text=text,
                    document_id=str(candidate["document_id"]),
                    reference_answers=[str(value) for value in source["reference_answers"]],
                    relevant_document_ids=[
                        str(value) for value in source["relevant_document_ids"]
                    ],
                ),
            }
            labeled.append(add_row_hash(payload))
    return labeled


def _candidate_sort_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    chunker = str(row.get("chunker", ""))
    try:
        chunker_index = CHUNKERS.index(chunker)
    except ValueError:
        chunker_index = len(CHUNKERS)
    return (
        str(row["question_id"]),
        int(row["fused_rank"]),
        chunker_index,
        str(row["chunk_id"]),
    )


def deduplicate_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in (dict(value) for value in rows):
        question_id = str(row["question_id"])
        normalized = normalize_answer(str(row["text"]))
        if not normalized:
            # Keep normalization-empty candidates in the immutable evidence-label
            # table for accounting, but exclude them from listwise supervision:
            # they contain no rankable token evidence and all collapse to the
            # same empty deduplication key.
            continue
        key = (question_id, normalized)
        current = best.get(key)
        selection_key = (
            -int(row.get("grade", 0)),
            int(row["fused_rank"]),
            CHUNKERS.index(str(row["chunker"])) if str(row.get("chunker")) in CHUNKERS else len(CHUNKERS),
            str(row["chunk_id"]),
        )
        if current is None:
            best[key] = row
            continue
        current_key = (
            -int(current.get("grade", 0)),
            int(current["fused_rank"]),
            CHUNKERS.index(str(current["chunker"])) if str(current.get("chunker")) in CHUNKERS else len(CHUNKERS),
            str(current["chunk_id"]),
        )
        if selection_key < current_key:
            best[key] = row
    return sorted(best.values(), key=_candidate_sort_key)


def build_listwise_groups(
    rows: Iterable[dict[str, Any]],
    *,
    max_negatives: int = 7,
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive")
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grade = int(row["grade"])
        if grade not in {0, 1, 2}:
            raise ValueError(f"Unsupported evidence grade: {grade}")
        by_question[str(row["question_id"])].append(dict(row))

    groups: list[dict[str, Any]] = []
    for question_id in sorted(by_question):
        candidates = sorted(
            by_question[question_id],
            key=lambda row: (int(row["fused_rank"]), str(row["chunk_id"])),
        )
        question_values = {str(row["question"]) for row in candidates}
        if len(question_values) != 1:
            raise ValueError(f"Question text mismatch within {question_id}")
        top_grade = max(int(row["grade"]) for row in candidates)
        if top_grade == 0:
            continue
        positives = [row for row in candidates if int(row["grade"]) == top_grade]
        negatives = [row for row in candidates if int(row["grade"]) < top_grade]
        if not negatives:
            continue
        negatives = negatives[:max_negatives]
        for positive in positives:
            messages: list[dict[str, str]] = []
            if instruction:
                messages.append({"role": "system", "content": instruction.strip()})
            messages.append({"role": "user", "content": str(positive["question"])})
            groups.append(
                {
                    "question_id": question_id,
                    "positive_grade": top_grade,
                    "positive_chunk_id": str(positive["chunk_id"]),
                    "negative_chunk_ids": [str(row["chunk_id"]) for row in negatives],
                    "messages": messages,
                    "positive_messages": [
                        [{"role": "assistant", "content": str(positive["text"])}]
                    ],
                    "negative_messages": [
                        [{"role": "assistant", "content": str(row["text"])}]
                        for row in negatives
                    ],
                }
            )
    return groups


def ndcg_at_k(grades: Sequence[int | float], *, k: int = 4) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    observed = [float(value) for value in grades[:k]]
    if any(not math.isfinite(value) or value < 0.0 for value in observed):
        raise ValueError("Grades must be finite and non-negative")

    def dcg(values: Sequence[float]) -> float:
        return sum((2.0**value - 1.0) / math.log2(index + 2.0) for index, value in enumerate(values))

    ideal = sorted((float(value) for value in grades), reverse=True)[:k]
    denominator = dcg(ideal)
    if denominator == 0.0:
        return 0.0
    return dcg(observed) / denominator
