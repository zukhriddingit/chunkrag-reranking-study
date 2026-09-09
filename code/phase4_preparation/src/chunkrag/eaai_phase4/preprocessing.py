"""Approved, label-blind Phase 4 pair preprocessing.

The scientific policy is deliberately narrow:

* choose duplicate representatives without consulting B/E labels;
* render the exact Qwen3 reranker training prompt;
* measure its full tokenized length, including the literal chat/control tokens;
* prune overlength pairs individually; and
* retain a group only when its positive and at least one negative survive.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "label_blind_dedup_pair_prune_v1"

QWEN_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_answer(text: str) -> str:
    """Phase 3 answer normalization retained verbatim in behavior."""

    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalize_chunk_text(text: str) -> str:
    return normalize_answer(text)


def _label_blind_representative_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ordering key that is structurally incapable of consulting a grade."""

    return (
        int(row.get("fused_rank", 10**12)),
        str(row.get("chunker", "")),
        str(row.get("chunk_id", "")),
        str(row.get("doc_id", "")),
        sha256_json(
            {
                "qid": str(row.get("qid", "")),
                "chunk_id": str(row.get("chunk_id", "")),
                "text": str(row.get("text", "")),
            }
        ),
    )


def label_blind_deduplicate(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate normalized chunk text within question without using B/E labels."""

    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["qid"]), normalize_chunk_text(str(row.get("text", ""))))
        buckets.setdefault(key, []).append(row)
    selected = [dict(min(bucket, key=_label_blind_representative_key)) for bucket in buckets.values()]
    return sorted(selected, key=lambda row: (str(row["qid"]), _label_blind_representative_key(row)))


def training_pair_text(question: str, document: str, instruction: str) -> str:
    body = f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {document}"
    return QWEN_PREFIX + body + QWEN_SUFFIX


def training_pair_token_ids(
    tokenizer: Any,
    question: str,
    document: str,
    instruction: str,
) -> list[int]:
    text = training_pair_text(question, document, instruction)
    return list(tokenizer.encode(text, add_special_tokens=False))


def training_pair_token_count(
    tokenizer: Any,
    question: str,
    document: str,
    instruction: str,
) -> int:
    return len(training_pair_token_ids(tokenizer, question, document, instruction))


@dataclass(frozen=True)
class PrunedGroup:
    qid: str
    positive: dict[str, Any]
    negatives: tuple[dict[str, Any], ...]
    pair_token_counts: tuple[int, ...]

    @property
    def size(self) -> int:
        return 1 + len(self.negatives)

    def pairs(self) -> tuple[dict[str, Any], ...]:
        return (self.positive,) + self.negatives


@dataclass(frozen=True)
class PruneDecision:
    status: str
    reason: str | None
    group: PrunedGroup | None
    positive_length: int
    negative_lengths: tuple[int, ...]


def prune_group_pairs(
    group: Mapping[str, Any],
    tokenizer: Any,
    *,
    instruction: str,
    max_length: int = 512,
) -> PruneDecision:
    """Prune pairs independently and preserve positive-at-index-zero grouping."""

    positive = dict(group["positive"])
    negatives = tuple(dict(row) for row in group.get("negatives", []))
    question = str(group["question"])
    positive_length = training_pair_token_count(
        tokenizer, question, str(positive["text"]), instruction
    )
    negative_lengths = tuple(
        training_pair_token_count(tokenizer, question, str(row["text"]), instruction)
        for row in negatives
    )
    if positive_length > max_length:
        return PruneDecision(
            status="excluded",
            reason="overlength_positive_pair",
            group=None,
            positive_length=positive_length,
            negative_lengths=negative_lengths,
        )

    kept = tuple(row for row, length in zip(negatives, negative_lengths) if length <= max_length)
    kept_lengths = tuple(length for length in negative_lengths if length <= max_length)
    if not kept:
        return PruneDecision(
            status="excluded",
            reason="no_surviving_negative_pair",
            group=None,
            positive_length=positive_length,
            negative_lengths=negative_lengths,
        )

    return PruneDecision(
        status="retained",
        reason=None,
        group=PrunedGroup(
            qid=str(group["qid"]),
            positive=positive,
            negatives=kept,
            pair_token_counts=(positive_length,) + kept_lengths,
        ),
        positive_length=positive_length,
        negative_lengths=negative_lengths,
    )


def group_sequence(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the positive first, followed by the surviving negatives."""

    return [group["positive"], *list(group.get("negatives", []))]


def assert_positive_indexing(group: Mapping[str, Any]) -> None:
    sequence = group_sequence(group)
    if len(sequence) < 2:
        raise ValueError("A listwise group must contain one positive and at least one negative")
    if int(sequence[0].get("binary_label", 1)) != 1:
        raise ValueError("Positive pair is not at group index zero")
    if any(int(row.get("binary_label", 0)) != 0 for row in sequence[1:]):
        raise ValueError("A negative position contains a positive label")


def dedup_identity(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return [(str(row["qid"]), str(row["chunk_id"])) for row in label_blind_deduplicate(rows)]
