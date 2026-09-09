"""Leakage-safe Doc2Dial conversion and common generation contracts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


RETRIEVAL_CANDIDATES = 20
FINAL_CONTEXTS = 4
MAX_INPUT_TOKENS = 1536
MAX_OUTPUT_TOKENS = 512
TARGET_SOURCE_FIELD = "doc2dial_dialogue_agent_utterance"


def dialogue_query(history: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(f"{str(turn['role']).upper()}: {str(turn['text'])}" for turn in history)


def convert_dialogue_case(
    turns: Sequence[Mapping[str, str]],
    *,
    selected_user_index: int,
    grounding_annotations: Sequence[Mapping[str, Any]],
    gold_document_metadata: Mapping[str, Any],
    doc2dial_rc_answer: str | None = None,
) -> dict[str, Any]:
    """Convert a dialogue transition while separating inference and evaluation fields."""

    if selected_user_index < 0 or selected_user_index >= len(turns) - 1:
        raise ValueError("The selected user turn must have a following turn")
    current = turns[selected_user_index]
    target = turns[selected_user_index + 1]
    if str(current["role"]).lower() != "user" or str(target["role"]).lower() != "agent":
        raise ValueError("A case must select a user turn followed by an agent turn")
    history = [dict(turn) for turn in turns[: selected_user_index + 1]]
    inference = {
        "dialogue_history": history,
        "retrieval_query": dialogue_query(history),
    }
    evaluation = {
        "target_agent_response": str(target["text"]),
        "target_source_field": TARGET_SOURCE_FIELD,
        "grounding_annotations": [dict(item) for item in grounding_annotations],
        "gold_document_metadata": dict(gold_document_metadata),
        "doc2dial_rc_answer": doc2dial_rc_answer,
    }
    forbidden = {
        "target_agent_response",
        "grounding_annotations",
        "gold_document_metadata",
        "doc2dial_rc_answer",
    }
    if forbidden.intersection(inference):
        raise AssertionError("Evaluation-only fields leaked into inference")
    return {"inference": inference, "evaluation": evaluation}


def inference_payload(case: Mapping[str, Any], retrieved_context: str) -> dict[str, Any]:
    allowed = case["inference"]
    return {
        "dialogue_history": [dict(turn) for turn in allowed["dialogue_history"]],
        "retrieval_query": str(allowed["retrieval_query"]),
        "retrieved_context": str(retrieved_context),
    }


def generation_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded question answering assistant. "
                "Use only the provided context. Give a concise but complete answer "
                "containing the information needed to resolve the question. "
                "Do not explain your reasoning or add citations. If the answer is not "
                "fully supported, reply with exactly 'unanswerable'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Answer the following question using only the context.\n\n"
                f"Question: {question}\n\n"
                "Context passages:\n"
                f"{context}\n\n"
                "Return only the final answer."
            ),
        },
    ]


def generation_prompt_skeleton(question: str) -> str:
    return json.dumps(generation_messages(question, "<RETRIEVED_CONTEXT>"), sort_keys=True)


def prompt_contract_hash(question: str) -> str:
    return hashlib.sha256(generation_prompt_skeleton(question).encode("utf-8")).hexdigest()


def family_key(domain: str, document_title: str) -> str:
    """Outcome-independent, domain-scoped title-family definition."""

    normalized = re.sub(r"\s+#\d+$", "", str(document_title).strip().lower())
    normalized = " ".join(normalized.split())
    return f"{str(domain).strip().lower()}::{normalized}"


def candidate_pool_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(
        (
            int(row["fused_rank"]),
            str(row["chunk_id"]),
            str(row.get("doc_id", "")),
            hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
        )
        for row in rows
    )
    blob = json.dumps(identities, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
