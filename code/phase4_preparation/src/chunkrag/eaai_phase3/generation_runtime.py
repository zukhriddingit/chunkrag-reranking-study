from __future__ import annotations

import time
import re
from typing import Any


def build_phase3_qa_messages(question: str, context: str) -> list[dict[str, str]]:
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


def _normalize_complete_response(text: str) -> str:
    cleaned = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    cleaned = re.sub(r"^\s*\[\d+\]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*the answer is\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'`").strip()
    if cleaned.lower() in {"unanswerable", "not answerable", "not supported"}:
        return "unanswerable"
    return cleaned


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    return len(tokens)


def pack_generation_messages(
    tokenizer: Any,
    *,
    question: str,
    context: str,
    answer_style: str,
    max_input_tokens: int,
) -> dict[str, Any]:
    if answer_style != "complete":
        raise ValueError("Phase 3 generation supports only the frozen complete-answer style")
    full_messages = build_phase3_qa_messages(question, context)
    full_tokens = _chat_token_count(tokenizer, full_messages)
    if full_tokens <= max_input_tokens:
        return {
            "messages": full_messages,
            "packed_context": context,
            "full_prompt_tokens": full_tokens,
            "used_prompt_tokens": full_tokens,
            "context_truncated": False,
        }

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    lo, hi = 0, len(context_ids)
    best_messages: list[dict[str, str]] | None = None
    best_context = ""
    best_count: int | None = None
    while lo <= hi:
        midpoint = (lo + hi) // 2
        candidate_context = tokenizer.decode(
            context_ids[:midpoint],
            skip_special_tokens=True,
        ).strip()
        candidate_messages = build_phase3_qa_messages(question, candidate_context)
        count = _chat_token_count(tokenizer, candidate_messages)
        if count <= max_input_tokens:
            best_messages = candidate_messages
            best_context = candidate_context
            best_count = count
            lo = midpoint + 1
        else:
            hi = midpoint - 1
    if best_messages is None or best_count is None:
        raise RuntimeError("Question and fixed prompt exceed the frozen generation input budget")
    return {
        "messages": best_messages,
        "packed_context": best_context,
        "full_prompt_tokens": full_tokens,
        "used_prompt_tokens": best_count,
        "context_truncated": True,
    }


class FrozenQwen35Generator:
    """OpenAI-compatible client enforcing the frozen Qwen3.5 decoding contract."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        base_url: str,
        api_key: str,
        max_input_tokens: int = 1536,
        max_new_tokens: int = 512,
    ) -> None:
        try:
            from openai import OpenAI
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - GPU endpoint dependency
            raise ImportError("Install the Phase 3 GPU requirements for reserve generation") from error
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, *, question: str, context: str) -> dict[str, Any]:
        packed = pack_generation_messages(
            self.tokenizer,
            question=question,
            context=context,
            answer_style="complete",
            max_input_tokens=self.max_input_tokens,
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=packed["messages"],
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "seed": 20260904,
            },
        )
        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        raw = choice.message.content or ""
        normalized = _normalize_complete_response(raw)
        usage = getattr(response, "usage", None)
        return {
            "raw_output": raw,
            "normalized_output": normalized,
            "packed_context": packed["packed_context"],
            "full_prompt_tokens": packed["full_prompt_tokens"],
            "used_prompt_tokens": packed["used_prompt_tokens"],
            "context_truncated": packed["context_truncated"],
            "generated_tokens": (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            ),
            "finish_reason": getattr(choice, "finish_reason", None),
            "latency_seconds": elapsed,
            "tokens_per_second": (
                getattr(usage, "completion_tokens", 0) / elapsed
                if usage is not None and elapsed and getattr(usage, "completion_tokens", None)
                else None
            ),
            "thinking_enabled": False,
            "temperature": 0.0,
            "max_new_tokens": self.max_new_tokens,
        }
