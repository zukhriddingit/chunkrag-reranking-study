from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence


QWEN_RERANKER_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
QWEN_RERANKER_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def format_qwen_reranker_input(question: str, document: str, instruction: str) -> str:
    return (
        f"<Instruct>: {instruction.strip()}\n"
        f"<Query>: {question.strip()}\n"
        f"<Document>: {document.strip()}"
    )


def rank_scored_candidates(
    candidates: Sequence[dict[str, Any]],
    scores: Sequence[float],
) -> list[dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError("Candidate and score counts differ")
    scored: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores, strict=True):
        row = dict(candidate)
        row["fused_rank"] = int(candidate.get("fused_rank", candidate["rank"]))
        row["reranker_score"] = float(score)
        scored.append(row)
    ordered = sorted(
        scored,
        key=lambda row: (
            -float(row["reranker_score"]),
            int(row["fused_rank"]),
            str(row["chunk_id"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


class Qwen3Reranker:
    """Official yes/no-logit Qwen3 reranker with an optional frozen LoRA adapter."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        instruction: str,
        adapter_path: str | Path | None = None,
        max_length: int = 512,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - GPU-only dependency
            raise ImportError("Install the Phase 3 GPU requirements to load Qwen3Reranker") from error

        self.torch = torch
        self.instruction = instruction
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            padding_side="left",
        )
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": torch.bfloat16,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:  # pragma: no cover - GPU-only dependency
                raise ImportError("Install peft to load the evidence-aware LoRA adapter") from error
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=False)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.prefix_tokens = self.tokenizer.encode(QWEN_RERANKER_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(QWEN_RERANKER_SUFFIX, add_special_tokens=False)
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_token_id == self.tokenizer.unk_token_id or self.no_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Qwen3 tokenizer does not expose the required yes/no tokens")
        self.last_trace: dict[str, Any] = {}

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def _encode(self, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
        available = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if available <= 0:
            raise ValueError("Reranker max_length is shorter than the frozen prompt wrapper")
        bodies = [
            format_qwen_reranker_input(question, document, self.instruction)
            for question, document in pairs
        ]
        encoded = self.tokenizer(
            bodies,
            padding=False,
            truncation=True,
            max_length=available,
            add_special_tokens=False,
        )
        features = [
            {
                "input_ids": [*self.prefix_tokens, *ids, *self.suffix_tokens],
                "attention_mask": [1] * (len(self.prefix_tokens) + len(ids) + len(self.suffix_tokens)),
            }
            for ids in encoded["input_ids"]
        ]
        return self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> Any:
        del show_progress_bar
        values = list(pairs)
        scores: list[float] = []
        started = time.perf_counter()
        cuda = self.device.type == "cuda"
        if cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)
        for start in range(0, len(values), int(batch_size)):
            batch = self._encode(values[start : start + int(batch_size)])
            batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
            with self.torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
                yes_no = logits[:, [self.no_token_id, self.yes_token_id]].float()
                probabilities = self.torch.softmax(yes_no, dim=1)[:, 1]
            scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
        elapsed = time.perf_counter() - started
        self.last_trace = {
            "pairs": len(values),
            "elapsed_seconds": elapsed,
            "pairs_per_second": len(values) / elapsed if elapsed else None,
            "peak_gpu_memory_bytes": (
                int(self.torch.cuda.max_memory_allocated(self.device)) if cuda else None
            ),
        }
        if convert_to_numpy:
            import numpy as np

            return np.asarray(scores, dtype=float)
        return scores
