"""Explicit group-aware batching and listwise-loss contract for Phase 4."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .preprocessing import assert_positive_indexing, training_pair_text


def hand_calculated_group_cross_entropy(scores: Sequence[float], temperature: float = 1.0) -> float:
    """Cross entropy for a group whose positive is at index zero."""

    if len(scores) < 2:
        raise ValueError("Listwise loss requires at least two candidates")
    scaled = [float(value) / temperature for value in scores]
    anchor = max(scaled)
    log_sum_exp = anchor + math.log(sum(math.exp(value - anchor) for value in scaled))
    return log_sum_exp - scaled[0]


def hand_calculated_listwise_loss(
    scores: Sequence[float], group_sizes: Sequence[int], temperature: float = 1.0
) -> float:
    if sum(group_sizes) != len(scores):
        raise ValueError("Group sizes do not partition the score vector")
    offset = 0
    losses: list[float] = []
    for size in group_sizes:
        if size < 2:
            raise ValueError("Each listwise group must have at least two pairs")
        losses.append(hand_calculated_group_cross_entropy(scores[offset : offset + size], temperature))
        offset += size
    return sum(losses) / len(losses)


def torch_listwise_loss(scores: Any, group_sizes: Sequence[int], temperature: float = 1.0) -> Any:
    """Mean within-group CE; never constructs groups from batch-global positives."""

    import torch
    import torch.nn.functional as functional

    if scores.ndim != 1:
        raise ValueError("Expected one scalar score per candidate pair")
    if int(scores.numel()) != int(sum(group_sizes)):
        raise ValueError("Group sizes do not partition the score tensor")
    offset = 0
    losses = []
    for size in group_sizes:
        if int(size) < 2:
            raise ValueError("Each listwise group must have at least two pairs")
        group_scores = scores[offset : offset + int(size)] / float(temperature)
        target = torch.zeros(1, dtype=torch.long, device=scores.device)
        losses.append(functional.cross_entropy(group_scores.unsqueeze(0), target))
        offset += int(size)
    return torch.stack(losses).mean()


def collate_listwise_groups(
    groups: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    instruction: str,
    max_length: int = 512,
) -> dict[str, Any]:
    """Flatten intact groups, then pad; group boundaries stay explicit."""

    encoded: list[dict[str, list[int]]] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    group_keys: list[str] = []
    token_counts: list[int] = []
    for group_index, group in enumerate(groups):
        assert_positive_indexing(group)
        pairs = [group["positive"], *list(group["negatives"])]
        group_sizes.append(len(pairs))
        group_keys.append(str(group.get("group_id", f"group-{group_index}")))
        for pair_index, pair in enumerate(pairs):
            text = training_pair_text(str(group["question"]), str(pair["text"]), instruction)
            item = tokenizer(text, add_special_tokens=False, truncation=False)
            ids = list(item["input_ids"])
            if len(ids) > max_length:
                raise ValueError("An overlength pair reached the collator")
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            labels.append(1 if pair_index == 0 else 0)
            token_counts.append(len(ids))
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    batch["group_sizes"] = group_sizes
    batch["binary_labels"] = labels
    batch["group_keys"] = group_keys
    batch["token_counts"] = token_counts
    return batch


def final_token_margin(logits: Any, attention_mask: Any, *, yes_token_id: int, no_token_id: int) -> Any:
    """Return yes-minus-no logits at each pair's final non-padding token."""

    import torch

    if logits.ndim != 3:
        raise ValueError("Expected causal-LM logits shaped [pairs, sequence, vocabulary]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_positions = (positions * attention_mask.long()).max(dim=1).values
    rows = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[rows, last_positions]
    return final_logits[:, int(yes_token_id)] - final_logits[:, int(no_token_id)]
