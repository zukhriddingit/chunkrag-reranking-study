from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    dataset: str


@dataclass(slots=True)
class QAExample:
    example_id: str
    dataset: str
    question: str
    answers: list[str]
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    dataset: str
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricSummary:
    value: float
    ci_low: float
    ci_high: float

    def to_flat_fields(self, prefix: str) -> dict[str, float]:
        return {
            prefix: self.value,
            f"{prefix}_ci_low": self.ci_low,
            f"{prefix}_ci_high": self.ci_high,
        }


@dataclass(slots=True)
class AggregateMetricSummary:
    mean: float
    std: float
    min: float
    max: float

    def to_flat_fields(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
        }


@dataclass(slots=True)
class PredictionRecord:
    seed: int
    retriever: str
    example_id: str
    question: str
    gold_answers: list[str]
    prediction: str
    chunker: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_titles: list[str] = field(default_factory=list)
    exact_match: float = 0.0
    f1: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    supporting_doc_coverage: float = 0.0
    all_supporting_docs_found: float = 0.0
    answer_string_visible_at_k: float = 0.0
    raw_prediction: str | None = None
    full_prompt_tokens: int | None = None
    used_prompt_tokens: int | None = None
    context_truncated: bool | None = None
    refinement_applied: bool | None = None
    generated_tokens: int | None = None
    generation_max_new_tokens: int | None = None
    generation_length_capped: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SummaryRow:
    dataset: str
    system: str
    seed: int
    retriever: str
    chunker: str | None
    num_examples: int
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    num_documents: int | None = None
    num_chunks: int | None = None
    avg_chunk_tokens: float | None = None
    avg_retrieval_latency_s: float | None = None
    avg_generation_latency_s: float | None = None

    def numeric_fields(self) -> dict[str, float]:
        values: dict[str, float] = {"num_examples": float(self.num_examples)}
        for field_name in (
            "num_documents",
            "num_chunks",
            "avg_chunk_tokens",
            "avg_retrieval_latency_s",
            "avg_generation_latency_s",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                values[field_name] = float(field_value)
        for metric_name, metric_summary in self.metrics.items():
            values[metric_name] = metric_summary.value
        return values

    def to_flat_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "system": self.system,
            "seed": self.seed,
            "retriever": self.retriever,
            "chunker": self.chunker,
            "num_examples": self.num_examples,
        }
        for field_name in (
            "num_documents",
            "num_chunks",
            "avg_chunk_tokens",
            "avg_retrieval_latency_s",
            "avg_generation_latency_s",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                payload[field_name] = field_value
        for metric_name, metric_summary in self.metrics.items():
            payload.update(metric_summary.to_flat_fields(metric_name))
        return payload


@dataclass(slots=True)
class AggregateSummaryRow:
    dataset: str
    system: str
    retriever: str | None
    chunker: str | None
    num_seeds: int
    seed_values: list[int]
    aggregates: dict[str, AggregateMetricSummary] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "system": self.system,
            "retriever": self.retriever,
            "chunker": self.chunker,
            "num_seeds": self.num_seeds,
            "seed_values": self.seed_values,
        }
        for metric_name, metric_summary in self.aggregates.items():
            payload.update(metric_summary.to_flat_fields(metric_name))
        return payload
