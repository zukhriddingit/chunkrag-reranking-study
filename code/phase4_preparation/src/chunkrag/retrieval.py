from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunkrag.schemas import Chunk


def lexical_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@runtime_checkable
class Retriever(Protocol):
    def build(self, chunks: list[Chunk]) -> None:
        ...

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        ...


class DenseRetriever:
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        encoder: SentenceTransformer | None = None,
        encoder_identifier: str | None = None,
        cache_dir: str | Path | None = None,
        cache_namespace: str | None = None,
        query_prefix: str = "",
    ) -> None:
        if encoder is None:
            if model_name is None:
                raise ValueError("DenseRetriever requires either an encoder instance or a model name.")
            encoder = SentenceTransformer(model_name, device=device)
        self.encoder = encoder
        self.batch_size = batch_size
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self.encoder_identifier = encoder_identifier or model_name or type(encoder).__name__
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_namespace = cache_namespace
        self.query_prefix = query_prefix

    def _cache_prefix(self, chunks: list[Chunk]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self.encoder_identifier.encode("utf-8"))
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.doc_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\0")
        namespace = self.cache_namespace or "default"
        safe_namespace = re.sub(r"[^a-zA-Z0-9_.-]+", "_", namespace).strip("_") or "default"
        return self.cache_dir / safe_namespace / digest.hexdigest()

    def _load_cached_index(self, prefix: Path) -> bool:
        embeddings_path = prefix.with_suffix(".npy")
        index_path = prefix.with_suffix(".faiss")
        metadata_path = prefix.with_suffix(".json")
        if not embeddings_path.exists() or not index_path.exists() or not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("encoder_identifier") != self.encoder_identifier:
                return False
            self.index = faiss.read_index(str(index_path))
            return True
        except Exception:
            return False

    def _save_cached_index(self, prefix: Path, embeddings: np.ndarray) -> None:
        if self.index is None:
            return
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(prefix.with_suffix(".npy"), embeddings)
        faiss.write_index(self.index, str(prefix.with_suffix(".faiss")))
        metadata = {
            "encoder_identifier": self.encoder_identifier,
            "num_chunks": len(self.chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.index = None
            return
        cache_prefix = self._cache_prefix(chunks)
        if cache_prefix is not None and self._load_cached_index(cache_prefix):
            return
        embeddings = self.encoder.encode(
            [chunk.text for chunk in chunks],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        if cache_prefix is not None:
            self._save_cached_index(cache_prefix, embeddings)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("The dense retriever index has not been built yet.")
        query_embedding = self.encoder.encode(
            [f"{self.query_prefix}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]


class BM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus_tokens = [lexical_tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("The BM25 index has not been built yet.")
        scores = np.asarray(self.index.get_scores(lexical_tokenize(query)), dtype=float)
        if len(scores) == 0:
            return []
        top_k = min(top_k, len(scores))
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return [(self.chunks[idx], float(scores[idx])) for idx in ranked_indices]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        candidate_pool_size: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: float = 60.0,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.candidate_pool_size = candidate_pool_size
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def build(self, chunks: list[Chunk]) -> None:
        self.dense_retriever.build(chunks)
        self.sparse_retriever.build(chunks)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        candidate_k = max(top_k, self.candidate_pool_size)
        fused = mean_reciprocal_rank_fusion(
            [
                self.dense_retriever.retrieve(query, candidate_k),
                self.sparse_retriever.retrieve(query, candidate_k),
            ],
            [self.dense_weight, self.sparse_weight],
            self.rrf_k,
        )
        return fused[:top_k]


class RerankRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str,
        model_revision: str | None,
        device: str,
        candidate_pool_size: int = 20,
        batch_size: int = 16,
    ) -> None:
        self.base_retriever = base_retriever
        self.cross_encoder = CrossEncoder(
            model_name,
            revision=model_revision,
            device=device,
        )
        self.candidate_pool_size = candidate_pool_size
        self.batch_size = batch_size

    def build(self, chunks: list[Chunk]) -> None:
        self.base_retriever.build(chunks)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        candidate_k = max(top_k, self.candidate_pool_size)
        candidates = self.base_retriever.retrieve(query, candidate_k)
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [(chunk, float(score)) for (chunk, _), score in ranked]


def mean_reciprocal_rank_fusion(
    result_sets: list[list[tuple[Chunk, float]]],
    weights: list[float],
    rrf_k: float,
) -> list[tuple[Chunk, float]]:
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}
    for weight, results in zip(weights, result_sets, strict=True):
        for rank, (chunk, _) in enumerate(results, start=1):
            chunk_lookup[chunk.chunk_id] = chunk
            fused_scores.setdefault(chunk.chunk_id, 0.0)
            fused_scores[chunk.chunk_id] += weight / (rrf_k + rank)
    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(chunk_lookup[chunk_id], fused_scores[chunk_id]) for chunk_id in ranked_chunk_ids]


@dataclass(slots=True)
class RetrieverFactoryContext:
    encoder: SentenceTransformer
    encoder_identifier: str
    device: str
    embedding_batch_size: int = 32
    retrieval_top_k: int = 4
    cache_dir: Path | None = None
    cache_namespace: str | None = None
    query_prefix: str = ""


RetrieverBuilder = Callable[[dict[str, Any], "RetrieverFactory"], Retriever]


class RetrieverFactory:
    def __init__(self, chunks: list[Chunk], context: RetrieverFactoryContext) -> None:
        self.chunks = chunks
        self.context = context
        self._shared_retrievers: dict[str, Retriever] = {}

    def get_dense(self) -> Retriever:
        if "dense" not in self._shared_retrievers:
            dense = DenseRetriever(
                encoder=self.context.encoder,
                encoder_identifier=self.context.encoder_identifier,
                device=self.context.device,
                batch_size=self.context.embedding_batch_size,
                cache_dir=self.context.cache_dir,
                cache_namespace=self.context.cache_namespace,
                query_prefix=self.context.query_prefix,
            )
            dense.build(self.chunks)
            self._shared_retrievers["dense"] = dense
        return self._shared_retrievers["dense"]

    def get_sparse(self) -> Retriever:
        if "bm25" not in self._shared_retrievers:
            sparse = BM25Retriever()
            sparse.build(self.chunks)
            self._shared_retrievers["bm25"] = sparse
        return self._shared_retrievers["bm25"]

    def create(self, spec: dict[str, Any]) -> Retriever:
        retriever_type = spec.get("type", "dense")
        try:
            builder = RETRIEVER_REGISTRY[retriever_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported retriever type: {retriever_type}") from exc
        return builder(spec, self)


def _build_dense_retriever(spec: dict[str, Any], factory: RetrieverFactory) -> Retriever:
    return factory.get_dense()


def _build_bm25_retriever(spec: dict[str, Any], factory: RetrieverFactory) -> Retriever:
    return factory.get_sparse()


def _build_hybrid_retriever(spec: dict[str, Any], factory: RetrieverFactory) -> Retriever:
    candidate_pool_size = spec.get("candidate_pool_size", max(factory.context.retrieval_top_k * 5, 20))
    return HybridRetriever(
        dense_retriever=factory.get_dense(),
        sparse_retriever=factory.get_sparse(),
        candidate_pool_size=candidate_pool_size,
        dense_weight=spec.get("dense_weight", 0.5),
        sparse_weight=spec.get("sparse_weight", 0.5),
        rrf_k=spec.get("rrf_k", 60.0),
    )


def _build_rerank_retriever(spec: dict[str, Any], factory: RetrieverFactory) -> Retriever:
    base_retriever_spec = dict(spec.get("base_retriever", {"type": "dense"}))
    base_retriever = factory.create(base_retriever_spec)
    candidate_pool_size = spec.get("candidate_pool_size", max(factory.context.retrieval_top_k * 5, 20))
    return RerankRetriever(
        base_retriever=base_retriever,
        model_name=spec.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        model_revision=spec.get("model_revision"),
        device=factory.context.device,
        candidate_pool_size=candidate_pool_size,
        batch_size=spec.get("batch_size", 16),
    )


RETRIEVER_REGISTRY: dict[str, RetrieverBuilder] = {
    "dense": _build_dense_retriever,
    "bm25": _build_bm25_retriever,
    "hybrid": _build_hybrid_retriever,
    "rerank": _build_rerank_retriever,
}


def create_retriever(
    chunks: list[Chunk],
    spec: dict[str, Any],
    context: RetrieverFactoryContext,
) -> Retriever:
    return RetrieverFactory(chunks, context).create(spec)
