from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from chunkrag.schemas import Chunk, Document

try:
    from chonkie import RecursiveChunker as ChonkieRecursiveChunker
    from chonkie import SemanticChunker as ChonkieSemanticChunker
    from chonkie.embeddings import AutoEmbeddings as ChonkieAutoEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    ChonkieRecursiveChunker = None
    ChonkieSemanticChunker = None
    ChonkieAutoEmbeddings = None


_SPACY_PIPELINE = None
_CHONKIE_EMBEDDINGS_CACHE = {}


@dataclass(slots=True)
class ChunkingContext:
    tokenizer: PreTrainedTokenizerBase
    semantic_encoder: SentenceTransformer | None = None


def get_sentencizer():
    global _SPACY_PIPELINE
    if _SPACY_PIPELINE is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _SPACY_PIPELINE = nlp
    return _SPACY_PIPELINE


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sentence_split(text: str) -> list[str]:
    doc = get_sentencizer()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _decode_prefix_with_token_limit(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    max_tokens: int,
) -> tuple[str, int]:
    """Decode the longest prefix whose round-trip token count fits the limit."""
    for consumed in range(min(len(token_ids), max_tokens), 0, -1):
        text = tokenizer.decode(token_ids[:consumed], skip_special_tokens=True).strip()
        if text and count_tokens(tokenizer, text) <= max_tokens:
            return text, consumed
    raise ValueError("Could not decode a non-empty token-bounded fragment")


def _split_text_to_token_limit(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
) -> list[str]:
    """Split text with a post-decode token-count guarantee."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    fragments: list[str] = []
    start = 0
    while start < len(token_ids):
        fragment, consumed = _decode_prefix_with_token_limit(
            tokenizer,
            token_ids[start : start + max_tokens],
            max_tokens,
        )
        fragments.append(fragment)
        start += consumed
    return fragments


def _chunk_from_texts(
    document: Document,
    texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    chunker_name: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        clean_text = text.strip()
        if not clean_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
                doc_id=document.doc_id,
                title=document.title,
                dataset=document.dataset,
                text=clean_text,
                token_count=count_tokens(tokenizer, clean_text),
            )
        )
    return chunks


def fixed_token_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
    chunker_name: str,
    enforce_token_limit: bool = False,
) -> list[Chunk]:
    token_ids = tokenizer.encode(document.text, add_special_tokens=False)
    texts: list[str] = []
    if enforce_token_limit:
        start = 0
        while start < len(token_ids):
            text, consumed = _decode_prefix_with_token_limit(
                tokenizer,
                token_ids[start : start + chunk_size],
                chunk_size,
            )
            texts.append(text)
            if start + consumed >= len(token_ids):
                break
            start += max(1, consumed - min(chunk_overlap, consumed - 1))
    else:
        stride = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(token_ids), stride):
            window = token_ids[start : start + chunk_size]
            if not window:
                continue
            texts.append(tokenizer.decode(window, skip_special_tokens=True))
            if start + chunk_size >= len(token_ids):
                break
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def recursive_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
    chunker_name: str,
    enforce_token_limit: bool = False,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(document.text)
    if enforce_token_limit:
        texts = [
            fragment
            for text in texts
            for fragment in _split_text_to_token_limit(tokenizer, text, chunk_size)
        ]
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def sentence_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunker_name: str,
    enforce_token_limit: bool = False,
) -> list[Chunk]:
    texts: list[str] = []
    current: list[str] = []
    for sentence in sentence_split(document.text):
        if enforce_token_limit and count_tokens(tokenizer, sentence) > chunk_size:
            if current:
                texts.append(" ".join(current))
                current = []
            texts.extend(_split_text_to_token_limit(tokenizer, sentence, chunk_size))
            continue
        candidate = " ".join(current + [sentence])
        if current and count_tokens(tokenizer, candidate) > chunk_size:
            texts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        texts.append(" ".join(current))
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def _sentence_token_count_tables(
    tokenizer: PreTrainedTokenizerBase,
    sentences: list[str],
) -> tuple[list[int], list[int]]:
    first_position_counts = [count_tokens(tokenizer, sentence) for sentence in sentences]
    continuation_counts = [0] + [count_tokens(tokenizer, f" {sentence}") for sentence in sentences[1:]]
    return first_position_counts, continuation_counts


def semantic_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    similarity_threshold: float,
    embedding_model: SentenceTransformer,
    chunker_name: str,
    min_chunk_tokens: int | None = None,
) -> list[Chunk]:
    sentences = sentence_split(document.text)
    if len(sentences) <= 1:
        return _chunk_from_texts(document, sentences, tokenizer, chunker_name)

    min_chunk_tokens = min_chunk_tokens or max(32, chunk_size // 2)
    embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    first_position_counts, continuation_counts = _sentence_token_count_tables(tokenizer, sentences)
    texts: list[str] = []
    current_start = 0
    current_sentences = [sentences[0]]
    current_token_count = first_position_counts[0]
    for idx in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[idx - 1], embeddings[idx]))
        candidate_token_count = current_token_count + continuation_counts[idx]
        similarity_break = similarity < similarity_threshold and current_token_count >= min_chunk_tokens
        size_break = candidate_token_count > chunk_size
        should_split = similarity_break or size_break
        if should_split:
            texts.append(" ".join(current_sentences))
            current_start = idx
            current_sentences = [sentences[idx]]
            current_token_count = first_position_counts[current_start]
        else:
            current_sentences.append(sentences[idx])
            current_token_count = candidate_token_count
    if current_sentences:
        texts.append(" ".join(current_sentences))
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def get_chonkie_embeddings(model_name: str):
    if ChonkieAutoEmbeddings is None:
        raise ImportError("Chonkie is not installed. Install it with `pip install 'chonkie[semantic]'`.")
    if model_name not in _CHONKIE_EMBEDDINGS_CACHE:
        _CHONKIE_EMBEDDINGS_CACHE[model_name] = ChonkieAutoEmbeddings.get_embeddings(model_name)
    return _CHONKIE_EMBEDDINGS_CACHE[model_name]


def chonkie_recursive_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunker_name: str,
) -> list[Chunk]:
    if ChonkieRecursiveChunker is None:
        raise ImportError("Chonkie is not installed. Install it with `pip install chonkie`.")
    chunker = ChonkieRecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=24,
    )
    raw_chunks = chunker.chunk(document.text)
    return [
        Chunk(
            chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
            doc_id=document.doc_id,
            title=document.title,
            dataset=document.dataset,
            text=chunk.text.strip(),
            token_count=int(chunk.token_count),
        )
        for index, chunk in enumerate(raw_chunks)
        if chunk.text.strip()
    ]


def chonkie_semantic_chunks(
    document: Document,
    chunk_size: int,
    chunker_name: str,
    embedding_model_name: str,
    threshold: float = 0.7,
    min_sentences_per_chunk: int = 1,
    similarity_window: int = 3,
    skip_window: int = 0,
) -> list[Chunk]:
    if ChonkieSemanticChunker is None:
        raise ImportError("Chonkie semantic support is not installed. Install it with `pip install 'chonkie[semantic]'`.")
    chunker = ChonkieSemanticChunker(
        embedding_model=get_chonkie_embeddings(embedding_model_name),
        threshold=threshold,
        chunk_size=chunk_size,
        similarity_window=similarity_window,
        min_sentences_per_chunk=min_sentences_per_chunk,
        skip_window=skip_window,
    )
    raw_chunks = chunker.chunk(document.text)
    return [
        Chunk(
            chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
            doc_id=document.doc_id,
            title=document.title,
            dataset=document.dataset,
            text=chunk.text.strip(),
            token_count=int(chunk.token_count),
        )
        for index, chunk in enumerate(raw_chunks)
        if chunk.text.strip()
    ]


ChunkerBuilder = Callable[[Document, dict[str, Any], ChunkingContext], list[Chunk]]


def _build_fixed_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    return fixed_token_chunks(
        document,
        context.tokenizer,
        chunk_size=spec["chunk_size"],
        chunk_overlap=spec.get("chunk_overlap", 0),
        chunker_name=spec["name"],
        enforce_token_limit=spec.get("enforce_token_limit", False),
    )


def _build_recursive_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    return recursive_chunks(
        document,
        context.tokenizer,
        chunk_size=spec["chunk_size"],
        chunk_overlap=spec.get("chunk_overlap", 0),
        chunker_name=spec["name"],
        enforce_token_limit=spec.get("enforce_token_limit", False),
    )


def _build_sentence_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    return sentence_chunks(
        document,
        context.tokenizer,
        chunk_size=spec["chunk_size"],
        chunker_name=spec["name"],
        enforce_token_limit=spec.get("enforce_token_limit", False),
    )


def _build_semantic_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    if context.semantic_encoder is None:
        raise ValueError("Semantic chunking requires a semantic encoder in the chunking context.")
    return semantic_chunks(
        document,
        context.tokenizer,
        chunk_size=spec["chunk_size"],
        similarity_threshold=spec.get("similarity_threshold", 0.72),
        embedding_model=context.semantic_encoder,
        chunker_name=spec["name"],
        min_chunk_tokens=spec.get("min_chunk_tokens"),
    )


def _build_chonkie_recursive_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    return chonkie_recursive_chunks(
        document,
        context.tokenizer,
        chunk_size=spec["chunk_size"],
        chunker_name=spec["name"],
    )


def _build_chonkie_semantic_chunks(document: Document, spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    return chonkie_semantic_chunks(
        document,
        chunk_size=spec["chunk_size"],
        chunker_name=spec["name"],
        embedding_model_name=spec.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        threshold=spec.get("similarity_threshold", 0.7),
        min_sentences_per_chunk=spec.get("min_sentences_per_chunk", 1),
        similarity_window=spec.get("similarity_window", 3),
        skip_window=spec.get("skip_window", 0),
    )


CHUNKER_REGISTRY: dict[str, ChunkerBuilder] = {
    "fixed": _build_fixed_chunks,
    "recursive": _build_recursive_chunks,
    "sentence": _build_sentence_chunks,
    "semantic": _build_semantic_chunks,
    "chonkie_recursive": _build_chonkie_recursive_chunks,
    "chonkie_semantic": _build_chonkie_semantic_chunks,
}


def build_document_chunks(document: Document, chunker_spec: dict[str, Any], context: ChunkingContext) -> list[Chunk]:
    chunker_type = chunker_spec["type"]
    try:
        builder = CHUNKER_REGISTRY[chunker_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported chunker type: {chunker_type}") from exc
    return builder(document, chunker_spec, context)


def build_chunks(
    documents: list[Document],
    chunker_spec: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    semantic_encoder: SentenceTransformer | None = None,
) -> list[Chunk]:
    context = ChunkingContext(tokenizer=tokenizer, semantic_encoder=semantic_encoder)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(build_document_chunks(document, chunker_spec, context))
    return chunks
