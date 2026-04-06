from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    source: str
    content: str
    path: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    path: str
    chunk_index: int
    text: str
    word_start: int
    word_end: int


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


@dataclass(slots=True)
class EvidenceSnippet:
    evidence_id: str
    retrieved_chunk: RetrievedChunk
    sentence_index: int
    sentence: str
    score: float


@dataclass(slots=True)
class RetrievalResult:
    retrieved_chunks: list[RetrievedChunk]
    top_score: float
    confidence: float
    refusal_reason: str | None = None
    applied_floor: float = 0.0
    considered_chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    doc_id: str
    title: str
    source: str
    chunk_index: int
    score: float
    note: str | None = None


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: str | None = None
