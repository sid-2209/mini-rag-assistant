from __future__ import annotations

import re

from sklearn.metrics.pairwise import cosine_similarity

from mini_rag_assistant.types import AnswerResult, Citation, RetrievalResult, RetrievedChunk

REFUSAL_MESSAGE = "I don’t have enough information in the provided documents to answer this question."
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


class GroundedAnswerGenerator:
    def __init__(self, *, vectorizer) -> None:
        self.vectorizer = vectorizer

    def generate(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        max_sentences: int = 3,
    ) -> AnswerResult:
        if not retrieval.retrieved_chunks:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason=retrieval.refusal_reason,
            )

        candidate_sentences = self._collect_candidate_sentences(question, retrieval.retrieved_chunks)
        if not candidate_sentences:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason="Retrieved chunks did not contain a clear supporting sentence.",
            )

        sentence_floor = max(0.12, retrieval.applied_floor * 0.7)
        selected: list[tuple[float, RetrievedChunk, int, str]] = []
        seen_sentences: set[str] = set()
        used_chunks: set[str] = set()

        for score, retrieved, sentence_index, sentence in candidate_sentences:
            normalized = _normalize_sentence(sentence)
            if score < sentence_floor or normalized in seen_sentences:
                continue

            if retrieved.chunk.chunk_id in used_chunks and len(selected) >= 2:
                continue

            selected.append((score, retrieved, sentence_index, sentence))
            seen_sentences.add(normalized)
            used_chunks.add(retrieved.chunk.chunk_id)
            if len(selected) >= max_sentences:
                break

        if not selected:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason="Supporting evidence was too weak after sentence-level filtering.",
            )

        ordered_sentences = [item[3] for item in selected]
        answer_text = _compose_answer(ordered_sentences)
        citations = _build_citations(selected)
        answer_confidence = round(min(retrieval.confidence, max(item[0] for item in selected)), 3)
        return AnswerResult(
            answer=answer_text,
            citations=citations,
            confidence=answer_confidence,
            refused=False,
        )

    def _collect_candidate_sentences(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[tuple[float, RetrievedChunk, int, str]]:
        query_vector = self.vectorizer.transform([question.strip()])
        candidates: list[tuple[float, RetrievedChunk, int, str]] = []

        for retrieved in retrieved_chunks:
            sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(retrieved.chunk.text) if sentence.strip()]
            if not sentences:
                continue

            sentence_matrix = self.vectorizer.transform(sentences)
            sentence_scores = cosine_similarity(query_vector, sentence_matrix).ravel()
            for index, (sentence, score) in enumerate(zip(sentences, sentence_scores, strict=False), start=1):
                if len(sentence.split()) < 5:
                    continue
                combined_score = round((float(score) * 0.75) + (retrieved.score * 0.25), 3)
                candidates.append((combined_score, retrieved, index, sentence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates


def _compose_answer(sentences: list[str]) -> str:
    cleaned = [_clean_sentence(sentence) for sentence in sentences if sentence.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in cleaned:
        key = _normalize_sentence(sentence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return " ".join(deduped).strip()


def _build_citations(selected: list[tuple[float, RetrievedChunk, int, str]]) -> list[Citation]:
    citations: list[Citation] = []
    seen_chunks: set[str] = set()
    for score, retrieved, _, _ in selected:
        chunk = retrieved.chunk
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(score, 3),
            )
        )
    return citations


def _clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().lower()

