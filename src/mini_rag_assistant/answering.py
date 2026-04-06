from __future__ import annotations

import re

from sklearn.metrics.pairwise import cosine_similarity

from mini_rag_assistant.ollama_client import OllamaClient, OllamaError
from mini_rag_assistant.text_utils import extract_keywords
from mini_rag_assistant.types import (
    AnswerResult,
    Citation,
    EvidenceSnippet,
    RetrievalResult,
    RetrievedChunk,
)

REFUSAL_MESSAGE = "I don’t have enough information in the provided documents to answer this question."
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "refused": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
    "required": ["refused", "answer", "citations"],
}


class GroundedAnswerGenerator:
    def __init__(
        self,
        *,
        vectorizer,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
        answer_mode: str = "ollama",
    ) -> None:
        self.vectorizer = vectorizer
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.answer_mode = answer_mode

    def generate(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        max_sentences: int = 2,
        max_evidence: int = 6,
    ) -> AnswerResult:
        if not retrieval.retrieved_chunks:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                citations=_build_refusal_citations(retrieval),
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason=retrieval.refusal_reason,
            )

        evidence_snippets = self._select_evidence(question, retrieval.retrieved_chunks, max_evidence=max_evidence)
        if not evidence_snippets:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                citations=_build_refusal_citations(retrieval),
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason="Retrieved chunks did not contain a clear supporting sentence.",
            )

        if self.answer_mode == "ollama" and self.llm_client and self.llm_model:
            llm_result = self._generate_with_ollama(
                question,
                retrieval,
                evidence_snippets,
                max_sentences=max_sentences,
            )
            if llm_result is not None:
                return llm_result

        return self._generate_extractive_answer(
            question,
            retrieval,
            evidence_snippets,
            max_sentences=max_sentences,
        )

    def _generate_with_ollama(
        self,
        question: str,
        retrieval: RetrievalResult,
        evidence_snippets: list[EvidenceSnippet],
        *,
        max_sentences: int,
    ) -> AnswerResult | None:
        system_prompt = (
            "You are a grounded answer composer for a retrieval-augmented QA system.\n"
            "Use only the provided evidence snippets.\n"
            "Do not use background knowledge.\n"
            "If the evidence is incomplete or the answer is not explicitly supported, refuse.\n"
            f"When refusing, answer exactly: {REFUSAL_MESSAGE}\n"
            "Keep successful answers concise and limited to one or two sentences.\n"
            "Return only valid JSON matching the schema."
        )
        prompt_lines = [
            f"Question: {question}",
            "",
            "Evidence snippets:",
        ]
        for snippet in evidence_snippets:
            chunk = snippet.retrieved_chunk.chunk
            prompt_lines.extend(
                [
                    (
                        f"[{snippet.evidence_id}] {chunk.title} | {chunk.source} | "
                        f"chunk {chunk.chunk_index} | sentence {snippet.sentence_index} | score {snippet.score:.3f}"
                    ),
                    snippet.sentence,
                    "",
                ]
            )
        prompt_lines.extend(
            [
                "Rules:",
                "- Cite the evidence IDs that directly support the answer.",
                "- Do not mention evidence IDs inside the answer text.",
                "- Do not add facts, dates, people, or numbers that are absent from the cited snippets.",
            ]
        )

        try:
            payload = self.llm_client.generate_json(
                model=self.llm_model,
                system=system_prompt,
                prompt="\n".join(prompt_lines).strip(),
                schema=GROUNDING_SCHEMA,
                options={
                    "temperature": 0,
                    "top_p": 0.9,
                    "num_predict": 220,
                },
            )
        except OllamaError:
            return None

        answer_text = str(payload.get("answer", "")).strip()
        refused = bool(payload.get("refused"))
        citation_ids = _dedupe_ids(payload.get("citations", []))

        if refused or answer_text == REFUSAL_MESSAGE:
            return None

        cited_snippets = [snippet for snippet in evidence_snippets if snippet.evidence_id in citation_ids]
        if not answer_text or not cited_snippets:
            return None

        cleaned_answer = _limit_to_sentence_count(_clean_sentence(answer_text), max_sentences)
        if not _is_answer_grounded(cleaned_answer, cited_snippets, question):
            return None

        answer_confidence = round(min(retrieval.confidence, max(item.score for item in cited_snippets)), 3)
        return AnswerResult(
            answer=cleaned_answer,
            citations=_build_citations_from_evidence(cited_snippets),
            confidence=answer_confidence,
            refused=False,
        )

    def _generate_extractive_answer(
        self,
        question: str,
        retrieval: RetrievalResult,
        evidence_snippets: list[EvidenceSnippet],
        *,
        max_sentences: int,
    ) -> AnswerResult:
        selected = evidence_snippets[:max_sentences]
        if not selected:
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                citations=_build_refusal_citations(retrieval),
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason="Supporting evidence was too weak after sentence-level filtering.",
            )

        ordered_sentences = [item.sentence for item in selected]
        answer_text = _compose_answer(ordered_sentences)
        if not _is_answer_grounded(answer_text, selected, question):
            return AnswerResult(
                answer=REFUSAL_MESSAGE,
                citations=_build_refusal_citations(retrieval),
                confidence=retrieval.confidence,
                refused=True,
                refusal_reason="Supporting evidence was too weak after sentence-level filtering.",
            )

        citations = _build_citations_from_evidence(selected)
        answer_confidence = round(min(retrieval.confidence, max(item.score for item in selected)), 3)
        return AnswerResult(
            answer=answer_text,
            citations=citations,
            confidence=answer_confidence,
            refused=False,
        )

    def _select_evidence(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        *,
        max_evidence: int,
    ) -> list[EvidenceSnippet]:
        candidate_sentences = self._collect_candidate_sentences(question, retrieved_chunks)
        if not candidate_sentences:
            return []

        sentence_floor = max(0.12, candidate_sentences[0][0] * 0.55)
        question_keywords = extract_keywords(question)
        selected: list[EvidenceSnippet] = []
        seen_sentences: set[str] = set()
        covered_keywords: set[str] = set()

        for score, retrieved, sentence_index, sentence in candidate_sentences:
            normalized = _normalize_sentence(sentence)
            if score < sentence_floor or normalized in seen_sentences:
                continue

            sentence_keywords = extract_keywords(sentence)
            keyword_overlap = sentence_keywords & question_keywords
            keyword_gain = keyword_overlap - covered_keywords

            if question_keywords and not keyword_overlap:
                continue
            if selected and question_keywords and not keyword_gain:
                continue

            selected.append(
                EvidenceSnippet(
                    evidence_id=f"E{len(selected) + 1}",
                    retrieved_chunk=retrieved,
                    sentence_index=sentence_index,
                    sentence=_clean_sentence(sentence),
                    score=round(score, 3),
                )
            )
            seen_sentences.add(normalized)
            covered_keywords.update(keyword_overlap)
            if len(selected) >= max_evidence:
                break

        return selected

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
                combined_score = round((float(score) * 0.7) + (retrieved.score * 0.3), 3)
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


def _build_citations_from_evidence(selected: list[EvidenceSnippet]) -> list[Citation]:
    citations: list[Citation] = []
    seen_chunks: set[str] = set()
    for item in selected:
        chunk = item.retrieved_chunk.chunk
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(item.score, 3),
            )
        )
    return citations


def _build_refusal_citations(retrieval: RetrievalResult) -> list[Citation]:
    candidates = retrieval.retrieved_chunks or retrieval.considered_chunks
    if not candidates:
        return []

    citations: list[Citation] = []
    seen_chunks: set[str] = set()
    for retrieved in candidates[:2]:
        chunk = retrieved.chunk
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)
        note = retrieval.refusal_reason or "Insufficient support to answer confidently."
        if retrieved.score <= 0:
            note = "No similarity match."
        elif retrieved.score < retrieval.applied_floor:
            note = "Below relevance threshold."
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(retrieved.score, 3),
                note=note,
            )
        )
    return citations


def _is_answer_grounded(answer: str, snippets: list[EvidenceSnippet], question: str) -> bool:
    answer_keywords = extract_keywords(answer)
    if not answer_keywords:
        return False

    support_text = " ".join(snippet.sentence for snippet in snippets)
    support_keywords = extract_keywords(support_text)
    question_keywords = extract_keywords(question)
    if question_keywords and not (answer_keywords & question_keywords):
        return False
    unsupported_keywords = answer_keywords - support_keywords - question_keywords
    if unsupported_keywords:
        return False

    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    support_numbers = set(NUMBER_PATTERN.findall(support_text))
    return answer_numbers.issubset(support_numbers)


def _limit_to_sentence_count(text: str, max_sentences: int) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return text.strip()
    return " ".join(_clean_sentence(sentence) for sentence in sentences[:max_sentences]).strip()


def _clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip().strip('"')
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().lower()


def _coerce_reason(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _dedupe_ids(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped
