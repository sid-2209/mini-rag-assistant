from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from mini_rag_assistant.ollama_client import OllamaClient, OllamaError
from mini_rag_assistant.text_utils import extract_keywords
from mini_rag_assistant.types import Chunk, RetrievalResult, RetrievedChunk

INDEX_FILE_NAME = "index.pkl"
MANIFEST_FILE_NAME = "manifest.json"
DENSE_WEIGHT = 0.72
LEXICAL_WEIGHT = 0.28


class LocalVectorStore:
    def __init__(
        self,
        *,
        lexical_vectorizer: FeatureUnion,
        lexical_matrix,
        chunks: list[Chunk],
        dense_matrix: np.ndarray | None = None,
        embedding_backend: str = "tfidf",
        embedding_model: str | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.lexical_vectorizer = lexical_vectorizer
        self.lexical_matrix = lexical_matrix
        self.chunks = chunks
        self.dense_matrix = dense_matrix
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.ollama_client = ollama_client

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        *,
        embedding_backend: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        ollama_client: OllamaClient | None = None,
    ) -> "LocalVectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store with zero chunks.")

        lexical_vectorizer = _build_lexical_vectorizer()
        texts = [chunk.text for chunk in chunks]
        lexical_matrix = lexical_vectorizer.fit_transform(texts)

        normalized_backend = embedding_backend.lower()
        dense_matrix: np.ndarray | None = None
        if normalized_backend == "ollama":
            client = ollama_client or OllamaClient()
            dense_matrix = _normalize_dense_rows(np.asarray(client.embed(embedding_model, texts), dtype=np.float32))
        elif normalized_backend != "tfidf":
            raise ValueError("embedding_backend must be either 'ollama' or 'tfidf'")

        return cls(
            lexical_vectorizer=lexical_vectorizer,
            lexical_matrix=lexical_matrix,
            chunks=chunks,
            dense_matrix=dense_matrix,
            embedding_backend=normalized_backend,
            embedding_model=embedding_model if normalized_backend == "ollama" else None,
            ollama_client=ollama_client,
        )

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        ollama_client: OllamaClient | None = None,
    ) -> "LocalVectorStore":
        path = Path(index_dir).expanduser().resolve() / INDEX_FILE_NAME
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with path.open("rb") as file:
            payload = pickle.load(file)

        lexical_vectorizer = payload.get("lexical_vectorizer", payload.get("vectorizer"))
        lexical_matrix = payload.get("lexical_matrix", payload.get("chunk_matrix"))
        dense_matrix = payload.get("dense_matrix")
        embedding_backend = payload.get("embedding_backend", "tfidf")
        embedding_model = payload.get("embedding_model")

        if dense_matrix is not None:
            dense_matrix = np.asarray(dense_matrix, dtype=np.float32)

        return cls(
            lexical_vectorizer=lexical_vectorizer,
            lexical_matrix=lexical_matrix,
            chunks=[Chunk(**item) for item in payload["chunks"]],
            dense_matrix=dense_matrix,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            ollama_client=ollama_client,
        )

    @staticmethod
    def exists(index_dir: str | Path) -> bool:
        return (Path(index_dir).expanduser().resolve() / INDEX_FILE_NAME).exists()

    def save(self, index_dir: str | Path, *, manifest: dict[str, object] | None = None) -> None:
        root = Path(index_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        payload = {
            "lexical_vectorizer": self.lexical_vectorizer,
            "lexical_matrix": self.lexical_matrix,
            "dense_matrix": self.dense_matrix,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        with (root / INDEX_FILE_NAME).open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

        output_manifest = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(self.chunks),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
        }
        if manifest:
            output_manifest.update(manifest)

        with (root / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as file:
            json.dump(output_manifest, file, indent=2)

    def search(
        self,
        question: str,
        *,
        top_k: int = 4,
        min_score: float = 0.15,
        relative_score_floor: float = 0.55,
    ) -> RetrievalResult:
        query = question.strip()
        if not query:
            raise ValueError("Question cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not 0 <= relative_score_floor <= 1:
            raise ValueError("relative_score_floor must be between 0 and 1")

        lexical_query = self.lexical_vectorizer.transform([query])
        lexical_scores = cosine_similarity(lexical_query, self.lexical_matrix).ravel()
        dense_scores = self._dense_scores(query)
        raw_scores = _combine_scores(dense_scores, lexical_scores)

        if raw_scores.size == 0:
            return RetrievalResult([], 0.0, 0.0, refusal_reason="No chunks are available in the index.")

        ranked_indexes = np.argsort(raw_scores)[::-1]
        top_score = float(raw_scores[ranked_indexes[0]])
        question_keywords = extract_keywords(query)
        considered_chunks = [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(raw_scores[index]),
                dense_score=float(dense_scores[index]),
                lexical_score=float(lexical_scores[index]),
            )
            for index in ranked_indexes[: max(top_k, 2)]
        ]

        if top_score <= 0:
            return RetrievalResult(
                [],
                top_score=0.0,
                confidence=0.0,
                refusal_reason="No relevant chunk was retrieved for this question.",
                applied_floor=min_score,
                considered_chunks=considered_chunks,
            )

        applied_floor = max(min_score, top_score * relative_score_floor)
        retrieved_chunks: list[RetrievedChunk] = []
        rejected_for_keyword_mismatch = False
        for index in ranked_indexes:
            score = float(raw_scores[index])
            if score < applied_floor:
                continue
            chunk = self.chunks[index]
            if question_keywords and not (extract_keywords(chunk.text) & question_keywords):
                rejected_for_keyword_mismatch = True
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=float(dense_scores[index]),
                    lexical_score=float(lexical_scores[index]),
                )
            )
            if len(retrieved_chunks) >= top_k:
                break

        if not retrieved_chunks:
            refusal_reason = "Top matches were below the minimum relevance threshold."
            if rejected_for_keyword_mismatch:
                refusal_reason = "Top matches did not overlap the question's key terms."
            return RetrievalResult(
                [],
                top_score=top_score,
                confidence=round(top_score, 3),
                refusal_reason=refusal_reason,
                applied_floor=applied_floor,
                considered_chunks=considered_chunks,
            )

        mean_score = float(np.mean([item.score for item in retrieved_chunks]))
        confidence = round((top_score * 0.7) + (mean_score * 0.3), 3)
        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            top_score=top_score,
            confidence=confidence,
            applied_floor=applied_floor,
            considered_chunks=considered_chunks,
        )

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.embedding_backend != "ollama":
            return np.zeros(len(self.chunks), dtype=np.float32)
        if self.dense_matrix is None or not len(self.chunks):
            return np.zeros(len(self.chunks), dtype=np.float32)
        if self.ollama_client is None or not self.embedding_model:
            raise OllamaError("This index requires an Ollama client and embedding model to search.")

        query_embedding = np.asarray(self.ollama_client.embed(self.embedding_model, [query])[0], dtype=np.float32)
        query_embedding = _normalize_dense_rows(query_embedding.reshape(1, -1))[0]
        dense_scores = np.clip(self.dense_matrix @ query_embedding, 0.0, 1.0)
        return dense_scores.astype(np.float32)


def load_manifest(index_dir: str | Path) -> dict[str, object]:
    path = Path(index_dir).expanduser().resolve() / MANIFEST_FILE_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_lexical_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _combine_scores(dense_scores: np.ndarray, lexical_scores: np.ndarray) -> np.ndarray:
    clipped_lexical = np.clip(lexical_scores.astype(np.float32), 0.0, 1.0)
    if dense_scores.size == 0 or not np.any(dense_scores):
        return clipped_lexical
    clipped_dense = np.clip(dense_scores.astype(np.float32), 0.0, 1.0)
    return (clipped_dense * DENSE_WEIGHT) + (clipped_lexical * LEXICAL_WEIGHT)


def _normalize_dense_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms
