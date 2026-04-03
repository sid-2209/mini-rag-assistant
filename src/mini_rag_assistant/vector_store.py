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

from mini_rag_assistant.types import Chunk, RetrievalResult, RetrievedChunk

INDEX_FILE_NAME = "index.pkl"
MANIFEST_FILE_NAME = "manifest.json"


class LocalVectorStore:
    def __init__(self, *, vectorizer: FeatureUnion, chunk_matrix, chunks: list[Chunk]) -> None:
        self.vectorizer = vectorizer
        self.chunk_matrix = chunk_matrix
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "LocalVectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store with zero chunks.")

        vectorizer = _build_vectorizer()
        chunk_matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(vectorizer=vectorizer, chunk_matrix=chunk_matrix, chunks=chunks)

    @classmethod
    def load(cls, index_dir: str | Path) -> "LocalVectorStore":
        path = Path(index_dir).expanduser().resolve() / INDEX_FILE_NAME
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            vectorizer=payload["vectorizer"],
            chunk_matrix=payload["chunk_matrix"],
            chunks=[Chunk(**item) for item in payload["chunks"]],
        )

    @staticmethod
    def exists(index_dir: str | Path) -> bool:
        return (Path(index_dir).expanduser().resolve() / INDEX_FILE_NAME).exists()

    def save(self, index_dir: str | Path, *, manifest: dict[str, object] | None = None) -> None:
        root = Path(index_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        payload = {
            "vectorizer": self.vectorizer,
            "chunk_matrix": self.chunk_matrix,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        with (root / INDEX_FILE_NAME).open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

        output_manifest = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(self.chunks),
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

        query_matrix = self.vectorizer.transform([query])
        raw_scores = cosine_similarity(query_matrix, self.chunk_matrix).ravel()
        if raw_scores.size == 0:
            return RetrievalResult([], 0.0, 0.0, refusal_reason="No chunks are available in the index.")

        ranked_indexes = np.argsort(raw_scores)[::-1]
        top_score = float(raw_scores[ranked_indexes[0]])
        if top_score <= 0:
            return RetrievalResult(
                [],
                top_score=0.0,
                confidence=0.0,
                refusal_reason="No relevant chunk was retrieved for this question.",
                applied_floor=min_score,
            )

        applied_floor = max(min_score, top_score * relative_score_floor)
        retrieved_chunks: list[RetrievedChunk] = []
        for index in ranked_indexes:
            score = float(raw_scores[index])
            if score < applied_floor:
                continue
            retrieved_chunks.append(RetrievedChunk(chunk=self.chunks[index], score=score))
            if len(retrieved_chunks) >= top_k:
                break

        if not retrieved_chunks:
            return RetrievalResult(
                [],
                top_score=top_score,
                confidence=top_score,
                refusal_reason="Top matches were below the minimum relevance threshold.",
                applied_floor=applied_floor,
            )

        mean_score = float(np.mean([item.score for item in retrieved_chunks]))
        confidence = round((top_score * 0.7) + (mean_score * 0.3), 3)
        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            top_score=top_score,
            confidence=confidence,
            applied_floor=applied_floor,
        )


def load_manifest(index_dir: str | Path) -> dict[str, object]:
    path = Path(index_dir).expanduser().resolve() / MANIFEST_FILE_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_vectorizer() -> FeatureUnion:
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
