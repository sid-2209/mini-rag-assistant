from __future__ import annotations

from pathlib import Path

from mini_rag_assistant.answering import GroundedAnswerGenerator
from mini_rag_assistant.chunking import chunk_documents
from mini_rag_assistant.document_loader import load_documents
from mini_rag_assistant.types import AnswerResult, RetrievalResult
from mini_rag_assistant.vector_store import LocalVectorStore, load_manifest


class MiniRAGAssistant:
    def __init__(self, store: LocalVectorStore) -> None:
        self.store = store
        self.answer_generator = GroundedAnswerGenerator(vectorizer=store.vectorizer)

    def answer(
        self,
        question: str,
        *,
        top_k: int = 4,
        min_score: float = 0.15,
        relative_score_floor: float = 0.55,
    ) -> tuple[AnswerResult, RetrievalResult]:
        retrieval = self.store.search(
            question,
            top_k=top_k,
            min_score=min_score,
            relative_score_floor=relative_score_floor,
        )
        answer = self.answer_generator.generate(question, retrieval)
        return answer, retrieval


def build_index(
    docs_dir: str | Path,
    *,
    index_dir: str | Path = ".rag_store",
    chunk_size: int = 140,
    chunk_overlap: int = 30,
) -> dict[str, object]:
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = LocalVectorStore.build(chunks)
    manifest = {
        "docs_dir": str(Path(docs_dir).expanduser().resolve()),
        "document_count": len(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "source": document.source,
                "path": document.path,
            }
            for document in documents
        ],
    }
    store.save(index_dir, manifest=manifest)
    return load_manifest(index_dir)


def load_assistant(index_dir: str | Path = ".rag_store") -> MiniRAGAssistant:
    store = LocalVectorStore.load(index_dir)
    return MiniRAGAssistant(store)
