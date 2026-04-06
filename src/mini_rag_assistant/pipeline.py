from __future__ import annotations

from pathlib import Path

from mini_rag_assistant.answering import GroundedAnswerGenerator
from mini_rag_assistant.chunking import chunk_documents
from mini_rag_assistant.document_loader import fingerprint_documents, load_documents
from mini_rag_assistant.ollama_client import OllamaClient
from mini_rag_assistant.types import AnswerResult, RetrievalResult
from mini_rag_assistant.vector_store import LocalVectorStore, load_manifest


class MiniRAGAssistant:
    def __init__(
        self,
        store: LocalVectorStore,
        *,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
        answer_mode: str = "ollama",
    ) -> None:
        self.store = store
        self.answer_generator = GroundedAnswerGenerator(
            vectorizer=store.lexical_vectorizer,
            llm_client=llm_client,
            llm_model=llm_model,
            answer_mode=answer_mode,
        )

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
    embedding_backend: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    ollama_host: str = "http://127.0.0.1:11434",
) -> dict[str, object]:
    documents = load_documents(docs_dir)
    document_fingerprints = fingerprint_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ollama_client = OllamaClient(host=ollama_host) if embedding_backend == "ollama" else None
    store = LocalVectorStore.build(
        chunks,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        ollama_client=ollama_client,
    )
    manifest = {
        "docs_dir": str(Path(docs_dir).expanduser().resolve()),
        "document_count": len(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_fingerprints": document_fingerprints,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model if embedding_backend == "ollama" else None,
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


def load_assistant(
    index_dir: str | Path = ".rag_store",
    *,
    answer_mode: str = "ollama",
    llm_model: str = "llama3.2:3b",
    ollama_host: str = "http://127.0.0.1:11434",
) -> MiniRAGAssistant:
    ollama_client = OllamaClient(host=ollama_host)
    store = LocalVectorStore.load(index_dir, ollama_client=ollama_client)
    llm_client = ollama_client if answer_mode == "ollama" else None
    return MiniRAGAssistant(store, llm_client=llm_client, llm_model=llm_model, answer_mode=answer_mode)
