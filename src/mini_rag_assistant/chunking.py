from __future__ import annotations

from mini_rag_assistant.types import Chunk, Document


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 140,
    chunk_overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        words = document.content.split()
        if not words:
            continue

        start = 0
        chunk_index = 1
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_text = " ".join(words[start:end]).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source=document.source,
                    path=document.path,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    word_start=start,
                    word_end=end,
                )
            )

            if end >= len(words):
                break

            start = end - chunk_overlap
            chunk_index += 1

    return chunks

