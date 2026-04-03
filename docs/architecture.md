# Architecture

## Flow

`document_loader.py`
Parses `.md` and `.txt` files, supporting YAML front matter, `Title:` / `Source:` headers, and filename fallbacks.

`chunking.py`
Splits each document into overlapping word windows and keeps stable chunk metadata for citations.

`vector_store.py`
Builds a local index from TF-IDF word and character n-gram vectors, persists it to `.rag_store`, and retrieves chunks by cosine similarity.

`answering.py`
Scores sentences inside retrieved chunks, selects the strongest supporting sentences, and refuses when the evidence is weak.

`pipeline.py`
Connects ingestion, retrieval, and answer generation behind one small interface.

`cli.py`
Exposes `ingest`, `ask`, `chat`, and `evaluate`.

## Why This Design

- Local retrieval keeps the project runnable without API keys.
- Sparse embeddings are simple, fast, and dependable for a small document set.
- Sentence-level answer construction reduces drift from retrieved evidence.
- Separate retrieval and generation modules make the logic easy to review and extend.

