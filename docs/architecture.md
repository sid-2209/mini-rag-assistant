# Architecture

## Flow

`document_loader.py`
Parses `.md` and `.txt` files, supporting YAML front matter, `Title:` / `Source:` headers, and filename fallbacks.

`chunking.py`
Splits each document into overlapping word windows and keeps stable chunk metadata for citations.

`vector_store.py`
Builds a local index from TF-IDF word and character n-gram vectors, persists it to `.rag_store`, retrieves chunks by cosine similarity, and applies two retrieval guardrails before answer generation:

- an absolute and relative score floor
- a keyword-overlap check against the question's meaningful terms

When nothing passes, it still preserves the top considered chunks so refusals can cite why the answer was blocked.

`answering.py`
Scores sentences inside retrieved chunks, requires sentence-level keyword overlap, selects the strongest supporting sentences, and refuses when the evidence is weak or off-topic.

Refusals use the same fixed message and include source citations, including zero-similarity cases.

`pipeline.py`
Connects ingestion, retrieval, and answer generation behind one small interface.

`cli.py`
Exposes `ingest`, `ask`, `chat`, and `evaluate`.

If the CLI needs to build an index and no document folder is supplied, it prompts for one interactively and warns when the folder does not match the expected 3-5 document evaluation setup.

## Why This Design

- Local retrieval keeps the project runnable without API keys.
- Sparse embeddings are simple, fast, and dependable for a small document set.
- Sentence-level answer construction reduces drift from retrieved evidence.
- Keyword-overlap filtering adds a lightweight hallucination-control layer without introducing a model dependency.
- Separate retrieval and generation modules make the logic easy to review and extend.
