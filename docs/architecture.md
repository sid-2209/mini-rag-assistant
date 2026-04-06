# Architecture

## Flow

`document_loader.py`
Parses `.md` and `.txt` files, supporting YAML front matter, `Title:` / `Source:` headers, and filename fallbacks.

`chunking.py`
Splits each document into overlapping word windows and keeps stable chunk metadata for citations.

`ollama_client.py`
Wraps the local Ollama HTTP API for embeddings and grounded JSON generation.

`vector_store.py`
Builds and persists a hybrid local index:

- dense chunk embeddings from Ollama
- lexical TF-IDF word and character n-gram features

At query time it combines dense and lexical scores, applies:

- an absolute and relative score floor
- a keyword-overlap check against the question's meaningful terms

When nothing passes, it still preserves the top considered chunks so refusals can cite why the answer was blocked.

`answering.py`
Scores sentences inside retrieved chunks, picks the strongest evidence snippets, and then:

- asks Ollama for a JSON answer that cites evidence snippet IDs, or
- falls back to extractive composition if the LLM output is missing, invalid, or unsupported

The final answer is validated against the cited evidence so unsupported keywords or numbers are rejected before the user sees them.

`pipeline.py`
Connects ingestion, retrieval, and grounded generation behind one small interface.

`cli.py`
Exposes `ingest`, `ask`, `chat`, and `evaluate`, plus knobs for embedding backend, Ollama host, and answer mode.

If the CLI needs to build an index and no document folder is supplied, it prompts for one interactively and warns when the folder does not match the expected 3-5 document evaluation setup.

## Why This Design

- Hybrid retrieval improves recall over lexical-only TF-IDF while preserving exact-match behavior.
- Ollama keeps the whole stack local and avoids external search or hosted LLM dependencies.
- Sentence-level evidence selection narrows the context before generation, which reduces drift.
- JSON-only LLM output makes the grounding contract explicit and easy to validate.
- Extractive fallback keeps the assistant usable even when the local LLM is unavailable or produces weak output.
- Separate retrieval and generation modules keep the logic readable and testable.
