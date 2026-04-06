# Architecture

## Flow

`document_loader.py`
Parses `.md` and `.txt` files, supporting YAML front matter, `Title:` / `Source:` headers, and filename fallbacks.

`chunking.py`
Splits each document into overlapping word windows and keeps stable chunk metadata for citations.

`ollama_client.py`
Wraps the local Ollama HTTP API for:

- dense embeddings
- grounded JSON answer generation
- lightweight health checks such as model listing

`vector_store.py`
Builds and persists a hybrid local index:

- dense chunk embeddings from Ollama
- lexical TF-IDF word and character n-gram features

At query time it combines dense and lexical scores, then applies:

- an absolute and relative score floor
- a keyword-overlap check against the question's meaningful terms

If nothing passes, it still preserves the best considered chunks so refusals can explain what was rejected.

`answering.py`
Scores sentences inside retrieved chunks, picks the strongest evidence snippets, and then:

- asks Ollama for a grounded JSON answer that cites evidence snippet IDs, or
- falls back to extractive composition if the LLM output is missing, invalid, over-refuses, or introduces unsupported content

The final answer is validated against the cited evidence so unsupported keywords or numbers are rejected before the user sees them.

`config.py`
Stores project-local defaults such as docs folder, index path, eval file, retrieval backend, and models inside `.mini-rag/settings.json`.

`cli.py`
Provides two layers of UX:

- a guided default menu via `mini-rag`
- explicit subcommands such as `setup`, `doctor`, `chat`, `ask`, `evaluate`, and `ingest`

The CLI can detect when Ollama is down, show the exact recovery command, ask for `y/yes/n/no`, and execute the fix automatically when the user agrees.

`pipeline.py`
Connects ingestion, retrieval, and grounded generation behind one small interface.

## Why This Design

- Hybrid retrieval improves recall over lexical-only TF-IDF while keeping exact-match behavior strong.
- Ollama keeps the whole stack local and avoids external search or hosted LLM dependencies.
- Sentence-level evidence selection narrows the context before generation, which reduces drift.
- JSON-only LLM output makes the grounding contract explicit and easy to validate.
- Extractive fallback keeps the assistant usable even when the local LLM is unavailable or misbehaves.
- Project-local saved defaults reduce command complexity for end users without removing scriptable explicit commands.
- The guided CLI follows progressive disclosure: simple default flows for humans, detailed flags for debugging and automation.
