# Mini RAG Assistant

A small Python CLI RAG assistant that answers questions only from local `.md` and `.txt` documents. It uses hybrid retrieval, grounded local answer generation through Ollama, explicit refusal handling, and source citations on every answer.

## Run It

### 1. Start Ollama

Make sure the local Ollama server is running and the required models are available:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 2. Install the project

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 3. Build the index

```bash
mini-rag ingest data/sample_docs
```

If you omit the document folder while building an index, the CLI prompts for one. The intended evaluation flow expects a folder containing 3-5 `.txt` or `.md` files.

### 4. Ask questions

```bash
mini-rag ask "When does payroll run?" --debug
mini-rag chat --debug
mini-rag evaluate data/eval/sample_eval.jsonl --docs-dir data/sample_docs
```

### Optional fallback

If you want to skip Ollama for retrieval or tests, build with lexical-only retrieval:

```bash
mini-rag ingest data/sample_docs --embedding-backend tfidf
mini-rag ask "When does payroll run?" --answer-mode extractive
```

## Architecture

The pipeline is intentionally small but now uses a stronger RAG flow:

1. Documents are loaded from a folder and parsed for `title`, `source`, and content.
2. Content is chunked with overlap.
3. Chunks are indexed with hybrid retrieval data:
   - dense embeddings from Ollama (`nomic-embed-text` by default)
   - lexical TF-IDF features for exact-match support
4. A question is embedded and scored with hybrid dense + lexical retrieval.
5. Low-score or keyword-mismatched results are filtered before answer generation.
6. High-signal evidence sentences are selected from retrieved chunks.
7. An Ollama LLM (`llama3.2:3b` by default) generates a JSON answer using only those evidence snippets.
8. The answer is validated against the cited evidence before it is returned.

More detail is in [`docs/architecture.md`](docs/architecture.md).

## Hallucination Control

- The assistant never uses external search or background knowledge.
- Retrieval uses both an absolute score threshold and a relative floor against the best match.
- Retrieved chunks must overlap the question's meaningful terms before they can be used.
- Empty retrieval and low-confidence retrieval produce the same fixed refusal message.
- The LLM sees only selected evidence snippets, not the whole corpus.
- LLM output must be valid JSON and must cite the evidence snippet IDs it used.
- Answers are rejected if they introduce unsupported keywords or numbers that are absent from the cited evidence.
- If the Ollama answer is invalid, the system falls back to a strict extractive answer instead of guessing.
- Refusals still include the closest rejected chunk citations so the decision remains inspectable.

## CLI

Build an index:

```bash
mini-rag ingest /path/to/documents
```

Or let the CLI prompt you:

```bash
mini-rag ingest
```

Ask one question:

```bash
mini-rag ask "What date is Republic Day celebrated in India?" --docs-dir /path/to/documents
```

Run interactive mode:

```bash
mini-rag chat --docs-dir /path/to/documents
```

Show retrieval diagnostics:

```bash
mini-rag ask "When does payroll run?" --docs-dir /path/to/documents --debug
```

Switch models:

```bash
mini-rag ingest /path/to/documents --embedding-model nomic-embed-text
mini-rag ask "When does payroll run?" --llm-model llama3.2:3b
```

## Assumptions And Limitations

- Documents are plain text or Markdown and should include either explicit metadata or filenames that can serve as fallbacks.
- The default setup assumes a local Ollama server is available.
- The answer generator is grounded, but still intentionally conservative. It may refuse borderline questions rather than risk unsupported synthesis.
- The index is rebuilt when documents change; incremental re-indexing is not implemented yet.
- The hybrid retriever is optimized for small document collections, not very large corpora.

## With More Time

- Add a cross-encoder or reranker pass for even tighter evidence ordering.
- Cache query embeddings and generation traces for repeated evaluations.
- Add automatic model detection and friendlier setup checks for Ollama.
- Expand the evaluation suite with retrieval recall, citation accuracy, and grounding regression cases.
- Support multiple local backends behind the same answer-generation interface.
