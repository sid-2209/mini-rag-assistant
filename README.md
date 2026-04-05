# Mini RAG Assistant

A small Python CLI that answers questions only from local `.md` and `.txt` documents. It retrieves the most relevant chunks, applies explicit relevance thresholds, and refuses when evidence is missing or weak.

## Run It

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
mini-rag ingest data/sample_docs
mini-rag ask "When does payroll run?"
```

If you omit the document folder while building an index, the CLI will prompt for it. The evaluation setup expects a folder containing 3-5 `.txt` or `.md` files.

## Architecture

The pipeline is intentionally small:

1. Documents are loaded from a folder and parsed for `title`, `source`, and content.
2. Content is chunked with overlap.
3. Chunks are embedded with local TF-IDF vectors and stored in a persisted local index.
4. A question retrieves top chunks by cosine similarity.
5. Low-score results are filtered before answer generation.
6. The answer is built only from retrieved sentences and always includes citations.

More detail is in [`docs/architecture.md`](docs/architecture.md).

## Hallucination Control

- The assistant never uses external search or background knowledge.
- Retrieval uses both an absolute score threshold and a relative floor against the best match.
- Retrieved chunks must overlap the question's meaningful terms before they can be used for answering.
- Empty retrieval and low-confidence retrieval return the same refusal message.
- Refusals include the closest rejected chunk citations, including zero-match cases, so the decision is inspectable.
- The answer is composed only from sentences inside accepted chunks.
- Citations are always printed, including chunk numbers and retrieval scores.

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

Run the sample evaluation:

```bash
mini-rag evaluate data/eval/sample_eval.jsonl --docs-dir data/sample_docs
```

## Assumptions And Limitations

- Documents are plain text or Markdown and include either metadata lines or a filename that can serve as a fallback title.
- The default embeddings are sparse TF-IDF vectors, chosen for reliability and zero model-download overhead.
- Answer generation is extractive by design. It favors groundedness over polished abstraction.
- Rebuild the local index after documents change by running `mini-rag ingest` again or using `--rebuild`.

## With More Time

- Swap in a stronger dense embedding model behind the same interface.
- Add document-change detection and incremental re-indexing.
- Add richer sentence compression for more concise answers.
- Expand the evaluation suite with retrieval and refusal metrics.
