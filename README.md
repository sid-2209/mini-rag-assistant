# Mini RAG Assistant

A small local RAG CLI that answers questions only from `.md` and `.txt` documents. It combines hybrid retrieval, grounded Ollama answer generation, explicit refusal handling, and source citations on every answer.

## Best Way To Use It

The CLI is designed to be easy to remember:

```bash
mini-rag
```

That opens a guided terminal menu with simple commands shown next to each option:

- `mini-rag setup`
- `mini-rag chat`
- `mini-rag ask`
- `mini-rag evaluate`
- `mini-rag doctor`

The menu and setup flow save your defaults under `.mini-rag/settings.json`, so you do not need to keep retyping long commands.

## Quick Start

### 1. Install the project

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 2. Start the assistant

```bash
mini-rag
```

Then choose:

```text
1. Guided Setup              (mini-rag setup)
2. Chat With Documents       (mini-rag chat)
3. Ask One Question          (mini-rag ask)
4. Run Evaluation            (mini-rag evaluate)
5. System Doctor             (mini-rag doctor)
6. Exit                      (exit)
```

### 3. Let setup do the heavy lifting

`mini-rag setup` will:

- check whether `ollama` is installed
- detect whether the Ollama server is down
- show the exact command needed to start it
- ask `y/yes/n/no` whether it should run that command automatically
- verify required models such as `nomic-embed-text` and `llama3.2:3b`
- prompt for the documents folder if needed
- build the local index
- save the working defaults for future runs

## Simple Commands

After setup, the short commands are enough:

```bash
mini-rag chat
mini-rag ask
mini-rag evaluate
mini-rag doctor
```

Examples:

```bash
mini-rag ask
mini-rag ask "When does payroll run?"
mini-rag chat --debug
mini-rag evaluate
mini-rag doctor --fix
```

`mini-rag ask` prompts for the question if you omit it.

`mini-rag evaluate` uses the saved evaluation file or the sample evaluation file if one is available.

## Advanced Commands

The explicit command-line flags are still supported for power users and scripts:

```bash
mini-rag ingest /path/to/documents --embedding-model nomic-embed-text
mini-rag ask "When does payroll run?" --docs-dir /path/to/documents --index-dir .rag_store --debug
mini-rag chat --docs-dir /path/to/documents --answer-mode extractive
mini-rag evaluate data/eval/sample_eval.jsonl --docs-dir data/sample_docs
```

## Architecture

The current pipeline is:

1. Load documents from a folder and parse `title`, `source`, and content.
2. Chunk content with overlap.
3. Build a hybrid local index:
   - dense embeddings from Ollama (`nomic-embed-text` by default)
   - lexical TF-IDF features for exact-match support
4. Retrieve top chunks with hybrid dense + lexical scoring.
5. Apply hallucination controls:
   - absolute score threshold
   - relative score floor
   - keyword overlap checks
6. Select high-signal evidence sentences from retrieved chunks.
7. Ask an Ollama LLM (`llama3.2:3b` by default) for a grounded JSON answer using only those snippets.
8. Validate the answer against the cited evidence before returning it.
9. Fall back to strict extractive composition if the LLM output is weak, invalid, or unsupported.

More detail is in [`docs/architecture.md`](docs/architecture.md).

## Hallucination Control

- The assistant never uses external search or background knowledge.
- Empty retrieval and low-confidence retrieval both produce the same fixed refusal message.
- Retrieved chunks must overlap the question's meaningful terms before they are used.
- The LLM only sees filtered evidence snippets, not the whole corpus.
- LLM output must be valid JSON and cite evidence snippet IDs.
- Answers are rejected if they introduce unsupported keywords or numbers.
- When the LLM fails or over-refuses, the system falls back to a stricter extractive answer instead of guessing.
- Refusals still cite the closest rejected chunks so the decision is inspectable.

## Assumptions And Limitations

- Documents are plain text or Markdown and should include explicit metadata or filenames that can act as fallbacks.
- The default path assumes a local Ollama installation.
- The assistant is intentionally conservative and may refuse borderline questions rather than invent unsupported detail.
- The index is rebuilt when documents change; incremental re-indexing is not implemented yet.
- The hybrid retriever is designed for small local collections, not very large corpora.

## With More Time

- Add a reranker or cross-encoder pass for even tighter evidence ordering.
- Add richer citation auditing and grounding regression tests.
- Add incremental indexing so document changes do not require a full rebuild.
- Add multiple local provider backends behind the same guided CLI.
- Add richer accessibility affordances for long-running terminal operations.
