from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mini_rag_assistant.answering import REFUSAL_MESSAGE
from mini_rag_assistant.config import AssistantSettings
from mini_rag_assistant.document_loader import SUPPORTED_EXTENSIONS
from mini_rag_assistant.evaluation import load_evaluation_cases, run_evaluation
from mini_rag_assistant.ollama_client import OllamaError
from mini_rag_assistant.pipeline import build_index, load_assistant
from mini_rag_assistant.vector_store import LocalVectorStore


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, OllamaError) as exc:
        raise SystemExit(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    settings = AssistantSettings()
    parser = argparse.ArgumentParser(
        prog="mini-rag",
        description="Answer questions using only the provided local documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local vector index from a document folder.")
    ingest_parser.add_argument("docs_dir", nargs="?", help="Folder containing .md or .txt documents.")
    ingest_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    ingest_parser.add_argument("--chunk-size", type=int, default=140, help="Chunk size in words.")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=30, help="Chunk overlap in words.")
    _add_retrieval_backend_args(
        ingest_parser,
        settings=settings,
        include_answer_mode=False,
    )
    ingest_parser.set_defaults(func=_run_ingest)

    ask_parser = subparsers.add_parser("ask", help="Answer a single question.")
    ask_parser.add_argument("question", help="The user question.")
    ask_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    ask_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Maximum number of retrieved chunks.")
    ask_parser.add_argument("--min-score", type=float, default=0.15, help="Absolute hybrid relevance floor.")
    ask_parser.add_argument(
        "--relative-score-floor",
        type=float,
        default=0.55,
        help="Relative floor applied against the best retrieved score.",
    )
    ask_parser.add_argument("--rebuild", action="store_true", help="Rebuild the index before answering.")
    ask_parser.add_argument("--debug", action="store_true", help="Print retrieval details before the final answer.")
    _add_retrieval_backend_args(
        ask_parser,
        settings=settings,
        include_answer_mode=True,
    )
    ask_parser.set_defaults(func=_run_ask)

    chat_parser = subparsers.add_parser("chat", help="Run an interactive CLI session.")
    chat_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    chat_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.add_argument("--min-score", type=float, default=0.15)
    chat_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    chat_parser.add_argument("--rebuild", action="store_true")
    chat_parser.add_argument("--debug", action="store_true")
    _add_retrieval_backend_args(
        chat_parser,
        settings=settings,
        include_answer_mode=True,
    )
    chat_parser.set_defaults(func=_run_chat)

    eval_parser = subparsers.add_parser("evaluate", help="Run a lightweight evaluation set.")
    eval_parser.add_argument("eval_file", help="Path to a JSONL evaluation file.")
    eval_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    eval_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--min-score", type=float, default=0.15)
    eval_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    eval_parser.add_argument("--rebuild", action="store_true")
    _add_retrieval_backend_args(
        eval_parser,
        settings=settings,
        include_answer_mode=True,
    )
    eval_parser.set_defaults(func=_run_evaluate)

    return parser


def _add_retrieval_backend_args(
    parser: argparse.ArgumentParser,
    *,
    settings: AssistantSettings,
    include_answer_mode: bool,
) -> None:
    parser.add_argument(
        "--embedding-backend",
        choices=("ollama", "tfidf"),
        default=settings.embedding_backend,
        help="Retrieval embedding backend used when building or rebuilding the index.",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help="Embedding model used when --embedding-backend=ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default=settings.ollama_host,
        help="Base URL for the local Ollama server.",
    )
    if include_answer_mode:
        parser.add_argument(
            "--answer-mode",
            choices=("ollama", "extractive"),
            default=settings.answer_mode,
            help="Answer generation strategy after retrieval.",
        )
        parser.add_argument(
            "--llm-model",
            default=settings.llm_model,
            help="Ollama model used for grounded answer generation when --answer-mode=ollama.",
        )


def _run_ingest(args: argparse.Namespace) -> None:
    docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="build the index")
    args.docs_dir = docs_dir
    manifest = build_index(
        docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    print(f"Indexed {manifest.get('document_count', 0)} documents into {Path(args.index_dir).resolve()}")
    print(f"Chunks: {manifest.get('chunk_count', 0)}")
    print(
        "Retrieval: "
        f"{manifest.get('embedding_backend', 'tfidf')}"
        + (
            f" ({manifest.get('embedding_model')})"
            if manifest.get("embedding_model")
            else ""
        )
    )


def _run_ask(args: argparse.Namespace) -> None:
    assistant = _ensure_assistant(args)
    answer, retrieval = assistant.answer(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        relative_score_floor=args.relative_score_floor,
    )

    if args.debug:
        _print_debug(retrieval, args)

    _print_answer(answer.answer, answer.citations, answer.confidence)


def _run_chat(args: argparse.Namespace) -> None:
    assistant = _ensure_assistant(args)
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            print()
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return

        answer, retrieval = assistant.answer(
            question,
            top_k=args.top_k,
            min_score=args.min_score,
            relative_score_floor=args.relative_score_floor,
        )
        if args.debug:
            _print_debug(retrieval, args)
        _print_answer(answer.answer, answer.citations, answer.confidence)


def _run_evaluate(args: argparse.Namespace) -> None:
    assistant = _ensure_assistant(args)
    cases = load_evaluation_cases(args.eval_file)
    summary = run_evaluation(
        assistant,
        cases,
        top_k=args.top_k,
        min_score=args.min_score,
        relative_score_floor=args.relative_score_floor,
    )
    print(json.dumps(summary, indent=2))


def _ensure_assistant(args: argparse.Namespace):
    if args.rebuild:
        docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="rebuild the index")
        args.docs_dir = docs_dir
        build_index(
            docs_dir,
            index_dir=args.index_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            ollama_host=args.ollama_host,
        )

    elif not LocalVectorStore.exists(args.index_dir):
        docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="build the index")
        args.docs_dir = docs_dir
        build_index(
            docs_dir,
            index_dir=args.index_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            ollama_host=args.ollama_host,
        )

    return load_assistant(
        args.index_dir,
        answer_mode=getattr(args, "answer_mode", "extractive"),
        llm_model=getattr(args, "llm_model", "llama3.2:3b"),
        ollama_host=args.ollama_host,
    )


def _resolve_docs_dir_for_build(docs_dir: str | None, *, action_label: str) -> str:
    if docs_dir:
        _warn_if_unexpected_doc_count(docs_dir)
        return docs_dir

    if not sys.stdin.isatty():
        raise SystemExit(
            f"A document folder is required to {action_label}. Pass --docs-dir or run the CLI interactively."
        )

    prompted_docs_dir = input("Documents folder (expected 3-5 .txt/.md files)> ").strip()
    if not prompted_docs_dir:
        raise SystemExit("A document folder is required.")

    _warn_if_unexpected_doc_count(prompted_docs_dir)
    return prompted_docs_dir


def _warn_if_unexpected_doc_count(docs_dir: str) -> None:
    root = Path(docs_dir).expanduser()
    if not root.exists():
        return

    doc_count = sum(
        1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if 3 <= doc_count <= 5:
        return

    print(
        f"Note: the evaluation setup expects 3-5 documents; found {doc_count} supported files in {root.resolve()}.",
        file=sys.stderr,
    )


def _print_answer(answer: str, citations, confidence: float) -> None:
    print("\nAnswer:")
    print(answer)
    print("\nSources:")

    if not citations:
        print("- No supporting chunks were found.")
    else:
        for citation in citations:
            details = f"chunk {citation.chunk_index}, score {citation.score:.3f}"
            if citation.note:
                details = f"{details}; {citation.note}"
            print(f"- {citation.title} — {citation.source} ({details})")

    print(f"\nConfidence: {confidence:.3f}")
    if answer == REFUSAL_MESSAGE:
        print("Status: refused")


def _print_debug(retrieval, args: argparse.Namespace) -> None:
    print("\nRetrieval Debug:")
    print(f"- top score: {retrieval.top_score:.3f}")
    print(f"- applied floor: {retrieval.applied_floor:.3f}")
    print(f"- confidence: {retrieval.confidence:.3f}")
    print(f"- answer mode: {getattr(args, 'answer_mode', 'extractive')}")
    print(f"- embedding backend: {args.embedding_backend}")
    if args.embedding_backend == "ollama":
        print(f"- embedding model: {args.embedding_model}")
    if getattr(args, "answer_mode", "extractive") == "ollama":
        print(f"- llm model: {args.llm_model}")
    if retrieval.refusal_reason:
        print(f"- refusal reason: {retrieval.refusal_reason}")
    chunks_to_show = retrieval.retrieved_chunks or retrieval.considered_chunks
    if chunks_to_show:
        for item in chunks_to_show:
            preview = item.chunk.text[:140].strip()
            print(
                f"- {item.chunk.title} / chunk {item.chunk.chunk_index}: "
                f"score={item.score:.3f}, dense={item.dense_score:.3f}, lexical={item.lexical_score:.3f} :: {preview}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
