from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mini_rag_assistant.answering import REFUSAL_MESSAGE
from mini_rag_assistant.evaluation import load_evaluation_cases, run_evaluation
from mini_rag_assistant.pipeline import build_index, load_assistant
from mini_rag_assistant.vector_store import LocalVectorStore


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-rag",
        description="Answer questions using only the provided local documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local vector index from a document folder.")
    ingest_parser.add_argument("docs_dir", help="Folder containing .md or .txt documents.")
    ingest_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    ingest_parser.add_argument("--chunk-size", type=int, default=140, help="Chunk size in words.")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=30, help="Chunk overlap in words.")
    ingest_parser.set_defaults(func=_run_ingest)

    ask_parser = subparsers.add_parser("ask", help="Answer a single question.")
    ask_parser.add_argument("question", help="The user question.")
    ask_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    ask_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Maximum number of retrieved chunks.")
    ask_parser.add_argument("--min-score", type=float, default=0.15, help="Absolute cosine-similarity floor.")
    ask_parser.add_argument(
        "--relative-score-floor",
        type=float,
        default=0.55,
        help="Relative floor applied against the best retrieved score.",
    )
    ask_parser.add_argument("--rebuild", action="store_true", help="Rebuild the index before answering.")
    ask_parser.add_argument("--debug", action="store_true", help="Print retrieval details before the final answer.")
    ask_parser.set_defaults(func=_run_ask)

    chat_parser = subparsers.add_parser("chat", help="Run an interactive CLI session.")
    chat_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    chat_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    chat_parser.add_argument("--top-k", type=int, default=4)
    chat_parser.add_argument("--min-score", type=float, default=0.15)
    chat_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    chat_parser.add_argument("--rebuild", action="store_true")
    chat_parser.add_argument("--debug", action="store_true")
    chat_parser.set_defaults(func=_run_chat)

    eval_parser = subparsers.add_parser("evaluate", help="Run a lightweight evaluation set.")
    eval_parser.add_argument("eval_file", help="Path to a JSONL evaluation file.")
    eval_parser.add_argument("--docs-dir", help="Document folder. Required when creating or rebuilding the index.")
    eval_parser.add_argument("--index-dir", default=".rag_store", help="Directory where the local index is stored.")
    eval_parser.add_argument("--top-k", type=int, default=4)
    eval_parser.add_argument("--min-score", type=float, default=0.15)
    eval_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    eval_parser.add_argument("--rebuild", action="store_true")
    eval_parser.set_defaults(func=_run_evaluate)

    return parser


def _run_ingest(args: argparse.Namespace) -> None:
    manifest = build_index(
        args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Indexed {manifest.get('document_count', 0)} documents into {Path(args.index_dir).resolve()}")
    print(f"Chunks: {manifest.get('chunk_count', 0)}")


def _run_ask(args: argparse.Namespace) -> None:
    assistant = _ensure_assistant(args)
    answer, retrieval = assistant.answer(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        relative_score_floor=args.relative_score_floor,
    )

    if args.debug:
        _print_debug(retrieval)

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
            _print_debug(retrieval)
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
        if not args.docs_dir:
            raise SystemExit("--docs-dir is required when using --rebuild.")
        build_index(args.docs_dir, index_dir=args.index_dir)

    elif not LocalVectorStore.exists(args.index_dir):
        if not args.docs_dir:
            raise SystemExit(
                f"No index found in {Path(args.index_dir).resolve()}. Run ingest first or pass --docs-dir."
            )
        build_index(args.docs_dir, index_dir=args.index_dir)

    return load_assistant(args.index_dir)


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


def _print_debug(retrieval) -> None:
    print("\nRetrieval Debug:")
    print(f"- top score: {retrieval.top_score:.3f}")
    print(f"- applied floor: {retrieval.applied_floor:.3f}")
    print(f"- confidence: {retrieval.confidence:.3f}")
    if retrieval.refusal_reason:
        print(f"- refusal reason: {retrieval.refusal_reason}")
    if retrieval.retrieved_chunks:
        for item in retrieval.retrieved_chunks:
            preview = item.chunk.text[:140].strip()
            print(
                f"- {item.chunk.title} / chunk {item.chunk.chunk_index}: "
                f"score={item.score:.3f} :: {preview}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
