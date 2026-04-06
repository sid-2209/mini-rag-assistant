from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mini_rag_assistant.answering import REFUSAL_MESSAGE
from mini_rag_assistant.config import AssistantSettings
from mini_rag_assistant.document_loader import SUPPORTED_EXTENSIONS, fingerprint_documents
from mini_rag_assistant.evaluation import load_evaluation_cases, run_evaluation
from mini_rag_assistant.ollama_client import OllamaClient, OllamaError
from mini_rag_assistant.pipeline import build_index, load_assistant
from mini_rag_assistant.vector_store import LocalVectorStore, load_manifest

YES_ANSWERS = {"y", "yes"}
NO_ANSWERS = {"n", "no"}
MENU_ITEMS = [
    ("1", "Guided Setup", "mini-rag setup"),
    ("2", "Chat With Documents", "mini-rag chat"),
    ("3", "Ask One Question", "mini-rag ask"),
    ("4", "Run Evaluation", "mini-rag evaluate"),
    ("5", "System Doctor", "mini-rag doctor"),
    ("6", "Exit", "exit"),
]


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _hydrate_args(args)
        args.func(args)
    except KeyboardInterrupt as exc:
        raise SystemExit("\nCancelled.") from exc
    except (FileNotFoundError, ValueError, OllamaError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    settings = AssistantSettings.load()
    parser = argparse.ArgumentParser(
        prog="mini-rag",
        description="Answer questions using only the provided local documents.",
    )
    parser.set_defaults(func=_run_menu)
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="Guided setup for the project.")
    setup_parser.add_argument("docs_dir", nargs="?", default=settings.docs_dir, help="Folder containing .md or .txt documents.")
    setup_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
    setup_parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Chunk size in words.")
    setup_parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap, help="Chunk overlap in words.")
    setup_parser.add_argument("--rebuild", action="store_true", help="Force a rebuild of the local index.")
    _add_retrieval_backend_args(setup_parser, settings=settings, include_answer_mode=True)
    setup_parser.set_defaults(func=_run_setup)

    doctor_parser = subparsers.add_parser("doctor", help="Check project health and suggest fixes.")
    doctor_parser.add_argument("--docs-dir", default=settings.docs_dir, help="Saved or active document folder.")
    doctor_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
    doctor_parser.add_argument("--eval-file", default=settings.eval_file, help="Evaluation file to check.")
    doctor_parser.add_argument("--fix", action="store_true", help="Offer interactive fixes when possible.")
    _add_retrieval_backend_args(doctor_parser, settings=settings, include_answer_mode=True)
    doctor_parser.set_defaults(func=_run_doctor)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local vector index from a document folder.")
    ingest_parser.add_argument("docs_dir", nargs="?", default=settings.docs_dir, help="Folder containing .md or .txt documents.")
    ingest_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
    ingest_parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Chunk size in words.")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap, help="Chunk overlap in words.")
    _add_retrieval_backend_args(ingest_parser, settings=settings, include_answer_mode=False)
    ingest_parser.set_defaults(func=_run_ingest)

    ask_parser = subparsers.add_parser("ask", help="Answer a single question.")
    ask_parser.add_argument("question", nargs="?", help="The user question.")
    ask_parser.add_argument("--docs-dir", default=settings.docs_dir, help="Document folder. Required when creating or rebuilding the index.")
    ask_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
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
    _add_retrieval_backend_args(ask_parser, settings=settings, include_answer_mode=True)
    ask_parser.set_defaults(func=_run_ask)

    chat_parser = subparsers.add_parser("chat", help="Run an interactive CLI session.")
    chat_parser.add_argument("--docs-dir", default=settings.docs_dir, help="Document folder. Required when creating or rebuilding the index.")
    chat_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.add_argument("--min-score", type=float, default=0.15)
    chat_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    chat_parser.add_argument("--rebuild", action="store_true")
    chat_parser.add_argument("--debug", action="store_true")
    _add_retrieval_backend_args(chat_parser, settings=settings, include_answer_mode=True)
    chat_parser.set_defaults(func=_run_chat)

    eval_parser = subparsers.add_parser("evaluate", help="Run a lightweight evaluation set.")
    eval_parser.add_argument("eval_file", nargs="?", default=settings.eval_file, help="Path to a JSONL evaluation file.")
    eval_parser.add_argument("--docs-dir", default=settings.docs_dir, help="Document folder. Required when creating or rebuilding the index.")
    eval_parser.add_argument("--index-dir", default=settings.index_dir, help="Directory where the local index is stored.")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--min-score", type=float, default=0.15)
    eval_parser.add_argument("--relative-score-floor", type=float, default=0.55)
    eval_parser.add_argument("--rebuild", action="store_true")
    _add_retrieval_backend_args(eval_parser, settings=settings, include_answer_mode=True)
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


def _run_menu(args: argparse.Namespace) -> None:
    if not sys.stdin.isatty():
        raise SystemExit("No command provided. Run `mini-rag --help` or use `mini-rag` in an interactive terminal.")

    while True:
        settings = AssistantSettings.load()
        _print_header("Mini RAG Assistant")
        _print_saved_defaults(settings)
        print("Menu")
        print("----")
        for key, label, command in MENU_ITEMS:
            print(f"{key}. {label:<24} ({command})")
        print()
        choice = input("Choose an option> ").strip().lower()
        if choice in {"6", "exit", "quit", "q"}:
            return

        action = {
            "1": "setup",
            "setup": "setup",
            "2": "chat",
            "chat": "chat",
            "3": "ask",
            "ask": "ask",
            "4": "evaluate",
            "evaluate": "evaluate",
            "5": "doctor",
            "doctor": "doctor",
        }.get(choice)
        if action is None:
            print("Please choose 1, 2, 3, 4, 5, or 6.")
            print()
            continue

        command_args = _make_menu_namespace(action, settings)
        if action == "doctor":
            command_args.fix = True
        if action == "setup":
            _run_setup(command_args)
        elif action == "chat":
            _run_chat(command_args)
        elif action == "ask":
            _run_ask(command_args)
        elif action == "evaluate":
            _run_evaluate(command_args)
        elif action == "doctor":
            _run_doctor(command_args)
        _pause_for_menu_return()


def _run_setup(args: argparse.Namespace) -> None:
    _hydrate_args(args)
    _print_header("Project Setup")
    manifest = _load_manifest_if_exists(args.index_dir)
    if not args.docs_dir and isinstance(manifest.get("docs_dir"), str):
        args.docs_dir = str(manifest["docs_dir"])

    args.docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="set up the assistant")
    _print_status("OK", "Documents", str(Path(args.docs_dir).expanduser().resolve()))
    _warn_if_unexpected_doc_count(args.docs_dir)

    _ensure_ollama_requirements(args, manifest=None, building=True, allow_fix=True)
    manifest = _ensure_index_ready(args, sync_with_current_config=True, allow_fix=True)
    _print_ingest_summary(manifest, args)
    _save_settings_from_args(args)

    print()
    _print_status("OK", "Setup complete", "Your project is ready.")
    print("Next commands")
    print("-------------")
    print("- Chat with your docs: mini-rag chat")
    print("- Ask one question: mini-rag ask")
    print("- Run a health check: mini-rag doctor")


def _run_doctor(args: argparse.Namespace) -> None:
    _hydrate_args(args)
    _print_header("System Doctor")
    allow_fix = bool(args.fix and sys.stdin.isatty())
    settings_path = AssistantSettings.config_path()
    manifest = _load_manifest_if_exists(args.index_dir)

    _print_status(
        "OK" if settings_path.exists() else "WARN",
        "Saved settings",
        str(settings_path) if settings_path.exists() else f"No saved settings yet. Run `mini-rag setup`.",
    )

    ollama_binary = shutil.which("ollama")
    if ollama_binary:
        _print_status("OK", "Ollama CLI", ollama_binary)
    else:
        _print_status("ERROR", "Ollama CLI", "Not installed or not on PATH.")

    try:
        if _requires_ollama(args, manifest=manifest, building=False):
            _ensure_ollama_requirements(args, manifest=manifest, building=False, allow_fix=allow_fix)
            _print_status("OK", "Ollama runtime", "Server reachable and required models are available.")
        else:
            _print_status("OK", "Ollama runtime", "Current configuration does not require Ollama.")
    except SystemExit as exc:
        _print_status("WARN", "Ollama runtime", str(exc))

    docs_dir = _resolve_existing_docs_dir(args.docs_dir, manifest)
    if docs_dir:
        doc_count = _supported_doc_count(docs_dir)
        status = "OK" if 3 <= doc_count <= 5 else "WARN"
        _print_status(status, "Documents", f"{docs_dir} ({doc_count} supported files)")
    else:
        _print_status("WARN", "Documents", "No saved documents folder. Run `mini-rag setup`.")

    if args.eval_file and Path(args.eval_file).expanduser().exists():
        _print_status("OK", "Evaluation file", str(Path(args.eval_file).expanduser().resolve()))
    else:
        _print_status("WARN", "Evaluation file", "No evaluation file configured.")

    if LocalVectorStore.exists(args.index_dir):
        detail = str(Path(args.index_dir).expanduser().resolve())
        if isinstance(manifest.get("embedding_backend"), str):
            detail += f" [{manifest.get('embedding_backend')}"
            if manifest.get("embedding_model"):
                detail += f": {manifest.get('embedding_model')}"
            detail += "]"
        _print_status("OK", "Index", detail)
    else:
        _print_status("WARN", "Index", f"Missing local index at {Path(args.index_dir).expanduser().resolve()}")

    print()
    print("Recommended next step: mini-rag setup" if not LocalVectorStore.exists(args.index_dir) else "Recommended next step: mini-rag chat")


def _run_ingest(args: argparse.Namespace) -> None:
    _hydrate_args(args)
    _print_header("Build Index")
    args.docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="build the index")
    _print_status("OK", "Documents", str(Path(args.docs_dir).expanduser().resolve()))
    _warn_if_unexpected_doc_count(args.docs_dir)
    _ensure_ollama_requirements(args, manifest=None, building=True, allow_fix=sys.stdin.isatty())
    manifest = _build_index_with_feedback(args)
    _save_settings_from_args(args)
    _print_ingest_summary(manifest, args)


def _run_ask(args: argparse.Namespace) -> None:
    _hydrate_args(args)
    if not args.question:
        if not sys.stdin.isatty():
            raise SystemExit("A question is required. Pass one on the command line or run `mini-rag ask` interactively.")
        args.question = _prompt_text("Question> ", allow_blank=False)

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
    _save_settings_from_args(args)


def _run_chat(args: argparse.Namespace) -> None:
    _hydrate_args(args)
    assistant = _ensure_assistant(args)
    docs_dir = _resolve_existing_docs_dir(args.docs_dir, _load_manifest_if_exists(args.index_dir))
    print("Chat Mode")
    print("---------")
    if docs_dir:
        print(f"Documents: {docs_dir}")
    print("Type `exit` or `quit` to stop.")
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
    _hydrate_args(args)
    if not args.eval_file:
        if not sys.stdin.isatty():
            raise SystemExit("An evaluation file is required. Pass one on the command line or run interactively.")
        args.eval_file = _prompt_text("Evaluation file> ", allow_blank=False)

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
    _save_settings_from_args(args)


def _ensure_assistant(args: argparse.Namespace):
    _hydrate_args(args)
    manifest = _load_manifest_if_exists(args.index_dir)
    _ensure_ollama_requirements(args, manifest=manifest, building=False, allow_fix=sys.stdin.isatty())
    manifest = _ensure_index_ready(args, sync_with_current_config=False, allow_fix=sys.stdin.isatty())
    _save_settings_from_args(args)
    return load_assistant(
        args.index_dir,
        answer_mode=getattr(args, "answer_mode", "extractive"),
        llm_model=getattr(args, "llm_model", "llama3.2:3b"),
        ollama_host=args.ollama_host,
    )


def _ensure_index_ready(
    args: argparse.Namespace,
    *,
    sync_with_current_config: bool,
    allow_fix: bool,
) -> dict[str, object]:
    manifest = _load_manifest_if_exists(args.index_dir)
    index_exists = LocalVectorStore.exists(args.index_dir)
    resolved_docs_dir = _resolve_existing_docs_dir(args.docs_dir, manifest)

    if resolved_docs_dir and args.docs_dir != resolved_docs_dir:
        args.docs_dir = resolved_docs_dir
    elif not args.docs_dir and isinstance(manifest.get("docs_dir"), str):
        args.docs_dir = str(manifest["docs_dir"])

    rebuild_reason: str | None = None
    if args.rebuild:
        rebuild_reason = "Rebuild requested."
    elif not index_exists:
        rebuild_reason = "Local index is missing."
    elif resolved_docs_dir and isinstance(manifest.get("docs_dir"), str):
        selected_docs = str(Path(resolved_docs_dir).expanduser().resolve())
        if selected_docs != manifest.get("docs_dir"):
            rebuild_reason = "Documents folder changed."
        elif _documents_changed_since_last_build(selected_docs, manifest):
            rebuild_reason = "Documents changed."

    if sync_with_current_config and index_exists:
        if manifest.get("embedding_backend") != args.embedding_backend:
            rebuild_reason = "Embedding backend changed."
        elif args.embedding_backend == "ollama" and manifest.get("embedding_model") != args.embedding_model:
            rebuild_reason = "Embedding model changed."

    if rebuild_reason:
        args.docs_dir = _resolve_docs_dir_for_build(args.docs_dir, action_label="build the index")
        _warn_if_unexpected_doc_count(args.docs_dir)
        _ensure_ollama_requirements(args, manifest=None, building=True, allow_fix=allow_fix)
        print()
        _print_status("RUN", "Index", rebuild_reason)
        manifest = _build_index_with_feedback(args)
    else:
        if index_exists:
            _print_status("OK", "Index", str(Path(args.index_dir).expanduser().resolve()))
        else:
            raise SystemExit("A document folder is required to build the local index.")

    if not args.docs_dir and isinstance(manifest.get("docs_dir"), str):
        args.docs_dir = str(manifest["docs_dir"])
    return manifest


def _build_index_with_feedback(args: argparse.Namespace) -> dict[str, object]:
    manifest = build_index(
        args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    return manifest


def _ensure_ollama_requirements(
    args: argparse.Namespace,
    *,
    manifest: dict[str, object] | None,
    building: bool,
    allow_fix: bool,
) -> None:
    if not _requires_ollama(args, manifest=manifest, building=building):
        return

    binary = shutil.which("ollama")
    if not binary:
        raise SystemExit("Ollama is required for the current configuration, but the `ollama` command was not found.")

    client = OllamaClient(host=args.ollama_host)
    if not client.is_available():
        print()
        _print_status("WARN", "Ollama server", f"Not reachable at {args.ollama_host}")
        start_command = ["ollama", "serve"]
        print("Suggested command:")
        print(f"  {_format_command(start_command)}")
        if allow_fix and _prompt_yes_no("Run this now in the background? [y/n]> "):
            log_path = _start_ollama_service(binary, args.ollama_host)
            _print_status("OK", "Ollama server", f"Started. Logs: {log_path}")
        else:
            raise SystemExit("Ollama must be running to continue with the current configuration.")

    available_models = set(client.list_models())
    for model in _required_models(args, manifest=manifest, building=building):
        if _model_is_available(model, available_models):
            continue

        print()
        _print_status("WARN", "Missing model", model)
        pull_command = ["ollama", "pull", model]
        print("Suggested command:")
        print(f"  {_format_command(pull_command)}")
        if allow_fix and _prompt_yes_no("Run this now? [y/n]> "):
            _run_command(pull_command, env=_ollama_env(args))
            available_models.add(model)
            _print_status("OK", "Model ready", model)
        else:
            raise SystemExit(f"Model `{model}` is required to continue.")


def _requires_ollama(
    args: argparse.Namespace,
    *,
    manifest: dict[str, object] | None,
    building: bool,
) -> bool:
    return bool(_required_models(args, manifest=manifest, building=building))


def _required_models(
    args: argparse.Namespace,
    *,
    manifest: dict[str, object] | None,
    building: bool,
) -> list[str]:
    models: list[str] = []
    embedding_backend = args.embedding_backend
    embedding_model = args.embedding_model
    if not building and manifest:
        embedding_backend = str(manifest.get("embedding_backend") or embedding_backend)
        if manifest.get("embedding_model"):
            embedding_model = str(manifest["embedding_model"])

    if embedding_backend == "ollama":
        models.append(embedding_model)
    if getattr(args, "answer_mode", "extractive") == "ollama":
        models.append(getattr(args, "llm_model", "llama3.2:3b"))

    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def _model_is_available(model: str, available_models: set[str]) -> bool:
    if model in available_models:
        return True
    if ":" not in model and f"{model}:latest" in available_models:
        return True
    if model.endswith(":latest") and model[:-7] in available_models:
        return True
    return False


def _start_ollama_service(binary: str, host: str) -> Path:
    log_dir = AssistantSettings.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ollama.log"
    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            [binary, "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_ollama_env_from_host(host),
            start_new_session=True,
        )
    finally:
        log_file.close()

    deadline = time.time() + 15
    client = OllamaClient(host=host)
    while time.time() < deadline:
        if process.poll() is not None:
            break
        if client.is_available():
            return log_path
        time.sleep(0.5)

    raise SystemExit(
        f"Started `ollama serve`, but the server did not become ready in time. Check the log at {log_path}."
    )


def _run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"Running: {_format_command(command)}")
    subprocess.run(command, check=True, env=env)


def _ollama_env(args: argparse.Namespace) -> dict[str, str]:
    return _ollama_env_from_host(args.ollama_host)


def _ollama_env_from_host(host: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    return env


def _resolve_docs_dir_for_build(docs_dir: str | None, *, action_label: str) -> str:
    if docs_dir:
        root = Path(docs_dir).expanduser()
        if root.exists():
            return str(root.resolve())
        if not sys.stdin.isatty():
            raise SystemExit(f"Document folder does not exist: {root.resolve()}")
        _print_status("WARN", "Documents", f"Saved folder not found: {root}")

    if not sys.stdin.isatty():
        raise SystemExit(
            f"A document folder is required to {action_label}. Pass --docs-dir or run the CLI interactively."
        )

    prompted_docs_dir = _prompt_text("Documents folder (expected 3-5 .txt/.md files)> ", allow_blank=False)
    resolved = Path(prompted_docs_dir).expanduser()
    if not resolved.exists():
        raise SystemExit(f"Document folder does not exist: {resolved.resolve()}")
    return str(resolved.resolve())


def _resolve_existing_docs_dir(docs_dir: str | None, manifest: dict[str, object]) -> str | None:
    candidate = docs_dir or manifest.get("docs_dir")
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    if not path.exists():
        return None
    return str(path.resolve())


def _warn_if_unexpected_doc_count(docs_dir: str) -> None:
    doc_count = _supported_doc_count(docs_dir)
    if 3 <= doc_count <= 5:
        return
    _print_status(
        "WARN",
        "Documents",
        f"The evaluation setup expects 3-5 documents, but found {doc_count} supported files.",
    )


def _supported_doc_count(docs_dir: str | Path) -> int:
    root = Path(docs_dir).expanduser()
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _documents_changed_since_last_build(docs_dir: str | Path, manifest: dict[str, object]) -> bool:
    current_fingerprints = fingerprint_documents(docs_dir)
    saved_fingerprints = _manifest_document_fingerprints(manifest)
    if saved_fingerprints is not None:
        return current_fingerprints != saved_fingerprints

    current_paths = [item["path"] for item in current_fingerprints]
    saved_paths = _manifest_document_paths(manifest)
    if saved_paths is None:
        return False
    if current_paths != saved_paths:
        return True

    built_at = _manifest_saved_timestamp(manifest)
    if built_at is None:
        return False
    return any(Path(path).stat().st_mtime > built_at for path in current_paths)


def _manifest_document_fingerprints(manifest: dict[str, object]) -> list[dict[str, object]] | None:
    raw_value = manifest.get("document_fingerprints")
    if not isinstance(raw_value, list):
        return None

    fingerprints: list[dict[str, object]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            return None
        path = item.get("path")
        sha256 = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if not isinstance(path, str) or not isinstance(sha256, str) or not isinstance(size_bytes, int):
            return None
        fingerprints.append(
            {
                "path": path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return fingerprints


def _manifest_document_paths(manifest: dict[str, object]) -> list[str] | None:
    raw_value = manifest.get("documents")
    if not isinstance(raw_value, list):
        return None

    paths: list[str] = []
    for item in raw_value:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return None
        paths.append(str(item["path"]))
    return paths


def _manifest_saved_timestamp(manifest: dict[str, object]) -> float | None:
    saved_at = manifest.get("saved_at")
    if not isinstance(saved_at, str):
        return None
    try:
        return datetime.fromisoformat(saved_at).timestamp()
    except ValueError:
        return None


def _load_manifest_if_exists(index_dir: str | Path) -> dict[str, object]:
    if not LocalVectorStore.exists(index_dir):
        return {}
    return load_manifest(index_dir)


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


def _print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _print_status(level: str, label: str, detail: str) -> None:
    print(f"[{level}] {label}: {detail}")


def _print_saved_defaults(settings: AssistantSettings) -> None:
    print("Saved Defaults")
    print("--------------")
    print(f"- Documents: {settings.docs_dir or 'Not set'}")
    print(f"- Index: {settings.index_dir}")
    print(f"- Retrieval: {settings.embedding_backend} ({settings.embedding_model})")
    answer_detail = settings.answer_mode
    if settings.answer_mode == "ollama":
        answer_detail += f" ({settings.llm_model})"
    print(f"- Answers: {answer_detail}")
    print(f"- Eval file: {settings.eval_file or 'Not set'}")
    print()


def _print_ingest_summary(manifest: dict[str, object], args: argparse.Namespace) -> None:
    print()
    _print_status("OK", "Indexed documents", str(manifest.get("document_count", 0)))
    _print_status("OK", "Chunks", str(manifest.get("chunk_count", 0)))
    _print_status("OK", "Index path", str(Path(args.index_dir).expanduser().resolve()))
    retrieval = manifest.get("embedding_backend", "tfidf")
    if manifest.get("embedding_model"):
        retrieval = f"{retrieval} ({manifest.get('embedding_model')})"
    _print_status("OK", "Retrieval backend", retrieval)


def _prompt_text(prompt: str, *, allow_blank: bool) -> str:
    while True:
        value = input(prompt).strip()
        if value or allow_blank:
            return value
        print("Please enter a value.")


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in YES_ANSWERS:
            return True
        if answer in NO_ANSWERS:
            return False
        print("Please answer with y, yes, n, or no.")


def _pause_for_menu_return() -> None:
    if sys.stdin.isatty():
        print()
        input("Press Enter to return to the menu...")


def _save_settings_from_args(args: argparse.Namespace) -> Path:
    settings = AssistantSettings.load()
    for field_name in (
        "embedding_backend",
        "embedding_model",
        "answer_mode",
        "llm_model",
        "ollama_host",
        "docs_dir",
        "index_dir",
        "eval_file",
        "chunk_size",
        "chunk_overlap",
    ):
        if hasattr(args, field_name):
            value = getattr(args, field_name)
            if value is not None:
                setattr(settings, field_name, value)
    return settings.save()


def _make_menu_namespace(command: str, settings: AssistantSettings) -> argparse.Namespace:
    return argparse.Namespace(
        command=command,
        func={
            "setup": _run_setup,
            "chat": _run_chat,
            "ask": _run_ask,
            "evaluate": _run_evaluate,
            "doctor": _run_doctor,
        }[command],
        docs_dir=settings.docs_dir,
        index_dir=settings.index_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
        answer_mode=settings.answer_mode,
        llm_model=settings.llm_model,
        ollama_host=settings.ollama_host,
        top_k=5,
        min_score=0.15,
        relative_score_floor=0.55,
        rebuild=False,
        debug=False,
        eval_file=settings.eval_file,
        question=None,
        fix=False,
    )


def _format_command(command: list[str]) -> str:
    return shlex.join(command)


def _hydrate_args(args: argparse.Namespace) -> None:
    settings = AssistantSettings.load()
    defaults: dict[str, object] = {
        "docs_dir": settings.docs_dir,
        "index_dir": settings.index_dir,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "answer_mode": settings.answer_mode,
        "llm_model": settings.llm_model,
        "ollama_host": settings.ollama_host,
        "eval_file": settings.eval_file,
        "top_k": 5,
        "min_score": 0.15,
        "relative_score_floor": 0.55,
        "rebuild": False,
        "debug": False,
        "fix": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)


if __name__ == "__main__":
    main(sys.argv[1:])
