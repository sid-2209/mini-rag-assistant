from __future__ import annotations

import hashlib
import re
from pathlib import Path

from mini_rag_assistant.types import Document

SUPPORTED_EXTENSIONS = {".md", ".txt"}
FRONT_MATTER_PATTERN = re.compile(r"(?s)\A---\n(.*?)\n---\n?")
KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.+?)\s*$")
TITLE_LINE_PATTERN = re.compile(r"(?i)^title\s*:\s*(.+)$")
SOURCE_LINE_PATTERN = re.compile(r"(?i)^source\s*:\s*(.+)$")
HEADING_PATTERN = re.compile(r"^#\s+(.+)$")


def load_documents(folder: str | Path) -> list[Document]:
    paths = discover_document_paths(folder)

    documents: list[Document] = []
    for index, path in enumerate(paths, start=1):
        raw_text = path.read_text(encoding="utf-8")
        try:
            title, source, content = parse_document_text(
                raw_text,
                fallback_title=_humanize_filename(path.stem),
                fallback_source=path.name,
            )
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        documents.append(
            Document(
                doc_id=f"doc-{index}",
                title=title,
                source=source,
                content=content,
                path=str(path),
            )
        )
    return documents


def discover_document_paths(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Document folder does not exist: {root}")

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No .md or .txt files found in {root}")
    return paths


def fingerprint_documents(folder: str | Path) -> list[dict[str, object]]:
    fingerprints: list[dict[str, object]] = []
    for path in discover_document_paths(folder):
        raw_bytes = path.read_bytes()
        fingerprints.append(
            {
                "path": str(path),
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        )
    return fingerprints


def parse_document_text(
    raw_text: str,
    *,
    fallback_title: str,
    fallback_source: str,
) -> tuple[str, str, str]:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    metadata: dict[str, str] = {}
    body = text

    front_matter_match = FRONT_MATTER_PATTERN.match(text)
    if front_matter_match:
        metadata.update(_parse_key_values(front_matter_match.group(1)))
        body = text[front_matter_match.end() :].strip()

    lines = body.splitlines()
    consumed_indexes: set[int] = set()

    title = metadata.get("title", "").strip()
    source = metadata.get("source", "").strip()

    for index, line in enumerate(lines[:12]):
        stripped = line.strip()
        if not stripped:
            continue

        if not title:
            title_match = TITLE_LINE_PATTERN.match(stripped)
            if title_match:
                title = title_match.group(1).strip()
                consumed_indexes.add(index)
                continue

            heading_match = HEADING_PATTERN.match(stripped)
            if heading_match and index < 3:
                title = heading_match.group(1).strip()
                consumed_indexes.add(index)
                continue

        if not source:
            source_match = SOURCE_LINE_PATTERN.match(stripped)
            if source_match:
                source = source_match.group(1).strip()
                consumed_indexes.add(index)
                continue

    cleaned_lines = [line for index, line in enumerate(lines) if index not in consumed_indexes]
    content = "\n".join(cleaned_lines).strip()

    if not title:
        title = fallback_title
    if not source:
        source = fallback_source
    if not content:
        raise ValueError("Document content is empty after metadata parsing.")

    return title, source, content


def _parse_key_values(block: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in block.splitlines():
        match = KEY_VALUE_PATTERN.match(line.strip())
        if match:
            metadata[match.group(1).lower()] = match.group(2).strip()
    return metadata


def _humanize_filename(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title()
