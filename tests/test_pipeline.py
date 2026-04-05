from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from mini_rag_assistant.answering import REFUSAL_MESSAGE
from mini_rag_assistant.cli import _ensure_assistant
from mini_rag_assistant.document_loader import load_documents
from mini_rag_assistant.pipeline import build_index, load_assistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


class MiniRAGAssistantTests(unittest.TestCase):
    def test_document_loader_extracts_title_and_source(self) -> None:
        documents = load_documents(SAMPLE_DOCS)
        self.assertEqual(len(documents), 4)
        first = documents[0]
        self.assertTrue(first.title)
        self.assertTrue(first.source)
        self.assertTrue(first.content)

    def test_answer_contains_grounded_content_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("When does payroll run?")

        self.assertFalse(answer.refused)
        self.assertIn("last working day", answer.answer.lower())
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_answer_stays_tight_to_the_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("What date is Republic Day celebrated in India?")

        self.assertFalse(answer.refused)
        self.assertIn("26 january", answer.answer.lower())
        self.assertNotIn("independence day", answer.answer.lower())

    def test_unanswerable_question_refuses_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("What is the office dress code?")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertGreaterEqual(len(answer.citations), 1)
        self.assertTrue(any(citation.note for citation in answer.citations))

    def test_keyword_mismatch_question_refuses_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("Who is the CEO?")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertIn("key terms", answer.refusal_reason or "")
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_zero_similarity_refusal_still_includes_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("zzqvbbn wkjhqxz")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertGreaterEqual(len(answer.citations), 1)
        self.assertTrue(all(citation.score == 0.0 for citation in answer.citations))
        self.assertTrue(all(citation.note for citation in answer.citations))

    def test_cli_prompts_for_docs_folder_when_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = Namespace(
                docs_dir=None,
                index_dir=temp_dir,
                rebuild=False,
            )
            with patch("builtins.input", return_value=str(SAMPLE_DOCS)), patch(
                "mini_rag_assistant.cli.sys.stdin.isatty",
                return_value=True,
            ):
                assistant = _ensure_assistant(args)
                answer, _ = assistant.answer("When does payroll run?")

        self.assertEqual(Path(args.docs_dir).resolve(), SAMPLE_DOCS.resolve())
        self.assertFalse(answer.refused)


if __name__ == "__main__":
    unittest.main()
