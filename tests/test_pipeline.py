from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_rag_assistant.answering import REFUSAL_MESSAGE
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

    def test_unanswerable_question_refuses_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(SAMPLE_DOCS, index_dir=temp_dir, chunk_size=80, chunk_overlap=20)
            assistant = load_assistant(temp_dir)
            answer, _ = assistant.answer("What is the office dress code?")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertEqual(answer.citations, [])


if __name__ == "__main__":
    unittest.main()

