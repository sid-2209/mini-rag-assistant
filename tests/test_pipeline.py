from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from mini_rag_assistant.answering import REFUSAL_MESSAGE
from mini_rag_assistant.cli import _ensure_assistant
from mini_rag_assistant.document_loader import load_documents
from mini_rag_assistant.pipeline import MiniRAGAssistant, build_index, load_assistant
from mini_rag_assistant.vector_store import LocalVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


class FakeLLMClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(self, **_: object) -> dict[str, object]:
        self.calls += 1
        return self.payload


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
            assistant = self._build_extractive_assistant(temp_dir)
            answer, _ = assistant.answer("When does payroll run?")

        self.assertFalse(answer.refused)
        self.assertIn("last working day", answer.answer.lower())
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_answer_stays_tight_to_the_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = self._build_extractive_assistant(temp_dir)
            answer, _ = assistant.answer("What date is Republic Day celebrated in India?")

        self.assertFalse(answer.refused)
        self.assertIn("26 january", answer.answer.lower())
        self.assertNotIn("independence day", answer.answer.lower())

    def test_unanswerable_question_refuses_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = self._build_extractive_assistant(temp_dir)
            answer, _ = assistant.answer("What is the office dress code?")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertGreaterEqual(len(answer.citations), 1)
        self.assertTrue(any(citation.note for citation in answer.citations))

    def test_keyword_mismatch_question_refuses_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = self._build_extractive_assistant(temp_dir)
            answer, _ = assistant.answer("Who is the CEO?")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertIn("key terms", answer.refusal_reason or "")
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_zero_similarity_refusal_still_includes_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = self._build_extractive_assistant(temp_dir)
            answer, _ = assistant.answer("zzqvbbn wkjhqxz")

        self.assertTrue(answer.refused)
        self.assertEqual(answer.answer, REFUSAL_MESSAGE)
        self.assertGreaterEqual(len(answer.citations), 1)
        self.assertTrue(all(citation.score == 0.0 for citation in answer.citations))
        self.assertTrue(all(citation.note for citation in answer.citations))

    def test_grounded_ollama_answer_uses_validated_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(
                SAMPLE_DOCS,
                index_dir=temp_dir,
                chunk_size=80,
                chunk_overlap=20,
                embedding_backend="tfidf",
            )
            store = LocalVectorStore.load(temp_dir)
            llm_client = FakeLLMClient(
                {
                    "refused": False,
                    "answer": "The company runs payroll on the last working day of each month.",
                    "citations": ["E1"],
                }
            )
            assistant = MiniRAGAssistant(store, llm_client=llm_client, llm_model="fake-llm", answer_mode="ollama")
            answer, _ = assistant.answer("When does payroll run?")

        self.assertFalse(answer.refused)
        self.assertEqual(llm_client.calls, 1)
        self.assertIn("last working day", answer.answer.lower())
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_unsupported_ollama_answer_falls_back_to_extractive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(
                SAMPLE_DOCS,
                index_dir=temp_dir,
                chunk_size=80,
                chunk_overlap=20,
                embedding_backend="tfidf",
            )
            store = LocalVectorStore.load(temp_dir)
            llm_client = FakeLLMClient(
                {
                    "refused": False,
                    "answer": "Payroll runs every Friday.",
                    "citations": ["E1"],
                }
            )
            assistant = MiniRAGAssistant(store, llm_client=llm_client, llm_model="fake-llm", answer_mode="ollama")
            answer, _ = assistant.answer("When does payroll run?")

        self.assertFalse(answer.refused)
        self.assertEqual(llm_client.calls, 1)
        self.assertIn("last working day", answer.answer.lower())
        self.assertNotIn("every friday", answer.answer.lower())

    def test_cli_prompts_for_docs_folder_when_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = Namespace(
                docs_dir=None,
                index_dir=temp_dir,
                rebuild=False,
                embedding_backend="tfidf",
                embedding_model="nomic-embed-text",
                ollama_host="http://127.0.0.1:11434",
                answer_mode="extractive",
                llm_model="llama3.2:3b",
            )
            with patch("builtins.input", return_value=str(SAMPLE_DOCS)), patch(
                "mini_rag_assistant.cli.sys.stdin.isatty",
                return_value=True,
            ), patch("mini_rag_assistant.cli._save_settings_from_args"):
                assistant = _ensure_assistant(args)
                answer, _ = assistant.answer("When does payroll run?")

        self.assertEqual(Path(args.docs_dir).resolve(), SAMPLE_DOCS.resolve())
        self.assertFalse(answer.refused)

    def _build_extractive_assistant(self, temp_dir: str) -> MiniRAGAssistant:
        build_index(
            SAMPLE_DOCS,
            index_dir=temp_dir,
            chunk_size=80,
            chunk_overlap=20,
            embedding_backend="tfidf",
        )
        return load_assistant(temp_dir, answer_mode="extractive")


if __name__ == "__main__":
    unittest.main()
