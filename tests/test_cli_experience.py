from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from mini_rag_assistant.cli import (
    _ensure_index_ready,
    _ensure_ollama_requirements,
    _model_is_available,
    _prompt_yes_no,
)
from mini_rag_assistant.config import AssistantSettings
from mini_rag_assistant.pipeline import build_index
from mini_rag_assistant.vector_store import load_manifest


class CLIExperienceTests(unittest.TestCase):
    def test_prompt_yes_no_accepts_common_variants(self) -> None:
        with patch("builtins.input", side_effect=["maybe", "YES"]):
            self.assertTrue(_prompt_yes_no("Run now? "))

        with patch("builtins.input", return_value="n"):
            self.assertFalse(_prompt_yes_no("Run now? "))

    def test_settings_round_trip_through_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AssistantSettings(
                docs_dir="/tmp/docs",
                index_dir="/tmp/index",
                eval_file="/tmp/eval.jsonl",
            )
            path = settings.save(temp_dir)
            loaded = AssistantSettings.load(temp_dir)

        self.assertTrue(path.name.endswith("settings.json"))
        self.assertEqual(loaded.docs_dir, "/tmp/docs")
        self.assertEqual(loaded.index_dir, "/tmp/index")
        self.assertEqual(loaded.eval_file, "/tmp/eval.jsonl")

    def test_ensure_ollama_requirements_can_start_service_after_yes_prompt(self) -> None:
        args = Namespace(
            embedding_backend="ollama",
            embedding_model="nomic-embed-text",
            answer_mode="ollama",
            llm_model="llama3.2:3b",
            ollama_host="http://127.0.0.1:11434",
        )

        with patch("mini_rag_assistant.cli.shutil.which", return_value="/opt/homebrew/bin/ollama"), patch(
            "mini_rag_assistant.cli.OllamaClient.is_available",
            return_value=False,
        ), patch(
            "mini_rag_assistant.cli.OllamaClient.list_models",
            return_value=["nomic-embed-text", "llama3.2:3b"],
        ), patch(
            "mini_rag_assistant.cli._prompt_yes_no",
            return_value=True,
        ) as prompt_mock, patch(
            "mini_rag_assistant.cli._start_ollama_service",
            return_value=Path("/tmp/ollama.log"),
        ) as start_mock:
            _ensure_ollama_requirements(args, manifest=None, building=True, allow_fix=True)

        prompt_mock.assert_called_once()
        start_mock.assert_called_once()

    def test_model_availability_accepts_latest_tag_alias(self) -> None:
        available = {"nomic-embed-text:latest", "llama3.2:3b"}
        self.assertTrue(_model_is_available("nomic-embed-text", available))
        self.assertTrue(_model_is_available("llama3.2:3b", available))
        self.assertFalse(_model_is_available("llama3.2", available))

    def test_ensure_index_ready_rebuilds_when_docs_are_replaced_from_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            index_dir = Path(temp_dir) / "index"
            docs_dir.mkdir()
            self._write_doc(
                docs_dir / "old.md",
                "# Legacy Notes\nSource: Test\nThe office Wi-Fi password rotates monthly.\n",
            )
            build_index(docs_dir, index_dir=index_dir, embedding_backend="tfidf")

            legacy_manifest = load_manifest(index_dir)
            legacy_manifest.pop("document_fingerprints", None)
            (index_dir / "manifest.json").write_text(json.dumps(legacy_manifest, indent=2), encoding="utf-8")

            (docs_dir / "old.md").unlink()
            self._write_doc(
                docs_dir / "new.md",
                "# Intake Runbook\nSource: Test\nSlack triage starts at 10 AM every weekday.\n",
            )

            args = self._index_args(docs_dir, index_dir)
            with patch("mini_rag_assistant.cli._print_status"):
                manifest = _ensure_index_ready(args, sync_with_current_config=False, allow_fix=False)

        self.assertEqual(manifest["document_count"], 1)
        self.assertEqual(Path(manifest["documents"][0]["path"]).name, "new.md")

    def test_ensure_index_ready_rebuilds_when_doc_contents_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            index_dir = Path(temp_dir) / "index"
            docs_dir.mkdir()
            doc_path = docs_dir / "policy.md"
            self._write_doc(
                doc_path,
                "# Support Policy\nSource: Test\nEscalations are reviewed every Tuesday.\n",
            )
            build_index(docs_dir, index_dir=index_dir, embedding_backend="tfidf")
            original_manifest = load_manifest(index_dir)

            self._write_doc(
                doc_path,
                "# Support Policy\nSource: Test\nEscalations are reviewed every Thursday.\n",
            )

            args = self._index_args(docs_dir, index_dir)
            with patch("mini_rag_assistant.cli._print_status"):
                updated_manifest = _ensure_index_ready(args, sync_with_current_config=False, allow_fix=False)

        self.assertEqual(updated_manifest["document_count"], 1)
        self.assertNotEqual(
            original_manifest["document_fingerprints"][0]["sha256"],
            updated_manifest["document_fingerprints"][0]["sha256"],
        )

    def _index_args(self, docs_dir: Path, index_dir: Path) -> Namespace:
        return Namespace(
            docs_dir=str(docs_dir),
            index_dir=str(index_dir),
            rebuild=False,
            embedding_backend="tfidf",
            embedding_model="nomic-embed-text",
            ollama_host="http://127.0.0.1:11434",
            chunk_size=140,
            chunk_overlap=30,
        )

    def _write_doc(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
