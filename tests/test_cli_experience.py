from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from mini_rag_assistant.cli import _ensure_ollama_requirements, _model_is_available, _prompt_yes_no
from mini_rag_assistant.config import AssistantSettings


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


if __name__ == "__main__":
    unittest.main()
