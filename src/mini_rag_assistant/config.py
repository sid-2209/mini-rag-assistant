from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_DIR_NAME = ".mini-rag"
CONFIG_FILE_NAME = "settings.json"
LOG_DIR_NAME = "logs"
ENV_OVERRIDES = {
    "embedding_backend": "MINI_RAG_EMBEDDING_BACKEND",
    "embedding_model": "MINI_RAG_EMBEDDING_MODEL",
    "answer_mode": "MINI_RAG_ANSWER_MODE",
    "llm_model": "MINI_RAG_LLM_MODEL",
    "ollama_host": "MINI_RAG_OLLAMA_HOST",
    "docs_dir": "MINI_RAG_DOCS_DIR",
    "index_dir": "MINI_RAG_INDEX_DIR",
    "eval_file": "MINI_RAG_EVAL_FILE",
}


@dataclass(slots=True)
class AssistantSettings:
    embedding_backend: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    answer_mode: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    docs_dir: str | None = None
    index_dir: str = ".rag_store"
    eval_file: str | None = None
    chunk_size: int = 140
    chunk_overlap: int = 30

    @classmethod
    def load(cls, cwd: str | Path | None = None) -> "AssistantSettings":
        settings = cls()
        path = cls.config_path(cwd)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

        for field_name, env_var in ENV_OVERRIDES.items():
            env_value = os.getenv(env_var)
            if env_value:
                setattr(settings, field_name, env_value)

        if not settings.eval_file:
            default_eval = Path(cwd or Path.cwd()) / "data" / "eval" / "sample_eval.jsonl"
            if default_eval.exists():
                settings.eval_file = str(default_eval)
        return settings

    def save(self, cwd: str | Path | None = None) -> Path:
        config_dir = self.config_dir(cwd)
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / CONFIG_FILE_NAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def config_dir(cls, cwd: str | Path | None = None) -> Path:
        root = Path(cwd or Path.cwd()).resolve()
        return root / CONFIG_DIR_NAME

    @classmethod
    def config_path(cls, cwd: str | Path | None = None) -> Path:
        return cls.config_dir(cwd) / CONFIG_FILE_NAME

    @classmethod
    def log_dir(cls, cwd: str | Path | None = None) -> Path:
        return cls.config_dir(cwd) / LOG_DIR_NAME
