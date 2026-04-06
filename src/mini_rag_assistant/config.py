from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class AssistantSettings:
    embedding_backend: str = os.getenv("MINI_RAG_EMBEDDING_BACKEND", "ollama")
    embedding_model: str = os.getenv("MINI_RAG_EMBEDDING_MODEL", "nomic-embed-text")
    answer_mode: str = os.getenv("MINI_RAG_ANSWER_MODE", "ollama")
    llm_model: str = os.getenv("MINI_RAG_LLM_MODEL", "llama3.2:3b")
    ollama_host: str = os.getenv("MINI_RAG_OLLAMA_HOST", "http://127.0.0.1:11434")
