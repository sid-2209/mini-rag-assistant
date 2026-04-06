from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


@dataclass(slots=True)
class OllamaClient:
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = self._post_json(
            "/api/embed",
            {
                "model": model,
                "input": texts,
            },
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama did not return one embedding per input text.")
        return embeddings

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any] | str = "json",
        system: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str = "5m",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if system:
            body["system"] = system
        if options:
            body["options"] = options

        payload = self._post_json("/api/generate", body)
        raw_response = payload.get("response", "")
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise OllamaError("Ollama returned an empty generation payload.")

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON: {raw_response!r}") from exc

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.host.rstrip('/')}{path}"
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                charset = response.headers.get_content_charset("utf-8")
                response_text = response.read().decode(charset)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise OllamaError(f"Ollama request failed with status {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaError(
                "Could not reach the local Ollama server. Start it with `ollama serve` and make sure the model is pulled."
            ) from exc

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON: {response_text!r}") from exc

        if isinstance(data, dict) and data.get("error"):
            raise OllamaError(str(data["error"]))
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an unexpected response shape.")
        return data
