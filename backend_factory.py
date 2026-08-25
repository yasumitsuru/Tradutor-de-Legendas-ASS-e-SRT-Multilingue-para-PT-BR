"""Selected-backend construction without UI dependencies or network calls."""

from __future__ import annotations

from typing import Any

from backend_config import BackendSettings
from inference_backend import BackendConfigurationError, InferenceBackend


def create_backend(
    settings: BackendSettings,
    *,
    ollama_client: Any | None = None,
    http_opener: Any | None = None,
) -> InferenceBackend:
    """Create only the requested backend; service checks belong to callers."""

    if settings.backend == "ollama":
        from ollama_backend import OllamaBackend

        return OllamaBackend(client=ollama_client, host=settings.api_base)
    if settings.backend == "llama-swap":
        from llama_swap_backend import LlamaSwapBackend

        return LlamaSwapBackend(settings.api_base, opener=http_opener)
    raise BackendConfigurationError(f"Backend desconhecido: {settings.backend}")
