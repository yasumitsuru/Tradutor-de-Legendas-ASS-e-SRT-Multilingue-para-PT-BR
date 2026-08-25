"""Ollama implementation of the provider-agnostic inference contract."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

import ollama

from inference_backend import (
    BackendModelNotFoundError,
    BackendResponseError,
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)


def is_model_not_found_error(exc: Exception) -> bool:
    """Detect Ollama errors that represent a missing model."""

    return isinstance(exc, BackendModelNotFoundError) or (
        getattr(exc, "status_code", None) == 404
        or ("model" in str(exc).lower() and "not found" in str(exc).lower())
    )


class OllamaBackend:
    """Adapter that keeps Ollama-specific calls out of translation orchestration."""

    backend_id = "ollama"

    def __init__(self, client: Any | None = None, host: str | None = None) -> None:
        self.client = client if client is not None else (ollama.Client(host=host) if host else ollama)
        self._selected_model: str | None = None
        self._closed = False

    def ensure_available(self, model: str) -> None:
        try:
            self.client.show(model)
        except Exception as exc:
            if is_model_not_found_error(exc):
                raise BackendModelNotFoundError(str(exc)) from exc
            raise BackendUnavailableError(str(exc)) from exc
        self._selected_model = model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = self.client.generate(
                model=request.model,
                prompt=request.prompt,
                system=request.system_prompt,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p,
                },
            )
        except Exception as exc:
            if is_model_not_found_error(exc):
                raise BackendModelNotFoundError(str(exc)) from exc
            raise
        self._selected_model = request.model
        response_text = response.get("response") if isinstance(response, Mapping) else getattr(response, "response", None)
        if not isinstance(response_text, str):
            raise BackendResponseError("Ollama retornou uma resposta sem campo textual válido.")
        return GenerationResult(text=response_text.strip(), backend=self.backend_id, model=request.model)

    def close(self, model: str | None = None) -> None:
        """Best-effort unload of the selected model; repeated calls are no-ops.

        The optional model argument is retained for compatibility but never selects
        a model for shutdown.
        """

        model_name = self._selected_model
        if self._closed or not model_name:
            return
        self._closed = True
        print(f"\n🛑 Encerrando modelo Ollama: {model_name}")
        try:
            running = self.client.ps()
            loaded_models: list[str] = []
            models = running.get("models", []) if isinstance(running, Mapping) else getattr(running, "models", [])
            for model_info in models:
                for attribute in ("model", "name"):
                    value = model_info.get(attribute) if isinstance(model_info, Mapping) else getattr(model_info, attribute, None)
                    if value:
                        loaded_models.append(value)
            if model_name not in loaded_models:
                print(f"ℹ️ Modelo '{model_name}' já não está carregado.")
                return
        except Exception as exc:
            print(f"⚠️ Não foi possível consultar modelos ativos ({exc}). Tentando encerrar mesmo assim...")

        try:
            self.client.generate(model=model_name, prompt="", keep_alive=0)
            print(f"✅ Modelo '{model_name}' descarregado via API.")
            return
        except Exception as exc:
            print(f"⚠️ Falha ao descarregar via API: {exc}")

        ollama_cli = shutil.which("ollama")
        if not ollama_cli:
            print("⚠️ Comando 'ollama' não encontrado para fallback de encerramento.")
            return
        try:
            result = subprocess.run(
                [ollama_cli, "stop", model_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"⚠️ Erro ao tentar encerrar via CLI: {exc}")
            return
        details = (result.stdout or result.stderr).strip()
        if result.returncode == 0:
            print(f"✅ Modelo '{model_name}' encerrado via CLI.")
        else:
            print(f"⚠️ Não foi possível encerrar o modelo via CLI (código {result.returncode}).")
        if details:
            print(f"   ↳ {details}")
