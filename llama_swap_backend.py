"""Read-only llama-swap HTTP adapter using the OpenAI-compatible API."""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend_config import normalize_backend_endpoint
from inference_backend import (
    BackendModelNotFoundError,
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
)


class LlamaSwapBackend:
    """Adapter for llama-swap's read-only discovery and chat-completions API."""

    backend_id = "llama-swap"

    def __init__(self, api_base: str, *, opener: Any | None = None) -> None:
        endpoint = normalize_backend_endpoint(self.backend_id, api_base.rstrip("/"))
        self.api_base = endpoint.api_base
        self.service_root = endpoint.service_root
        self._opener = opener if opener is not None else urlopen

    def health(self) -> None:
        self._get(f"{self.service_root}/health")

    def list_models(self) -> tuple[ModelInfo, ...]:
        payload = self._get_json(f"{self.api_base}/models")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise BackendResponseError("llama-swap retornou uma lista de modelos inválida.")
        models: list[ModelInfo] = []
        for item in payload["data"]:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                raise BackendResponseError("llama-swap retornou uma lista de modelos inválida.")
            models.append(ModelInfo(model=item["id"], backend=self.backend_id))
        return tuple(models)

    def ensure_available(self, model: str) -> None:
        self.health()
        if model not in {info.model for info in self.list_models()}:
            raise BackendModelNotFoundError(f"Modelo não encontrado no llama-swap: {model}")
        self._get(f"{self.service_root}/running")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        response = self._post_json(f"{self.api_base}/chat/completions", payload, request.timeout)
        try:
            content = response["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError):
            raise BackendResponseError("llama-swap retornou uma conclusão inválida.") from None
        if not isinstance(content, str) or not (text := content.strip()):
            raise BackendResponseError("llama-swap retornou uma conclusão sem texto.")

        usage = response.get("usage") if isinstance(response, Mapping) else None
        metadata = {"usage": usage} if isinstance(usage, Mapping) else {}
        return GenerationResult(
            text=text,
            backend=self.backend_id,
            model=request.model,
            prompt_tokens=self._usage_value(usage, "prompt_tokens"),
            completion_tokens=self._usage_value(usage, "completion_tokens"),
            total_tokens=self._usage_value(usage, "total_tokens"),
            metadata=metadata,
        )

    def close(self) -> None:
        """llama-swap lifecycle remains owned by its service manager."""

    def _get(self, url: str) -> bytes:
        return self._open(Request(url))

    def _get_json(self, url: str) -> object:
        return self._decode_json(self._get(url))

    def _post_json(self, url: str, payload: Mapping[str, object], timeout: float) -> Mapping[str, object]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = self._decode_json(
            self._open(Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="POST"), timeout)
        )
        if not isinstance(response, Mapping):
            raise BackendResponseError("llama-swap retornou uma resposta JSON inválida.")
        return response

    def _open(self, request: Request, timeout: float | None = None) -> bytes:
        try:
            if timeout is None:
                response = self._opener(request)
            else:
                response = self._opener(request, timeout=timeout)
            with response:
                status = response.getcode()
                if not isinstance(status, int) or not 200 <= status < 300:
                    raise BackendUnavailableError(f"llama-swap retornou HTTP {status}.")
                return response.read()
        except BackendUnavailableError:
            raise
        except HTTPError as exc:
            raise BackendUnavailableError(f"llama-swap retornou HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise BackendTimeoutError("Tempo limite ao comunicar com llama-swap.") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendTimeoutError("Tempo limite ao comunicar com llama-swap.") from exc
            raise BackendTransportError("Falha de transporte ao comunicar com llama-swap.") from exc
        except OSError as exc:
            raise BackendTransportError("Falha de transporte ao comunicar com llama-swap.") from exc

    @staticmethod
    def _decode_json(body: bytes) -> object:
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendResponseError("llama-swap retornou JSON inválido.") from exc

    @staticmethod
    def _usage_value(usage: object, name: str) -> int | None:
        value = usage.get(name) if isinstance(usage, Mapping) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None
