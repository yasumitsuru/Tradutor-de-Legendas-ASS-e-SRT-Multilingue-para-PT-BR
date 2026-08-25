from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_backend import (
    BackendModelNotFoundError,
    BackendResponseError,
    GenerationRequest,
)
from ollama_backend import OllamaBackend
from translation_engine import ensure_ollama_model_available, is_model_not_found_error


def request() -> GenerationRequest:
    return GenerationRequest(
        model="qwen2.5:14b",
        system_prompt="Translate subtitles.",
        prompt="<<<ITEM_0001>>>\nHello\n<<<END_ITEM_0001>>>",
        temperature=0.35,
        top_p=0.9,
        max_tokens=2048,
        timeout=300,
    )


class FakeOllamaClient:
    def __init__(self, response: object = None) -> None:
        self.response = response if response is not None else {"response": "Olá"}
        self.show_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.unload_calls: list[dict] = []

    def show(self, model: str) -> dict[str, str]:
        self.show_calls.append(model)
        return {"model": model}

    def generate(self, **kwargs) -> object:
        if kwargs.get("keep_alive") == 0:
            self.unload_calls.append(kwargs)
            return {"response": ""}
        self.generate_calls.append(kwargs)
        return self.response

    def ps(self) -> SimpleNamespace:
        return SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:14b", name=None)])


def test_ensure_available_checks_model_with_the_supplied_client() -> None:
    client = FakeOllamaClient()

    backend = OllamaBackend(client=client)
    backend.ensure_available("qwen2.5:14b")

    assert backend.backend_id == "ollama"
    assert client.show_calls == ["qwen2.5:14b"]


def test_ensure_available_normalizes_missing_model_errors() -> None:
    class MissingModelClient(FakeOllamaClient):
        def show(self, model: str) -> dict[str, str]:
            error = RuntimeError(f"model {model} not found")
            error.status_code = 404
            raise error

    with pytest.raises(BackendModelNotFoundError, match="not found"):
        OllamaBackend(client=MissingModelClient()).ensure_available("missing:model")


def test_generate_maps_request_options_and_normalizes_response() -> None:
    client = FakeOllamaClient({"response": "  Olá  "})

    result = OllamaBackend(client=client).generate(request())

    assert result.text == "Olá"
    assert result.backend == "ollama"
    assert result.model == "qwen2.5:14b"
    assert client.generate_calls == [
        {
            "model": "qwen2.5:14b",
            "prompt": "<<<ITEM_0001>>>\nHello\n<<<END_ITEM_0001>>>",
            "system": "Translate subtitles.",
            "options": {"temperature": 0.35, "num_predict": 2048, "top_p": 0.9},
        }
    ]


def test_generate_rejects_provider_responses_without_text() -> None:
    with pytest.raises(BackendResponseError, match="campo textual"):
        OllamaBackend(client=FakeOllamaClient({"response": None})).generate(request())


def test_close_unloads_the_selected_model_only_once() -> None:
    client = FakeOllamaClient()
    backend = OllamaBackend(client=client)
    backend.ensure_available("qwen2.5:14b")

    backend.close()
    backend.close()

    assert client.unload_calls == [
        {"model": "qwen2.5:14b", "prompt": "", "keep_alive": 0}
    ]


def test_close_does_not_unload_an_explicit_model_without_selection() -> None:
    client = FakeOllamaClient()

    OllamaBackend(client=client).close("qwen2.5:14b")

    assert client.unload_calls == []


def test_legacy_validation_aliases_use_the_adapter_classification() -> None:
    class MissingModelError(RuntimeError):
        status_code = 404

    class MissingModelClient:
        def show(self, _model: str) -> None:
            raise MissingModelError("anything")

    error = MissingModelError("model disappeared")

    assert is_model_not_found_error(error)
    with pytest.raises(RuntimeError, match='Modelo "qwen2.5:14b"'):
        ensure_ollama_model_available("qwen2.5:14b", client=MissingModelClient())
