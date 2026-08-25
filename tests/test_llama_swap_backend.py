from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import pytest

from inference_backend import (
    BackendModelNotFoundError,
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    BackendUnavailableError,
    GenerationRequest,
)
from llama_swap_backend import LlamaSwapBackend


def request() -> GenerationRequest:
    return GenerationRequest(
        model="qwen3.6:latest",
        system_prompt="Translate subtitles.",
        prompt="<<<ITEM_0001>>>\nHello\n<<<END_ITEM_0001>>>",
        temperature=0.35,
        top_p=0.9,
        max_tokens=2048,
        timeout=37.5,
    )


class FakeResponse:
    def __init__(self, body: object, status: int = 200) -> None:
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeOpener:
    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[object, float | None]] = []

    def __call__(self, http_request: object, timeout: float | None = None) -> FakeResponse:
        self.calls.append((http_request, timeout))
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def paths(self) -> list[str]:
        return [urlsplit(call.full_url).path for call, _timeout in self.calls]


def test_health_uses_service_root_and_accepts_a_successful_response() -> None:
    opener = FakeOpener(FakeResponse({"status": "ok"}))
    backend = LlamaSwapBackend("http://127.0.0.1:9292/v1/", opener=opener)

    backend.health()

    assert backend.api_base == "http://127.0.0.1:9292/v1"
    assert backend.service_root == "http://127.0.0.1:9292"
    assert opener.paths == ["/health"]
    assert opener.calls[0][1] is None


@pytest.mark.parametrize("status", (404, 500))
def test_health_normalizes_http_errors_without_exposing_response_data(status: int) -> None:
    opener = FakeOpener(HTTPError("http://127.0.0.1:9292/health", status, "secret details", None, None))

    with pytest.raises(BackendUnavailableError) as error:
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).health()

    assert "secret details" not in str(error.value)
    assert opener.paths == ["/health"]


def test_list_models_returns_advertised_model_ids() -> None:
    opener = FakeOpener(FakeResponse({"data": [{"id": "qwen3.6:latest"}]}))

    models = LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).list_models()

    assert models == (type(models[0])(model="qwen3.6:latest", backend="llama-swap"),)
    assert opener.paths == ["/v1/models"]


@pytest.mark.parametrize("body", ({"data": [{"id": 42}]}, {"data": [{"name": "qwen3.6:latest"}]}, {"data": ["qwen3.6:latest"]}))
def test_list_models_rejects_entries_without_a_string_id(body: object) -> None:
    opener = FakeOpener(FakeResponse(body))

    with pytest.raises(BackendResponseError):
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).list_models()

    assert opener.paths == ["/v1/models"]


def test_list_models_rejects_malformed_json() -> None:
    opener = FakeOpener(FakeResponse(b"not-json"))

    with pytest.raises(BackendResponseError):
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).list_models()


def test_ensure_available_accepts_only_an_exact_advertised_model_after_read_only_preflight() -> None:
    opener = FakeOpener(
        FakeResponse({"status": "ok"}),
        FakeResponse({"data": [{"id": "qwen3.6:latest"}, {"id": "qwen3.6:latest-extra"}]}),
        FakeResponse({"running": []}),
    )

    LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).ensure_available("qwen3.6:latest")

    assert opener.paths == ["/health", "/v1/models", "/running"]
    assert len(opener.calls) == 3


def test_ensure_available_rejects_partial_or_missing_model_matches() -> None:
    opener = FakeOpener(
        FakeResponse({"status": "ok"}),
        FakeResponse({"data": [{"id": "qwen3.6:latest-extra"}]}),
    )

    with pytest.raises(BackendModelNotFoundError):
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).ensure_available("qwen3.6:latest")

    assert opener.paths == ["/health", "/v1/models"]


def test_generate_posts_openai_messages_and_returns_trimmed_content_with_usage() -> None:
    opener = FakeOpener(
        FakeResponse(
            {
                "choices": [{"message": {"content": "  Olá  "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        )
    )

    result = LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).generate(request())

    sent = json.loads(opener.calls[0][0].data.decode("utf-8"))
    assert opener.paths == ["/v1/chat/completions"]
    assert opener.calls[0][0].get_method() == "POST"
    assert opener.calls[0][0].get_header("Content-type") == "application/json"
    assert opener.calls[0][1] == 37.5
    assert sent == {
        "model": "qwen3.6:latest",
        "messages": [
            {"role": "system", "content": "Translate subtitles."},
            {"role": "user", "content": "<<<ITEM_0001>>>\nHello\n<<<END_ITEM_0001>>>"},
        ],
        "temperature": 0.35,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": False,
    }
    assert result.text == "Olá"
    assert result.backend == "llama-swap"
    assert result.model == "qwen3.6:latest"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (10, 4, 14)
    assert result.metadata == {"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}


@pytest.mark.parametrize(
    "body",
    (
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "   "}}]},
        b"not-json",
    ),
)
def test_generate_normalizes_malformed_or_empty_completion_responses(body: object) -> None:
    opener = FakeOpener(FakeResponse(body))

    with pytest.raises(BackendResponseError):
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).generate(request())

    assert opener.paths == ["/v1/chat/completions"]


@pytest.mark.parametrize("status", (404, 500))
def test_generate_normalizes_http_errors_in_one_attempt(status: int) -> None:
    opener = FakeOpener(HTTPError("http://127.0.0.1:9292/v1/chat/completions", status, "secret details", None, None))

    with pytest.raises(BackendUnavailableError) as error:
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).generate(request())

    assert "secret details" not in str(error.value)
    assert opener.paths == ["/v1/chat/completions"]
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    ("failure", "error_type"),
    (
        (TimeoutError("late"), BackendTimeoutError),
        (socket.timeout("late"), BackendTimeoutError),
        (OSError("offline"), BackendTransportError),
        (URLError(socket.timeout("late")), BackendTimeoutError),
        (URLError(OSError("offline")), BackendTransportError),
    ),
)
def test_generate_normalizes_timeout_and_connection_errors_without_retry(failure: Exception, error_type: type[Exception]) -> None:
    opener = FakeOpener(failure)

    with pytest.raises(error_type):
        LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener).generate(request())

    assert len(opener.calls) == 1


def test_close_is_a_no_op_without_lifecycle_requests() -> None:
    opener = FakeOpener()
    backend = LlamaSwapBackend("http://127.0.0.1:9292/v1", opener=opener)

    backend.close()
    backend.close()

    assert opener.calls == []
