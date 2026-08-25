from typing import TypeVar

from inference_backend import (
    BackendConfigurationError,
    BackendError,
    BackendModelNotFoundError,
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
)


def test_generation_request_stores_normalized_generation_inputs() -> None:
    request = GenerationRequest(
        model="qwen3.6",
        system_prompt="Translate faithfully.",
        prompt="Hello, world!",
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        timeout=30.0,
    )

    assert request.model == "qwen3.6"
    assert request.system_prompt == "Translate faithfully."
    assert request.prompt == "Hello, world!"
    assert request.temperature == 0.2
    assert request.top_p == 0.9
    assert request.max_tokens == 512
    assert request.timeout == 30.0


def test_generation_result_stores_normalized_text_backend_and_model() -> None:
    result = GenerationResult(
        text="Olá, mundo!",
        backend="llama-swap",
        model="qwen3.6",
    )

    assert result.text == "Olá, mundo!"
    assert result.backend == "llama-swap"
    assert result.model == "qwen3.6"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None


def test_backend_errors_share_the_backend_error_base_class() -> None:
    for error_type in (
        BackendConfigurationError,
        BackendUnavailableError,
        BackendModelNotFoundError,
        BackendTransportError,
        BackendTimeoutError,
        BackendResponseError,
    ):
        assert issubclass(error_type, BackendError)


BackendProtocol = TypeVar("BackendProtocol", bound=InferenceBackend)


def _accepts_backend(backend: BackendProtocol) -> BackendProtocol:
    return backend


class FakeBackend:
    def ensure_available(self, model: str) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(text=request.prompt, backend="fake", model=request.model)

    def close(self) -> None:
        pass


def test_provider_agnostic_fake_satisfies_backend_protocol() -> None:
    backend = _accepts_backend(FakeBackend())

    assert backend.generate(
        GenerationRequest(
            model="qwen3.6",
            system_prompt="Translate faithfully.",
            prompt="Hello",
            temperature=0.2,
            top_p=0.9,
            max_tokens=512,
            timeout=30.0,
        )
    ).backend == "fake"
