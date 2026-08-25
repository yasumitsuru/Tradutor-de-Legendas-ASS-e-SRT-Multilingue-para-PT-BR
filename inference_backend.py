"""Provider-agnostic inference backend contract."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    system_prompt: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout: float


@dataclass(frozen=True)
class GenerationResult:
    text: str
    backend: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    model: str
    backend: str


class BackendError(Exception):
    """Base exception raised by inference backends."""


class BackendConfigurationError(BackendError):
    """Backend configuration is invalid or incomplete."""


class BackendUnavailableError(BackendError):
    """Backend service is unavailable."""


class BackendModelNotFoundError(BackendError):
    """Requested model is not available from the backend."""


class BackendTransportError(BackendError):
    """Backend request could not be transported."""


class BackendTimeoutError(BackendError):
    """Backend request timed out."""


class BackendResponseError(BackendError):
    """Backend returned an invalid response."""


class InferenceBackend(Protocol):
    def ensure_available(self, model: str) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def close(self) -> None: ...
