from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from backend_config import (
    BackendSettings,
    load_backend_settings,
    normalize_backend_endpoint,
    save_backend_settings,
)
from backend_factory import create_backend
from inference_backend import BackendConfigurationError


@pytest.mark.parametrize(
    "raw",
    (
        "http://127.0.0.1:9292",
        "http://127.0.0.1:9292/",
        "http://127.0.0.1:9292/v1",
    ),
)
def test_normalize_llama_swap_endpoint_has_one_api_base_and_service_root(raw: str) -> None:
    endpoint = normalize_backend_endpoint("llama-swap", raw)

    assert endpoint.api_base == "http://127.0.0.1:9292/v1"
    assert endpoint.service_root == "http://127.0.0.1:9292"


@pytest.mark.parametrize(
    "raw",
    (
        "http://127.0.0.1:9292/v2",
        "http://user:secret@127.0.0.1:9292/v1",
        "http://127.0.0.1:9292/v1?model=qwen3.6",
        "http://127.0.0.1:9292/v1#models",
        "ftp://127.0.0.1:9292/v1",
    ),
)
def test_normalize_llama_swap_endpoint_rejects_unsafe_or_noncanonical_urls(raw: str) -> None:
    with pytest.raises(BackendConfigurationError):
        normalize_backend_endpoint("llama-swap", raw)


def test_normalize_ollama_endpoint_preserves_local_and_remote_forms() -> None:
    local = normalize_backend_endpoint("ollama", "")
    remote = normalize_backend_endpoint("ollama", "192.168.1.10:11434/")

    assert local.api_base is None
    assert local.service_root is None
    assert remote.api_base == "http://192.168.1.10:11434"
    assert remote.service_root == "http://192.168.1.10:11434"


def test_load_backend_settings_defaults_to_local_ollama_and_existing_model(tmp_path) -> None:
    settings = load_backend_settings(tmp_path / "missing.json")

    assert settings.backend == "ollama"
    assert settings.api_base is None
    assert settings.model == "qwen2.5:14b"
    assert settings.batch_size == 15
    assert settings.timeout == 300


def test_load_backend_settings_migrates_legacy_local_and_remote_ollama_values(tmp_path) -> None:
    local_path = tmp_path / "local.json"
    remote_path = tmp_path / "remote.json"
    local_path.write_text(json.dumps({"ollama_mode": "local", "ollama_endpoint": ""}), encoding="utf-8")
    remote_path.write_text(
        json.dumps({"ollama_mode": "remote", "ollama_endpoint": "https://ollama.example.test:11434/"}),
        encoding="utf-8",
    )

    local = load_backend_settings(local_path)
    remote = load_backend_settings(remote_path)

    assert (local.backend, local.api_base) == ("ollama", None)
    assert (remote.backend, remote.api_base) == ("ollama", "https://ollama.example.test:11434")


def test_load_backend_settings_rejects_an_unknown_legacy_ollama_mode(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"ollama_mode": "hosted", "ollama_endpoint": "https://ollama.example.test:11434"}),
        encoding="utf-8",
    )

    with pytest.raises(BackendConfigurationError):
        load_backend_settings(path)


@pytest.mark.parametrize("endpoint", (None, ""))
def test_load_backend_settings_rejects_legacy_remote_ollama_without_endpoint(tmp_path, endpoint: str | None) -> None:
    path = tmp_path / "settings.json"
    legacy_settings: dict[str, str] = {"ollama_mode": "remote"}
    if endpoint is not None:
        legacy_settings["ollama_endpoint"] = endpoint
    path.write_text(json.dumps(legacy_settings), encoding="utf-8")

    with pytest.raises(BackendConfigurationError):
        load_backend_settings(path)


def test_save_backend_settings_keeps_canonical_and_legacy_ollama_keys(tmp_path) -> None:
    path = tmp_path / "settings.json"
    settings = BackendSettings(backend="ollama", api_base="https://ollama.example.test:11434", model="qwen2.5:14b")

    save_backend_settings(path, settings)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["backend"] == "ollama"
    assert saved["api_base"] == "https://ollama.example.test:11434"
    assert saved["ollama_mode"] == "remote"
    assert saved["ollama_endpoint"] == "https://ollama.example.test:11434"


def test_factory_constructs_only_the_selected_ollama_backend() -> None:
    class FakeOllamaClient:
        pass

    backend = create_backend(BackendSettings(), ollama_client=FakeOllamaClient())

    assert backend.backend_id == "ollama"
    assert backend.client.__class__ is FakeOllamaClient


def test_default_factory_does_not_import_or_construct_llama_swap(monkeypatch) -> None:
    module = ModuleType("llama_swap_backend")

    class UnexpectedLlamaSwapBackend:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("llama-swap must not be constructed for Ollama")

    module.LlamaSwapBackend = UnexpectedLlamaSwapBackend
    monkeypatch.setitem(sys.modules, "llama_swap_backend", module)

    backend = create_backend(BackendSettings(), ollama_client=object())

    assert backend.backend_id == "ollama"


def test_factory_lazily_resolves_the_selected_llama_swap_backend(monkeypatch) -> None:
    module = ModuleType("llama_swap_backend")

    class FakeLlamaSwapBackend:
        backend_id = "llama-swap"

        def __init__(self, api_base, *, opener) -> None:
            self.api_base = api_base
            self.opener = opener

    module.LlamaSwapBackend = FakeLlamaSwapBackend
    monkeypatch.setitem(sys.modules, "llama_swap_backend", module)
    opener = object()

    backend = create_backend(
        BackendSettings(backend="llama-swap", api_base="http://127.0.0.1:9292/v1"),
        http_opener=opener,
    )

    assert backend.api_base == "http://127.0.0.1:9292/v1"
    assert backend.opener is opener


def test_factory_rejects_unknown_backend_without_creating_a_client() -> None:
    with pytest.raises(BackendConfigurationError):
        create_backend(BackendSettings(backend="other"))
