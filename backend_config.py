"""Canonical, provider-neutral inference backend settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from inference_backend import BackendConfigurationError


DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = "qwen2.5:14b"


@dataclass(frozen=True)
class NormalizedEndpoint:
    api_base: str | None
    service_root: str | None


@dataclass(frozen=True)
class BackendSettings:
    backend: str = DEFAULT_BACKEND
    api_base: str | None = None
    model: str = DEFAULT_MODEL
    batch_size: int = 15
    timeout: int = 300
    enable_cache: bool = True
    cache_file: str = "translation_cache.json"


def _parse_http_url(raw: str, *, allow_missing_scheme: bool) -> tuple[str, str]:
    candidate = raw.strip()
    if allow_missing_scheme and "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BackendConfigurationError("O endpoint deve usar http:// ou https://.")
    if parsed.username is not None or parsed.password is not None:
        raise BackendConfigurationError("O endpoint não pode conter credenciais.")
    if parsed.params or parsed.query or parsed.fragment:
        raise BackendConfigurationError("O endpoint não pode conter parâmetros, query ou fragmento.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BackendConfigurationError("A porta do endpoint é inválida.") from exc
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    authority = host if port is None else f"{host}:{port}"
    return f"{parsed.scheme}://{authority}", parsed.path


def normalize_backend_endpoint(backend: str, raw: str | None) -> NormalizedEndpoint:
    """Validate provider-specific endpoint forms and return canonical URLs."""

    if backend == "ollama":
        if not raw or not raw.strip():
            return NormalizedEndpoint(api_base=None, service_root=None)
        root, path = _parse_http_url(raw, allow_missing_scheme=True)
        if path not in {"", "/"}:
            raise BackendConfigurationError("O endpoint Ollama não pode conter path.")
        return NormalizedEndpoint(api_base=root, service_root=root)

    if backend == "llama-swap":
        if not raw or not raw.strip():
            raise BackendConfigurationError("Informe o endpoint do llama-swap.")
        root, path = _parse_http_url(raw, allow_missing_scheme=False)
        if path not in {"", "/", "/v1"}:
            raise BackendConfigurationError("O endpoint llama-swap deve apontar para a raiz ou /v1.")
        return NormalizedEndpoint(api_base=f"{root}/v1", service_root=root)

    raise BackendConfigurationError(f"Backend desconhecido: {backend}")


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}:
        return False
    return default


def _migrate_settings(raw: dict[str, object]) -> BackendSettings:
    backend = str(raw.get("backend") or DEFAULT_BACKEND)
    model = str(raw.get("model") or DEFAULT_MODEL)
    api_base = raw.get("api_base")
    if "backend" not in raw:
        mode = str(raw.get("ollama_mode") or "local").strip().lower()
        backend = "ollama"
        api_base = raw.get("ollama_endpoint") if mode == "remote" else None
    endpoint = normalize_backend_endpoint(backend, str(api_base) if api_base is not None else None)
    return BackendSettings(
        backend=backend,
        api_base=endpoint.api_base,
        model=model,
        batch_size=_as_int(raw.get("batch_size"), 15),
        timeout=_as_int(raw.get("timeout"), 300),
        enable_cache=_as_bool(raw.get("enable_cache"), True),
        cache_file=str(raw.get("cache_file") or "translation_cache.json"),
    )


def load_backend_settings(path: str | Path) -> BackendSettings:
    """Load canonical settings, migrating legacy Ollama-only configuration."""

    config_path = Path(path)
    if not config_path.exists():
        return BackendSettings()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return BackendSettings()
    if not isinstance(raw, dict):
        return BackendSettings()
    return _migrate_settings(raw)


def save_backend_settings(path: str | Path, settings: BackendSettings) -> None:
    """Persist canonical settings while retaining legacy Ollama rollback keys."""

    endpoint = normalize_backend_endpoint(settings.backend, settings.api_base)
    payload: dict[str, object] = {
        "backend": settings.backend,
        "api_base": endpoint.api_base,
        "model": settings.model,
        "batch_size": settings.batch_size,
        "timeout": settings.timeout,
        "enable_cache": settings.enable_cache,
        "cache_file": settings.cache_file,
        "ollama_mode": "remote" if settings.backend == "ollama" and endpoint.api_base else "local",
        "ollama_endpoint": endpoint.api_base if settings.backend == "ollama" and endpoint.api_base else "",
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
