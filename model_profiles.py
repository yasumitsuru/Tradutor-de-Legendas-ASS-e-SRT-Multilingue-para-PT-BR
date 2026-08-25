"""Small registry for known model identities and compatibility state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping


BACKEND_SUPPORTED = "BACKEND_SUPPORTED"
MODEL_DISCOVERED = "MODEL_DISCOVERED"
MODEL_PROFILE_KNOWN = "MODEL_PROFILE_KNOWN"
STRUCTURALLY_VALIDATED = "STRUCTURALLY_VALIDATED"
PRODUCTION_APPROVED = "PRODUCTION_APPROVED"

EXPERIMENTAL = "EXPERIMENTAL"
UNVALIDATED = "UNVALIDATED"
NOT_PRODUCTION_APPROVED = "NOT_PRODUCTION_APPROVED"

QWEN36_MODEL_ID = "Qwen3.6-28B-REAP20-A3B-Q4_K_M"
SUPPORTED_BACKEND_IDS = frozenset({"ollama", "llama-swap"})


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    display_name: str
    backend: str
    status: FrozenSet[str]
    generation_defaults: Mapping[str, object]
    max_parallel: int | None
    provenance: str
    is_default: bool = False


@dataclass(frozen=True)
class ModelSelection:
    model_id: str
    backend: str
    profile: ModelProfile | None
    status: FrozenSet[str]
    provenance: str


QWEN36_PROFILE = ModelProfile(
    model_id=QWEN36_MODEL_ID,
    display_name="Qwen3.6 28B REAP20 A3B Q4_K_M",
    backend="llama-swap",
    status=frozenset(
        {
            BACKEND_SUPPORTED,
            MODEL_DISCOVERED,
            MODEL_PROFILE_KNOWN,
            EXPERIMENTAL,
            UNVALIDATED,
            NOT_PRODUCTION_APPROVED,
        }
    ),
    generation_defaults=MappingProxyType({}),
    max_parallel=None,
    provenance="registered",
)

_PROFILES = {(QWEN36_PROFILE.backend, QWEN36_PROFILE.model_id): QWEN36_PROFILE}


def get_model_profile(backend: str, model: str) -> ModelProfile | None:
    """Return a registered profile only for an exact backend/model identity."""

    return _PROFILES.get((backend, model))


def select_model(backend: str, model: str, *, discovered: bool = False) -> ModelSelection:
    """Describe a selected model without assigning unknown models a profile."""

    profile = get_model_profile(backend, model)
    if profile is not None:
        return ModelSelection(
            model_id=model,
            backend=backend,
            profile=profile,
            status=profile.status,
            provenance=profile.provenance,
        )

    status = set()
    if backend in SUPPORTED_BACKEND_IDS:
        status.add(BACKEND_SUPPORTED)
    if discovered:
        status.add(MODEL_DISCOVERED)
    return ModelSelection(
        model_id=model,
        backend=backend,
        profile=None,
        status=frozenset(status),
        provenance="runtime-discovered" if discovered else "user-provided",
    )
