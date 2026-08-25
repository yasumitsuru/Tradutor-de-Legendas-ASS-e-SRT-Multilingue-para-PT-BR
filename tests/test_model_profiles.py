from __future__ import annotations

from model_profiles import (
    BACKEND_SUPPORTED,
    EXPERIMENTAL,
    MODEL_DISCOVERED,
    MODEL_PROFILE_KNOWN,
    NOT_PRODUCTION_APPROVED,
    QWEN36_MODEL_ID,
    QWEN36_PROFILE,
    UNVALIDATED,
    get_model_profile,
    select_model,
)


def test_qwen36_profile_is_registered_only_for_llama_swap_and_is_not_default() -> None:
    profile = get_model_profile("llama-swap", QWEN36_MODEL_ID)

    assert profile == QWEN36_PROFILE
    assert profile.model_id == "Qwen3.6-28B-REAP20-A3B-Q4_K_M"
    assert profile.display_name == "Qwen3.6 28B REAP20 A3B Q4_K_M"
    assert profile.backend == "llama-swap"
    assert profile.is_default is False
    assert get_model_profile("ollama", QWEN36_MODEL_ID) is None


def test_qwen36_profile_marks_unverified_compatibility_without_runtime_defaults() -> None:
    profile = QWEN36_PROFILE

    assert {EXPERIMENTAL, UNVALIDATED, NOT_PRODUCTION_APPROVED}.issubset(profile.status)
    assert {BACKEND_SUPPORTED, MODEL_DISCOVERED, MODEL_PROFILE_KNOWN}.issubset(profile.status)
    assert profile.generation_defaults == {}
    assert profile.max_parallel is None
    assert profile.provenance == "registered"


def test_unknown_llama_swap_model_is_discovered_without_a_fabricated_profile() -> None:
    selection = select_model("llama-swap", "operator-provided-model", discovered=True)

    assert selection.profile is None
    assert selection.model_id == "operator-provided-model"
    assert selection.backend == "llama-swap"
    assert selection.status == frozenset({BACKEND_SUPPORTED, MODEL_DISCOVERED})
    assert selection.provenance == "runtime-discovered"
