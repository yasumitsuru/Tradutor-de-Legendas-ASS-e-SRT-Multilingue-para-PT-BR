from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from subtitle_formats import ASSFormatHandler, SRTFormatHandler
from inference_backend import GenerationResult
from translation_engine import (
    CACHE_SCHEMA_VERSION,
    CONFIG,
    ITEM_BLOCK_RE,
    FixedASSTranslator,
    IncompleteTranslationError,
)


FIXTURES = Path(__file__).parent / "fixtures"


class TranslatingOllama:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def show(self, _model: str) -> dict[str, str]:
        return {"model": "available"}

    def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
        self.prompts.append(prompt)
        blocks: list[str] = []
        for match in ITEM_BLOCK_RE.finditer(prompt):
            item_id = match.group(1)
            body = match.group(2)
            translated = (
                body.replace("Hello, world.", "Olá, mundo.")
                .replace("How are you?", "Como você está?")
                .replace("I am here.", "Eu estou aqui.")
                .replace("I am here", "Eu estou aqui")
                .replace("Are you ready?", "Você está pronto?")
                .replace("Yes.", "Sim.")
                .replace("No!", "Não!")
                .replace("Wait", "Espere")
                .replace("for me.", "por mim.")
                .replace("Visit ", "Visite ")
            )
            blocks.append(
                f"<<<ITEM_{item_id}>>>\n{translated}\n<<<END_ITEM_{item_id}>>>"
            )
        return {"response": "\n".join(blocks)}


def translator_config(tmp_path: Path, **overrides) -> dict:
    return {
        **CONFIG,
        "cache_file": str(tmp_path / "cache.json"),
        "enable_cache": False,
        "retry_count": 2,
        "retry_delay": 0,
        "batch_size": 10,
        **overrides,
    }


def test_translator_uses_an_injected_backend_for_generation(tmp_path: Path) -> None:
    class FakeBackend:
        backend_id = "fake"

        def __init__(self) -> None:
            self.requests = []

        def generate(self, request) -> GenerationResult:
            self.requests.append(request)
            return GenerationResult(text="resultado", backend="fake", model=request.model)

        def ensure_available(self, _model: str) -> None:
            return None

        def close(self) -> None:
            return None

    backend = FakeBackend()
    translator = FixedASSTranslator(translator_config(tmp_path), backend=backend)

    response = asyncio.run(translator._generate("prompt", temperature=0.0))

    assert response == "resultado"
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.model == translator.config["model"]
    assert request.system_prompt == translator.config["system_prompt"]
    assert request.prompt == "prompt"
    assert request.temperature == 0.0
    assert request.top_p == 0.9
    assert request.max_tokens == translator.config["max_tokens"]
    assert request.timeout == translator.config["timeout"]


def test_backend_result_metadata_never_reaches_item_validation(tmp_path: Path) -> None:
    class MetadataBackend:
        backend_id = "llama-swap"

        def generate(self, request) -> GenerationResult:
            return GenerationResult(
                text="<<<ITEM_0001>>>\nTraduzido\n<<<END_ITEM_0001>>>",
                backend=self.backend_id,
                model=request.model,
                metadata={"response": "metadata must not be parsed"},
            )

        def ensure_available(self, _model: str) -> None:
            return None

        def close(self) -> None:
            return None

    handler = SRTFormatHandler()
    translator = FixedASSTranslator(translator_config(tmp_path), backend=MetadataBackend())

    outcomes = asyncio.run(
        translator.translate_single_batch([handler.prepare_text("Original")], handler, 1, 1)
    )

    assert [outcome.text for outcome in outcomes] == ["Traduzido"]
    assert outcomes[0].used_fallback is False


def test_default_language_configuration_and_system_prompt_are_multilingual() -> None:
    prompt = CONFIG["system_prompt"].casefold()

    assert CONFIG["source_language"] == "auto"
    assert "idioma ou idiomas" in prompt
    assert "português" in prompt
    assert "nome" in prompt
    assert "honorífico" in prompt
    assert "do inglês" not in prompt


def test_auto_prompts_detect_multiple_languages_per_item_and_keep_protocol(
    tmp_path: Path,
) -> None:
    handler = SRTFormatHandler()
    translator = FixedASSTranslator(translator_config(tmp_path))
    prepared = handler.prepare_text("Ich brauche the key para abrir a porta.")

    batch_prompt = translator._build_item_prompt([(1, prepared)])
    individual_prompt = translator._build_individual_prompt(1, prepared)

    for prompt in (batch_prompt, individual_prompt):
        normalized = prompt.casefold()
        assert "detecte" in normalized
        assert "idioma ou idiomas" in normalized
        assert "mesma fala" in normalized or "mesmo item" in normalized
        assert "já estiver em português" in normalized
        assert "palavras isoladas significativas" in normalized
        assert "yes" in normalized
        assert "ja" in normalized
        assert "sim" in normalized
        assert "primeira pessoa" in normalized
        for first_person_form in ("I", "I'm", "I've", "I'll", "I'd"):
            assert first_person_form in prompt
        assert "nome" in normalized
        assert "honorífico" in normalized
        assert "exclusivamente inglês" not in normalized
        assert prompt.count("<<<ITEM_0001>>>") == 1
        assert prompt.count("<<<END_ITEM_0001>>>") == 1
        assert prepared.model_text in prompt


def test_manual_source_language_remains_available_without_a_rigid_language_list(
    tmp_path: Path,
) -> None:
    handler = SRTFormatHandler()
    translator = FixedASSTranslator(
        translator_config(tmp_path, source_language="Klingon"),
    )

    prompt = translator._build_item_prompt([(1, handler.prepare_text("nuqneH"))])

    assert "Klingon" in prompt
    assert "detecção automática" not in prompt.casefold()


def test_multilingual_items_are_translated_as_complete_semantic_units(
    tmp_path: Path,
) -> None:
    translations = {
        "Ich brauche the key para abrir a porta.": "Preciso da chave para abrir a porta.",
        "I don't know, aber ele já foi embora.": "Eu não sei, mas ele já foi embora.",
        "Eu já encontrei the key.": "Eu já encontrei a chave.",
        "Eu não sei o que aconteceu.": "Eu não sei o que aconteceu.",
        "Oogami-kun!": "Oogami-kun!",
    }

    class MultilingualOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            blocks = []
            for match in ITEM_BLOCK_RE.finditer(prompt):
                item_id, body = match.group(1), match.group(2).strip()
                blocks.append(
                    f"<<<ITEM_{item_id}>>>\n{translations[body]}\n"
                    f"<<<END_ITEM_{item_id}>>>"
                )
            return {"response": "\n".join(blocks)}

    handler = SRTFormatHandler()
    prepared = [handler.prepare_text(source) for source in translations]
    translator = FixedASSTranslator(
        translator_config(tmp_path),
        MultilingualOllama(),
    )

    outcomes = asyncio.run(translator.translate_single_batch(prepared, handler, 1, 1))

    assert [outcome.text for outcome in outcomes] == list(translations.values())
    assert all(not outcome.used_fallback for outcome in outcomes)


def test_auto_prompt_preserves_each_ass_marker_exactly_once(tmp_path: Path) -> None:
    handler = ASSFormatHandler()
    prepared = handler.prepare_text(r"{\i1}Ich brauche the key{\i0}\NJá volto.")
    translator = FixedASSTranslator(translator_config(tmp_path))

    prompt = translator._build_item_prompt([(1, prepared)])

    assert prepared.markers
    assert all(prompt.count(marker.token) == 1 for marker in prepared.markers)


def test_english_first_person_forms_reach_the_model_unchanged_and_translate(
    tmp_path: Path,
) -> None:
    translations = {
        "I": "Eu",
        "I'm ready.": "Estou pronto.",
        "I've arrived.": "Cheguei.",
        "I'll wait.": "Vou esperar.",
        "I'd rather stay.": "Eu preferiria ficar.",
    }
    seen: list[str] = []

    class FirstPersonOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            blocks = []
            for match in ITEM_BLOCK_RE.finditer(prompt):
                item_id, body = match.group(1), match.group(2).strip()
                seen.append(body)
                translated = translations.get(body, body)
                blocks.append(
                    f"<<<ITEM_{item_id}>>>\n{translated}\n<<<END_ITEM_{item_id}>>>"
                )
            return {"response": "\n".join(blocks)}

    handler = SRTFormatHandler()
    prepared = [handler.prepare_text(source) for source in translations]
    translator = FixedASSTranslator(translator_config(tmp_path), FirstPersonOllama())

    outcomes = asyncio.run(translator.translate_single_batch(prepared, handler, 1, 1))

    assert seen == list(translations)
    assert [outcome.text for outcome in outcomes] == list(translations.values())


@pytest.mark.parametrize("text", ["I", "A", "É", "Yes", "Ja", "Sim"])
def test_meaningful_isolated_words_are_not_skipped(tmp_path: Path, text: str) -> None:
    translator = FixedASSTranslator(translator_config(tmp_path))

    assert translator.should_skip_line(text) == (False, "")


@pytest.mark.parametrize(
    "text",
    ["", " ", ".", "...", "♪", "-", "_", "(door closes)", "[music]", "uh", "wow"],
)
def test_multilingual_short_text_support_preserves_existing_skip_rules(
    tmp_path: Path, text: str
) -> None:
    translator = FixedASSTranslator(translator_config(tmp_path))

    assert translator.should_skip_line(text)[0] is True


@pytest.mark.parametrize(
    ("format_name", "source_text"),
    [
        ("ass", r"{\an8}Hello\N- There"),
        ("srt", r"<i>Hello</i>\N- There"),
    ],
)
def test_cache_key_changes_for_every_context_factor(
    tmp_path: Path, format_name: str, source_text: str
) -> None:
    handler = ASSFormatHandler() if format_name == "ass" else SRTFormatHandler()
    prepared = handler.prepare_text(source_text)
    base = FixedASSTranslator(translator_config(tmp_path))
    base_key = base._get_text_hash(prepared)

    variants = [
        FixedASSTranslator(translator_config(tmp_path, model="another-model")),
        FixedASSTranslator(translator_config(tmp_path, source_language="Spanish")),
        FixedASSTranslator(translator_config(tmp_path, target_language="European Portuguese")),
        FixedASSTranslator(translator_config(tmp_path, prompt_version="future-prompt")),
    ]
    assert all(variant._get_text_hash(prepared) != base_key for variant in variants)

    other_handler = SRTFormatHandler() if format_name == "ass" else ASSFormatHandler()
    other_prepared = other_handler.prepare_text(source_text)
    assert base._get_text_hash(other_prepared) != base_key


def test_cache_uses_versioned_schema(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    translator = FixedASSTranslator(
        translator_config(tmp_path, cache_file=str(cache_path), enable_cache=True)
    )
    translator.cache["key"] = "value"
    translator._save_cache()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert CACHE_SCHEMA_VERSION == 5
    assert payload["schema_version"] == 5
    assert payload["prompt_version"] == CONFIG["prompt_version"]
    assert payload["entries"] == {"key": "value"}


def test_previous_cache_schema_is_ignored(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION - 1,
                "prompt_version": "subtitle-items-v2",
                "entries": {"old-key": "Old cached translation"},
            }
        ),
        encoding="utf-8",
    )

    translator = FixedASSTranslator(
        translator_config(tmp_path, cache_file=str(cache_path), enable_cache=True)
    )

    assert translator.cache == {}


def test_cache_key_distinguishes_auto_from_explicit_source_language(tmp_path: Path) -> None:
    prepared = SRTFormatHandler().prepare_text("Hello")
    automatic = FixedASSTranslator(translator_config(tmp_path, source_language="auto"))
    manual = FixedASSTranslator(translator_config(tmp_path, source_language="English"))

    assert automatic._get_text_hash(prepared) != manual._get_text_hash(prepared)


def test_cache_key_isolated_by_backend_and_generation_profile(tmp_path: Path) -> None:
    class OllamaBackend:
        backend_id = "ollama"

        def generate(self, _request) -> GenerationResult:
            raise AssertionError("cache key generation must not call a backend")

        def ensure_available(self, _model: str) -> None:
            return None

        def close(self) -> None:
            return None

    class LlamaSwapBackend(OllamaBackend):
        backend_id = "llama-swap"

    prepared = SRTFormatHandler().prepare_text("Same subtitle")
    ollama = FixedASSTranslator(
        translator_config(tmp_path), backend=OllamaBackend()
    )
    llama_swap = FixedASSTranslator(
        translator_config(tmp_path), backend=LlamaSwapBackend()
    )
    changed_profile = FixedASSTranslator(
        translator_config(tmp_path, temperature=0.3), backend=OllamaBackend()
    )

    assert ollama._get_text_hash(prepared) != llama_swap._get_text_hash(prepared)
    assert ollama._get_text_hash(prepared) != changed_profile._get_text_hash(prepared)


def test_legacy_ass_escape_helpers_remain_compatible(tmp_path: Path) -> None:
    translator = FixedASSTranslator(translator_config(tmp_path))
    source = r"First line\NSecond line\hnow"
    protected = translator.protect_ass_breaks_for_model(source)

    assert protected == "First line[[[ASS_BR]]]Second line[[[ASS_NBSP]]]now"
    assert translator.repair_ass_escapes(source, "Primeira[[ASS_BR]]Segunda[[ASS_NBSP]]agora") == (
        r"Primeira\NSegunda\hagora"
    )
    repaired = translator.repair_ass_escapes(r"First\NSecond", "Primeira Segunda")
    assert repaired.count(r"\N") == 1


def test_batch_retries_only_the_missing_item(tmp_path: Path) -> None:
    class MissingOnceOllama:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            self.prompts.append(prompt)
            matches = list(ITEM_BLOCK_RE.finditer(prompt))
            selected = matches[:1] if len(self.prompts) == 1 else matches
            response = "\n".join(
                f"<<<ITEM_{match.group(1)}>>>\nTraduzido {match.group(1)}\n"
                f"<<<END_ITEM_{match.group(1)}>>>"
                for match in selected
            )
            return {"response": response}

    client = MissingOnceOllama()
    translator = FixedASSTranslator(translator_config(tmp_path), client)
    handler = SRTFormatHandler()
    prepared = [handler.prepare_text("First"), handler.prepare_text("Second")]

    outcomes = asyncio.run(translator.translate_single_batch(prepared, handler, 1, 1))

    assert [outcome.used_fallback for outcome in outcomes] == [False, False]
    assert len(client.prompts) == 2
    assert "<<<ITEM_0001>>>" not in client.prompts[1]
    assert "<<<ITEM_0002>>>" in client.prompts[1]


def test_explicit_batch_envelope_contract_prevents_real_missing_start_response(
    tmp_path: Path,
) -> None:
    real_malformed_response = (
        "ITEM_0001\n"
        "Antes que você entre no estado de perda e então "
        "[[[ASS_TAG_0001]]]me[[[ASS_TAG_0002]]] atrapalhe.\n"
        "<<<END_ITEM_0001>>>"
    )

    class ProtocolSensitiveOllama:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            self.prompts.append(prompt)
            if "exatamente um par completo por ID" not in prompt:
                return {"response": real_malformed_response}
            return {
                "response": (
                    "<<<ITEM_0001>>>\n"
                    "Antes que você entre no estado de perda e então "
                    "[[[ASS_TAG_0001]]]me[[[ASS_TAG_0002]]] atrapalhe.\n"
                    "<<<END_ITEM_0001>>>"
                )
            }

    source = r"Bevor du in den Lost-Zustand gerätst und {\i1}mir{\i0} im Weg stehst."
    client = ProtocolSensitiveOllama()
    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), client
    )
    handler = ASSFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text(source)], handler, 1, 1
        )
    )[0]

    assert outcome.text == (
        r"Antes que você entre no estado de perda e então {\i1}me{\i0} atrapalhe."
    )
    assert not outcome.used_fallback
    assert len(client.prompts) == 1


def test_batch_parser_rejects_nested_item_instead_of_contaminating_outer_body() -> None:
    response = (
        "<<<ITEM_0001>>>\n"
        "ASTRONOMIA_ALPHA\n"
        "<<<ITEM_0002>>>\n"
        "CULINARIA_BETA\n"
        "<<<END_ITEM_0002>>>\n"
        "<<<END_ITEM_0001>>>"
    )

    parsed, warning = FixedASSTranslator._parse_item_response(response)

    assert parsed == {}
    assert warning is not None
    assert "aninhado" in warning.casefold()


@pytest.mark.parametrize(
    "response",
    [
        (
            "<<<ITEM_0001>>>\nPrimeiro\n<<<END_ITEM_0001>>>\n"
            "<<<ITEM_0001>>>\nDuplicado\n<<<END_ITEM_0001>>>"
        ),
        "<<<ITEM_0001>>>\nBloco parcial",
        "Primeiro corpo sem ID\nSegundo corpo sem ID",
        "Here is the translation: Primeiro corpo sem ID",
        "```text\nPrimeiro corpo sem ID\n```",
    ],
)
def test_batch_parser_does_not_map_ambiguous_or_unidentified_bodies(
    response: str,
) -> None:
    parsed, _warning = FixedASSTranslator._parse_item_response(response)

    assert 1 not in parsed


def test_synthetic_domains_remain_bound_to_ids_even_when_blocks_are_reordered() -> None:
    response = (
        "<<<ITEM_0003>>>\nFUTEBOL_GAMMA: gol no estádio\n<<<END_ITEM_0003>>>\n"
        "<<<ITEM_0001>>>\nASTRONOMIA_ALPHA: estrela distante\n<<<END_ITEM_0001>>>\n"
        "<<<ITEM_0002>>>\nCULINARIA_BETA: panela no fogo\n<<<END_ITEM_0002>>>"
    )

    parsed, warning = FixedASSTranslator._parse_item_response(response)

    assert warning is None
    assert parsed == {
        1: "ASTRONOMIA_ALPHA: estrela distante",
        2: "CULINARIA_BETA: panela no fogo",
        3: "FUTEBOL_GAMMA: gol no estádio",
    }


def test_external_response_residue_does_not_invalidate_valid_item(tmp_path: Path) -> None:
    class ResidueOllama:
        def generate(self, **_kwargs) -> dict[str, str]:
            return {
                "response": (
                    "Here is the translation:\n\n"
                    "<<<ITEM_0001>>>\nOlá\n<<<END_ITEM_0001>>>"
                )
            }

    translator = FixedASSTranslator(translator_config(tmp_path), ResidueOllama())
    handler = SRTFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch([handler.prepare_text("Hello")], handler, 1, 1)
    )[0]

    assert outcome.text == "Olá"
    assert not outcome.used_fallback


def test_reordered_items_are_mapped_by_id_without_cross_contamination(
    tmp_path: Path,
) -> None:
    class ReorderedOllama:
        def generate(self, **_kwargs) -> dict[str, str]:
            return {
                "response": (
                    "<<<ITEM_0002>>>\nSegundo: BETAQUASAR\n<<<END_ITEM_0002>>>\n"
                    "<<<ITEM_0001>>>\nPrimeiro: ALPHAORCHID\n<<<END_ITEM_0001>>>"
                )
            }

    handler = SRTFormatHandler()
    translator = FixedASSTranslator(
        translator_config(tmp_path), ReorderedOllama()
    )

    outcomes = asyncio.run(
        translator.translate_single_batch(
            [
                handler.prepare_text("First: ALPHAORCHID"),
                handler.prepare_text("Second: BETAQUASAR"),
            ],
            handler,
            1,
            1,
        )
    )

    assert [item.text for item in outcomes] == [
        "Primeiro: ALPHAORCHID",
        "Segundo: BETAQUASAR",
    ]


def test_valid_items_survive_residue_and_only_missing_item_is_retried(
    tmp_path: Path,
) -> None:
    class PartialResidueOllama:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return {
                    "response": (
                        "Here is the translation:\n\n"
                        "<<<ITEM_0001>>>\nPrimeiro\n<<<END_ITEM_0001>>>"
                    )
                }
            return {
                "response": "<<<ITEM_0002>>>\nSegundo\n<<<END_ITEM_0002>>>"
            }

    client = PartialResidueOllama()
    translator = FixedASSTranslator(translator_config(tmp_path), client)
    handler = SRTFormatHandler()
    prepared = [handler.prepare_text("First"), handler.prepare_text("Second")]

    outcomes = asyncio.run(translator.translate_single_batch(prepared, handler, 1, 1))

    assert [outcome.text for outcome in outcomes] == ["Primeiro", "Segundo"]
    assert len(client.prompts) == 2
    assert "<<<ITEM_0001>>>" not in client.prompts[1]
    assert "<<<ITEM_0002>>>" in client.prompts[1]


def test_unexpected_item_is_discarded_and_logged_without_invalidating_expected_item(
    tmp_path: Path, capsys
) -> None:
    class ExtraItemOllama:
        def generate(self, **_kwargs) -> dict[str, str]:
            return {
                "response": (
                    "<<<ITEM_0001>>>\nEsperado\n<<<END_ITEM_0001>>>\n"
                    "<<<ITEM_9999>>>\nNão deve entrar\n<<<END_ITEM_9999>>>"
                )
            }

    handler = SRTFormatHandler()
    translator = FixedASSTranslator(translator_config(tmp_path), ExtraItemOllama())

    outcome = asyncio.run(
        translator.translate_single_batch([handler.prepare_text("Expected")], handler, 1, 1)
    )[0]
    output = capsys.readouterr().out

    assert outcome.text == "Esperado"
    assert not outcome.used_fallback
    assert "ITEM inesperado 9999 descartado" in output


def test_individual_recovery_uses_zero_temperature_without_mutating_config(
    tmp_path: Path,
) -> None:
    class RecoveringOllama:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float]] = []

        def generate(self, *, prompt: str, options: dict, **_kwargs) -> dict[str, str]:
            self.calls.append((prompt, options["temperature"]))
            if options["temperature"] != 0.0:
                return {"response": "invalid response"}
            item_id = ITEM_BLOCK_RE.search(prompt).group(1)
            return {
                "response": f"<<<ITEM_{item_id}>>>\nRecuperado\n<<<END_ITEM_{item_id}>>>"
            }

    client = RecoveringOllama()
    translator = FixedASSTranslator(
        translator_config(tmp_path, temperature=0.35, retry_count=2), client
    )
    handler = SRTFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch([handler.prepare_text("Recover me")], handler, 1, 1)
    )[0]

    assert outcome.text == "Recuperado"
    assert not outcome.used_fallback
    assert [temperature for _prompt, temperature in client.calls] == [0.35, 0.0]
    assert translator.config["temperature"] == 0.35


def test_individual_recovery_accepts_guarded_bare_body_observed_from_real_ollama(
    tmp_path: Path,
) -> None:
    class RealBareRecoveryOllama:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                return {"response": "invalid batch response"}
            return {
                "response": (
                    "Sakurakouji-san, infelizmente ninguém consegue me entender"
                    "[[[ASS_LINE_BREAK_0001]]]."
                )
            }

    client = RealBareRecoveryOllama()
    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), client
    )
    handler = ASSFormatHandler()
    prepared = handler.prepare_text(
        r"Sakurakouji-san, unglücklicherweise kann mich \Nweder irgendjemand verstehen,"
    )

    outcome = asyncio.run(
        translator.translate_single_batch([prepared], handler, 1, 1)
    )[0]

    assert outcome.text == (
        "Sakurakouji-san, infelizmente ninguém consegue me entender\\N."
    )
    assert not outcome.used_fallback
    assert client.calls == 2


def test_individual_bare_recovery_still_rejects_missing_protected_marker(
    tmp_path: Path,
) -> None:
    class MissingMarkerRecoveryOllama:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                return {"response": "invalid batch response"}
            return {
                "response": (
                    "Sakura, isso é a primeira vez que você está tão ocupada com um menino, "
                    "ou não?"
                )
            }

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), MissingMarkerRecoveryOllama()
    )
    handler = ASSFormatHandler()
    prepared = handler.prepare_text(
        r"Sakura, das ist ja das erste Mal, \Ndass dich ein Junge so beschäftigt, oder?"
    )

    outcome = asyncio.run(
        translator.translate_single_batch([prepared], handler, 1, 1)
    )[0]

    assert outcome.used_fallback
    assert "marcador protegido" in str(outcome.error)


def test_missing_ass_line_break_recovers_by_translating_unambiguous_segments(
    tmp_path: Path,
) -> None:
    source = (
        r"Sakura, das ist ja das erste Mal, \N"
        r"dass dich ein Junge so beschäftigt, oder?"
    )
    observed_missing_marker_response = (
        "Sakura, esta é a primeira vez que um garoto mexe tanto com você, não é?"
    )
    segment_translations = {
        "Sakura, das ist ja das erste Mal,": "Sakura, esta é a primeira vez,",
        "dass dich ein Junge so beschäftigt, oder?": (
            "que um garoto mexe tanto com você, não é?"
        ),
    }

    class ObservedLineBreakFailureOllama:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            self.prompts.append(prompt)
            matches = list(ITEM_BLOCK_RE.finditer(prompt))
            if "[[[ASS_LINE_BREAK_0001]]]" in prompt:
                item_id = matches[0].group(1)
                return {
                    "response": (
                        f"<<<ITEM_{item_id}>>>\n{observed_missing_marker_response}\n"
                        f"<<<END_ITEM_{item_id}>>>"
                    )
                }
            return {
                "response": "\n".join(
                    f"<<<ITEM_{match.group(1)}>>>\n"
                    f"{segment_translations[match.group(2).strip()]}\n"
                    f"<<<END_ITEM_{match.group(1)}>>>"
                    for match in matches
                )
            }

    client = ObservedLineBreakFailureOllama()
    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), client
    )
    handler = ASSFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text(source)], handler, 1, 1
        )
    )[0]

    assert outcome.text == (
        r"Sakura, esta é a primeira vez,\N"
        r"que um garoto mexe tanto com você, não é?"
    )
    assert not outcome.used_fallback
    assert "[[[ASS_LINE_BREAK_0001]]]" in client.prompts[0]
    assert "[[[ASS_LINE_BREAK_0001]]]" not in client.prompts[-1]
    recovery_prompts = [
        prompt
        for prompt in client.prompts
        if "[[[ASS_LINE_BREAK_0001]]]" not in prompt
    ]
    assert len(recovery_prompts) == 2
    assert all(len(list(ITEM_BLOCK_RE.finditer(prompt))) == 1 for prompt in recovery_prompts)


def test_ass_line_break_segment_recovery_preserves_multiple_breaks_tags_and_prefixes(
    tmp_path: Path,
) -> None:
    source = r"{\i1}Start\Nend{\i0}\N- Wait"

    class StructuredSegmentsOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            matches = list(ITEM_BLOCK_RE.finditer(prompt))
            if "ASS_LINE_BREAK" in prompt:
                item_id = matches[0].group(1)
                return {
                    "response": (
                        f"<<<ITEM_{item_id}>>>\n"
                        "[[[ASS_TAG_0001]]]Comece, termine e "
                        "espere[[[ASS_TAG_0003]]][[[ASS_PREFIX_0005]]]\n"
                        f"<<<END_ITEM_{item_id}>>>"
                    )
                }
            blocks = []
            for match in matches:
                item_id = match.group(1)
                translated = (
                    match.group(2).strip()
                    .replace("Start", "Comece")
                    .replace("end", "fim")
                    .replace("Wait", "Espere")
                )
                blocks.append(
                    f"<<<ITEM_{item_id}>>>\n{translated}\n<<<END_ITEM_{item_id}>>>"
                )
            return {"response": "\n".join(blocks)}

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), StructuredSegmentsOllama()
    )
    handler = ASSFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text(source)], handler, 1, 1
        )
    )[0]

    assert outcome.text == r"{\i1}Comece\Nfim{\i0}\N- Espere"
    assert not outcome.used_fallback


def test_ass_line_break_segment_recovery_is_atomic_when_one_segment_fails(
    tmp_path: Path,
) -> None:
    source = r"First\NSecond"

    class OneFailedSegmentOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            matches = list(ITEM_BLOCK_RE.finditer(prompt))
            if "ASS_LINE_BREAK" in prompt:
                item_id = matches[0].group(1)
                return {
                    "response": (
                        f"<<<ITEM_{item_id}>>>\nPrimeiro e segundo\n"
                        f"<<<END_ITEM_{item_id}>>>"
                    )
                }
            blocks = []
            for match in matches:
                body = match.group(2).strip()
                if body == "Second":
                    continue
                item_id = match.group(1)
                blocks.append(
                    f"<<<ITEM_{item_id}>>>\nPrimeiro\n<<<END_ITEM_{item_id}>>>"
                )
            return {
                "response": "\n".join(blocks) or "Here is the translation: invalid"
            }

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), OneFailedSegmentOllama()
    )
    handler = ASSFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text(source)], handler, 1, 1
        )
    )[0]

    assert outcome.text == source
    assert outcome.used_fallback
    assert "segmento" in str(outcome.error).casefold()


def test_repositioned_ass_line_break_recovers_at_the_original_segment_boundary(
    tmp_path: Path,
) -> None:
    source = r"First\NSecond"

    class RepositionedLineBreakOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            matches = list(ITEM_BLOCK_RE.finditer(prompt))
            if "ASS_LINE_BREAK" in prompt:
                item_id = matches[0].group(1)
                return {
                    "response": (
                        f"<<<ITEM_{item_id}>>>\n"
                        "[[[ASS_LINE_BREAK_0001]]]Primeiro segundo\n"
                        f"<<<END_ITEM_{item_id}>>>"
                    )
                }
            translations = {"First": "Primeiro", "Second": "Segundo"}
            return {
                "response": "\n".join(
                    f"<<<ITEM_{match.group(1)}>>>\n"
                    f"{translations[match.group(2).strip()]}\n"
                    f"<<<END_ITEM_{match.group(1)}>>>"
                    for match in matches
                )
            }

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), RepositionedLineBreakOllama()
    )
    handler = ASSFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text(source)], handler, 1, 1
        )
    )[0]

    assert outcome.text == r"Primeiro\NSegundo"
    assert not outcome.used_fallback


def test_individual_recovery_rejects_multiple_item_blocks(tmp_path: Path) -> None:
    class MultipleItemsRecoveryOllama:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                return {"response": "invalid batch response"}
            return {
                "response": (
                    "<<<ITEM_0001>>>\nRecuperado\n<<<END_ITEM_0001>>>\n"
                    "<<<ITEM_9999>>>\nOutra fala\n<<<END_ITEM_9999>>>"
                )
            }

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), MultipleItemsRecoveryOllama()
    )
    handler = SRTFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text("Recover me")], handler, 1, 1
        )
    )[0]

    assert outcome.used_fallback
    assert "múltiplos" in str(outcome.error).casefold()


@pytest.mark.parametrize(
    "unsafe_response",
    [
        "Here is the translation: Recuperado",
        "Resultado: Recuperado",
        "# Recuperado",
        "**Recuperado**",
        "*Recuperado*",
        "__Recuperado__",
        "_Recuperado_",
        "~~Recuperado~~",
        "`Recuperado`",
        "[Recuperado](https://example.com)",
        "```text\nRecuperado\n```",
        "Recuperado\nOutra fala",
        "Recuperado\n<<<END_ITEM_0001>>>",
    ],
)
def test_individual_bare_recovery_rejects_unsafe_unwrapped_content(
    tmp_path: Path, unsafe_response: str
) -> None:
    class UnsafeBareRecoveryOllama:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs) -> dict[str, str]:
            self.calls += 1
            return {
                "response": "invalid batch response" if self.calls == 1 else unsafe_response
            }

    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), UnsafeBareRecoveryOllama()
    )
    handler = SRTFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text("Recover me")], handler, 1, 1
        )
    )[0]

    assert outcome.used_fallback


def test_optional_trace_observes_selective_retry_and_individual_recovery(
    tmp_path: Path,
) -> None:
    class TraceableRecoveryOllama:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "response": "<<<ITEM_0001>>>\nPrimeiro\n<<<END_ITEM_0001>>>"
                }
            if self.calls == 2:
                return {
                    "response": (
                        "<<<ITEM_0002>>>\n[[[SRT_TAG_9999]]]\n"
                        "<<<END_ITEM_0002>>>"
                    )
                }
            return {
                "response": "<<<ITEM_0002>>>\nSegundo\n<<<END_ITEM_0002>>>"
            }

    trace: list[dict] = []
    translator = FixedASSTranslator(
        translator_config(
            tmp_path,
            retry_count=2,
            trace_hook=trace.append,
        ),
        TraceableRecoveryOllama(),
    )
    handler = SRTFormatHandler()

    outcomes = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text("First"), handler.prepare_text("Second")],
            handler,
            3,
            9,
        )
    )

    assert [outcome.text for outcome in outcomes] == ["Primeiro", "Segundo"]
    attempts = [item for item in trace if item["event"] == "batch_attempt_started"]
    assert [item["item_ids"] for item in attempts] == [[1, 2], [2]]
    assert [item["attempt"] for item in attempts] == [1, 2]
    assert all(item["batch_num"] == 3 and item["total_batches"] == 9 for item in attempts)
    assert all("<<<ITEM_" in item["prompt"] for item in attempts)
    rejected = [item for item in trace if item["event"] == "item_rejected"]
    assert [(item["item_id"], item["mode"]) for item in rejected] == [
        (2, "batch"),
        (2, "batch"),
    ]
    assert any("ausente" in item["reason"].casefold() for item in rejected)
    assert any("desconhecido" in item["reason"].casefold() for item in rejected)
    individual = [item for item in trace if item["event"] == "individual_attempt_started"]
    assert len(individual) == 1
    assert individual[0]["item_ids"] == [2]
    assert individual[0]["temperature"] == 0.0
    accepted = [item for item in trace if item["event"] == "item_accepted"]
    assert [(item["item_id"], item["mode"]) for item in accepted] == [
        (1, "batch"),
        (2, "individual"),
    ]
    individual_recovered = [
        item for item in trace if item["event"] == "individual_recovery_completed"
    ]
    assert [
        (item["item_id"], item["mode"], item["batch_num"], item["total_batches"])
        for item in individual_recovered
    ] == [(2, "individual", 3, 9)]
    responses = [item for item in trace if item["event"] == "model_response"]
    assert len(responses) == 3
    assert all(item["response"] for item in responses)


def test_definitive_failure_trace_retains_source_event_descriptor(tmp_path: Path) -> None:
    class AlwaysMissingOllama:
        def generate(self, **_kwargs) -> dict[str, str]:
            return {"response": ""}

    trace: list[dict] = []
    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1, trace_hook=trace.append),
        AlwaysMissingOllama(),
    )
    handler = SRTFormatHandler()
    prepared = handler.prepare_text("Never returned")

    outcome = asyncio.run(
        translator.translate_single_batch(
            [prepared],
            handler,
            1,
            1,
            trace_items=[{"event_index": 42, "text_hash": "a" * 64}],
        )
    )[0]

    assert outcome.used_fallback
    failed = next(item for item in trace if item["event"] == "item_failed")
    assert failed["items"] == [
        {"item_id": 1, "event_index": 42, "text_hash": "a" * 64}
    ]


def test_trace_hook_failure_never_changes_translation_behavior(tmp_path: Path) -> None:
    client = TranslatingOllama()

    def broken_trace(_event: dict) -> None:
        raise RuntimeError("trace sink unavailable")

    translator = FixedASSTranslator(
        translator_config(tmp_path, trace_hook=broken_trace),
        client,
    )
    handler = SRTFormatHandler()

    outcome = asyncio.run(
        translator.translate_single_batch(
            [handler.prepare_text("Hello, world.")], handler, 1, 1
        )
    )[0]

    assert outcome.text == "Olá, mundo."
    assert len(client.prompts) == 1


def test_file_trace_links_batch_items_to_source_event_indices(tmp_path: Path) -> None:
    trace: list[dict] = []
    source = FIXTURES / "sample.ass"
    output = tmp_path / "sample.pt.ass"
    translator = FixedASSTranslator(
        translator_config(tmp_path, trace_hook=trace.append),
        TranslatingOllama(),
    )

    stats = asyncio.run(translator.translate_file(source, output))

    file_started = [item for item in trace if item["event"] == "file_started"]
    file_completed = [item for item in trace if item["event"] == "file_completed"]
    attempts = [item for item in trace if item["event"] == "batch_attempt_started"]
    assert len(file_started) == len(file_completed) == 1
    assert Path(file_started[0]["file"]).name == "sample.ass"
    assert Path(file_completed[0]["output"]).name == "sample.pt.ass"
    assert file_completed[0]["stats"] == stats
    traced_items = [item for attempt in attempts for item in attempt["items"]]
    assert {item["event_index"] for item in traced_items} == {0, 2}
    assert all(len(item["text_hash"]) == 64 for item in traced_items)


def test_permanently_missing_item_falls_back_without_losing_valid_item(tmp_path: Path) -> None:
    class AlwaysMissingSecondOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            first = ITEM_BLOCK_RE.search(prompt)
            if first is None or first.group(1) != "0001":
                return {"response": ""}
            return {
                "response": "<<<ITEM_0001>>>\nPrimeiro traduzido\n<<<END_ITEM_0001>>>"
            }

    handler = SRTFormatHandler()
    prepared = [handler.prepare_text("First"), handler.prepare_text("Second")]
    translator = FixedASSTranslator(
        translator_config(tmp_path), AlwaysMissingSecondOllama()
    )

    outcomes = asyncio.run(translator.translate_single_batch(prepared, handler, 1, 1))

    assert outcomes[0].text == "Primeiro traduzido"
    assert not outcomes[0].used_fallback
    assert outcomes[1].text == "Second"
    assert outcomes[1].used_fallback


@pytest.mark.parametrize(
    "bad_body",
    [
        "[[[SRT_TAG_9999]]] tradução",
        r"{\an8} tradução",
        r"\N tradução",
        "1. tradução",
        "```json tradução ```",
        "Here is the translation: tradução",
        "tradução\nlinha extra",
    ],
)
def test_invalid_model_artifacts_fall_back_per_item(tmp_path: Path, bad_body: str) -> None:
    class BrokenOllama:
        def generate(self, *, prompt: str, **_kwargs) -> dict[str, str]:
            item_id = ITEM_BLOCK_RE.search(prompt).group(1)
            return {
                "response": f"<<<ITEM_{item_id}>>>\n{bad_body}\n<<<END_ITEM_{item_id}>>>"
            }

    handler = SRTFormatHandler()
    prepared = handler.prepare_text("Translate me")
    translator = FixedASSTranslator(translator_config(tmp_path), BrokenOllama())

    outcome = asyncio.run(translator.translate_single_batch([prepared], handler, 1, 1))[0]

    assert outcome.used_fallback
    assert outcome.text == prepared.original_text


def test_full_mocked_ollama_srt_pipeline_preserves_structure(tmp_path: Path) -> None:
    source = tmp_path / "Episode's Sample.SRT"
    source.write_bytes((FIXTURES / "sample.srt").read_bytes())
    destination = tmp_path / "Episode's Sample.pt.SRT"
    client = TranslatingOllama()
    translator = FixedASSTranslator(translator_config(tmp_path), client)
    original = SRTFormatHandler().load(source)

    stats = asyncio.run(translator.translate_file(source, destination))

    translated = SRTFormatHandler().load(destination)
    assert stats["failed"] == 0
    assert len(translated.events) == len(original.events)
    assert translated._srt_cue_indices == original._srt_cue_indices
    assert [(event.start, event.end) for event in translated.events] == [
        (event.start, event.end) for event in original.events
    ]
    assert translated.events[0].text == r"<i>Olá, mundo.</i>\N- Como você está?"
    assert r'{\an8}<font color="#ffcc00">Eu estou aqui.</font>' in translated.events[1].text
    raw = destination.read_text(encoding="utf-8")
    assert "[[[" not in raw
    assert "ITEM_" not in raw
    assert r"\N" not in raw
    assert all("<i>" not in prompt and r"{\an8}" not in prompt for prompt in client.prompts)
    assert all("https://example.com" not in prompt for prompt in client.prompts)


def test_full_mocked_ollama_ass_pipeline_preserves_regressions(tmp_path: Path) -> None:
    source = FIXTURES / "sample.ass"
    destination = tmp_path / "sample.pt.ass"
    translator = FixedASSTranslator(translator_config(tmp_path), TranslatingOllama())
    original = ASSFormatHandler().load(source)

    stats = asyncio.run(translator.translate_file(source, destination))

    translated = ASSFormatHandler().load(destination)
    assert stats["failed"] == 0
    assert translated.styles.keys() == original.styles.keys()
    assert [(event.start, event.end) for event in translated.events] == [
        (event.start, event.end) for event in original.events
    ]
    assert translated.events[0].text == (
        r"{\an8}{\i1}Eu estou aqui{\i0}\N- Você está pronto?"
    )
    assert translated.events[0].marginl == 12
    assert translated.events[0].marginr == 14
    assert translated.events[0].marginv == 16
    assert translated.events[0].effect == "fade"
    assert translated.events[1].type == "Comment"
    assert translated.events[2].text == r"Espere\hpor mim."


class AlwaysInvalidOllama:
    def generate(self, **_kwargs) -> dict[str, str]:
        return {"response": "invalid response"}


def test_definitive_failure_keeps_existing_output_unchanged_by_default(
    tmp_path: Path,
) -> None:
    source = FIXTURES / "sample.ass"
    destination = tmp_path / "sample.pt.ass"
    old_bytes = b"existing output from an earlier run"
    destination.write_bytes(old_bytes)
    translator = FixedASSTranslator(
        translator_config(tmp_path, retry_count=1), AlwaysInvalidOllama()
    )

    with pytest.raises(IncompleteTranslationError, match="Arquivo não será salvo"):
        asyncio.run(translator.translate_file(source, destination))

    assert destination.read_bytes() == old_bytes


def test_explicit_original_fallback_preserves_legacy_save_behavior(tmp_path: Path) -> None:
    source = FIXTURES / "sample.ass"
    destination = tmp_path / "sample.pt.ass"
    translator = FixedASSTranslator(
        translator_config(
            tmp_path,
            retry_count=1,
            allow_original_fallback=True,
        ),
        AlwaysInvalidOllama(),
    )

    stats = asyncio.run(translator.translate_file(source, destination))

    assert stats["failed"] > 0
    assert destination.exists()
    assert ASSFormatHandler().load(destination).events[0].text == (
        r"{\an8}{\i1}I am here{\i0}\N- Are you ready?"
    )


def test_serialized_candidate_failure_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = FIXTURES / "sample.ass"
    destination = tmp_path / "sample.pt.ass"
    old_bytes = b"existing output from an earlier run"
    destination.write_bytes(old_bytes)
    translator = FixedASSTranslator(translator_config(tmp_path), TranslatingOllama())
    original_validate = translator._validate_rebuilt_document
    validation_calls = 0

    def fail_second_validation(original, rebuilt, handler) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise RuntimeError("serialized candidate rejected")
        original_validate(original, rebuilt, handler)

    monkeypatch.setattr(translator, "_validate_rebuilt_document", fail_second_validation)

    with pytest.raises(RuntimeError, match="candidate rejected"):
        asyncio.run(translator.translate_file(source, destination))

    assert destination.read_bytes() == old_bytes
