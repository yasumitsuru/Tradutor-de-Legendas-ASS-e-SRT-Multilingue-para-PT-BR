from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from subtitle_formats import ASSFormatHandler, SRTFormatHandler
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

    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
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
            if len(self.calls) <= 2:
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
    assert [temperature for _prompt, temperature in client.calls] == [0.35, 0.35, 0.0]
    assert translator.config["temperature"] == 0.35


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
