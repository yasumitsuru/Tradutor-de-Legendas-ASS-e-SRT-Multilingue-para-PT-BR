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
