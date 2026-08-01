from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_formats import (
    ASSFormatHandler,
    SRTFormatHandler,
    SubtitleFormatError,
    SubtitleValidationError,
    assert_no_internal_artifacts,
    count_subtitle_formats,
    get_format_handler,
    is_supported_subtitle,
    iter_subtitle_files,
    preserve_extension_output_path,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_supported_files_are_discovered_case_insensitively(tmp_path: Path) -> None:
    for filename in ("episode.ASS", "episode2.sRt", "notes.txt"):
        (tmp_path / filename).write_text("fixture", encoding="utf-8")

    discovered = iter_subtitle_files(tmp_path)

    assert [path.name for path in discovered] == ["episode.ASS", "episode2.sRt"]
    assert [path.name for path in iter_subtitle_files(tmp_path, "srt")] == ["episode2.sRt"]
    assert count_subtitle_formats(discovered) == {"ass": 1, "srt": 1}
    assert is_supported_subtitle("EPISODE.SRT")
    assert preserve_extension_output_path("EPISODE.SRT", tmp_path).name == "EPISODE.pt.SRT"


def test_invalid_filter_and_extension_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(SubtitleFormatError, match="Filtro"):
        iter_subtitle_files(tmp_path, "vtt")
    with pytest.raises(SubtitleFormatError, match="não suportado"):
        get_format_handler("episode.vtt")


def test_srt_round_trip_preserves_timing_order_text_and_markup(tmp_path: Path) -> None:
    handler = SRTFormatHandler()
    subs = handler.load(FIXTURES / "sample.srt")
    original_events = [(event.start, event.end, event.text) for event in subs.events]

    destination = tmp_path / "roundtrip.srt"
    handler.save(subs, destination)
    reloaded = handler.load(destination)

    assert [(event.start, event.end, event.text) for event in reloaded.events] == original_events
    raw_output = destination.read_text(encoding="utf-8")
    assert '<font color="#ffcc00">' in raw_output
    assert r"{\an8}" in raw_output
    assert r"\N" not in raw_output
    assert "SRT_SOURCE_" not in raw_output


def test_srt_protection_restores_tags_breaks_and_dialogue_hyphens() -> None:
    handler = SRTFormatHandler()
    source = r'{\an8}<i>Hello.</i>\N- How are you?'
    prepared = handler.prepare_text(source)

    assert "Hello." in prepared.cleaned_text
    assert prepared.line_structure == (r"\N",)
    assert any(marker.kind == "prefix" and marker.value == "- " for marker in prepared.markers)

    translated_model_text = prepared.model_text.replace("Hello.", "Olá.").replace(
        "How are you?", "Como você está?"
    )
    restored = handler.restore_text(prepared, translated_model_text)

    assert restored == r'{\an8}<i>Olá.</i>\N- Como você está?'
    assert "SRT_TAG_" not in restored


def test_srt_protection_keeps_urls_emails_and_variables_out_of_model_text() -> None:
    handler = SRTFormatHandler()
    source = "Open https://example.com/a?q=1 for dev@example.com and ${USER}."
    prepared = handler.prepare_text(source)

    assert "https://" not in prepared.model_text
    assert "dev@example.com" not in prepared.model_text
    assert "${USER}" not in prepared.model_text
    translated = prepared.model_text.replace("Open ", "Abra ").replace(" for ", " para ").replace(
        " and ", " e "
    )
    assert handler.restore_text(prepared, translated) == (
        "Abra https://example.com/a?q=1 para dev@example.com e ${USER}."
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("SRT_LINE_BREAK_", "SRT_BROKEN_", 1),
        lambda text: text + r" {\an8}",
        lambda text: text + "\ntexto extra",
        lambda text: text + " <<<ITEM_9999>>>",
    ],
)
def test_srt_validation_rejects_damaged_or_invented_structure(mutator) -> None:
    handler = SRTFormatHandler()
    prepared = handler.prepare_text(r"Hello\N- There")

    with pytest.raises(SubtitleValidationError):
        handler.restore_text(prepared, mutator(prepared.model_text))


def test_ass_round_trip_preserves_styles_comments_and_commands(tmp_path: Path) -> None:
    handler = ASSFormatHandler()
    subs = handler.load(FIXTURES / "sample.ass")
    prepared = handler.prepare_text(subs.events[0].text)
    translated = prepared.model_text.replace("I am here", "Eu estou aqui").replace(
        "Are you ready?", "Você está pronto?"
    )
    restored = handler.restore_text(prepared, translated)
    rebuilt = handler.rebuild(subs, {0: restored})
    destination = tmp_path / "translated.ass"
    handler.save(rebuilt, destination)
    reloaded = handler.load(destination)

    assert reloaded.styles.keys() == subs.styles.keys()
    assert reloaded.events[0].start == subs.events[0].start
    assert reloaded.events[0].end == subs.events[0].end
    assert reloaded.events[0].text == r"{\an8}<i>Eu estou aqui</i>\N- Você está pronto?"
    assert reloaded.events[1].type == "Comment"
    assert reloaded.events[1].text == "Translator note"
    assert reloaded.events[2].text == r"Wait\hfor me."


def test_internal_artifacts_are_never_accepted() -> None:
    for artifact in (
        "[[[SRT_TAG_0001]]]",
        "[[ASS_LB_1]]",
        "<<<ITEM_0001>>>",
        "<<<END_ITEM_0001>>>",
    ):
        with pytest.raises(SubtitleValidationError):
            assert_no_internal_artifacts(artifact)
