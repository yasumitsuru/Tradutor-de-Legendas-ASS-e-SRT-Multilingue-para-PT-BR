from __future__ import annotations

from pathlib import Path

import pytest
import pysubs2

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
    sanitize_ass_text,
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


@pytest.mark.parametrize(
    ("newline", "with_bom"),
    [("\n", False), ("\r\n", False), ("\n", True), ("\r\n", True)],
)
def test_srt_loads_lf_crlf_and_utf8_bom(
    tmp_path: Path, newline: str, with_bom: bool
) -> None:
    content = newline.join(
        [
            "1",
            "00:00:01,000 --> 00:00:02,000",
            "Olá.",
            "",
            "2",
            "00:00:03,000 --> 00:00:04,000",
            "- First line",
            "- Second line",
            "",
        ]
    )
    payload = content.encode("utf-8")
    if with_bom:
        payload = b"\xef\xbb\xbf" + payload
    source = tmp_path / "Episode's Test.SRT"
    source.write_bytes(payload)

    subs = SRTFormatHandler().load(source)

    assert len(subs.events) == 2
    assert subs.events[0].text == "Olá."
    assert subs.events[1].text == r"- First line\N- Second line"


def test_srt_preserves_empty_and_unusual_blocks(tmp_path: Path) -> None:
    source = tmp_path / "unusual.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n[MUSIC] ♪\n\n",
        encoding="utf-8",
    )
    handler = SRTFormatHandler()
    loaded = handler.load(source)
    destination = tmp_path / "unusual.pt.srt"
    handler.save(loaded, destination)
    reloaded = handler.load(destination)

    assert len(reloaded.events) == 2
    assert reloaded.events[0].text == ""
    assert reloaded.events[1].text == "[MUSIC] ♪"


def test_srt_preserves_non_sequential_original_cue_indices(tmp_path: Path) -> None:
    source = tmp_path / "indices.SRT"
    source.write_text(
        "007\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
        "42\n00:00:03,000 --> 00:00:04,000\nSecond\n\n",
        encoding="utf-8",
    )
    handler = SRTFormatHandler()
    loaded = handler.load(source)
    destination = tmp_path / "indices.pt.SRT"

    handler.save(loaded, destination)
    reloaded = handler.load(destination)

    assert loaded._srt_cue_indices == ("007", "42")
    assert reloaded._srt_cue_indices == ("007", "42")
    raw = destination.read_text(encoding="utf-8")
    assert raw.startswith("007\n")
    assert "\n42\n00:00:03,000" in raw


@pytest.mark.parametrize(
    "markup",
    [
        "<i>Hello</i>",
        "<b>Hello</b>",
        "<u>Hello</u>",
        '<font color="#FF0000">Hello</font>',
        "<i><b>Hello</b></i>",
    ],
)
def test_srt_supported_html_markup_keeps_exact_position(markup: str) -> None:
    handler = SRTFormatHandler()
    prepared = handler.prepare_text(markup)

    restored = handler.restore_text(prepared, prepared.model_text.replace("Hello", "Olá"))

    assert restored == markup.replace("Hello", "Olá")
    assert "<" not in prepared.model_text


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


def test_srt_validation_rejects_reordered_markers() -> None:
    handler = SRTFormatHandler()
    prepared = handler.prepare_text("<i>Hello</i>")
    first, second = prepared.markers
    reordered = prepared.model_text.replace(first.token, "TEMP", 1).replace(
        second.token, first.token, 1
    ).replace("TEMP", second.token, 1)

    with pytest.raises(SubtitleValidationError, match="ordem"):
        handler.restore_text(prepared, reordered)


def test_srt_save_rejects_invalid_timing_and_broken_simple_html(tmp_path: Path) -> None:
    handler = SRTFormatHandler()
    invalid_timing = pysubs2.SSAFile()
    invalid_timing.append(pysubs2.SSAEvent(start=2000, end=1000, text="Hello"))
    with pytest.raises(SubtitleValidationError, match="horário final"):
        handler.save(invalid_timing, tmp_path / "invalid-time.srt")

    broken_html = pysubs2.SSAFile()
    broken_html.append(pysubs2.SSAEvent(start=1000, end=2000, text="<i>Hello"))
    with pytest.raises(SubtitleValidationError, match="sem fechamento"):
        handler.save(broken_html, tmp_path / "invalid-html.srt")


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
    assert reloaded.events[0].text == r"{\an8}{\i1}Eu estou aqui{\i0}\N- Você está pronto?"
    assert reloaded.events[0].marginl == 12
    assert reloaded.events[0].marginr == 14
    assert reloaded.events[0].marginv == 16
    assert reloaded.events[0].effect == "fade"
    assert reloaded.events[1].type == "Comment"
    assert reloaded.events[1].text == "Translator note"
    assert reloaded.events[2].text == r"Wait\hfor me."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Warte, Oogami! {Oogami! Wait, Oogami!}", "Warte, Oogami!"),
        (r"{\i1}Hello{\i0}", r"{\i1}Hello{\i0}"),
        (
            r"{\i1}Der „Schlüssel“ ist im Korb,{The key is in the basket,\Nif you wouldn’t mind.\i0}",
            r"{\i1}Der „Schlüssel“ ist im Korb,{\i0}",
        ),
        (
            r"{\i1}wir verlassen uns auf dich. {The key is in the basket,\Nif you wouldn’t mind.}{wieder Fail vom Engsub}{\i0}",
            r"{\i1}wir verlassen uns auf dich. {\i0}",
        ),
    ],
)
def test_ass_text_comments_are_removed_without_losing_real_overrides(
    source: str, expected: str
) -> None:
    assert sanitize_ass_text(source) == expected


@pytest.mark.parametrize(
    "tag",
    [
        r"{\pos(100,200)}",
        r"{\move(10,20,100,200)}",
        r"{\clip(0,0,1920,1080)}",
        r"{\t(0,500,\fs40\bord3)}",
        r"{\fad(200,300)}",
        r"{\fade(0,255,0,0,200,700,900)}",
        r"{\c&HFFFFFF&}",
        r"{\1c&HFFFFFF&}",
        r"{\alpha&H80&}",
    ],
)
def test_ass_complex_override_commands_are_preserved_exactly(tag: str) -> None:
    assert sanitize_ass_text(f"{tag}Hello") == f"{tag}Hello"


def test_ass_sanitizer_is_idempotent() -> None:
    source = r"{\i1}Text {English,\Ncomment\i0}{\pos(100,200)}"

    once = sanitize_ass_text(source)

    assert sanitize_ass_text(once) == once


def test_ass_sanitizer_preserves_unrelated_trailing_whitespace() -> None:
    assert sanitize_ass_text("Visible text  ") == "Visible text  "


def test_ass_prepare_document_sanitizes_dialogue_once_and_preserves_comments() -> None:
    source = pysubs2.SSAFile()
    source.append(
        pysubs2.SSAEvent(
            start=1000,
            end=2000,
            text="Hello {English annotation}",
            type="Dialogue",
        )
    )
    source.append(
        pysubs2.SSAEvent(
            start=2000,
            end=3000,
            text="Translator {note}",
            type="Comment",
        )
    )

    prepared = ASSFormatHandler().prepare_document(source)

    assert prepared is not source
    assert source.events[0].text == "Hello {English annotation}"
    assert prepared.events[0].text == "Hello"
    assert prepared.events[1].text == "Translator {note}"


def test_internal_artifacts_are_never_accepted() -> None:
    for artifact in (
        "[[[SRT_TAG_0001]]]",
        "[[ASS_LB_1]]",
        "SRT_TAG_0001",
        "ASS_NBSP",
        "ITEM_0001",
        "<<<ITEM_0001>>>",
        "<<<END_ITEM_0001>>>",
    ):
        with pytest.raises(SubtitleValidationError):
            assert_no_internal_artifacts(artifact)
