from __future__ import annotations

import json
from pathlib import Path

import pysubs2
import pytest

from ass_regression import (
    AnalysisContext,
    CompressedTraceRecorder,
    RegressionReport,
    analyze_ass_pair,
    discover_ass_files,
    event_stable_id,
    file_sha256,
    trace_failure_diagnostics,
    write_reports,
)
from subtitle_formats import ASSFormatHandler


FIXTURES = Path(__file__).parent / "fixtures"


def _translated_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = FIXTURES / "regression_ass_cases.ass"
    handler = ASSFormatHandler()
    subs = handler.load(source)
    working = handler.prepare_document(subs)
    replacements = {
        0: ("Hello.", "Olá."),
        2: ("Move now.", "Mova-se agora."),
        3: ("Open ", "Abra "),
        4: ("Hallo.", "Olá."),
        5: ("the key", "a chave"),
    }
    for index, (before, after) in replacements.items():
        working.events[index].text = working.events[index].text.replace(before, after)
    working.events[0].text = working.events[0].text.replace("Hi.", "Oi.")
    working.events[2].text = working.events[2].text.replace(" White.", " Branco.")
    working.events[3].text = working.events[3].text.replace(
        "and mail", "e envie para"
    ).replace("now.", "agora.")
    output = tmp_path / "regression_ass_cases.pt.ass"
    handler.save(working, output)
    return source, output


def test_discovers_ass_files_case_insensitively_without_recursing(tmp_path: Path) -> None:
    for name in ("b.ass", "A.ASS", "notes.txt"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.ass").write_text("fixture", encoding="utf-8")

    assert [path.name for path in discover_ass_files(tmp_path)] == ["A.ASS", "b.ass"]


def test_file_hash_and_event_id_are_stable_and_traceable() -> None:
    source = FIXTURES / "regression_ass_cases.ass"
    handler = ASSFormatHandler()
    event = handler.load(source).events[0]

    first = event_stable_id(source.name, 0, event)
    second = event_stable_id(source.name, 0, event)

    assert first == second
    assert first.startswith("evt_")
    assert len(file_sha256(source)) == 64
    event.end += 1
    assert event_stable_id(source.name, 0, event) != first


def test_valid_pair_preserves_structure_tags_breaks_and_comments(tmp_path: Path) -> None:
    source, output = _translated_fixture(tmp_path)

    analysis = analyze_ass_pair(source, output)

    assert analysis.critical_count == 0
    assert analysis.metrics["events"] == 6
    assert analysis.metrics["comments"] == 1
    assert all(diagnostic.event_index != 1 for diagnostic in analysis.diagnostics)


def test_comment_changes_are_objective_but_comment_language_is_not_a_warning(
    tmp_path: Path,
) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    changed = handler.load(output)
    changed.events[1].text = "Comentário alterado"
    handler.save(changed, output)

    analysis = analyze_ass_pair(source, output)

    assert "COMMENT_MISMATCH" in {item.category for item in analysis.critical_diagnostics}
    assert all(item.event_index != 1 for item in analysis.semantic_warnings)


def test_structural_tag_break_and_placeholder_corruption_are_critical(
    tmp_path: Path,
) -> None:
    source, output = _translated_fixture(tmp_path)
    raw = output.read_text(encoding="utf-8")
    raw = raw.replace("0:00:03.00", "0:00:03.01", 1)
    raw = raw.replace(r"{\i0}\N- Oi.", r"{\b0}- Oi. [[[ASS_TAG_9999]]]", 1)
    output.write_text(raw, encoding="utf-8")

    categories = {item.category for item in analyze_ass_pair(source, output).critical_diagnostics}

    assert "STRUCTURE_MISMATCH" in categories
    assert "TAG_MISMATCH" in categories
    assert "BREAK_MISMATCH" in categories
    assert "PLACEHOLDER_LEAK" in categories


def test_foreign_protected_anchor_identifies_item_mapping_error(tmp_path: Path) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    changed = handler.load(output)
    changed.events[0].text += " https://example.com/a"
    handler.save(changed, output)

    categories = {item.category for item in analyze_ass_pair(source, output).critical_diagnostics}

    assert "ITEM_MAPPING_ERROR" in categories


def test_missing_and_invalid_outputs_are_critical(tmp_path: Path) -> None:
    source = FIXTURES / "regression_ass_cases.ass"
    missing = analyze_ass_pair(source, tmp_path / "missing.ass")
    invalid_path = tmp_path / "invalid.ass"
    invalid_path.write_text("not an ASS", encoding="utf-8")
    invalid = analyze_ass_pair(source, invalid_path)

    assert [item.category for item in missing.critical_diagnostics] == ["OUTPUT_MISSING"]
    assert missing.metrics == {
        "events": 6,
        "comments": 1,
        "translatable_events": 5,
    }
    assert "INVALID_OUTPUT_ASS" in {item.category for item in invalid.critical_diagnostics}


def test_translatable_metric_uses_the_production_skip_rules(tmp_path: Path) -> None:
    handler = ASSFormatHandler()
    source = handler.load(FIXTURES / "regression_ass_cases.ass")
    source.append(pysubs2.SSAEvent(start=14000, end=15000, text="uh"))
    source.append(pysubs2.SSAEvent(start=15000, end=16000, text="..."))
    source.append(pysubs2.SSAEvent(start=16000, end=17000, text="I"))
    source_path = tmp_path / "skip-rules.ass"
    handler.save(source, source_path)

    analysis = analyze_ass_pair(source_path, tmp_path / "missing.ass")

    assert analysis.metrics == {
        "events": 9,
        "comments": 1,
        "translatable_events": 6,
    }


def test_partial_translation_is_a_warning_by_default(tmp_path: Path) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    changed = handler.load(output)
    changed.events[0].text = r"{\an8}{\i1}I told you para não tocar nisso.{\i0}\N- Oi."
    handler.save(changed, output)

    analysis = analyze_ass_pair(source, output)

    assert analysis.critical_count == 0
    partial = [item for item in analysis.semantic_warnings if item.category == "PARTIAL_TRANSLATION"]
    assert partial
    assert partial[0].confidence in {"high", "medium"}
    report = RegressionReport.from_files(
        AnalysisContext(experiment="semantic", run_id="run-semantic"), [analysis]
    )
    assert report.exit_code() == 0
    assert report.exit_code(strict_semantic=True) == 1


def test_shared_words_and_preserved_terms_do_not_create_partial_false_positives(
    tmp_path: Path,
) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    changed = handler.load(output)
    changed.events[0].text = (
        r"{\an8}{\i1}As pessoas me chamam de Code Number One.{\i0}\N- Oi."
    )
    handler.save(changed, output)

    warnings = analyze_ass_pair(source, output).semantic_warnings

    assert all(item.category != "PARTIAL_TRANSLATION" for item in warnings)


def test_partial_german_and_portuguese_is_a_semantic_warning(tmp_path: Path) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    changed = handler.load(output)
    changed.events[0].text = r"{\an8}{\i1}Ich und du não sabemos disso.{\i0}\N- Oi."
    handler.save(changed, output)

    partial = [
        item
        for item in analyze_ass_pair(source, output).semantic_warnings
        if item.category == "PARTIAL_TRANSLATION"
    ]

    assert partial
    assert "alemão" in partial[0].message


def test_diagnostic_event_id_uses_raw_original_text_before_sanitization(
    tmp_path: Path,
) -> None:
    source, output = _translated_fixture(tmp_path)
    handler = ASSFormatHandler()
    original = handler.load(source)
    changed = handler.load(output)
    changed.events[4].text = ""
    handler.save(changed, output)

    analysis = analyze_ass_pair(source, output)
    diagnostic = next(item for item in analysis.critical_diagnostics if item.event_index == 4)

    assert diagnostic.event_id == event_stable_id(source.name, 4, original.events[4])
    assert diagnostic.original == "Hallo. {English annotation}"


def test_reports_are_compact_machine_and_human_readable(tmp_path: Path) -> None:
    source, output = _translated_fixture(tmp_path)
    file_analysis = analyze_ass_pair(source, output)
    report = RegressionReport.from_files(
        context=AnalysisContext(experiment="unit", run_id="run-001"),
        files=[file_analysis],
    )

    json_path, markdown_path = write_reports(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["context"]["experiment"] == "unit"
    assert payload["metrics"]["files"] == 1
    assert "ASS REAL-WORLD REGRESSION TEST" in markdown
    assert "Eventos analisados" in markdown
    assert "diagnostics" not in payload["files"][0] or not payload["files"][0]["diagnostics"]


def test_trace_failures_are_linked_to_raw_source_events_even_for_legacy_trace(
    tmp_path: Path,
) -> None:
    source = FIXTURES / "regression_ass_cases.ass"
    trace_path = tmp_path / "trace.jsonl.gz"
    file_name = source.name
    with CompressedTraceRecorder(trace_path) as trace:
        trace(
            {
                "event": "individual_attempt_started",
                "file": str(source),
                "batch_num": 4,
                "item_ids": [2],
                "items": [{"item_id": 2, "event_index": 4, "text_hash": "a" * 64}],
            }
        )
        # Old baseline traces did not repeat ``items`` on the terminal event.
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 4,
                "item_id": 2,
                "reason": "Item ausente ou duplicado na recuperação individual.",
            }
        )

    diagnostics = trace_failure_diagnostics(trace_path, {file_name: source})

    assert len(diagnostics[file_name]) == 1
    diagnostic = diagnostics[file_name][0]
    assert diagnostic.category == "MODEL_PROTOCOL_ERROR"
    assert diagnostic.event_index == 4
    assert diagnostic.original == "Hallo. {English annotation}"
    assert diagnostic.event_id is not None
    assert "batch 4" in diagnostic.message


@pytest.mark.parametrize(
    "completion_event",
    ["individual_recovery_completed", "line_break_recovery_completed"],
)
def test_trace_failure_is_not_terminal_after_matching_valid_recovery(
    tmp_path: Path, completion_event: str,
) -> None:
    source = FIXTURES / "regression_ass_cases.ass"
    trace_path = tmp_path / "trace.jsonl.gz"
    file_name = source.name
    with CompressedTraceRecorder(trace_path) as trace:
        trace(
            {
                "event": "individual_attempt_started",
                "file": str(source),
                "batch_num": 4,
                "item_ids": [2],
                "items": [{"item_id": 2, "event_index": 4, "text_hash": "a" * 64}],
            }
        )
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 4,
                "item_id": 2,
                "reason": "Item ausente ou duplicado na recuperação individual.",
            }
        )
        trace(
            {
                "event": completion_event,
                "file": str(source),
                "batch_num": 4,
                "item_id": 2,
            }
        )

    diagnostics = trace_failure_diagnostics(trace_path, {file_name: source})

    assert diagnostics[file_name] == []


def test_trace_recovery_only_clears_prior_failure_with_exact_identity(
    tmp_path: Path,
) -> None:
    source = FIXTURES / "regression_ass_cases.ass"
    trace_path = tmp_path / "trace.jsonl.gz"
    file_name = source.name
    cases = [
        (1, 1, 0),
        (2, 1, 2),
        (2, 2, 3),
        (3, 1, 4),
        (4, 1, 5),
    ]
    with CompressedTraceRecorder(trace_path) as trace:
        for batch_num, item_id, event_index in cases:
            trace(
                {
                    "event": "individual_attempt_started",
                    "file": str(source),
                    "batch_num": batch_num,
                    "item_ids": [item_id],
                    "items": [
                        {
                            "item_id": item_id,
                            "event_index": event_index,
                            "text_hash": f"{event_index:064x}",
                        }
                    ],
                }
            )

        # A conclusão anterior não pode esconder uma falha que ocorreu depois.
        trace(
            {
                "event": "line_break_recovery_completed",
                "file": str(source),
                "batch_num": 1,
                "item_id": 1,
            }
        )
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 1,
                "item_id": 1,
                "reason": "terminal depois da recuperação",
            }
        )

        # Uma recuperação de outro ITEM não resolve a falha pendente.
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 2,
                "item_id": 1,
                "reason": "identidade diferente",
            }
        )
        trace(
            {
                "event": "line_break_recovery_completed",
                "file": str(source),
                "batch_num": 2,
                "item_id": 2,
            }
        )

        # Uma recuperação que também falhou não resolve a falha original.
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 3,
                "item_id": 1,
                "reason": "recuperação sem sucesso",
            }
        )
        trace(
            {
                "event": "line_break_recovery_failed",
                "file": str(source),
                "batch_num": 3,
                "item_id": 1,
            }
        )

        # Somente esta falha tem uma recuperação posterior e inequívoca.
        trace(
            {
                "event": "item_failed",
                "file": str(source),
                "batch_num": 4,
                "item_id": 1,
                "reason": "recuperada depois",
            }
        )
        trace(
            {
                "event": "line_break_recovery_completed",
                "file": str(source),
                "batch_num": 4,
                "item_id": 1,
            }
        )

    diagnostics = trace_failure_diagnostics(trace_path, {file_name: source})

    assert [(item.event_index, item.message) for item in diagnostics[file_name]] == [
        (0, "Falha definitiva no batch 1, ITEM 0001: terminal depois da recuperação"),
        (2, "Falha definitiva no batch 2, ITEM 0001: identidade diferente"),
        (4, "Falha definitiva no batch 3, ITEM 0001: recuperação sem sucesso"),
    ]


def test_trace_summary_uses_compact_prompt_rollup(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl.gz"
    with CompressedTraceRecorder(trace_path) as trace:
        trace({"event": "batch_attempt_started", "prompt": "prompt A"})
        trace({"event": "batch_attempt_started", "prompt": "prompt B"})
        summary = trace.summary()

    assert summary["unique_prompt_count"] == 2
    assert len(summary["prompt_sha256_rollup"]) == 64
    assert "prompt_sha256" not in summary
