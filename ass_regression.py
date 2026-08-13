"""Deterministic analysis and reporting for real-world ASS regression runs."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pysubs2

from subtitle_formats import (
    ASSFormatHandler,
    PreparedSubtitleText,
    SubtitleFormatError,
    SubtitleValidationError,
    assert_no_internal_artifacts,
)


_SECTION_RE = re.compile(r"(?m)^\s*\[([^\]\r\n]+)\]\s*$")
_MODEL_PROTOCOL_RE = re.compile(
    r"(?:```|<<<\s*(?:END_)?ITEM_\d+\s*>>>|^\s*(?:translation|portuguese)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_REPEATED_WORD_RE = re.compile(r"\b([^\W\d_]{3,})\b(?:\s+\1\b){2,}", re.IGNORECASE)

_ENGLISH_WORDS = frozenset(
    {
        "a",
        "about",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "but",
        "can",
        "did",
        "do",
        "for",
        "from",
        "have",
        "hello",
        "here",
        "how",
        "i",
        "in",
        "is",
        "it",
        "know",
        "mail",
        "me",
        "my",
        "no",
        "not",
        "now",
        "of",
        "on",
        "open",
        "or",
        "that",
        "the",
        "this",
        "to",
        "told",
        "touch",
        "wait",
        "what",
        "white",
        "will",
        "with",
        "you",
        "your",
    }
)
_PORTUGUESE_WORDS = frozenset(
    {
        "a",
        "agora",
        "aqui",
        "as",
        "chave",
        "como",
        "com",
        "da",
        "de",
        "do",
        "e",
        "ela",
        "ele",
        "em",
        "encontrei",
        "está",
        "estou",
        "eu",
        "isso",
        "já",
        "me",
        "mova",
        "não",
        "o",
        "oi",
        "olá",
        "os",
        "para",
        "por",
        "que",
        "se",
        "sim",
        "tocar",
        "uma",
        "você",
    }
)
_GERMAN_WORDS = frozenset(
    {
        "aber",
        "das",
        "der",
        "die",
        "du",
        "ein",
        "eine",
        "für",
        "hallo",
        "ich",
        "ist",
        "mit",
        "nicht",
        "sie",
        "und",
        "wir",
    }
)
_SPANISH_WORDS = frozenset(
    {"como", "con", "de", "el", "ella", "en", "es", "hola", "la", "los", "no", "para", "que", "sí", "un", "una", "y"}
)
_EVENT_STRUCTURE_FIELDS = (
    "type",
    "start",
    "end",
    "style",
    "name",
    "marginl",
    "marginr",
    "marginv",
    "effect",
    "layer",
    "marked",
)


@dataclass(frozen=True)
class Diagnostic:
    """One objective failure or conservative semantic warning."""

    category: str
    severity: str
    confidence: str
    message: str
    file: str
    event_id: str | None = None
    event_index: int | None = None
    timestamp: str | None = None
    original: str | None = None
    translation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }


@dataclass
class FileAnalysis:
    """Analysis of one source/output pair."""

    source_path: str
    output_path: str
    source_hash: str
    output_hash: str | None
    metrics: dict[str, int]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def critical_diagnostics(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "FAIL"]

    @property
    def semantic_warnings(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "WARNING"]

    @property
    def critical_count(self) -> int:
        return len(self.critical_diagnostics)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source_path,
            "output": self.output_path,
            "source_sha256": self.source_hash,
            "output_sha256": self.output_hash,
            "passed": self.critical_count == 0,
            "metrics": dict(self.metrics),
        }
        if self.diagnostics:
            payload["diagnostics"] = [item.to_dict() for item in self.diagnostics]
        return payload


@dataclass
class AnalysisContext:
    """Reproducibility metadata for one experiment."""

    experiment: str
    run_id: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "started_at": self.started_at,
            **self.metadata,
        }


@dataclass
class RegressionReport:
    """Aggregate result for one isolated experiment."""

    context: AnalysisContext
    files: list[FileAnalysis]
    metrics: dict[str, Any]

    @classmethod
    def from_files(
        cls, context: AnalysisContext, files: Sequence[FileAnalysis]
    ) -> "RegressionReport":
        critical_categories: Counter[str] = Counter()
        warning_categories: Counter[str] = Counter()
        for file_analysis in files:
            critical_categories.update(
                item.category for item in file_analysis.critical_diagnostics
            )
            warning_categories.update(
                item.category for item in file_analysis.semantic_warnings
            )
        metrics: dict[str, Any] = {
            "files": len(files),
            "files_passed": sum(item.critical_count == 0 for item in files),
            "files_with_critical_failures": sum(item.critical_count > 0 for item in files),
            "events": sum(item.metrics.get("events", 0) for item in files),
            "comments": sum(item.metrics.get("comments", 0) for item in files),
            "translatable_events": sum(
                item.metrics.get("translatable_events", 0) for item in files
            ),
            "critical_failures": sum(critical_categories.values()),
            "semantic_warnings": sum(warning_categories.values()),
            "critical_by_category": dict(sorted(critical_categories.items())),
            "warnings_by_category": dict(sorted(warning_categories.items())),
        }
        return cls(context=context, files=list(files), metrics=metrics)

    def exit_code(self, strict_semantic: bool = False) -> int:
        if self.metrics["critical_failures"]:
            return 1
        if strict_semantic and any(
            item.confidence == "high"
            for file_analysis in self.files
            for item in file_analysis.semantic_warnings
        ):
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "metrics": self.metrics,
            "files": [item.to_dict() for item in self.files],
        }


class CompressedTraceRecorder:
    """Write diagnostic events as compact gzip JSONL and aggregate call metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = gzip.open(self.path, "wt", encoding="utf-8", newline="\n")
        self._counts: Counter[str] = Counter()
        self._prompt_hashes: set[str] = set()
        self._closed = False

    def __call__(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            return
        payload = dict(event)
        event_name = str(payload.get("event", "unknown"))
        self._counts[event_name] += 1
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            self._prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self._stream.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def summary(self) -> dict[str, Any]:
        prompt_rollup = hashlib.sha256(
            "\n".join(sorted(self._prompt_hashes)).encode("ascii")
        ).hexdigest()
        return {
            "batch_attempts": self._counts["batch_attempt_started"],
            "model_responses": self._counts["model_response"],
            "model_errors": self._counts["model_error"],
            "retries": self._counts["retry_scheduled"],
            "individual_attempts": self._counts["individual_attempt_started"],
            "items_accepted": self._counts["item_accepted"],
            "items_rejected": self._counts["item_rejected"],
            "items_failed": self._counts["item_failed"],
            "unique_prompt_count": len(self._prompt_hashes),
            "prompt_sha256_rollup": prompt_rollup,
            "events_by_type": dict(sorted(self._counts.items())),
        }

    def __enter__(self) -> "CompressedTraceRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def discover_ass_files(input_dir: str | Path) -> list[Path]:
    """Discover direct child ASS files case-insensitively and deterministically."""

    root = Path(input_dir)
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() == ".ass"
        ),
        key=lambda path: path.name.casefold(),
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_stable_id(filename: str, event_index: int, event: pysubs2.SSAEvent) -> str:
    text_hash = hashlib.sha256(event.text.encode("utf-8")).hexdigest()
    identity = "\0".join(
        (
            filename,
            str(event_index),
            str(event.start),
            str(event.end),
            event.style,
            text_hash,
        )
    )
    return f"evt_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _timestamp(event: pysubs2.SSAEvent) -> str:
    return f"{pysubs2.time.ms_to_str(event.start)} --> {pysubs2.time.ms_to_str(event.end)}"


def _diagnostic(
    category: str,
    severity: str,
    message: str,
    source_name: str,
    *,
    confidence: str = "objective",
    event_index: int | None = None,
    source_event: pysubs2.SSAEvent | None = None,
    output_text: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        category=category,
        severity=severity,
        confidence=confidence,
        message=message,
        file=source_name,
        event_id=(
            event_stable_id(source_name, event_index, source_event)
            if event_index is not None and source_event is not None
            else None
        ),
        event_index=event_index,
        timestamp=_timestamp(source_event) if source_event is not None else None,
        original=source_event.text if source_event is not None else None,
        translation=output_text,
    )


def _sections(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
    return tuple(match.group(1).strip() for match in _SECTION_RE.finditer(text))


def _style_snapshot(subs: pysubs2.SSAFile) -> dict[str, dict[str, Any]]:
    return {
        name: {key: repr(value) for key, value in sorted(vars(style).items())}
        for name, style in subs.styles.items()
    }


def _document_snapshot(subs: pysubs2.SSAFile) -> dict[str, Any]:
    return {
        "info": dict(subs.info),
        "aegisub_project": dict(subs.aegisub_project),
        "styles": _style_snapshot(subs),
        "fonts_opaque": dict(subs.fonts_opaque),
        "graphics_opaque": dict(subs.graphics_opaque),
    }


def _marker_groups(prepared: PreparedSubtitleText) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {
        "tags": [],
        "breaks": [],
        "prefixes": [],
        "anchors": [],
    }
    for marker in prepared.markers:
        if marker.kind in {"line_break", "non_break_space"}:
            groups["breaks"].append(marker.value)
        elif marker.kind == "prefix":
            groups["prefixes"].append(marker.value)
        elif marker.value.startswith("{") and marker.value.endswith("}"):
            groups["tags"].append(marker.value)
        else:
            groups["anchors"].append(marker.value)
    return {name: tuple(values) for name, values in groups.items()}


def _meaningful_text(text: str) -> bool:
    words = _WORD_RE.findall(text)
    return bool(words) and not (
        len(words) <= 3
        and text.strip().startswith(("(", "["))
        and text.strip().endswith((")", "]"))
    )


def _count_production_translatable_events(
    subs: pysubs2.SSAFile, handler: ASSFormatHandler
) -> int:
    """Use the backend's exact skip semantics instead of a second approximation."""

    from translation_engine import CONFIG, FixedASSTranslator

    translator = FixedASSTranslator({**CONFIG, "enable_cache": False})
    count = 0
    for event in subs.events:
        if not event.text or event.type == "Comment":
            continue
        prepared = handler.prepare_text(event.text)
        skip, _reason = translator.should_skip_line(prepared.cleaned_text)
        if not skip and prepared.cleaned_text:
            count += 1
    return count


def _word_hits(text: str, vocabulary: frozenset[str]) -> set[str]:
    return {word.casefold().replace("’", "'") for word in _WORD_RE.findall(text)} & vocabulary


def _semantic_diagnostics(
    source_name: str,
    event_index: int,
    source_event: pysubs2.SSAEvent,
    source_visible: str,
    output_visible: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    english_vocabulary = _ENGLISH_WORDS - _PORTUGUESE_WORDS
    german_vocabulary = _GERMAN_WORDS - _PORTUGUESE_WORDS - _ENGLISH_WORDS
    spanish_vocabulary = _SPANISH_WORDS - _PORTUGUESE_WORDS - _ENGLISH_WORDS
    portuguese_vocabulary = (
        _PORTUGUESE_WORDS - _ENGLISH_WORDS - _GERMAN_WORDS - _SPANISH_WORDS
    )
    foreign_by_language = {
        "inglês": _word_hits(output_visible, english_vocabulary),
        "alemão": _word_hits(output_visible, german_vocabulary),
        "espanhol": _word_hits(output_visible, spanish_vocabulary),
    }
    foreign_language, foreign_hits = max(
        foreign_by_language.items(), key=lambda item: len(item[1])
    )
    portuguese = _word_hits(output_visible, portuguese_vocabulary)
    source_foreign = (
        _word_hits(source_visible, _ENGLISH_WORDS)
        | _word_hits(source_visible, _GERMAN_WORDS)
        | _word_hits(source_visible, _SPANISH_WORDS)
    )
    normalized_source = " ".join(_WORD_RE.findall(source_visible.casefold()))
    normalized_output = " ".join(_WORD_RE.findall(output_visible.casefold()))

    if len(foreign_hits) >= 2 and len(portuguese) >= 2:
        diagnostics.append(
            _diagnostic(
                "PARTIAL_TRANSLATION",
                "WARNING",
                f"A saída mistura vocabulário provável em {foreign_language} "
                f"({', '.join(sorted(foreign_hits))}) e português.",
                source_name,
                confidence="high" if len(foreign_hits) >= 3 else "medium",
                event_index=event_index,
                source_event=source_event,
                output_text=output_visible,
            )
        )
    elif (
        normalized_source
        and normalized_source == normalized_output
        and len(source_foreign) >= 2
        and not portuguese
    ):
        diagnostics.append(
            _diagnostic(
                "UNTRANSLATED",
                "WARNING",
                "O texto visível permaneceu idêntico apesar de conter vocabulário estrangeiro provável.",
                source_name,
                confidence="high",
                event_index=event_index,
                source_event=source_event,
                output_text=output_visible,
            )
        )
    if len(source_visible) >= 40 and len(output_visible) < max(8, len(source_visible) // 4):
        diagnostics.append(
            _diagnostic(
                "TRUNCATION",
                "WARNING",
                "A saída é muito menor que o texto visível de origem; requer revisão humana.",
                source_name,
                confidence="medium",
                event_index=event_index,
                source_event=source_event,
                output_text=output_visible,
            )
        )
    if _REPEATED_WORD_RE.search(output_visible):
        diagnostics.append(
            _diagnostic(
                "DUPLICATION",
                "WARNING",
                "A saída contém repetição consecutiva improvável.",
                source_name,
                confidence="medium",
                event_index=event_index,
                source_event=source_event,
                output_text=output_visible,
            )
        )
    return diagnostics


def _read_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(Path(trace_path), "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _trace_item_identity(row: Mapping[str, Any]) -> tuple[str, int, int] | None:
    file_value = row.get("file")
    batch_num = row.get("batch_num")
    item_id = row.get("item_id")
    if (
        not isinstance(file_value, str)
        or not isinstance(batch_num, int)
        or not isinstance(item_id, int)
    ):
        return None
    return Path(file_value).name, batch_num, item_id


def _recovered_failure_row_indexes(rows: list[dict[str, Any]]) -> set[int]:
    """Find earlier failures superseded by an exact successful recovery."""

    successful_recovery_events = {
        "individual_recovery_completed",
        "line_break_recovery_completed",
    }
    pending: dict[tuple[str, int, int], list[int]] = {}
    recovered: set[int] = set()
    for row_index, row in enumerate(rows):
        identity = _trace_item_identity(row)
        if identity is None:
            continue
        event = row.get("event")
        if event == "item_failed":
            pending.setdefault(identity, []).append(row_index)
        elif event in successful_recovery_events:
            recovered.update(pending.pop(identity, ()))
    return recovered


def trace_failure_diagnostics(
    trace_path: str | Path,
    sources_by_name: Mapping[str, Path],
) -> dict[str, list[Diagnostic]]:
    """Turn unrecovered item failures into event-level objective diagnostics."""

    rows = _read_trace_rows(trace_path)
    recovered_failure_rows = _recovered_failure_row_indexes(rows)
    descriptors: dict[tuple[str, int, int], dict[str, Any]] = {}
    responses: dict[tuple[str, int, int], str] = {}
    loaded_sources: dict[str, pysubs2.SSAFile] = {}
    handler = ASSFormatHandler()
    result: dict[str, list[Diagnostic]] = {
        name: [] for name in sources_by_name
    }

    for row in rows:
        file_value = row.get("file")
        if not isinstance(file_value, str):
            continue
        file_name = Path(file_value).name
        batch_num = row.get("batch_num")
        if not isinstance(batch_num, int):
            continue
        items = row.get("items")
        if isinstance(items, list):
            for descriptor in items:
                if not isinstance(descriptor, dict):
                    continue
                item_id = descriptor.get("item_id")
                if isinstance(item_id, int):
                    descriptors[(file_name, batch_num, item_id)] = dict(descriptor)
        if row.get("event") == "model_response":
            response = row.get("response")
            item_ids = row.get("item_ids")
            if isinstance(response, str) and isinstance(item_ids, list):
                for item_id in item_ids:
                    if isinstance(item_id, int):
                        responses[(file_name, batch_num, item_id)] = response

    for row_index, row in enumerate(rows):
        if row_index in recovered_failure_rows:
            continue
        if row.get("event") != "item_failed":
            continue
        file_value = row.get("file")
        batch_num = row.get("batch_num")
        item_id = row.get("item_id")
        if not isinstance(file_value, str) or not isinstance(batch_num, int) or not isinstance(item_id, int):
            continue
        file_name = Path(file_value).name
        source_path = sources_by_name.get(file_name)
        if source_path is None:
            continue
        descriptor = descriptors.get((file_name, batch_num, item_id), {})
        event_index = descriptor.get("event_index")
        if not isinstance(event_index, int):
            continue
        if file_name not in loaded_sources:
            loaded_sources[file_name] = handler.load(source_path)
        source_document = loaded_sources[file_name]
        if not 0 <= event_index < len(source_document.events):
            continue
        source_event = source_document.events[event_index]
        reason = str(row.get("reason") or "Falha definitiva sem detalhe.")
        category = (
            "MODEL_PROTOCOL_ERROR"
            if "ausente ou duplicado" in reason.casefold()
            else "TRANSLATION_ITEM_FAILURE"
        )
        result.setdefault(file_name, []).append(
            _diagnostic(
                category,
                "FAIL",
                f"Falha definitiva no batch {batch_num}, ITEM {item_id:04d}: {reason}",
                file_name,
                event_index=event_index,
                source_event=source_event,
                output_text=responses.get((file_name, batch_num, item_id)),
            )
        )
    return result


def analyze_ass_pair(source_path: str | Path, output_path: str | Path) -> FileAnalysis:
    """Compare one translated ASS with the exact sanitized production base."""

    source = Path(source_path)
    output = Path(output_path)
    source_name = source.name
    source_hash = file_sha256(source)
    handler = ASSFormatHandler()
    try:
        original = handler.load(source)
        expected = handler.prepare_document(original)
    except (OSError, SubtitleFormatError, SubtitleValidationError, ValueError) as exc:
        return FileAnalysis(
            str(source),
            str(output),
            source_hash,
            file_sha256(output) if output.exists() else None,
            {"events": 0, "comments": 0, "translatable_events": 0},
            [
                _diagnostic(
                    "INVALID_SOURCE_ASS",
                    "FAIL",
                    f"A origem não pôde ser validada como ASS: {exc}",
                    source_name,
                )
            ],
        )
    base_metrics = {
        "events": len(expected.events),
        "comments": sum(event.type == "Comment" for event in expected.events),
        "translatable_events": _count_production_translatable_events(expected, handler),
    }
    if not output.exists():
        return FileAnalysis(
            str(source),
            str(output),
            source_hash,
            None,
            base_metrics,
            [_diagnostic("OUTPUT_MISSING", "FAIL", "Nenhum output foi produzido.", source_name)],
        )

    output_hash = file_sha256(output)
    try:
        if not {"Script Info", "Events"}.issubset(set(_sections(output))):
            raise SubtitleFormatError("As seções principais obrigatórias não foram encontradas.")
        translated = handler.load(output)
    except (OSError, SubtitleFormatError, SubtitleValidationError, ValueError) as exc:
        return FileAnalysis(
            str(source),
            str(output),
            source_hash,
            output_hash,
            base_metrics,
            [
                _diagnostic(
                    "INVALID_OUTPUT_ASS",
                    "FAIL",
                    f"O output não pôde ser validado como ASS: {exc}",
                    source_name,
                )
            ],
        )

    metrics = base_metrics
    diagnostics: list[Diagnostic] = []

    if _sections(source) != _sections(output):
        diagnostics.append(
            _diagnostic(
                "STRUCTURE_MISMATCH",
                "FAIL",
                f"As seções mudaram de {_sections(source)!r} para {_sections(output)!r}.",
                source_name,
            )
        )
    if _document_snapshot(expected) != _document_snapshot(translated):
        diagnostics.append(
            _diagnostic(
                "STRUCTURE_MISMATCH",
                "FAIL",
                "Metadados, estilos ou anexos ASS foram alterados.",
                source_name,
            )
        )
    if len(expected.events) != len(translated.events):
        diagnostics.append(
            _diagnostic(
                "STRUCTURE_MISMATCH",
                "FAIL",
                f"A quantidade de eventos mudou de {len(expected.events)} para {len(translated.events)}.",
                source_name,
            )
        )

    expected_anchor_owners: dict[str, set[int]] = {}
    for index, event in enumerate(expected.events):
        if event.type == "Comment":
            continue
        for anchor in _marker_groups(handler.prepare_text(event.text))["anchors"]:
            expected_anchor_owners.setdefault(anchor, set()).add(index)

    for index, (source_event, output_event) in enumerate(
        zip(expected.events, translated.events)
    ):
        raw_source_event = original.events[index]
        event_id = event_stable_id(source_name, index, raw_source_event)
        changed_fields = [
            field_name
            for field_name in _EVENT_STRUCTURE_FIELDS
            if getattr(source_event, field_name) != getattr(output_event, field_name)
        ]
        if changed_fields:
            diagnostics.append(
                _diagnostic(
                    "STRUCTURE_MISMATCH",
                    "FAIL",
                    f"Campos não textuais alterados: {', '.join(changed_fields)}.",
                    source_name,
                    event_index=index,
                    source_event=raw_source_event,
                    output_text=output_event.text,
                )
            )
        if source_event.type == "Comment":
            if source_event.text != output_event.text:
                diagnostics.append(
                    _diagnostic(
                        "COMMENT_MISMATCH",
                        "FAIL",
                        "O conteúdo de um evento Comment foi alterado.",
                        source_name,
                        event_index=index,
                        source_event=raw_source_event,
                        output_text=output_event.text,
                    )
                )
            continue

        try:
            assert_no_internal_artifacts(output_event.text)
        except SubtitleValidationError as exc:
            diagnostics.append(
                _diagnostic(
                    "PLACEHOLDER_LEAK",
                    "FAIL",
                    str(exc),
                    source_name,
                    event_index=index,
                    source_event=raw_source_event,
                    output_text=output_event.text,
                )
            )
        if _MODEL_PROTOCOL_RE.search(output_event.text):
            diagnostics.append(
                _diagnostic(
                    "MODEL_PROTOCOL_ERROR",
                    "FAIL",
                    "O output contém protocolo ITEM, Markdown ou rótulo do modelo.",
                    source_name,
                    event_index=index,
                    source_event=raw_source_event,
                    output_text=output_event.text,
                )
            )

        source_prepared = handler.prepare_text(source_event.text)
        output_prepared = handler.prepare_text(output_event.text)
        source_groups = _marker_groups(source_prepared)
        output_groups = _marker_groups(output_prepared)
        for group, category, label in (
            ("tags", "TAG_MISMATCH", "tags ASS"),
            ("breaks", "BREAK_MISMATCH", "quebras ASS"),
            ("prefixes", "PREFIX_MISMATCH", "prefixos de diálogo"),
            ("anchors", "PROTECTED_MARKER_MISMATCH", "âncoras protegidas"),
        ):
            if source_groups[group] != output_groups[group]:
                diagnostics.append(
                    _diagnostic(
                        category,
                        "FAIL",
                        f"A ordem ou o conteúdo de {label} foi alterado.",
                        source_name,
                        event_index=index,
                        source_event=raw_source_event,
                        output_text=output_event.text,
                    )
                )

        foreign_anchors = {
            anchor
            for anchor in output_groups["anchors"]
            if index not in expected_anchor_owners.get(anchor, set())
            and expected_anchor_owners.get(anchor)
        }
        if foreign_anchors:
            diagnostics.append(
                _diagnostic(
                    "ITEM_MAPPING_ERROR",
                    "FAIL",
                    "O evento recebeu âncora protegida pertencente a outro evento: "
                    + ", ".join(sorted(foreign_anchors)),
                    source_name,
                    event_index=index,
                    source_event=raw_source_event,
                    output_text=output_event.text,
                )
            )

        if source_prepared.cleaned_text and not output_prepared.cleaned_text:
            diagnostics.append(
                _diagnostic(
                    "EMPTY_OUTPUT",
                    "FAIL",
                    "O texto traduzível ficou vazio.",
                    source_name,
                    event_index=index,
                    source_event=raw_source_event,
                    output_text=output_event.text,
                )
            )
        elif _meaningful_text(source_prepared.cleaned_text):
            diagnostics.extend(
                _semantic_diagnostics(
                    source_name,
                    index,
                    raw_source_event,
                    source_prepared.cleaned_text,
                    output_prepared.cleaned_text,
                )
            )

        # Keep the ID calculation close to the comparison in tracebacks and debuggers.
        assert event_id.startswith("evt_")

    return FileAnalysis(
        source_path=str(source),
        output_path=str(output),
        source_hash=source_hash,
        output_hash=output_hash,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def collect_runtime_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Collect reproducibility context without modifying repository configuration."""

    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = git("status", "--porcelain=v1")
    code_digest = hashlib.sha256()
    relevant_roots = {"docs", "static", "templates", "tests", "tools"}
    relevant_suffixes = {".py", ".md", ".html", ".css", ".txt", ".toml", ".yaml", ".yml"}
    candidates = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    for relative_name in sorted(candidates, key=str.casefold):
        relative = Path(relative_name)
        if not relative.parts:
            continue
        if relative.parts[0] not in relevant_roots and len(relative.parts) > 1:
            continue
        if relative.suffix.casefold() not in relevant_suffixes:
            continue
        candidate = root / relative
        if not candidate.is_file():
            continue
        code_digest.update(relative.as_posix().encode("utf-8"))
        code_digest.update(b"\0")
        code_digest.update(candidate.read_bytes())
        code_digest.update(b"\0")
    code_hash = code_digest.hexdigest()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "git_state_sha256": code_hash,
        "python_version": sys.version,
        "platform": platform.platform(),
        "ollama_host": os.environ.get("OLLAMA_HOST", "default"),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown(report: RegressionReport) -> str:
    metrics = report.metrics
    lines = [
        "# ASS REAL-WORLD REGRESSION TEST",
        "",
        f"- Experimento: `{report.context.experiment}`",
        f"- Run ID: `{report.context.run_id}`",
        f"- Arquivos: {metrics['files']}",
        f"- Arquivos aprovados: {metrics['files_passed']}",
        f"- Arquivos com problemas críticos: {metrics['files_with_critical_failures']}",
        f"- Eventos analisados: {metrics['events']}",
        f"- Eventos traduzíveis: {metrics['translatable_events']}",
        f"- Eventos Comment: {metrics['comments']}",
        f"- Erros críticos: {metrics['critical_failures']}",
        f"- Warnings semânticos: {metrics['semantic_warnings']}",
        "",
        "## Diagnósticos",
        "",
    ]
    diagnostics = [
        item
        for file_analysis in report.files
        for item in file_analysis.diagnostics
    ]
    if not diagnostics:
        lines.append("Nenhum diagnóstico.")
    for item in diagnostics:
        event = "arquivo" if item.event_index is None else f"evento {item.event_index}"
        lines.extend(
            [
                f"### {item.severity} — {item.category}",
                "",
                f"- Arquivo: `{item.file}`",
                f"- Local: {event}",
                f"- Confiança: {item.confidence}",
                f"- Justificativa: {item.message}",
            ]
        )
        if item.timestamp:
            lines.append(f"- Timestamp: `{item.timestamp}`")
        if item.original is not None:
            lines.extend(["", "Original:", "", f"```text\n{item.original}\n```"])
        if item.translation is not None:
            lines.extend(["", "Resultado:", "", f"```text\n{item.translation}\n```"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(report: RegressionReport, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    _atomic_write(
        json_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, _markdown(report))
    return json_path, markdown_path


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    _atomic_write(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )
    return destination


def analyze_run(
    sources: Iterable[Path], outputs: Mapping[str, Path]
) -> list[FileAnalysis]:
    """Analyze a set of files using an explicit source-name to output mapping."""

    return [
        analyze_ass_pair(source, outputs.get(source.name, Path("__missing_output__.ass")))
        for source in sources
    ]
