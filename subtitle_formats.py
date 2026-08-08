"""Format-aware subtitle loading, protection, validation, and saving.

The translation model must only see natural-language text plus opaque markers.
This module owns those markers and guarantees that they are restored (or that
the original subtitle item is used as a safe fallback by the caller).
"""

from __future__ import annotations

import copy
import re
import uuid
from abc import ABC
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pysubs2


SUPPORTED_SUBTITLE_EXTENSIONS = frozenset({".ass", ".srt"})
SUPPORTED_FORMATS = ("ass", "srt")

_ITEM_ARTIFACT_RE = re.compile(r"<<<\s*(?:END_)?ITEM_\d+\s*>>>", re.IGNORECASE)
_INTERNAL_PLACEHOLDER_RE = re.compile(
    r"\[{1,3}\s*(?:ASS|SRT)_[A-Z0-9_-]+\s*\]{1,3}", re.IGNORECASE
)
_BARE_INTERNAL_ARTIFACT_RE = re.compile(
    r"(?:ASS_(?:BR|NBSP|TAG|LB|PREFIX)|SRT_(?:TAG|LB|PREFIX|SOURCE)|(?:END_)?ITEM_\d+)",
    re.IGNORECASE,
)
_ASS_COMMAND_RE = re.compile(r"\{\\[^}\r\n]+\}|\\[Nnh]")
_DIALOGUE_PREFIX_RE = re.compile(r"^([ \t]*-[ \t]+)")
_SRT_RAW_MARKUP_RE = re.compile(r"<[^>\r\n]+>|\{[^}\r\n]*\}")
_SIMPLE_HTML_TAG_RE = re.compile(r"<\s*(/?)\s*(i|b|u|font)\b[^>]*>", re.IGNORECASE)
_SRT_CUE_INDEX_RE = re.compile(
    r"(?m)^[ \t]*(\d+)[ \t]*\n(?=[ \t]*\d{1,3}:\d{2}:\d{2}[,.]\d{3}[ \t]+-->)"
)
_PROTECTED_CONTENT_PATTERN = (
    r"https?://[^\s<>]+|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"\$\{[^}\r\n]+\}|\{\{[^}\r\n]+\}\}|%[A-Z0-9_]+%"
)
_ASS_BRACE_BLOCK_RE = re.compile(r"\{([^{}\r\n]*)\}")
_ASS_PAREN_COMMANDS = frozenset(
    {"pos", "move", "org", "clip", "iclip", "fad", "fade", "t"}
)
_ASS_NUMERIC_COMMANDS = frozenset(
    {
        "i",
        "b",
        "u",
        "s",
        "an",
        "a",
        "q",
        "p",
        "pbo",
        "be",
        "blur",
        "bord",
        "xbord",
        "ybord",
        "shad",
        "xshad",
        "yshad",
        "fs",
        "fscx",
        "fscy",
        "fsp",
        "fr",
        "frx",
        "fry",
        "frz",
        "fax",
        "fay",
        "fe",
        "k",
        "kf",
        "ko",
        "kt",
    }
)
_ASS_COLOR_COMMANDS = frozenset({"c", "1c", "2c", "3c", "4c"})
_ASS_ALPHA_COMMANDS = frozenset({"alpha", "1a", "2a", "3a", "4a"})
_ASS_TEXT_COMMANDS = frozenset({"fn", "r"})
_ASS_COMMAND_NAMES = tuple(
    sorted(
        _ASS_PAREN_COMMANDS
        | _ASS_NUMERIC_COMMANDS
        | _ASS_COLOR_COMMANDS
        | _ASS_ALPHA_COMMANDS
        | _ASS_TEXT_COMMANDS,
        key=len,
        reverse=True,
    )
)
_ASS_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_ASS_NUMBER_RE = re.compile(rf"^{_ASS_NUMBER_PATTERN}$")
_ASS_COLOR_RE = re.compile(r"^&H[0-9A-F]+&?$", re.IGNORECASE)
_ASS_ALPHA_RE = re.compile(r"^&H[0-9A-F]{1,2}&?$", re.IGNORECASE)


def _scan_balanced_parentheses(value: str, start: int) -> int | None:
    """Return the first index after a balanced parenthesized argument."""

    if start >= len(value) or value[start] != "(":
        return None
    depth = 0
    for index in range(start, len(value)):
        character = value[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _is_number_list(value: str, allowed_counts: set[int]) -> bool:
    parts = [part.strip() for part in value.split(",")]
    return len(parts) in allowed_counts and all(_ASS_NUMBER_RE.fullmatch(part) for part in parts)


def _ass_command_name_at(value: str, start: int) -> str | None:
    if start >= len(value) or value[start] != "\\":
        return None
    remainder = value[start + 1 :].casefold()
    return next((name for name in _ASS_COMMAND_NAMES if remainder.startswith(name)), None)


def _ass_spans_cover_only_commands(value: str, spans: Sequence[tuple[int, int]]) -> bool:
    cursor = 0
    for start, end in spans:
        if value[cursor:start].strip():
            return False
        cursor = end
    return not value[cursor:].strip()


def _valid_parenthesized_ass_argument(command: str, argument: str) -> bool:
    inner = argument[1:-1].strip()
    if not inner:
        return False
    if command in {"pos", "org"}:
        return _is_number_list(inner, {2})
    if command == "move":
        return _is_number_list(inner, {4, 6})
    if command == "fad":
        return _is_number_list(inner, {2})
    if command == "fade":
        return _is_number_list(inner, {7})
    if command in {"clip", "iclip"}:
        return True
    if command == "t":
        first_override = inner.find("\\")
        if first_override < 0:
            return False
        numeric_prefix = inner[:first_override]
        if numeric_prefix and not re.fullmatch(
            rf"(?:{_ASS_NUMBER_PATTERN}\s*,\s*){{1,3}}", numeric_prefix
        ):
            return False
        modifiers = inner[first_override:]
        spans = _scan_ass_override_spans(modifiers)
        return bool(spans) and _ass_spans_cover_only_commands(modifiers, spans)
    return False


def _valid_simple_ass_argument(command: str, argument: str) -> bool:
    stripped = argument.strip()
    if command in _ASS_NUMERIC_COMMANDS:
        return not stripped or bool(_ASS_NUMBER_RE.fullmatch(stripped))
    if command in _ASS_COLOR_COMMANDS:
        return not stripped or bool(_ASS_COLOR_RE.fullmatch(stripped))
    if command in _ASS_ALPHA_COMMANDS:
        return not stripped or bool(_ASS_ALPHA_RE.fullmatch(stripped))
    if command in _ASS_TEXT_COMMANDS:
        return "{" not in argument and "}" not in argument
    return False


def _parse_ass_override_at(value: str, start: int) -> int | None:
    """Return the exact end of one recognized ASS override command."""

    command = _ass_command_name_at(value, start)
    if command is None:
        return None
    argument_start = start + 1 + len(command)
    if command in _ASS_PAREN_COMMANDS:
        command_end = _scan_balanced_parentheses(value, argument_start)
        if command_end is None:
            return None
        argument = value[argument_start:command_end]
        return command_end if _valid_parenthesized_ass_argument(command, argument) else None

    next_command = value.find("\\", argument_start)
    command_end = len(value) if next_command < 0 else next_command
    argument = value[argument_start:command_end]
    return command_end if _valid_simple_ass_argument(command, argument) else None


def _scan_ass_override_spans(value: str) -> tuple[tuple[int, int], ...]:
    """Find recognized ASS commands while leaving natural-language residue visible."""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        command_start = value.find("\\", cursor)
        if command_start < 0:
            break
        command_end = _parse_ass_override_at(value, command_start)
        if command_end is None:
            cursor = command_start + 1
            continue
        spans.append((command_start, command_end))
        cursor = command_end
    return tuple(spans)


def _matching_ass_brace_end(value: str, start: int) -> int | None:
    """Return the end of a brace region, accepting nested textual braces."""

    depth = 0
    for index in range(start, len(value)):
        character = value[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _remove_nested_ass_text_regions(text: str) -> tuple[str, bool]:
    """Remove whole nested comment regions before scanning flat override blocks."""

    visible_end = len(text.rstrip(" \t"))
    pieces: list[str] = []
    cursor = 0
    removed_terminal_region = False

    while cursor < len(text):
        region_start = text.find("{", cursor)
        if region_start < 0:
            pieces.append(text[cursor:])
            break

        region_end = _matching_ass_brace_end(text, region_start)
        candidate_end = len(text) if region_end is None else region_end
        nested_start = text.find("{", region_start + 1, candidate_end)
        if nested_start < 0:
            pieces.append(text[cursor : region_start + 1])
            cursor = region_start + 1
            continue

        prefix = text[region_start + 1 : nested_start]
        prefix_spans = _scan_ass_override_spans(prefix)
        prefix_is_text = bool(prefix.strip()) and not _ass_spans_cover_only_commands(
            prefix, prefix_spans
        )
        if not prefix_is_text:
            pieces.append(text[cursor : region_start + 1])
            cursor = region_start + 1
            continue

        pieces.append(text[cursor:region_start])
        cursor = candidate_end
        if candidate_end >= visible_end:
            removed_terminal_region = True

    return "".join(pieces), removed_terminal_region


def sanitize_ass_text(text: str) -> str:
    """Remove textual brace comments while preserving recognized ASS overrides exactly."""

    if not isinstance(text, str):
        raise SubtitleValidationError("O texto ASS deve ser uma string.")

    text, removed_terminal_text_block = _remove_nested_ass_text_regions(text)
    visible_end = len(text.rstrip(" \t"))

    def sanitize_block(match: re.Match[str]) -> str:
        nonlocal removed_terminal_text_block
        content = match.group(1)
        spans = _scan_ass_override_spans(content)
        if not spans:
            if match.end() == visible_end:
                removed_terminal_text_block = True
            return ""
        if _ass_spans_cover_only_commands(content, spans):
            return match.group(0)
        commands = "".join(content[start:end] for start, end in spans)
        return f"{{{commands}}}" if commands else ""

    sanitized = _ASS_BRACE_BLOCK_RE.sub(sanitize_block, text)
    return sanitized.rstrip(" \t") if removed_terminal_text_block else sanitized


class SubtitleFormatError(ValueError):
    """Raised when a subtitle format cannot be identified or decoded."""


class SubtitleValidationError(ValueError):
    """Raised when a model response would damage subtitle structure."""


@dataclass(frozen=True)
class ProtectedMarker:
    """One exact structural fragment replaced by an opaque model token."""

    token: str
    value: str
    kind: str


@dataclass(frozen=True)
class PreparedSubtitleText:
    """A subtitle item prepared for safe translation."""

    format_name: str
    original_text: str
    cleaned_text: str
    model_text: str
    markers: tuple[ProtectedMarker, ...]
    line_structure: tuple[str, ...]

    @property
    def protected_markings(self) -> tuple[tuple[str, str], ...]:
        """Stable representation used by the translation cache."""

        return tuple((marker.kind, marker.value) for marker in self.markers)


def is_supported_subtitle(path: str | Path) -> bool:
    """Return whether *path* has a supported extension, case-insensitively."""

    return Path(path).suffix.lower() in SUPPORTED_SUBTITLE_EXTENSIONS


def iter_subtitle_files(directory: str | Path, format_filter: str = "all") -> list[Path]:
    """List supported subtitle files without relying on case-sensitive globs."""

    root = Path(directory)
    normalized_filter = format_filter.lower()
    if normalized_filter not in {"all", *SUPPORTED_FORMATS}:
        raise SubtitleFormatError(f"Filtro de formato inválido: {format_filter}")
    if not root.exists() or not root.is_dir():
        return []

    allowed = SUPPORTED_SUBTITLE_EXTENSIONS
    if normalized_filter != "all":
        allowed = frozenset({f".{normalized_filter}"})
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.name.casefold(),
    )


def count_subtitle_formats(paths: Iterable[Path]) -> dict[str, int]:
    """Count ASS and SRT paths using stable lowercase format names."""

    counts = {format_name: 0 for format_name in SUPPORTED_FORMATS}
    for path in paths:
        extension = path.suffix.lower().removeprefix(".")
        if extension in counts:
            counts[extension] += 1
    return counts


def assert_no_internal_artifacts(text: str) -> None:
    """Reject control markers that must never reach a saved subtitle."""

    if (
        "[[[" in text
        or "]]]" in text
        or _ITEM_ARTIFACT_RE.search(text)
        or _INTERNAL_PLACEHOLDER_RE.search(text)
        or _BARE_INTERNAL_ARTIFACT_RE.search(text)
    ):
        raise SubtitleValidationError("A resposta contém marcadores internos de controle.")


def _read_text_with_fallback(path: Path) -> str:
    """Decode common subtitle encodings strictly, never replacing bytes silently."""

    payload = path.read_bytes()
    decoding_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError as exc:
            decoding_errors.append(f"{encoding}: {exc}")
    details = "; ".join(decoding_errors)
    raise SubtitleFormatError(f"Não foi possível decodificar {path.name}: {details}")


def _normalize_newlines(content: str) -> str:
    """Match Python's universal-newline behavior before handing text to pysubs2."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 content atomically in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        cleanup_error: OSError | None = None
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            cleanup_error = cleanup_exc
        details = f"Não foi possível salvar {path}: {exc}"
        if cleanup_error is not None:
            details += f"; também não foi possível remover o temporário: {cleanup_error}"
        raise SubtitleFormatError(details) from exc


def _replace_with_tokens(
    text: str,
    pattern: re.Pattern[str],
    prefix: str,
) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"[[[{prefix}_{len(replacements) + 1:06d}]]]"
        replacements[token] = match.group(0)
        return token

    return pattern.sub(replace, text), replacements


def _restore_tokens(text: str, replacements: Mapping[str, str]) -> str:
    restored = text
    for token, value in replacements.items():
        restored = restored.replace(token, value)
    return restored


class SubtitleFormatHandler(ABC):
    """Base handler for one subtitle serialization format."""

    format_name: str
    extension: str
    marker_pattern: re.Pattern[str]

    def prepare_document(self, subs: pysubs2.SSAFile) -> pysubs2.SSAFile:
        """Create the single working copy used by translation and reconstruction."""

        return copy.deepcopy(subs)

    def load(self, path: str | Path) -> pysubs2.SSAFile:
        """Load a subtitle document while preserving all supported structure."""

        source = Path(path)
        if source.suffix.lower() != self.extension:
            raise SubtitleFormatError(
                f"O handler {self.format_name.upper()} não aceita {source.suffix or 'sem extensão'}."
            )
        content = _normalize_newlines(_read_text_with_fallback(source))
        try:
            return pysubs2.SSAFile.from_string(content, format_=self.format_name)
        except Exception as exc:
            raise SubtitleFormatError(f"Falha ao carregar {source.name}: {exc}") from exc

    def save(self, subs: pysubs2.SSAFile, path: str | Path) -> None:
        """Serialize atomically using UTF-8."""

        destination = Path(path)
        self.validate_document(subs)
        try:
            content = subs.to_string(self.format_name)
        except Exception as exc:
            raise SubtitleFormatError(f"Falha ao serializar {destination.name}: {exc}") from exc
        assert_no_internal_artifacts(content)
        _atomic_write_text(destination, content)

    def validate_document(self, subs: pysubs2.SSAFile) -> None:
        """Validate timing and control artifacts before serialization."""

        for index, event in enumerate(subs.events, start=1):
            if event.end < event.start:
                raise SubtitleValidationError(f"O evento {index} possui horário final inválido.")
            assert_no_internal_artifacts(event.text)

    def prepare_text(self, text: str) -> PreparedSubtitleText:
        """Protect formatting, line structure, and dialogue prefixes for the model."""

        if not isinstance(text, str):
            raise SubtitleValidationError("O texto da legenda deve ser uma string.")

        markers: list[ProtectedMarker] = []
        model_parts: list[str] = []
        clean_parts: list[str] = []
        line_structure: list[str] = []
        cursor = 0
        at_line_start = True

        def add_marker(value: str, kind: str) -> str:
            token = f"[[[{self.format_name.upper()}_{kind.upper()}_{len(markers) + 1:04d}]]]"
            markers.append(ProtectedMarker(token=token, value=value, kind=kind))
            return token

        def add_plain(value: str, line_start: bool) -> bool:
            if not value:
                return line_start
            remaining = value
            if line_start:
                prefix_match = _DIALOGUE_PREFIX_RE.match(remaining)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    model_parts.append(add_marker(prefix, "prefix"))
                    clean_parts.append(prefix)
                    remaining = remaining[prefix_match.end() :]
            model_parts.append(remaining)
            clean_parts.append(remaining)
            return line_start and not bool(remaining.strip())

        for match in self.marker_pattern.finditer(text):
            at_line_start = add_plain(text[cursor : match.start()], at_line_start)
            value = match.group(0)
            kind = self._marker_kind(value)
            model_parts.append(add_marker(value, kind))
            if kind == "line_break":
                clean_parts.append("\n")
                line_structure.append(value)
                at_line_start = True
            elif kind == "non_break_space":
                clean_parts.append(" ")
            cursor = match.end()
        add_plain(text[cursor:], at_line_start)

        return PreparedSubtitleText(
            format_name=self.format_name,
            original_text=text,
            cleaned_text="".join(clean_parts).strip(),
            model_text="".join(model_parts).strip(),
            markers=tuple(markers),
            line_structure=tuple(line_structure),
        )

    def restore_text(self, prepared: PreparedSubtitleText, translated_text: str) -> str:
        """Validate every protected marker and restore the exact original structure."""

        if prepared.format_name != self.format_name:
            raise SubtitleValidationError("O texto preparado pertence a outro formato.")
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise SubtitleValidationError("A tradução retornada está vazia ou é inválida.")

        normalized = translated_text.strip()
        if "\n" in normalized or "\r" in normalized:
            raise SubtitleValidationError("A tradução alterou a quantidade de linhas do item.")
        if _ITEM_ARTIFACT_RE.search(normalized):
            raise SubtitleValidationError("A tradução contém delimitadores de item.")

        for marker in prepared.markers:
            marker_pattern = self._placeholder_pattern(marker)
            matches = list(marker_pattern.finditer(normalized))
            if len(matches) != 1:
                raise SubtitleValidationError(
                    f"O marcador protegido {marker.token} apareceu {len(matches)} vez(es)."
                )
            normalized = marker_pattern.sub(lambda _: marker.token, normalized, count=1)

        known_tokens = {marker.token for marker in prepared.markers}
        remaining_tokens = {
            match.group(0)
            for match in _INTERNAL_PLACEHOLDER_RE.finditer(normalized)
            if match.group(0) not in known_tokens
        }
        if remaining_tokens or _ITEM_ARTIFACT_RE.search(normalized):
            raise SubtitleValidationError("A tradução contém marcadores internos desconhecidos.")

        marker_positions = [normalized.index(marker.token) for marker in prepared.markers]
        if marker_positions != sorted(marker_positions):
            raise SubtitleValidationError("A tradução alterou a ordem das marcações protegidas.")

        # Raw ASS commands in a model response are always invented: legitimate source
        # commands were replaced with protected tokens above.
        if _ASS_COMMAND_RE.search(normalized):
            raise SubtitleValidationError("A tradução introduziu comandos ASS não protegidos.")

        restored = normalized
        for marker in prepared.markers:
            if marker.token not in known_tokens or restored.count(marker.token) != 1:
                raise SubtitleValidationError("A tradução corrompeu a estrutura protegida.")
            restored = restored.replace(marker.token, marker.value, 1)

        assert_no_internal_artifacts(restored)
        if self._line_structure(restored) != prepared.line_structure:
            raise SubtitleValidationError("A tradução alterou a estrutura de linhas.")
        return restored.strip()

    def rebuild(
        self,
        subs: pysubs2.SSAFile,
        translated_text_by_index: Mapping[int, str],
    ) -> pysubs2.SSAFile:
        """Clone a document and replace only translated event text."""

        rebuilt = copy.deepcopy(subs)
        for index, translated_text in translated_text_by_index.items():
            if not 0 <= index < len(rebuilt.events):
                raise SubtitleValidationError(f"Índice de legenda fora do documento: {index}")
            assert_no_internal_artifacts(translated_text)
            rebuilt.events[index].text = translated_text
        return rebuilt

    def _marker_kind(self, value: str) -> str:
        if value in {"\r\n", "\r", "\n", r"\N", r"\n"}:
            return "line_break"
        if value == r"\h":
            return "non_break_space"
        return "tag"

    def _line_structure(self, text: str) -> tuple[str, ...]:
        return tuple(
            match.group(0)
            for match in self.marker_pattern.finditer(text)
            if self._marker_kind(match.group(0)) == "line_break"
        )

    @staticmethod
    def _placeholder_pattern(marker: ProtectedMarker) -> re.Pattern[str]:
        label = marker.token.removeprefix("[[[").removesuffix("]]]")
        parts = label.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return re.compile(re.escape(marker.token))
        name, number = parts
        flexible_name = r"[\s_-]*".join(re.escape(part) for part in name.split("_"))
        flexible_number = rf"0*{int(number)}"
        return re.compile(
            rf"\[{{1,3}}\s*{flexible_name}[\s_-]*{flexible_number}\s*\]{{1,3}}",
            re.IGNORECASE,
        )


class ASSFormatHandler(SubtitleFormatHandler):
    """Preserve ASS override blocks and escaped layout commands exactly."""

    format_name = "ass"
    extension = ".ass"
    marker_pattern = re.compile(
        rf"{_PROTECTED_CONTENT_PATTERN}|\{{[^}}\r\n]*\}}|\\[Nnh]|\r\n|\r|\n"
    )

    def prepare_document(self, subs: pysubs2.SSAFile) -> pysubs2.SSAFile:
        prepared = super().prepare_document(subs)
        for event in prepared.events:
            if event.type != "Comment":
                event.text = sanitize_ass_text(event.text)
        return prepared


class SRTFormatHandler(SubtitleFormatHandler):
    """Load and save SRT without letting pysubs2 discard raw markup."""

    format_name = "srt"
    extension = ".srt"
    marker_pattern = re.compile(
        rf"{_PROTECTED_CONTENT_PATTERN}|<[^>\r\n]+>|\{{[^}}\r\n]*\}}|\\[Nn]|\r\n|\r|\n"
    )

    def validate_document(self, subs: pysubs2.SSAFile) -> None:
        super().validate_document(subs)
        cue_indices = getattr(subs, "_srt_cue_indices", None)
        if cue_indices is not None and len(cue_indices) != len(subs.events):
            raise SubtitleValidationError(
                "A quantidade de índices SRT não corresponde à quantidade de eventos."
            )
        for index, event in enumerate(subs.events, start=1):
            stack: list[str] = []
            for match in _SIMPLE_HTML_TAG_RE.finditer(event.text):
                closing = bool(match.group(1))
                tag_name = match.group(2).lower()
                if closing:
                    if not stack or stack[-1] != tag_name:
                        raise SubtitleValidationError(
                            f"O evento SRT {index} possui fechamento HTML incompatível: {match.group(0)}"
                        )
                    stack.pop()
                else:
                    stack.append(tag_name)
            if stack:
                raise SubtitleValidationError(
                    f"O evento SRT {index} possui tag(s) HTML sem fechamento: {', '.join(stack)}"
                )

    def load(self, path: str | Path) -> pysubs2.SSAFile:
        source = Path(path)
        if source.suffix.lower() != self.extension:
            raise SubtitleFormatError(
                f"O handler SRT não aceita {source.suffix or 'sem extensão'}."
            )
        content = _normalize_newlines(_read_text_with_fallback(source))
        cue_indices = tuple(_SRT_CUE_INDEX_RE.findall(content))
        protected, replacements = _replace_with_tokens(content, _SRT_RAW_MARKUP_RE, "SRT_SOURCE")
        try:
            subs = pysubs2.SSAFile.from_string(protected, format_="srt")
        except Exception as exc:
            raise SubtitleFormatError(f"Falha ao carregar {source.name}: {exc}") from exc
        for event in subs.events:
            event.text = _restore_tokens(event.text, replacements)
        if len(cue_indices) != len(subs.events):
            raise SubtitleFormatError(
                f"Não foi possível preservar todos os índices SRT de {source.name}: "
                f"{len(cue_indices)} índice(s) para {len(subs.events)} evento(s)."
            )
        subs._srt_cue_indices = cue_indices
        return subs

    def save(self, subs: pysubs2.SSAFile, path: str | Path) -> None:
        destination = Path(path)
        self.validate_document(subs)
        serializable = copy.deepcopy(subs)
        cue_indices = tuple(
            str(index)
            for index in getattr(
                subs, "_srt_cue_indices", tuple(range(1, len(subs.events) + 1))
            )
        )
        replacements: dict[str, str] = {}

        for event in serializable.events:
            protected, event_replacements = _replace_with_tokens(
                event.text,
                _SRT_RAW_MARKUP_RE,
                f"SRT_SOURCE_{len(replacements):06d}",
            )
            event.text = protected
            replacements.update(event_replacements)

        try:
            content = serializable.to_string("srt")
        except Exception as exc:
            raise SubtitleFormatError(f"Falha ao serializar {destination.name}: {exc}") from exc
        content = _restore_tokens(content, replacements)
        index_iterator = iter(cue_indices)

        def restore_index(match: re.Match[str]) -> str:
            return f"{next(index_iterator)}\n"

        content, restored_index_count = _SRT_CUE_INDEX_RE.subn(restore_index, content)
        if restored_index_count != len(cue_indices):
            raise SubtitleValidationError(
                "A serialização SRT alterou a quantidade de índices de blocos."
            )
        assert_no_internal_artifacts(content)
        if r"\N" in content:
            raise SubtitleValidationError("Uma quebra interna ASS vazou para o SRT final.")
        _atomic_write_text(destination, content)


_HANDLERS: dict[str, SubtitleFormatHandler] = {
    ".ass": ASSFormatHandler(),
    ".srt": SRTFormatHandler(),
}


def get_format_handler(path_or_extension: str | Path) -> SubtitleFormatHandler:
    """Resolve a handler from a path, extension, or bare format name."""

    raw = str(path_or_extension).strip()
    normalized = raw.lower()
    if normalized in SUPPORTED_FORMATS:
        normalized = f".{normalized}"
    else:
        normalized = Path(raw).suffix.lower()
    try:
        return _HANDLERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_SUBTITLE_EXTENSIONS))
        raise SubtitleFormatError(f"Formato de legenda não suportado: {raw}. Use {supported}.") from exc


def preserve_extension_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Build ``name.pt.ext`` while retaining the input extension's spelling."""

    source = Path(input_path)
    if not is_supported_subtitle(source):
        raise SubtitleFormatError(f"Formato de legenda não suportado: {source.name}")
    return Path(output_dir) / f"{source.stem}.pt{source.suffix}"


def marker_multiset(prepared: PreparedSubtitleText) -> Counter[tuple[str, str]]:
    """Expose exact protected structure for diagnostics and cache signatures."""

    return Counter(prepared.protected_markings)
