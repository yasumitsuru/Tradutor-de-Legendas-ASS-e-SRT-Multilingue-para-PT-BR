"""Shared ASS/SRT translation engine backed by Ollama."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import ollama
import pysubs2

from subtitle_formats import (
    ASSFormatHandler,
    PreparedSubtitleText,
    SubtitleFormatHandler,
    SubtitleValidationError,
    assert_no_internal_artifacts,
    get_format_handler,
    sanitize_ass_text,
)


CACHE_SCHEMA_VERSION = 4
PROMPT_VERSION = "subtitle-items-v4-multilingual-auto-item-envelope"
ITEM_BLOCK_RE = re.compile(
    r"<<<ITEM_(\d{4})>>>[ \t]*\r?\n?(.*?)[ \t]*\r?\n?<<<END_ITEM_\1>>>",
    re.IGNORECASE | re.DOTALL,
)
CONTROL_TOKEN_RE = re.compile(r"\[{1,3}\s*(?:ASS|SRT)_[A-Z0-9_-]+\s*\]{1,3}", re.I)
MODEL_MESSAGE_RE = re.compile(
    r"(?:here\s+is\s+the\s+translation|translation\s*:|tradu[cç][aã]o\s*:)", re.IGNORECASE
)
NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+[.)]\s+")
INDIVIDUAL_ITEM_ARTIFACT_RE = re.compile(
    r"(?:<{2,3}|>{2,3})|(?:END[_\s-]*)?ITEM[_\s-]*\d+", re.IGNORECASE
)
BATCH_NESTED_ITEM_ARTIFACT_RE = re.compile(
    r"(?:<{2,3}\s*(?:END[_\s-]*)?ITEM[_\s-]*[A-Z0-9_-]+\s*>{0,3})|"
    r"(?:(?:END_)?ITEM_[A-Z0-9_-]+\s*>{2,3})",
    re.IGNORECASE,
)
INDIVIDUAL_RAW_MARKDOWN_RE = re.compile(
    r"^\s*(?:#{1,6}\s|>\s|[-*+]\s)|"
    r"(?:\*\*[^*\r\n]+\*\*|__[^_\r\n]+__|~~[^~\r\n]+~~|"
    r"(?<!\*)\*[^*\r\n]+\*(?!\*)|(?<!_)_[^_\r\n]+_(?!_)|"
    r"`[^`\r\n]+`|\[[^\]\r\n]+\]\([^\)\r\n]+\))"
)
INDIVIDUAL_RAW_EXPLANATION_RE = re.compile(
    r"^\s*(?:resultado|result|resposta|answer|aqui está|here is)\s*:", re.IGNORECASE
)


CONFIG: dict[str, Any] = {
    "model": "qwen2.5:14b",
    "max_parallel": 4,
    "batch_size": 15,
    "temperature": 0.2,
    "max_tokens": 2048,
    "timeout": 300,
    "retry_count": 3,
    "retry_delay": 2.0,
    "enable_cache": True,
    "cache_file": "translation_cache.json",
    "min_text_length": 2,
    "skip_sfx": True,
    "turbo_mode": False,
    "source_language": "auto",
    "target_language": "Brazilian Portuguese",
    "allow_original_fallback": False,
    "prompt_version": PROMPT_VERSION,
    "system_prompt": (
        "Traduza texto natural de legendas para português do Brasil. Quando a origem estiver em "
        "modo automático, detecte o idioma ou idiomas presentes em cada item; quando um idioma de "
        "origem for informado explicitamente, respeite essa instrução. Uma mesma fala pode misturar "
        "vários idiomas. Traduza todo conteúdo natural que ainda não esteja em português, preserve "
        "o significado das partes que já estiverem em português e integre tudo em uma única fala "
        "natural, fluente e concisa. "
        "Preserve nomes próprios, honoríficos, siglas, termos técnicos, marcadores protegidos, tom, "
        "formalidade e informalidade. Não identifique os idiomas, não censure, não invente, não "
        "remova conteúdo e não explique. Responda somente no protocolo solicitado, sem Markdown."
    ),
}


class ModelUnavailableError(RuntimeError):
    """Fatal error raised when the configured model is absent from Ollama."""


class IncompleteTranslationError(SubtitleValidationError):
    """Raised when unresolved items make a safely completed output impossible."""

    def __init__(self, failed_count: int):
        self.failed_count = failed_count
        super().__init__(
            f"Falha definitiva em {failed_count} item(ns). Arquivo não será salvo como "
            "tradução concluída para evitar mistura de idiomas."
        )


def is_model_not_found_error(exc: Exception) -> bool:
    """Detect Ollama errors that represent a missing model."""

    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 404 or ("model" in message and "not found" in message)


def ensure_ollama_model_available(model_name: str, client: Any = ollama) -> None:
    """Validate the model before processing any user file."""

    try:
        client.show(model_name)
    except Exception as exc:
        if is_model_not_found_error(exc):
            raise ModelUnavailableError(
                f'❌ Modelo "{model_name}" não está disponível no Ollama. '
                f"Instale antes com: ollama pull {model_name}"
            ) from exc
        raise RuntimeError(f"❌ Não foi possível validar o modelo no Ollama: {exc}") from exc


@dataclass(frozen=True)
class TranslationOutcome:
    """Validated translation or an explicit per-item fallback."""

    text: str
    used_fallback: bool = False
    error: str | None = None
    line_break_recovery_eligible: bool = False


class FixedASSTranslator:
    """Backward-compatible name for the shared ASS/SRT translator."""

    def __init__(self, config: Mapping[str, Any], ollama_client: Any | None = None):
        self.config = {**CONFIG, **dict(config)}
        trace_hook = self.config.pop("trace_hook", None)
        self._trace_hook = trace_hook if callable(trace_hook) else None
        self._trace_file: str | None = None
        source_language = str(self.config.get("source_language", "auto")).strip()
        self.config["source_language"] = (
            "auto" if not source_language or source_language.casefold() == "auto" else source_language
        )
        self.ollama_client = ollama_client or ollama
        self.stats: dict[str, int] = {}
        self.reset_stats()
        self.cache: dict[str, str] = {}
        if self.config["enable_cache"]:
            self._load_cache()

        self.skip_patterns = [
            re.compile(r"^[\s♪♫…~\-_\.]+$"),
            re.compile(r"^\([^)]*\)$"),
            re.compile(r"^\[[^\]]*\]$"),
        ]
        self.sfx_words = {"uh", "ah", "oh", "huh", "eh", "woah", "wow", "hey"}

    def reset_stats(self) -> None:
        """Reset per-file counters."""

        self.stats = {"total": 0, "translated": 0, "failed": 0, "cached": 0, "skipped": 0}

    def _emit_trace(self, event: str, **details: Any) -> None:
        """Notify an optional diagnostic observer without affecting translation."""

        if self._trace_hook is None:
            return
        payload: dict[str, Any] = {
            "event": event,
            "recorded_at": datetime.now().astimezone().isoformat(),
        }
        if self._trace_file is not None:
            payload["file"] = self._trace_file
        payload.update(details)
        try:
            self._trace_hook(payload)
        except Exception as exc:
            print(f"⚠️ Instrumentação de diagnóstico desativada após falha: {exc}")
            self._trace_hook = None

    def _load_cache(self) -> None:
        """Load only the current versioned cache schema."""

        cache_path = Path(self.config["cache_file"])
        if not cache_path.exists():
            return
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ Erro ao carregar cache: {exc}")
            return

        if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            print("ℹ️ Cache antigo ou incompatível ignorado; um novo cache será criado.")
            return
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            print("⚠️ Cache ignorado porque a coleção de entradas é inválida.")
            return
        self.cache = {
            str(key): value for key, value in entries.items() if isinstance(key, str) and isinstance(value, str)
        }
        print(f"📦 Cache carregado: {len(self.cache)} traduções (schema {CACHE_SCHEMA_VERSION})")

    def _save_cache(self) -> None:
        """Persist cache metadata and entries using an atomic replacement."""

        if not self.config["enable_cache"]:
            return
        cache_path = Path(self.config["cache_file"])
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "prompt_version": self.config["prompt_version"],
            "entries": self.cache,
        }
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
            )
            temporary.replace(cache_path)
        except OSError as exc:
            cleanup_error: OSError | None = None
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
            message = f"⚠️ Erro ao salvar cache: {exc}"
            if cleanup_error is not None:
                message += f"; falha ao remover temporário: {cleanup_error}"
            print(message)

    def _get_text_hash(self, prepared: PreparedSubtitleText | str) -> str:
        """Build a contextual cache key that cannot cross formats or prompts."""

        if isinstance(prepared, str):
            prepared = ASSFormatHandler().prepare_text(prepared)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "format": prepared.format_name,
            "cleaned_text": prepared.cleaned_text,
            "line_structure": prepared.line_structure,
            "protected_markings": prepared.protected_markings,
            "source_language": self.config["source_language"],
            "target_language": self.config["target_language"],
            "model": self.config["model"],
            "prompt_version": self.config["prompt_version"],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Compatibility helpers retained for callers that used the former ASS-only class.
    def clean_ass_tags(self, text: str) -> str:
        cleaned = re.sub(r"\{[^}]*\}", "", text)
        if self.config["turbo_mode"]:
            cleaned = re.sub(r"\\[a-z]+[0-9]*", "", cleaned)
        return cleaned.strip()

    def protect_ass_breaks_for_model(self, text: str) -> str:
        protected = text.replace(r"\N", "[[[ASS_BR]]]")
        protected = protected.replace(r"\n", "[[[ASS_br]]]")
        return protected.replace(r"\h", "[[[ASS_NBSP]]]")

    def restore_ass_breaks_from_model(self, text: str) -> str:
        restored = re.sub(
            r"\s*\[{1,3}\s*ASS_BR\s*\]{1,3}\s*", lambda _match: r"\N", text
        )
        restored = re.sub(
            r"\s*\[{1,3}\s*ASS_br\s*\]{1,3}\s*", lambda _match: r"\n", restored
        )
        return re.sub(
            r"\s*\[{1,3}\s*ASS_NBSP\s*\]{1,3}\s*", lambda _match: r"\h", restored
        ).strip()

    def has_internal_ass_tokens(self, text: str) -> bool:
        return "ASS_" in text or "[[[" in text or "]]]" in text

    @staticmethod
    def _insert_missing_ass_breaks(text: str, token: str, missing_count: int) -> str:
        repaired = text.strip()
        for _ in range(max(0, missing_count)):
            segments = repaired.split(token)
            longest_index = max(range(len(segments)), key=lambda index: len(segments[index]))
            segment = segments[longest_index]
            if not segment:
                segments[longest_index] = token
                repaired = token.join(segments)
                continue
            middle = len(segment) // 2
            candidates = [match.start() for match in re.finditer(r"\s+", segment)]
            if candidates:
                split_at = min(candidates, key=lambda position: abs(position - middle))
                left = segment[:split_at].rstrip()
                right = segment[split_at:].lstrip()
                segments[longest_index] = f"{left}{token}{right}"
            else:
                segments[longest_index] = f"{segment[:middle]}{token}{segment[middle:]}"
            repaired = token.join(segments)
        return repaired.strip()

    def repair_ass_escapes(self, source_text: str, translated_text: str) -> str:
        """Restore the legacy ASS break tokens without adding unrequested commands."""

        repaired = self.restore_ass_breaks_from_model(translated_text)
        if r"\h" not in source_text:
            repaired = re.sub(r"\\h(?=\\[Nn])", "", repaired)
            repaired = repaired.replace(r"\h", " ")
        for token in (r"\N", r"\n"):
            expected = source_text.count(token)
            current = repaired.count(token)
            if current < expected:
                repaired = self._insert_missing_ass_breaks(repaired, token, expected - current)
            elif current > expected:
                for _ in range(current - expected):
                    repaired = repaired.replace(token, " ", 1)
        repaired = re.sub(r"\[{1,3}\s*ASS_[A-Za-z_]+\s*\]{1,3}", "", repaired)
        return re.sub(r"[ \t]{2,}", " ", repaired).strip()

    def should_skip_line(self, text: str) -> tuple[bool, str]:
        """Determine whether visible text is meaningful enough to translate."""

        clean = self.clean_ass_tags(text)
        if len(clean) < self.config["min_text_length"]:
            if any(character.isalpha() for character in clean):
                return False, ""
            return True, "too_short"
        for pattern in self.skip_patterns:
            if pattern.match(clean):
                return True, "symbols_only"
        if self.config["skip_sfx"]:
            words = clean.lower().split()
            if len(words) <= 2 and words and all(word in self.sfx_words for word in words):
                return True, "sfx"
        return False, ""

    def _valid_cached_translation(
        self,
        handler: SubtitleFormatHandler,
        prepared: PreparedSubtitleText,
        cached: str,
    ) -> bool:
        try:
            assert_no_internal_artifacts(cached)
            if handler.format_name == "ass" and sanitize_ass_text(cached) != cached:
                return False
            cached_prepared = handler.prepare_text(cached)
        except SubtitleValidationError:
            return False
        return (
            cached_prepared.line_structure == prepared.line_structure
            and cached_prepared.protected_markings == prepared.protected_markings
        )

    def parse_subtitle_file(
        self, input_path: str | Path
    ) -> tuple[pysubs2.SSAFile, list[dict[str, Any]], SubtitleFormatHandler]:
        """Load one subtitle and create cache-aware translation records."""

        source = Path(input_path)
        handler = get_format_handler(source)
        print(f"📖 Carregando arquivo {handler.format_name.upper()}: {source}")
        loaded = handler.load(source)
        subs = handler.prepare_document(loaded)
        lines_to_translate: list[dict[str, Any]] = []
        unique_by_hash: dict[str, dict[str, Any]] = {}

        for index, event in enumerate(subs.events):
            if not event.text or event.type == "Comment":
                self.stats["skipped"] += 1
                continue
            prepared = handler.prepare_text(event.text)
            skip, _reason = self.should_skip_line(prepared.cleaned_text)
            if skip or not prepared.cleaned_text:
                self.stats["skipped"] += 1
                continue

            text_hash = self._get_text_hash(prepared)
            cached_translation: str | None = None
            candidate = self.cache.get(text_hash) if self.config["enable_cache"] else None
            if candidate is not None:
                if self._valid_cached_translation(handler, prepared, candidate):
                    cached_translation = candidate
                    self.stats["cached"] += 1
                else:
                    self.cache.pop(text_hash, None)
                    print(f"⚠️ Cache inválido descartado para o item {index + 1}.")

            line_info: dict[str, Any] = {
                "index": index,
                "original_event": event,
                "original_text": event.text,
                "clean_text": prepared.cleaned_text,
                "prepared": prepared,
                "format": handler.format_name,
                "text_hash": text_hash,
                "cached_translation": cached_translation,
                "translated_text": cached_translation,
                "style": event.style,
                "start": event.start,
                "end": event.end,
            }
            original = unique_by_hash.get(text_hash)
            if cached_translation is None and original is not None:
                line_info["linked_to"] = original["index"]
            elif cached_translation is None:
                unique_by_hash[text_hash] = line_info
            lines_to_translate.append(line_info)

        self.stats["total"] = len(lines_to_translate)
        unique_count = sum(
            1
            for line in lines_to_translate
            if line["cached_translation"] is None and "linked_to" not in line
        )
        print(f"📊 Carregadas {len(lines_to_translate)} linhas traduzíveis")
        print(f"   → Linhas únicas: {unique_count}")
        print(f"   → Em cache: {self.stats['cached']}")
        print(f"   → Puladas: {self.stats['skipped']}")
        return subs, lines_to_translate, handler

    def parse_ass_file(self, input_path: str) -> tuple[pysubs2.SSAFile, list[dict[str, Any]]]:
        """Backward-compatible ASS parsing entrypoint."""

        subs, lines, handler = self.parse_subtitle_file(input_path)
        if handler.format_name != "ass":
            raise ValueError("parse_ass_file aceita somente arquivos ASS.")
        return subs, lines

    def _source_language_instruction(self) -> str:
        """Describe automatic or explicit source semantics without detecting in Python."""

        source_language = str(self.config["source_language"])
        first_person_instruction = (
            "Quando encontrar as formas inglesas I, I'm, I've, I'll e I'd, traduza "
            "explicitamente seu sentido de primeira pessoa para PT-BR; não as mantenha em inglês."
        )
        if source_language.casefold() == "auto":
            source_instruction = (
                "Detecte automaticamente o idioma ou idiomas presentes no texto natural de cada "
                "ITEM. Um mesmo ITEM e uma mesma fala podem misturar vários idiomas. Traduza todo "
                "conteúdo natural que não estiver em português para português do Brasil. Se uma "
                "parte já estiver em português, preserve seu significado e integre a fala completa "
                "em uma única versão natural em PT-BR. Preserve nomes próprios, honoríficos, "
                "siglas e termos técnicos. Palavras isoladas significativas também são conteúdo "
                "natural: traduza-as quando estiverem em outro idioma, como Yes ou Ja, e preserve-as "
                "quando já estiverem em português, como Sim."
            )
        else:
            source_instruction = (
                f"Considere {source_language} como o idioma de origem informado para cada ITEM e "
                f"traduza o texto natural completo para {self.config['target_language']}. Preserve "
                "nomes próprios, honoríficos, siglas e termos técnicos."
            )
        return f"{source_instruction} {first_person_instruction}"

    def _build_item_prompt(
        self, pending: Sequence[tuple[int, PreparedSubtitleText]]
    ) -> str:
        item_blocks = "\n".join(
            f"<<<ITEM_{item_id:04d}>>>\n{prepared.model_text}\n"
            f"<<<END_ITEM_{item_id:04d}>>>"
            for item_id, prepared in pending
        )
        return (
            f"{self._source_language_instruction()}\n"
            f"Destino obrigatório: {self.config['target_language']}.\n"
            "Regras obrigatórias:\n"
            "- cada tradução deve ficar dentro de exatamente um par completo por ID: "
            "abertura ITEM antes do texto e fechamento END_ITEM depois do texto, ambos "
            "com literalmente os mesmos quatro dígitos recebidos;\n"
            "- devolva exatamente os mesmos delimitadores ITEM, sem criar ou renumerar itens;\n"
            "- não omita, inverta, aninhe ou duplique delimitadores e nunca coloque a "
            "tradução fora do par correspondente;\n"
            "- devolva somente as traduções dentro dos itens, sem explicações, Markdown ou listas;\n"
            "- preserve literalmente todos os tokens entre colchetes triplos, na mesma ordem;\n"
            "- não crie quebras, tags, comandos ASS, numeração ou conteúdo adicional;\n"
            "- não misture o texto de itens diferentes.\n\n"
            f"{item_blocks}"
        )

    def _build_individual_prompt(
        self, item_id: int, prepared: PreparedSubtitleText
    ) -> str:
        """Build the strict minimal prompt used for final per-item recovery."""

        return (
            f"{self._source_language_instruction()}\n"
            f"Destino obrigatório: {self.config['target_language']}.\n"
            "Responda somente com o mesmo bloco ITEM, sem Markdown ou explicações. "
            "Preserve literalmente os tokens entre colchetes triplos e não crie "
            "tags, comandos ou quebras.\n\n"
            f"<<<ITEM_{item_id:04d}>>>\n"
            f"{prepared.model_text}\n"
            f"<<<END_ITEM_{item_id:04d}>>>"
        )

    async def _generate(
        self, prompt: str, *, temperature: float | None = None
    ) -> str:
        effective_temperature = (
            float(self.config["temperature"])
            if temperature is None
            else float(temperature)
        )
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.ollama_client.generate,
                model=self.config["model"],
                prompt=prompt,
                system=self.config["system_prompt"],
                options={
                    "temperature": effective_temperature,
                    "num_predict": self.config["max_tokens"],
                    "top_p": 0.9,
                },
            ),
            timeout=self.config["timeout"],
        )
        if isinstance(response, Mapping):
            response_text = response.get("response")
        else:
            response_text = getattr(response, "response", None)
        if not isinstance(response_text, str):
            raise ValueError("Ollama retornou uma resposta sem campo textual válido.")
        return response_text.strip()

    @staticmethod
    def _parse_item_response(response_text: str) -> tuple[dict[int, str], str | None]:
        parsed: dict[int, str] = {}
        duplicated: set[int] = set()
        nested: set[int] = set()
        for match in ITEM_BLOCK_RE.finditer(response_text):
            item_id = int(match.group(1))
            body = match.group(2).strip()
            if BATCH_NESTED_ITEM_ARTIFACT_RE.search(body):
                nested.add(item_id)
                continue
            if item_id in parsed:
                duplicated.add(item_id)
            else:
                parsed[item_id] = body
        for item_id in duplicated | nested:
            parsed.pop(item_id, None)

        residue = ITEM_BLOCK_RE.sub("", response_text).strip()
        if nested:
            nested_ids = ", ".join(f"{item_id:04d}" for item_id in sorted(nested))
            nested_warning = f"A resposta contém delimitador ITEM aninhado no(s) item(ns) {nested_ids}."
            if residue:
                nested_warning += " A resposta contém texto ou Markdown fora dos itens."
            return parsed, nested_warning
        if residue:
            return parsed, "A resposta contém texto ou Markdown fora dos itens."
        return parsed, None

    @staticmethod
    def _discard_unexpected_items(
        parsed: MutableMapping[int, str], expected_ids: Sequence[int]
    ) -> None:
        expected = set(expected_ids)
        for item_id in sorted(set(parsed) - expected):
            parsed.pop(item_id, None)
            print(f"   ⚠️ ITEM inesperado {item_id:04d} descartado.")

    @staticmethod
    def _parse_individual_response(
        response_text: str, expected_item_id: int
    ) -> tuple[str | None, list[int], str | None, str | None]:
        """Extract one strict recovery body, optionally without an ITEM envelope."""

        matches = list(ITEM_BLOCK_RE.finditer(response_text))
        parsed, residue_warning = FixedASSTranslator._parse_item_response(response_text)
        parsed_ids = sorted(parsed)
        if (
            len(matches) == 1
            and int(matches[0].group(1)) == expected_item_id
            and residue_warning is None
        ):
            return parsed.get(expected_item_id), parsed_ids, None, None
        if matches:
            if len(matches) > 1:
                error = "A recuperação individual contém múltiplos ITEMs."
            else:
                error = "A recuperação individual contém ITEM conflitante ou resíduo externo."
            return None, parsed_ids, residue_warning, error

        candidate = response_text.strip()
        if not candidate:
            return None, parsed_ids, residue_warning, "A recuperação individual está vazia."
        if INDIVIDUAL_ITEM_ARTIFACT_RE.search(candidate):
            return (
                None,
                parsed_ids,
                residue_warning,
                "A recuperação individual contém delimitador ITEM parcial ou conflitante.",
            )
        if "\n" in candidate or "\r" in candidate:
            return (
                None,
                parsed_ids,
                residue_warning,
                "A recuperação individual sem envelope contém múltiplas linhas físicas.",
            )
        visible_candidate = CONTROL_TOKEN_RE.sub("", candidate)
        if INDIVIDUAL_RAW_MARKDOWN_RE.search(visible_candidate):
            return None, parsed_ids, residue_warning, "A recuperação individual contém Markdown."
        if INDIVIDUAL_RAW_EXPLANATION_RE.search(candidate):
            return (
                None,
                parsed_ids,
                residue_warning,
                "A recuperação individual contém explicação externa.",
            )
        return (
            candidate,
            parsed_ids,
            "Resposta individual sem envelope; aplicando validação estrutural estrita.",
            None,
        )

    def _validate_model_body(
        self,
        handler: SubtitleFormatHandler,
        prepared: PreparedSubtitleText,
        body: str,
    ) -> str:
        if "```" in body or MODEL_MESSAGE_RE.search(body):
            raise SubtitleValidationError("A resposta contém Markdown ou mensagem do modelo.")
        visible_body = CONTROL_TOKEN_RE.sub("", body).strip()
        if NUMBERED_PREFIX_RE.match(visible_body) and not NUMBERED_PREFIX_RE.match(
            prepared.cleaned_text
        ):
            raise SubtitleValidationError("A resposta introduziu numeração de lista.")
        normalized_body = re.sub(r"[ \t]{2,}", " ", body.strip())
        return handler.restore_text(prepared, normalized_body)

    async def translate_single_batch(
        self,
        prepared_items: Sequence[PreparedSubtitleText],
        handler: SubtitleFormatHandler,
        batch_num: int,
        total_batches: int,
        trace_items: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[TranslationOutcome]:
        """Translate a batch, recovering failed ASS breaks by validated segments."""

        outcomes = await self._translate_single_batch_once(
            prepared_items,
            handler,
            batch_num,
            total_batches,
            trace_items,
        )
        if handler.format_name != "ass":
            return outcomes

        recovered = list(outcomes)
        for source_index, (prepared, outcome) in enumerate(
            zip(prepared_items, outcomes), start=1
        ):
            if not self._needs_ass_line_break_recovery(prepared, outcome):
                continue

            segments, separators = handler.split_at_line_breaks(prepared)
            translatable_indices = [
                index for index, segment in enumerate(segments) if segment.model_text.strip()
            ]
            self._emit_trace(
                "line_break_recovery_started",
                mode="segmented",
                batch_num=batch_num,
                total_batches=total_batches,
                item_id=source_index,
                segment_count=len(segments),
                separators=list(separators),
                reason=outcome.error,
            )

            segment_results: list[TranslationOutcome | None] = [None] * len(segments)
            if translatable_indices:
                source_trace = (
                    dict(trace_items[source_index - 1])
                    if trace_items is not None and source_index - 1 < len(trace_items)
                    else {}
                )
                for segment_index in translatable_indices:
                    segment_outcome = await self._translate_single_batch_once(
                        [segments[segment_index]],
                        handler,
                        batch_num,
                        total_batches,
                        [
                            {
                                **source_trace,
                                "source_item_id": source_index,
                                "segment_index": segment_index + 1,
                                "segment_count": len(segments),
                                "line_break_recovery": True,
                            }
                        ],
                    )
                    segment_results[segment_index] = segment_outcome[0]

            for segment_index, segment in enumerate(segments):
                if segment_results[segment_index] is None:
                    segment_results[segment_index] = TranslationOutcome(segment.original_text)

            failed_segments = [
                (index, result)
                for index, result in enumerate(segment_results, start=1)
                if result is not None and result.used_fallback
            ]
            if failed_segments:
                details = "; ".join(
                    f"segmento {index}/{len(segments)}: {result.error}"
                    for index, result in failed_segments
                    if result is not None
                )
                recovered[source_index - 1] = TranslationOutcome(
                    text=prepared.original_text,
                    used_fallback=True,
                    error=f"Recuperação segmentada falhou ({details}).",
                )
                self._emit_trace(
                    "line_break_recovery_failed",
                    mode="segmented",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    item_id=source_index,
                    reason=recovered[source_index - 1].error,
                )
                continue

            try:
                restored = handler.join_line_break_segments(
                    prepared,
                    [result.text for result in segment_results if result is not None],
                    separators,
                )
            except SubtitleValidationError as exc:
                recovered[source_index - 1] = TranslationOutcome(
                    text=prepared.original_text,
                    used_fallback=True,
                    error=f"Recuperação segmentada inválida: {exc}",
                )
                self._emit_trace(
                    "line_break_recovery_failed",
                    mode="segmented",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    item_id=source_index,
                    reason=recovered[source_index - 1].error,
                )
                continue

            recovered[source_index - 1] = TranslationOutcome(restored)
            self._emit_trace(
                "line_break_recovery_completed",
                mode="segmented",
                batch_num=batch_num,
                total_batches=total_batches,
                item_id=source_index,
                segment_count=len(segments),
            )
        return recovered

    @staticmethod
    def _needs_ass_line_break_recovery(
        prepared: PreparedSubtitleText, outcome: TranslationOutcome
    ) -> bool:
        """Limit segmented recovery to a line failure seen in an enveloped batch."""

        return bool(
            outcome.used_fallback
            and outcome.line_break_recovery_eligible
            and prepared.line_structure
        )

    @staticmethod
    def _is_protected_line_break_error(
        prepared: PreparedSubtitleText, error: str
    ) -> bool:
        """Identify validation errors caused specifically by protected line structure."""

        normalized_error = error.casefold()
        line_break_tokens = [
            marker.token.casefold()
            for marker in prepared.markers
            if marker.kind == "line_break"
        ]
        return any(token in normalized_error for token in line_break_tokens) or any(
            fragment in normalized_error
            for fragment in (
                "quantidade de linhas",
                "múltiplas linhas físicas",
                "multiplas linhas fisicas",
                "quebra protegida",
            )
        )

    async def _translate_single_batch_once(
        self,
        prepared_items: Sequence[PreparedSubtitleText],
        handler: SubtitleFormatHandler,
        batch_num: int,
        total_batches: int,
        trace_items: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[TranslationOutcome]:
        """Run the existing ITEM protocol once, including selective recovery."""

        outcomes: list[TranslationOutcome | None] = [None] * len(prepared_items)
        pending: dict[int, PreparedSubtitleText] = {
            index: prepared for index, prepared in enumerate(prepared_items, start=1)
        }
        errors: dict[int, str] = {}
        line_break_recovery_eligible: set[int] = set()
        trace_lookup = {
            item_id: {
                "item_id": item_id,
                **(
                    dict(trace_items[item_id - 1])
                    if trace_items is not None and item_id - 1 < len(trace_items)
                    else {}
                ),
            }
            for item_id in pending
        }

        def trace_descriptors(item_ids: Sequence[int]) -> list[dict[str, Any]]:
            return [dict(trace_lookup[item_id]) for item_id in item_ids]

        for attempt in range(1, int(self.config["retry_count"]) + 1):
            if not pending:
                break
            pending_items = list(pending.items())
            print(
                f"   🔄 Batch {batch_num}/{total_batches} - tentativa {attempt} "
                f"({len(pending_items)} item(ns))"
            )
            prompt = self._build_item_prompt(pending_items)
            trace_context = {
                "mode": "batch",
                "batch_num": batch_num,
                "total_batches": total_batches,
                "attempt": attempt,
                "item_ids": [item_id for item_id, _prepared in pending_items],
                "items": trace_descriptors(
                    [item_id for item_id, _prepared in pending_items]
                ),
            }
            self._emit_trace(
                "batch_attempt_started",
                **trace_context,
                temperature=float(self.config["temperature"]),
                prompt=prompt,
            )
            try:
                response_text = await self._generate(prompt)
            except asyncio.TimeoutError:
                message = f"Timeout após {self.config['timeout']}s"
                for item_id in pending:
                    errors[item_id] = message
                print(f"   ⏰ {message} no batch {batch_num}.")
                self._emit_trace(
                    "model_error",
                    **trace_context,
                    error_type="TimeoutError",
                    reason=message,
                )
            except Exception as exc:
                if is_model_not_found_error(exc):
                    raise ModelUnavailableError(
                        f'❌ Modelo "{self.config["model"]}" não está disponível no Ollama. '
                        "Encerrando para nova execução com um modelo válido."
                    ) from exc
                message = f"Erro do Ollama: {exc}"
                for item_id in pending:
                    errors[item_id] = message
                print(f"   ❌ {message}")
                self._emit_trace(
                    "model_error",
                    **trace_context,
                    error_type=type(exc).__name__,
                    reason=message,
                )
            else:
                self._emit_trace(
                    "model_response",
                    **trace_context,
                    response=response_text,
                )
                parsed, response_warning = self._parse_item_response(response_text)
                parsed_ids = sorted(parsed)
                unexpected_ids = sorted(set(parsed) - set(pending))
                self._discard_unexpected_items(parsed, tuple(pending))
                self._emit_trace(
                    "response_parsed",
                    **trace_context,
                    parsed_item_ids=parsed_ids,
                    unexpected_item_ids=unexpected_ids,
                    warning=response_warning,
                )
                if response_warning:
                    print(f"   ⚠️ {response_warning} O resíduo externo foi descartado.")
                resolved_ids: list[int] = []
                for item_id, prepared in pending.items():
                    body = parsed.get(item_id)
                    if body is None:
                        errors[item_id] = "Item ausente ou duplicado na resposta."
                        self._emit_trace(
                            "item_rejected",
                            **trace_context,
                            item_id=item_id,
                            reason=errors[item_id],
                        )
                        continue
                    try:
                        restored = self._validate_model_body(handler, prepared, body)
                    except SubtitleValidationError as exc:
                        errors[item_id] = str(exc)
                        if self._is_protected_line_break_error(prepared, errors[item_id]):
                            line_break_recovery_eligible.add(item_id)
                        self._emit_trace(
                            "item_rejected",
                            **trace_context,
                            item_id=item_id,
                            reason=errors[item_id],
                        )
                        continue
                    outcomes[item_id - 1] = TranslationOutcome(restored)
                    resolved_ids.append(item_id)
                    self._emit_trace(
                        "item_accepted",
                        **trace_context,
                        item_id=item_id,
                    )
                for item_id in resolved_ids:
                    pending.pop(item_id, None)

            single_item_id = pending_items[0][0] if len(pending_items) == 1 else None
            if (
                single_item_id is not None
                and single_item_id in pending
                and errors.get(single_item_id)
                == "Item ausente ou duplicado na resposta."
            ):
                self._emit_trace(
                    "individual_recovery_scheduled",
                    mode="batch",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    attempt=attempt,
                    item_ids=[single_item_id],
                    items=trace_descriptors([single_item_id]),
                    reason=errors[single_item_id],
                )
                break

            if pending and attempt < int(self.config["retry_count"]):
                self._emit_trace(
                    "retry_scheduled",
                    mode="batch",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    item_ids=sorted(pending),
                    items=trace_descriptors(sorted(pending)),
                )
                await asyncio.sleep(float(self.config["retry_delay"]))

        if pending:
            print(
                f"   🛟 Recuperação individual de {len(pending)} item(ns) "
                "com temperatura 0.0"
            )
        for item_id, prepared in list(pending.items()):
            prompt = self._build_individual_prompt(item_id, prepared)
            trace_context = {
                "mode": "individual",
                "batch_num": batch_num,
                "total_batches": total_batches,
                "attempt": 1,
                "item_ids": [item_id],
                "items": trace_descriptors([item_id]),
            }
            self._emit_trace(
                "individual_attempt_started",
                **trace_context,
                temperature=0.0,
                prompt=prompt,
            )
            try:
                response_text = await self._generate(
                    prompt,
                    temperature=0.0,
                )
            except asyncio.TimeoutError:
                errors[item_id] = (
                    f"Timeout após {self.config['timeout']}s na recuperação individual."
                )
                self._emit_trace(
                    "model_error",
                    **trace_context,
                    error_type="TimeoutError",
                    reason=errors[item_id],
                )
                continue
            except Exception as exc:
                if is_model_not_found_error(exc):
                    raise ModelUnavailableError(
                        f'❌ Modelo "{self.config["model"]}" não está disponível no Ollama. '
                        "Encerrando para nova execução com um modelo válido."
                    ) from exc
                errors[item_id] = f"Erro do Ollama na recuperação individual: {exc}"
                self._emit_trace(
                    "model_error",
                    **trace_context,
                    error_type=type(exc).__name__,
                    reason=errors[item_id],
                )
                continue

            self._emit_trace(
                "model_response",
                **trace_context,
                response=response_text,
            )
            body, parsed_ids, response_warning, protocol_error = (
                self._parse_individual_response(response_text, item_id)
            )
            unexpected_ids = sorted(set(parsed_ids) - {item_id})
            self._emit_trace(
                "response_parsed",
                **trace_context,
                parsed_item_ids=parsed_ids,
                unexpected_item_ids=unexpected_ids,
                warning=response_warning,
            )
            if response_warning:
                print(f"   ⚠️ {response_warning}")
            if protocol_error is not None or body is None:
                errors[item_id] = protocol_error or "Item ausente na recuperação individual."
                self._emit_trace(
                    "item_rejected",
                    **trace_context,
                    item_id=item_id,
                    reason=errors[item_id],
                )
                continue
            try:
                restored = self._validate_model_body(handler, prepared, body)
            except SubtitleValidationError as exc:
                errors[item_id] = str(exc)
                self._emit_trace(
                    "item_rejected",
                    **trace_context,
                    item_id=item_id,
                    reason=errors[item_id],
                )
                continue
            outcomes[item_id - 1] = TranslationOutcome(restored)
            pending.pop(item_id, None)
            self._emit_trace(
                "item_accepted",
                **trace_context,
                item_id=item_id,
            )
            self._emit_trace(
                "individual_recovery_completed",
                **trace_context,
                item_id=item_id,
            )

        for item_id, prepared in pending.items():
            error = errors.get(item_id, "Falha de tradução sem detalhe.")
            print(f"   ⚠️ Item {item_id:04d}: fallback para o original ({error})")
            outcomes[item_id - 1] = TranslationOutcome(
                text=prepared.original_text,
                used_fallback=True,
                error=error,
                line_break_recovery_eligible=item_id in line_break_recovery_eligible,
            )
            self._emit_trace(
                "item_failed",
                mode="fallback",
                batch_num=batch_num,
                total_batches=total_batches,
                attempt=int(self.config["retry_count"]) + 1,
                item_ids=[item_id],
                items=trace_descriptors([item_id]),
                item_id=item_id,
                reason=error,
            )

        return [
            outcome
            if outcome is not None
            else TranslationOutcome(
                text=prepared_items[index].original_text,
                used_fallback=True,
                error="Resultado ausente após o processamento.",
            )
            for index, outcome in enumerate(outcomes)
        ]

    async def process_lines_optimized(
        self,
        lines: list[dict[str, Any]],
        handler: SubtitleFormatHandler | None = None,
    ) -> list[dict[str, Any]]:
        """Translate unique items in batches, then resolve cache and duplicates."""

        if handler is None:
            format_name = lines[0]["format"] if lines else "ass"
            handler = get_format_handler(format_name)
        to_translate = [
            line
            for line in lines
            if line.get("cached_translation") is None and "linked_to" not in line
        ]
        if to_translate:
            batch_size = int(self.config["batch_size"])
            batches = [to_translate[index : index + batch_size] for index in range(0, len(to_translate), batch_size)]
            print(f"\n📦 Processando {len(batches)} batches de até {batch_size} itens")
            for batch_num, batch in enumerate(batches, start=1):
                prepared_items = [line["prepared"] for line in batch]
                outcomes = await self.translate_single_batch(
                    prepared_items,
                    handler,
                    batch_num,
                    len(batches),
                    trace_items=[
                        {
                            "event_index": int(line["index"]),
                            "text_hash": str(line["text_hash"]),
                        }
                        for line in batch
                    ],
                )
                for line, outcome in zip(batch, outcomes):
                    line["translated_text"] = outcome.text
                    if outcome.used_fallback:
                        self.stats["failed"] += 1
                        line["translation_error"] = outcome.error
                    else:
                        self.stats["translated"] += 1
                        if self.config["enable_cache"]:
                            self.cache[line["text_hash"]] = outcome.text
                if batch_num % 5 == 0:
                    self._save_cache()

        translated_by_index = {
            line["index"]: line.get("translated_text")
            for line in lines
            if isinstance(line.get("translated_text"), str)
        }
        for line in lines:
            if "linked_to" in line:
                linked = translated_by_index.get(line["linked_to"])
                if isinstance(linked, str):
                    line["translated_text"] = linked
            if not isinstance(line.get("translated_text"), str) or not line["translated_text"]:
                line["translated_text"] = line["original_text"]
                line["translation_error"] = "Tradução vazia ou vínculo não resolvido."
                self.stats["failed"] += 1
        self._save_cache()
        return lines

    @staticmethod
    def _validate_rebuilt_document(
        original: pysubs2.SSAFile,
        rebuilt: pysubs2.SSAFile,
        handler: SubtitleFormatHandler,
    ) -> None:
        if len(original.events) != len(rebuilt.events):
            raise SubtitleValidationError("A reconstrução alterou a quantidade de eventos.")
        if handler.format_name == "srt":
            original_indices = getattr(original, "_srt_cue_indices", None)
            rebuilt_indices = getattr(rebuilt, "_srt_cue_indices", None)
            if original_indices != rebuilt_indices:
                raise SubtitleValidationError("A reconstrução alterou os índices dos blocos SRT.")
        for index, (source_event, output_event) in enumerate(
            zip(original.events, rebuilt.events), start=1
        ):
            if output_event.start != source_event.start or output_event.end != source_event.end:
                raise SubtitleValidationError(f"Os horários do evento {index} foram alterados.")
            if output_event.end < output_event.start:
                raise SubtitleValidationError(f"O evento {index} possui horário final inválido.")
            assert_no_internal_artifacts(output_event.text)
            source_structure = handler.prepare_text(source_event.text)
            output_structure = handler.prepare_text(output_event.text)
            if source_structure.line_structure != output_structure.line_structure:
                raise SubtitleValidationError(f"As quebras do evento {index} foram alteradas.")
            if source_structure.protected_markings != output_structure.protected_markings:
                raise SubtitleValidationError(f"As marcações do evento {index} foram alteradas.")
            if source_event.text and not output_event.text:
                raise SubtitleValidationError(f"O evento {index} ficou vazio.")

    def rebuild_ass(
        self, subs: pysubs2.SSAFile, translated_lines: Sequence[Mapping[str, Any]]
    ) -> pysubs2.SSAFile:
        """Backward-compatible ASS rebuild method using the format handler."""

        handler = ASSFormatHandler()
        translations = {
            int(line["index"]): str(line.get("translated_text") or line.get("original_text", ""))
            for line in translated_lines
        }
        rebuilt = handler.rebuild(subs, translations)
        self._validate_rebuilt_document(subs, rebuilt, handler)
        return rebuilt

    async def translate_file(
        self, input_path: str | Path, output_path: str | Path | None = None
    ) -> dict[str, int]:
        """Run the complete format-aware pipeline for one subtitle file."""

        self.reset_stats()
        source = Path(input_path)
        handler = get_format_handler(source)
        if output_path is None:
            output_path = source.parent / f"{source.stem}_traduzido{source.suffix}"
        destination = Path(output_path)
        if destination.suffix.lower() != handler.extension:
            raise SubtitleValidationError(
                f"A extensão de saída deve permanecer {handler.extension} para {source.name}."
            )

        self._trace_file = str(source.resolve())
        self._emit_trace(
            "file_started",
            output=str(destination.resolve()),
            format=handler.format_name,
        )

        print(f"\n{'=' * 60}")
        print(f"🎬 Iniciando tradução de legenda {handler.format_name.upper()}")
        print(f"📁 Arquivo: {source.name}")
        print(f"🤖 Modelo: {self.config['model']}")
        print(f"📦 Batch: {self.config['batch_size']} itens/requisição")
        print(f"⏰ Timeout: {self.config['timeout']}s")
        print(f"🔄 Retries por item: {self.config['retry_count']}")
        print(f"💾 Cache: {'Ativado' if self.config['enable_cache'] else 'Desativado'}")
        print(f"{'=' * 60}")

        started_at = datetime.now()
        subs, lines, handler = self.parse_subtitle_file(source)
        translated_lines = await self.process_lines_optimized(lines, handler) if lines else lines
        if self.stats["failed"] and not bool(self.config["allow_original_fallback"]):
            print(
                f"❌ Falha definitiva em {self.stats['failed']} item(ns). "
                f"Nenhum novo output foi produzido para {source.name}."
            )
            self._emit_trace(
                "file_failed",
                output=str(destination.resolve()),
                reason="incomplete_translation",
                stats=dict(self.stats),
            )
            raise IncompleteTranslationError(self.stats["failed"])
        translations = {
            line["index"]: line.get("translated_text", line["original_text"])
            for line in translated_lines
        }
        rebuilt = handler.rebuild(subs, translations)
        self._validate_rebuilt_document(subs, rebuilt, handler)
        candidate = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.candidate{destination.suffix}"
        )
        try:
            handler.save(rebuilt, candidate)
            # Validate the exact serialization before atomically publishing it.
            saved = handler.load(candidate)
            self._validate_rebuilt_document(rebuilt, saved, handler)
            candidate.replace(destination)
        finally:
            candidate.unlink(missing_ok=True)

        elapsed = max((datetime.now() - started_at).total_seconds(), 0.001)
        print(f"\n✅ Arquivo {handler.format_name.upper()} concluído: {source.name}")
        print(f"📄 Saída: {destination}")
        print(
            "📊 Estatísticas "
            f"{handler.format_name.upper()}: total={self.stats['total']}, "
            f"traduzidas={self.stats['translated']}, cache={self.stats['cached']}, "
            f"puladas={self.stats['skipped']}, falhas={self.stats['failed']}"
        )
        print(f"⏱️ Tempo: {elapsed:.1f}s")
        result = dict(self.stats)
        self._emit_trace(
            "file_completed",
            output=str(destination.resolve()),
            stats=result,
            elapsed_seconds=elapsed,
        )
        self._trace_file = None
        return result


SubtitleTranslator = FixedASSTranslator
