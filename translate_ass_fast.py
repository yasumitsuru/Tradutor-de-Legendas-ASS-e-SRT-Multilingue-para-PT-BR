#!/usr/bin/env python3
"""CLI para tradução em lote de legendas ASS e SRT com Ollama."""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm import tqdm

from inference_backend import BackendModelNotFoundError
from run_manifest import initialize_run_manifest, record_run_output
from subtitle_formats import (
    SUPPORTED_FORMATS,
    count_subtitle_formats,
    iter_subtitle_files,
    preserve_extension_output_path,
)
from translation_engine import (
    CONFIG,
    FixedASSTranslator,
    ModelUnavailableError,
    SubtitleTranslator,
    ensure_ollama_model_available,
    is_model_not_found_error,
)
from ollama_backend import OllamaBackend


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace", line_buffering=True, write_through=True
    )
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8", errors="replace", line_buffering=True, write_through=True
    )


# Backward-compatible private names used by earlier integrations.
_is_model_not_found_error = is_model_not_found_error
_ensure_ollama_model_available = ensure_ollama_model_available


def shutdown_ollama_model(model_name: str) -> None:
    """Backward-compatible wrapper for the Ollama adapter shutdown path."""

    OllamaBackend().close()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser for local and packaged entrypoints."""

    parser = argparse.ArgumentParser(
        description="Tradutor multilíngue de legendas ASS/SRT para PT-BR"
    )
    parser.add_argument(
        "-i", "--input-dir", default="./entrada", help="Pasta com legendas de entrada"
    )
    parser.add_argument(
        "-o", "--output-dir", default="./saida", help="Pasta de saída das legendas traduzidas"
    )
    parser.add_argument("-m", "--model", default=CONFIG["model"], help="Modelo Ollama")
    parser.add_argument(
        "--source-language",
        default=CONFIG["source_language"],
        metavar="IDIOMA",
        help='Idioma de origem. Use "auto" para detecção automática por item',
    )
    parser.add_argument(
        "--batch-size", type=int, default=CONFIG["batch_size"], help="Itens por lote"
    )
    parser.add_argument(
        "--timeout", type=int, default=CONFIG["timeout"], help="Timeout em segundos"
    )
    parser.add_argument(
        "--format",
        choices=("all", *SUPPORTED_FORMATS),
        default="all",
        help="Filtrar por formato: all, ass ou srt",
    )
    parser.add_argument("--turbo", action="store_true", help="Reduz temperatura e tamanho do lote")
    parser.add_argument("--clear-cache", action="store_true", help="Limpa o cache antes de iniciar")
    parser.add_argument("--no-cache", action="store_true", help="Desativa o cache")
    parser.add_argument(
        "--allow-original-fallback",
        action="store_true",
        default=False,
        help="Permite salvar itens que falharam mantendo o texto original",
    )
    parser.add_argument(
        "--result-manifest",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def _ignored_input_files(input_dir: Path, selected: Sequence[Path]) -> list[str]:
    selected_names = {path.name for path in selected}
    return sorted(
        (
            path.name
            for path in input_dir.iterdir()
            if path.is_file() and path.name not in selected_names and path.name != ".gitkeep"
        ),
        key=str.casefold,
    )


def _new_format_summary() -> dict[str, dict[str, int]]:
    return {
        format_name: {
            "files": 0,
            "processed": 0,
            "failed_files": 0,
            "total": 0,
            "translated": 0,
            "cached": 0,
            "skipped": 0,
            "failed": 0,
        }
        for format_name in SUPPORTED_FORMATS
    }


def _add_stats(summary: dict[str, int], stats: dict[str, int]) -> None:
    for key in ("total", "translated", "cached", "skipped", "failed"):
        summary[key] += int(stats.get(key, 0))


async def main(
    argv: Sequence[str] | None = None,
    *,
    trace_hook: Callable[[dict[str, Any]], None] | None = None,
    cache_file: str | Path | None = None,
) -> int:
    """Validate CLI arguments and process every selected subtitle safely."""

    args = build_argument_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Pasta de entrada não encontrada: {input_dir}")
        return 1
    if args.batch_size < 1:
        print("❌ Batch size deve ser maior que zero.")
        return 1
    if args.timeout < 1:
        print("❌ Timeout deve ser maior que zero.")
        return 1
    source_language = args.source_language.strip()
    if not source_language:
        print('❌ Idioma de origem não pode ser vazio. Use "auto" para detecção automática.')
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.result_manifest is not None:
        initialize_run_manifest(args.result_manifest)
    subtitle_files = iter_subtitle_files(input_dir, args.format)
    ignored_files = _ignored_input_files(input_dir, subtitle_files)
    counts = count_subtitle_formats(subtitle_files)
    if not subtitle_files:
        print(f"⚠️ Nenhum arquivo ASS/SRT encontrado em {input_dir} para o filtro '{args.format}'.")
        if ignored_files:
            print(f"ℹ️ Arquivos ignorados: {', '.join(ignored_files)}")
        return 0

    print(f"📂 Encontrados {len(subtitle_files)} arquivos:")
    print(f"- {counts['ass']} ASS")
    print(f"- {counts['srt']} SRT")
    if ignored_files:
        print(f"ℹ️ Arquivos ignorados: {', '.join(ignored_files)}")
    print(f"📤 Saída: {output_dir.resolve()}")

    cache_path = Path(cache_file) if cache_file is not None else Path(CONFIG["cache_file"])
    if args.clear_cache and cache_path.exists():
        try:
            cache_path.unlink()
        except OSError as exc:
            print(f"❌ Não foi possível limpar o cache: {exc}")
            return 1
        print("🗑️ Cache limpo!")

    config: dict[str, Any] = {
        **CONFIG,
        "model": args.model,
        "source_language": source_language,
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "turbo_mode": args.turbo,
        "enable_cache": not args.no_cache,
        "cache_file": str(cache_path),
        "allow_original_fallback": args.allow_original_fallback,
    }
    if trace_hook is not None:
        config["trace_hook"] = trace_hook
    if args.turbo:
        config["temperature"] = 0.15
        config["batch_size"] = min(config["batch_size"], 10)
        print("🔥 MODO TURBO ATIVADO")

    if config["source_language"].casefold() == "auto":
        print("🌐 Idioma de origem: detecção automática por item")
    else:
        print(f"🌐 Idioma de origem informado: {config['source_language']}")

    backend = OllamaBackend()
    try:
        try:
            backend.ensure_available(config["model"])
        except BackendModelNotFoundError as exc:
            raise ModelUnavailableError(
                f'❌ Modelo "{config["model"]}" não está disponível no Ollama. '
                f"Instale antes com: ollama pull {config["model"]}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"❌ Não foi possível validar o modelo no Ollama: {exc}") from exc
    except (ModelUnavailableError, RuntimeError) as exc:
        print(str(exc))
        print("⛔ Encerrando backend para nova tentativa com um modelo válido.")
        return 2

    translator = FixedASSTranslator(config, backend=backend)
    season_started_at = datetime.now()
    summary = _new_format_summary()
    failed_files: list[str] = []
    for format_name, count in counts.items():
        summary[format_name]["files"] = count

    try:
        for subtitle_file in tqdm(subtitle_files, desc="📦 Processando legendas", unit="arquivo"):
            format_name = subtitle_file.suffix.lower().removeprefix(".")
            output_file = preserve_extension_output_path(subtitle_file, output_dir)
            try:
                stats = await translator.translate_file(subtitle_file, output_file)
                if args.result_manifest is not None:
                    record_run_output(args.result_manifest, output_dir, output_file)
            except ModelUnavailableError as exc:
                print(str(exc))
                print("⛔ Encerrando backend para nova tentativa com um modelo válido.")
                return 2
            except Exception as exc:
                print(f"❌ Erro crítico ao processar {subtitle_file.name}: {exc}")
                traceback.print_exc()
                failed_files.append(subtitle_file.name)
                summary[format_name]["failed_files"] += 1
            else:
                summary[format_name]["processed"] += 1
                _add_stats(summary[format_name], stats)
            translator._save_cache()

        elapsed = (datetime.now() - season_started_at).total_seconds()
        print(f"\n{'=' * 60}")
        print("✅ PROCESSAMENTO DE LEGENDAS CONCLUÍDO")
        for format_name in SUPPORTED_FORMATS:
            format_stats = summary[format_name]
            print(
                f"- {format_name.upper()}: arquivos={format_stats['processed']}/"
                f"{format_stats['files']}, linhas={format_stats['total']}, "
                f"traduzidas={format_stats['translated']}, cache={format_stats['cached']}, "
                f"puladas={format_stats['skipped']}, falhas={format_stats['failed']}"
            )
        if failed_files:
            print(f"⚠️ Arquivos com falha: {', '.join(failed_files)}")
        print(f"💾 Cache final: {len(translator.cache)} entradas")
        print(f"⏱️ Tempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"{'=' * 60}")
        return 1 if failed_files else 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
