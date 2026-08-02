#!/usr/bin/env python3
"""CLI para tradução em lote de legendas ASS e SRT com Ollama."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import ollama
from tqdm import tqdm

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
    """Best-effort model unload after a local or remote translation run."""

    if not model_name:
        return
    print(f"\n🛑 Encerrando modelo Ollama: {model_name}")
    try:
        running = ollama.ps()
        loaded_models: list[str] = []
        for model_info in getattr(running, "models", []):
            model = getattr(model_info, "model", None)
            name = getattr(model_info, "name", None)
            if model:
                loaded_models.append(model)
            if name:
                loaded_models.append(name)
        if model_name not in loaded_models:
            print(f"ℹ️ Modelo '{model_name}' já não está carregado.")
            return
    except Exception as exc:
        print(f"⚠️ Não foi possível consultar modelos ativos ({exc}). Tentando encerrar mesmo assim...")

    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        print(f"✅ Modelo '{model_name}' descarregado via API.")
        return
    except Exception as exc:
        print(f"⚠️ Falha ao descarregar via API: {exc}")

    ollama_cli = shutil.which("ollama")
    if not ollama_cli:
        print("⚠️ Comando 'ollama' não encontrado para fallback de encerramento.")
        return
    try:
        result = subprocess.run(
            [ollama_cli, "stop", model_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"⚠️ Erro ao tentar encerrar via CLI: {exc}")
        return
    details = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        print(f"✅ Modelo '{model_name}' encerrado via CLI.")
    else:
        print(f"⚠️ Não foi possível encerrar o modelo via CLI (código {result.returncode}).")
    if details:
        print(f"   ↳ {details}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser for local and packaged entrypoints."""

    parser = argparse.ArgumentParser(description="Tradutor de legendas ASS/SRT em lote")
    parser.add_argument(
        "-i", "--input-dir", default="./entrada", help="Pasta com legendas de entrada"
    )
    parser.add_argument(
        "-o", "--output-dir", default="./saida", help="Pasta de saída das legendas traduzidas"
    )
    parser.add_argument("-m", "--model", default=CONFIG["model"], help="Modelo Ollama")
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


async def main(argv: Sequence[str] | None = None) -> int:
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

    output_dir.mkdir(parents=True, exist_ok=True)
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

    cache_path = Path(CONFIG["cache_file"])
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
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "turbo_mode": args.turbo,
        "enable_cache": not args.no_cache,
    }
    if args.turbo:
        config["temperature"] = 0.15
        config["batch_size"] = min(config["batch_size"], 10)
        print("🔥 MODO TURBO ATIVADO")

    try:
        ensure_ollama_model_available(config["model"])
    except (ModelUnavailableError, RuntimeError) as exc:
        print(str(exc))
        print("⛔ Encerrando backend para nova tentativa com um modelo válido.")
        return 2

    translator = FixedASSTranslator(config)
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
        shutdown_ollama_model(config["model"])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
