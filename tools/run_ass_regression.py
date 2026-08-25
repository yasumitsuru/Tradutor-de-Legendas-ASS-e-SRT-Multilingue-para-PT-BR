#!/usr/bin/env python3
"""Run isolated real-world ASS regression experiments with the production CLI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# Allow direct execution with ``python tools/run_ass_regression.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ass_regression import (  # noqa: E402
    AnalysisContext,
    CompressedTraceRecorder,
    Diagnostic,
    RegressionReport,
    analyze_ass_pair,
    collect_runtime_metadata,
    discover_ass_files,
    file_sha256,
    trace_failure_diagnostics,
    write_json,
    write_reports,
)
from backend_config import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_LLAMA_SWAP_API_BASE,
    BackendSettings,
    get_configured_model_profile,
    normalize_backend_endpoint,
)
from backend_factory import create_backend  # noqa: E402
from inference_backend import BackendConfigurationError  # noqa: E402
from model_profiles import select_model  # noqa: E402
from subtitle_formats import preserve_extension_output_path  # noqa: E402
from translate_ass_fast import main as translation_cli_main  # noqa: E402
from translation_engine import CONFIG  # noqa: E402


EXIT_CRITICAL_FAILURE = 1
EXIT_OLLAMA_UNAVAILABLE = 2
EXIT_CONFIGURATION_ERROR = 3


class _Tee:
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    @property
    def encoding(self) -> str:
        return "utf-8"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bateria manual de regressão ASS usando o backend selecionado e a CLI de produção"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("entrada"))
    parser.add_argument("--file", type=Path, dest="input_file")
    parser.add_argument("--output-dir", type=Path, default=Path("test_results"))
    parser.add_argument("--model", default=CONFIG["model"])
    parser.add_argument(
        "--backend",
        choices=("ollama", "llama-swap"),
        default=DEFAULT_BACKEND,
    )
    parser.add_argument("--api-base", default=DEFAULT_LLAMA_SWAP_API_BASE)
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--source-language", default=CONFIG["source_language"])
    parser.add_argument("--turbo", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", dest="no_cache", action="store_true", default=True)
    cache.add_argument("--with-cache", dest="no_cache", action="store_false")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict-semantic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--experiment", default="baseline")
    return parser


def _safe_component(value: str, label: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError(f"{label} deve conter apenas letras, números, '_' ou '-'.")
    return value


def _effective_settings(args: argparse.Namespace, cache_file: Path | None) -> BackendSettings:
    api_base = args.api_base
    if args.backend == DEFAULT_BACKEND and api_base == DEFAULT_LLAMA_SWAP_API_BASE:
        api_base = None
    endpoint = normalize_backend_endpoint(args.backend, api_base)
    return BackendSettings(
        backend=args.backend,
        api_base=endpoint.api_base,
        model=args.model,
        batch_size=args.batch_size,
        timeout=args.timeout,
        enable_cache=not args.no_cache,
        cache_file=str(cache_file) if cache_file is not None else CONFIG["cache_file"],
    )


def _effective_context(
    args: argparse.Namespace, file_count: int, settings: BackendSettings
) -> dict[str, Any]:
    profile = get_configured_model_profile(settings)
    generation_config = {**CONFIG}
    if profile is not None:
        generation_config.update(profile.generation_defaults)
    effective_batch = min(args.batch_size, 10) if args.turbo else args.batch_size
    effective_temperature = 0.15 if args.turbo else float(generation_config["temperature"])
    system_prompt = str(generation_config["system_prompt"])
    selection = select_model(settings.backend, settings.model)
    return {
        "backend": settings.backend,
        "api_base": settings.api_base,
        "model": settings.model,
        "model_profile": {
            "known": selection.profile is not None,
            "status": sorted(selection.status),
            "provenance": selection.provenance,
        },
        "generation_parameters": {
            "temperature": effective_temperature,
            "num_predict": generation_config["max_tokens"],
            "num_ctx": generation_config.get("num_ctx"),
            "top_p": 0.9,
            "timeout_seconds": args.timeout,
            "retry_count": generation_config["retry_count"],
            "retry_delay_seconds": generation_config["retry_delay"],
        },
        "batch_size_requested": args.batch_size,
        "batch_size": effective_batch,
        "turbo": bool(args.turbo),
        "cache_enabled": not args.no_cache,
        "allow_original_fallback": False,
        "source_language": args.source_language,
        "target_language": generation_config["target_language"],
        "prompt_version": generation_config["prompt_version"],
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "files": file_count,
    }


def _snapshot_sources(sources: Sequence[Path], destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []
    for source in sources:
        snapshot = destination / source.name
        shutil.copy2(source, snapshot)
        source_hash = file_sha256(source)
        snapshot_hash = file_sha256(snapshot)
        if source_hash != snapshot_hash:
            raise OSError(f"O snapshot de {source.name} não preservou os bytes originais.")
        manifest.append(
            {
                "name": source.name,
                "original_path": str(source.resolve()),
                "snapshot_path": str(snapshot.resolve()),
                "sha256": source_hash,
                "size_bytes": source.stat().st_size,
            }
        )
    return manifest


def _print_summary(
    report: RegressionReport, experiment_dir: Path, *, verbose: bool = False
) -> None:
    metrics = report.metrics
    print("\n========================================")
    print("ASS REAL-WORLD REGRESSION TEST")
    print("========================================")
    print(f"Arquivos encontrados: {metrics['files']}")
    print(f"Arquivos aprovados: {metrics['files_passed']}")
    print(f"Arquivos com problemas: {metrics['files_with_critical_failures']}")
    print(f"Eventos analisados: {metrics['events']}")
    print(f"Eventos traduzíveis: {metrics['translatable_events']}")
    print(f"Erros críticos: {metrics['critical_failures']}")
    print(f"Warnings semânticos: {metrics['semantic_warnings']}")
    print(f"Relatório: {experiment_dir / 'report.md'}")
    if verbose:
        for file_analysis in report.files:
            for item in file_analysis.diagnostics:
                location = (
                    item.file
                    if item.event_index is None
                    else f"{item.file}: evento {item.event_index}"
                )
                print(
                    f"- {item.severity} {item.category} [{item.confidence}] "
                    f"{location}: {item.message}"
                )


async def run_experiment(args: argparse.Namespace) -> int:
    started_at = datetime.now().astimezone().isoformat()
    try:
        run_id = _safe_component(
            args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"), "run-id"
        )
        experiment = _safe_component(args.experiment, "experiment")
    except ValueError as exc:
        print(f"❌ {exc}")
        return EXIT_CONFIGURATION_ERROR
    if args.batch_size < 1 or args.timeout < 1 or (args.limit is not None and args.limit < 1):
        print("❌ batch-size, timeout e limit devem ser maiores que zero.")
        return EXIT_CONFIGURATION_ERROR

    if args.input_file is not None:
        if not args.input_file.is_file() or args.input_file.suffix.casefold() != ".ass":
            print(f"❌ O arquivo informado não é um ASS válido: {args.input_file}")
            return EXIT_CONFIGURATION_ERROR
        sources = [args.input_file]
    else:
        sources = discover_ass_files(args.input_dir)
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        print(f"❌ Nenhum arquivo ASS encontrado em {args.input_dir}.")
        return EXIT_CONFIGURATION_ERROR

    experiment_dir = args.output_dir / run_id / experiment
    try:
        experiment_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"❌ O experimento já existe e não será sobrescrito: {experiment_dir}")
        return EXIT_CONFIGURATION_ERROR

    snapshots_dir = experiment_dir / "sources"
    outputs_dir = experiment_dir / "outputs"
    outputs_dir.mkdir()
    isolated_cache_path: Path | None = None
    if not args.no_cache:
        isolated_cache_path = experiment_dir / "cache" / "translation_cache.json"
        isolated_cache_path.parent.mkdir()
    try:
        source_manifest = _snapshot_sources(sources, snapshots_dir)
    except OSError as exc:
        write_json(
            experiment_dir / "preflight_error.json",
            {"status": "NOT_EXECUTED", "reason": str(exc)},
        )
        return EXIT_CONFIGURATION_ERROR
    write_json(experiment_dir / "sources.json", {"files": source_manifest})

    try:
        settings = _effective_settings(args, isolated_cache_path)
    except BackendConfigurationError as exc:
        write_json(
            experiment_dir / "preflight_error.json",
            {"status": "NOT_EXECUTED", "reason": str(exc)},
        )
        return EXIT_CONFIGURATION_ERROR

    context = {
        **collect_runtime_metadata(PROJECT_ROOT),
        **_effective_context(args, len(sources), settings),
        "experiment": experiment,
        "run_id": run_id,
        "input_dir": str(args.input_dir.resolve()),
        "input_file": str(args.input_file.resolve()) if args.input_file else None,
        "output_dir": str(outputs_dir.resolve()),
        "cache_file": (
            str(isolated_cache_path.resolve()) if isolated_cache_path is not None else None
        ),
        "status": "PREFLIGHT",
        "started_at": started_at,
    }
    write_json(experiment_dir / "context.json", context)

    try:
        create_backend(settings).ensure_available(settings.model)
    except BackendConfigurationError as exc:
        failure = {"status": "NOT_EXECUTED", "reason": str(exc), "model": settings.model}
        context.update(failure)
        context["completed_at"] = datetime.now().astimezone().isoformat()
        write_json(experiment_dir / "context.json", context)
        write_json(experiment_dir / "preflight_error.json", failure)
        print(f"❌ Configuração do backend inválida: {exc}")
        return EXIT_CONFIGURATION_ERROR
    except Exception as exc:
        failure = {"status": "NOT_EXECUTED", "reason": str(exc), "model": settings.model}
        context.update(failure)
        context["completed_at"] = datetime.now().astimezone().isoformat()
        write_json(experiment_dir / "context.json", context)
        write_json(experiment_dir / "preflight_error.json", failure)
        print(f"❌ Regressão real não executada: {exc}")
        return EXIT_OLLAMA_UNAVAILABLE

    trace = CompressedTraceRecorder(experiment_dir / "trace.jsonl.gz")
    cli_argv = [
        "--input-dir",
        str(snapshots_dir),
        "--output-dir",
        str(outputs_dir),
        "--format",
        "ass",
        "--model",
        str(settings.model),
        "--backend",
        settings.backend,
        "--api-base",
        str(settings.api_base),
        "--source-language",
        str(args.source_language),
        "--batch-size",
        str(args.batch_size),
        "--timeout",
        str(args.timeout),
        "--result-manifest",
        str(experiment_dir / "run_outputs.json"),
    ]
    if args.no_cache:
        cli_argv.append("--no-cache")
    if args.turbo:
        cli_argv.append("--turbo")

    cli_exit_code = EXIT_CRITICAL_FAILURE
    log_path = experiment_dir / "run.log"
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
            stdout_tee = _Tee(sys.stdout, log_file)
            stderr_tee = _Tee(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                print(f"🧪 Experimento: {experiment}")
                print(f"🤖 Modelo validado: {args.model}")
                print(f"📁 Corpus: {len(sources)} arquivo(s)")
                cli_exit_code = await translation_cli_main(
                    cli_argv,
                    trace_hook=trace,
                    cache_file=isolated_cache_path,
                )
    except Exception as exc:
        print(f"❌ Falha ao executar a rota de produção: {exc}")
        context["runner_exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        trace.close()

    snapshot_sources = discover_ass_files(snapshots_dir)
    analyses = [
        analyze_ass_pair(
            source,
            preserve_extension_output_path(source, outputs_dir),
        )
        for source in snapshot_sources
    ]
    failures_by_file = trace_failure_diagnostics(
        experiment_dir / "trace.jsonl.gz",
        {source.name: source for source in snapshot_sources},
    )
    for analysis in analyses:
        analysis.diagnostics.extend(
            failures_by_file.get(Path(analysis.source_path).name, [])
        )
    if cli_exit_code not in (0, EXIT_OLLAMA_UNAVAILABLE) and analyses:
        analyses[0].diagnostics.append(
            Diagnostic(
                category="PIPELINE_EXECUTION_ERROR",
                severity="FAIL",
                confidence="objective",
                message=f"A CLI de produção terminou com exit code {cli_exit_code}.",
                file=analyses[0].source_path,
            )
        )

    source_integrity_errors: list[dict[str, str]] = []
    for item in source_manifest:
        original_path = Path(item["original_path"])
        try:
            current_hash = file_sha256(original_path)
        except OSError as exc:
            source_integrity_errors.append(
                {
                    "file": str(original_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if current_hash != item["sha256"]:
            source_integrity_errors.append(
                {
                    "file": str(original_path),
                    "reason": "SHA-256 diferente do snapshot inicial.",
                }
            )
    source_integrity = not source_integrity_errors
    if not source_integrity and analyses:
        analyses[0].diagnostics.append(
            Diagnostic(
                category="SOURCE_MUTATED",
                severity="FAIL",
                confidence="objective",
                message="Um ou mais arquivos originais mudaram durante a execução.",
                file=analyses[0].source_path,
            )
        )

    if cli_exit_code == EXIT_OLLAMA_UNAVAILABLE:
        late_unavailability = {
            "status": "NOT_EXECUTED",
            "reason": "A CLI de produção informou indisponibilidade do modelo.",
            "model": args.model,
        }
        write_json(experiment_dir / "preflight_error.json", late_unavailability)

    context.update(
        {
            "status": (
                "NOT_EXECUTED"
                if cli_exit_code == EXIT_OLLAMA_UNAVAILABLE
                else "COMPLETED" if cli_exit_code == 0 else "FAILED"
            ),
            "completed_at": datetime.now().astimezone().isoformat(),
            "cli_exit_code": cli_exit_code,
            "source_integrity_preserved": source_integrity,
            "source_integrity_errors": source_integrity_errors,
            "trace_metrics": trace.summary(),
        }
    )
    write_json(experiment_dir / "context.json", context)
    report = RegressionReport.from_files(
        AnalysisContext(
            experiment=experiment,
            run_id=run_id,
            metadata=context,
        ),
        analyses,
    )
    write_reports(report, experiment_dir)
    _print_summary(report, experiment_dir, verbose=args.verbose)
    if cli_exit_code == EXIT_OLLAMA_UNAVAILABLE:
        return EXIT_OLLAMA_UNAVAILABLE
    return report.exit_code(strict_semantic=args.strict_semantic)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    return asyncio.run(run_experiment(args))


if __name__ == "__main__":
    raise SystemExit(main())
