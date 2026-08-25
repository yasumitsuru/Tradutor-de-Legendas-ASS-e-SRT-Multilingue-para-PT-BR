from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import translate_ass_fast
from run_manifest import load_run_outputs
from translation_engine import IncompleteTranslationError


@pytest.fixture(autouse=True)
def available_ollama_backend(monkeypatch) -> None:
    class AvailableOllamaBackend:
        def ensure_available(self, _model: str) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "OllamaBackend", AvailableOllamaBackend)


def test_cli_processes_both_formats_preserving_extension_and_reporting_counts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "Alpha.ASS").write_text("fixture", encoding="utf-8")
    (input_dir / "Beta.SRT").write_text("fixture", encoding="utf-8")
    (input_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    destinations: list[Path] = []

    class FakeTranslator:
        def __init__(self, _config, **_kwargs) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destinations.append(destination)
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--no-cache",
            ]
        )
    )
    output = capsys.readouterr().out

    assert result == 0
    assert {path.name for path in destinations} == {"Alpha.pt.ASS", "Beta.pt.SRT"}
    assert "Encontrados 2 arquivos" in output
    assert "- 1 ASS" in output
    assert "- 1 SRT" in output
    assert "notes.txt" in output


def test_cli_format_filter_reports_other_supported_file_as_ignored(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "only.ass").write_text("fixture", encoding="utf-8")
    (input_dir / "skip.srt").write_text("fixture", encoding="utf-8")

    class FakeTranslator:
        def __init__(self, _config, **_kwargs) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 0, "translated": 0, "cached": 0, "skipped": 1, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--format",
                "ass",
                "--no-cache",
            ]
        )
    )
    output = capsys.readouterr().out

    assert result == 0
    assert (output_dir / "only.pt.ass").exists()
    assert not (output_dir / "skip.pt.srt").exists()
    assert "skip.srt" in output


def test_cli_original_fallback_defaults_false_and_is_propagated_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    parser = translate_ass_fast.build_argument_parser()
    assert parser.parse_args([]).allow_original_fallback is False
    assert parser.parse_args(["--allow-original-fallback"]).allow_original_fallback is True

    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_text("fixture", encoding="utf-8")
    received_configs: list[dict] = []

    class FakeTranslator:
        def __init__(self, config, **_kwargs) -> None:
            received_configs.append(config)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--allow-original-fallback",
                "--no-cache",
            ]
        )
    )

    assert result == 0
    assert received_configs[0]["allow_original_fallback"] is True


def test_cli_keeps_the_existing_ollama_model_as_its_default() -> None:
    assert translate_ass_fast.build_argument_parser().parse_args([]).model == "qwen2.5:14b"


def test_cli_source_language_defaults_to_auto_and_propagates_any_manual_value(
    tmp_path: Path, monkeypatch
) -> None:
    parser = translate_ass_fast.build_argument_parser()
    assert parser.parse_args([]).source_language == "auto"
    assert parser.parse_args(["--source-language", "Klingon"]).source_language == "Klingon"
    assert "detecção automática por item" in " ".join(parser.format_help().split())

    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "episode.srt").write_text("fixture", encoding="utf-8")
    received_configs: list[dict] = []

    class FakeTranslator:
        def __init__(self, config, **_kwargs) -> None:
            received_configs.append(config)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--source-language",
                "Klingon",
                "--no-cache",
            ]
        )
    )

    assert result == 0
    assert received_configs[0]["source_language"] == "Klingon"


def test_cli_passes_optional_trace_hook_without_exposing_a_public_flag(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_text("fixture", encoding="utf-8")
    trace: list[dict] = []
    trace_hook = trace.append
    received_configs: list[dict] = []

    class FakeTranslator:
        def __init__(self, config, **_kwargs) -> None:
            received_configs.append(config)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--no-cache",
            ],
            trace_hook=trace_hook,
        )
    )

    assert result == 0
    assert received_configs[0]["trace_hook"] is trace_hook
    assert "trace_hook" not in vars(translate_ass_fast.build_argument_parser().parse_args([]))


def test_cli_uses_an_internal_isolated_cache_path_when_provided(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_text("fixture", encoding="utf-8")
    cache_path = tmp_path / "experiment" / "translation_cache.json"
    cache_path.parent.mkdir()
    received_configs: list[dict] = []

    class FakeTranslator:
        def __init__(self, config, **_kwargs) -> None:
            received_configs.append(config)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            ["--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            cache_file=cache_path,
        )
    )

    assert result == 0
    assert received_configs[0]["cache_file"] == str(cache_path)


def test_readme_and_cli_help_present_the_project_as_multilingual() -> None:
    readme = (Path(translate_ass_fast.__file__).parent / "README.md").read_text(
        encoding="utf-8"
    )
    help_text = translate_ass_fast.build_argument_parser().format_help()

    assert readme.splitlines()[0] == "# Tradutor de Legendas ASS e SRT Multilíngue para PT-BR"
    assert "detecção automática por item" in readme
    assert "--source-language" in help_text
    assert "Inglês para Português" not in readme.splitlines()[0]


def test_cli_failed_run_keeps_old_output_and_leaves_manifest_empty(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "episode.ass").write_text("fixture", encoding="utf-8")
    old_output = output_dir / "episode.pt.ass"
    old_output.write_text("resultado antigo", encoding="utf-8")
    manifest = tmp_path / "run.json"

    class FailingTranslator:
        def __init__(self, _config, **_kwargs) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, _destination: Path) -> dict[str, int]:
            raise IncompleteTranslationError(1)

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FailingTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translate_ass_fast, "shutdown_ollama_model", lambda _model: None)

    result = asyncio.run(
        translate_ass_fast.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--result-manifest",
                str(manifest),
                "--no-cache",
            ]
        )
    )
    output = capsys.readouterr().out

    assert result == 1
    assert old_output.read_text(encoding="utf-8") == "resultado antigo"
    assert load_run_outputs(manifest, output_dir) == []
    assert "Arquivos com falha: episode.ass" in output


def test_cli_preflight_and_shutdown_use_the_same_selected_backend(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_text("fixture", encoding="utf-8")
    created_backends: list[object] = []
    translator_backends: list[object] = []

    class FakeBackend:
        def __init__(self) -> None:
            self.selected_model: str | None = None
            self.closed = False
            created_backends.append(self)

        def ensure_available(self, model: str) -> None:
            self.selected_model = model

        def close(self) -> None:
            self.closed = True

    class FakeTranslator:
        def __init__(self, _config, *, backend) -> None:
            translator_backends.append(backend)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "OllamaBackend", FakeBackend)
    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)

    result = asyncio.run(
        translate_ass_fast.main(
            ["--input-dir", str(input_dir), "--output-dir", str(output_dir), "--no-cache"]
        )
    )

    backend = created_backends[0]
    assert result == 0
    assert created_backends == [backend]
    assert translator_backends == [backend]
    assert backend.selected_model == "qwen2.5:14b"
    assert backend.closed is True


def test_legacy_shutdown_does_not_pass_an_unselected_model_to_an_adapter(monkeypatch) -> None:
    closed_backends: list[object] = []

    class FakeBackend:
        def close(self) -> None:
            closed_backends.append(self)

    monkeypatch.setattr(translate_ass_fast, "OllamaBackend", FakeBackend)

    translate_ass_fast.shutdown_ollama_model("qwen2.5:14b")

    assert len(closed_backends) == 1
