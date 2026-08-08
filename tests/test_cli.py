from __future__ import annotations

import asyncio
from pathlib import Path

import translate_ass_fast
from run_manifest import load_run_outputs
from translation_engine import IncompleteTranslationError


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
        def __init__(self, _config) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destinations.append(destination)
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda _model: None)
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
        def __init__(self, _config) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 0, "translated": 0, "cached": 0, "skipped": 1, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda _model: None)
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
        def __init__(self, config) -> None:
            received_configs.append(config)
            self.cache = {}

        async def translate_file(self, _source: Path, destination: Path) -> dict[str, int]:
            destination.write_text("translated", encoding="utf-8")
            return {"total": 1, "translated": 1, "cached": 0, "skipped": 0, "failed": 0}

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FakeTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda _model: None)
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
        def __init__(self, _config) -> None:
            self.cache = {}

        async def translate_file(self, _source: Path, _destination: Path) -> dict[str, int]:
            raise IncompleteTranslationError(1)

        def _save_cache(self) -> None:
            return None

    monkeypatch.setattr(translate_ass_fast, "FixedASSTranslator", FailingTranslator)
    monkeypatch.setattr(translate_ass_fast, "ensure_ollama_model_available", lambda _model: None)
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
