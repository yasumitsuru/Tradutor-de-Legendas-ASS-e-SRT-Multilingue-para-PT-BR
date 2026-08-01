from __future__ import annotations

from pathlib import Path

import app_flet
import app_gui


def test_desktop_progress_tracks_completed_ass_and_srt_files() -> None:
    logs = [
        "📂 Encontrados 4 arquivos:",
        "✅ Arquivo ASS concluído: one.ass",
        "✅ Arquivo SRT concluído: two.srt",
    ]

    flet_percent, flet_label = app_flet._compute_progress(logs, True, None)
    pyside_percent, pyside_label = app_gui._compute_progress(logs, True, None)

    assert flet_percent == 50
    assert "2/4 arquivos" in flet_label
    assert pyside_percent == 50
    assert "2/4 arquivos" in pyside_label


def test_desktop_commands_delegate_format_discovery_to_shared_backend(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    monkeypatch.setattr(app_flet, "ENTRY_DIR", input_dir)
    monkeypatch.setattr(app_flet, "OUTPUT_DIR", output_dir)
    form = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": "qwen2.5:14b",
        "batch_size": "10",
        "timeout": "60",
        "turbo": "off",
        "clear_cache": "off",
        "no_cache": "off",
    }

    flet_command = app_flet._build_command(form)
    pyside_command = app_gui._build_command(form)

    assert flet_command[flet_command.index("--input-dir") + 1] == str(input_dir)
    assert pyside_command[pyside_command.index("--input-dir") + 1] == str(input_dir)
    assert "--format" not in flet_command
    assert "--format" not in pyside_command


def test_desktop_sources_expose_both_file_picker_extensions() -> None:
    flet_source = Path(app_flet.__file__).read_text(encoding="utf-8")
    pyside_source = Path(app_gui.__file__).read_text(encoding="utf-8")

    assert 'allowed_extensions=["ass", "srt"]' in flet_source
    assert "*.ass *.ASS *.srt *.SRT" in pyside_source
    assert 'iter_subtitle_files(OUTPUT_DIR)' in flet_source
    assert 'iter_subtitle_files(OUTPUT_DIR)' in pyside_source
