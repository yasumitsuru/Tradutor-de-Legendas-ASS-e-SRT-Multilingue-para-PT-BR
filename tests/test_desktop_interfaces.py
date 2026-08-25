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
        "allow_original_fallback": "off",
    }

    flet_command = app_flet._build_command(form)
    pyside_command = app_gui._build_command(form)

    assert flet_command[flet_command.index("--input-dir") + 1] == str(input_dir)
    assert pyside_command[pyside_command.index("--input-dir") + 1] == str(input_dir)
    assert "--format" not in flet_command
    assert "--format" not in pyside_command
    assert "--allow-original-fallback" not in flet_command
    assert "--allow-original-fallback" not in pyside_command
    assert "--result-manifest" in flet_command
    assert "--result-manifest" in pyside_command

    flet_enabled = app_flet._build_command({**form, "allow_original_fallback": True})
    pyside_enabled = app_gui._build_command({**form, "allow_original_fallback": "on"})

    assert "--allow-original-fallback" in flet_enabled
    assert "--allow-original-fallback" in pyside_enabled


def test_desktop_interfaces_normalize_shared_backend_settings_and_keep_ollama_defaults() -> None:
    form = {
        "backend": "ollama",
        "api_base": "",
        "model": "qwen2.5:14b",
        "batch_size": "10",
        "timeout": "60",
    }

    for interface in (app_flet, app_gui):
        ok, normalized, error = interface._normalize_config(form)

        assert ok, error
        assert normalized["backend"] == "ollama"
        assert normalized["api_base"] == ""
        assert normalized["model"] == "qwen2.5:14b"


def test_desktop_interfaces_propagate_llama_swap_settings_without_changing_run_options(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    output_dir = tmp_path / "saida"
    form = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "backend": "llama-swap",
        "api_base": "http://127.0.0.1:9292",
        "model": "Qwen3.6-28B-REAP20-A3B-Q4_K_M",
        "batch_size": "10",
        "timeout": "60",
        "turbo": "on",
        "clear_cache": "on",
        "no_cache": "on",
        "allow_original_fallback": "on",
    }
    monkeypatch.setattr(app_flet, "ENTRY_DIR", input_dir)
    monkeypatch.setattr(app_flet, "OUTPUT_DIR", output_dir)

    for interface in (app_flet, app_gui):
        ok, normalized, error = interface._normalize_config(form)
        assert ok, error
        command = interface._build_command(
            {
                **form,
                **normalized,
                "allow_original_fallback": (
                    True if interface is app_flet else "on"
                ),
            }
        )

        assert command[command.index("--backend") + 1] == "llama-swap"
        assert command[command.index("--api-base") + 1] == "http://127.0.0.1:9292/v1"
        assert command[command.index("--model") + 1] == "Qwen3.6-28B-REAP20-A3B-Q4_K_M"
        assert command[command.index("--batch-size") + 1] == "10"
        assert command[command.index("--timeout") + 1] == "60"
        assert "--turbo" in command
        assert "--clear-cache" in command
        assert "--no-cache" in command
        assert "--allow-original-fallback" in command
        assert "--result-manifest" in command


def test_desktop_interfaces_use_shared_backend_configuration_without_ollama_shutdown() -> None:
    for interface in (app_flet, app_gui):
        source = Path(interface.__file__).read_text(encoding="utf-8")

        assert "load_backend_settings" in source
        assert "save_backend_settings" in source
        assert "normalize_backend_endpoint" in source
        assert "shutdown_ollama_model" not in source


def test_desktop_sources_expose_both_file_picker_extensions() -> None:
    flet_source = Path(app_flet.__file__).read_text(encoding="utf-8")
    pyside_source = Path(app_gui.__file__).read_text(encoding="utf-8")

    assert 'allowed_extensions=["ass", "srt"]' in flet_source
    assert "*.ass *.ASS *.srt *.SRT" in pyside_source
    assert 'load_run_outputs(_run_manifest_path(), OUTPUT_DIR)' in flet_source
    assert 'load_run_outputs(_run_manifest_path(), OUTPUT_DIR)' in pyside_source
    assert "Permitir manter texto original quando a tradução falhar" in flet_source
    assert "Permitir manter texto original quando a tradução falhar" in pyside_source
    assert "pode gerar legendas misturando PT-BR com o idioma original" in flet_source
    assert "pode gerar legendas misturando PT-BR com o idioma original" in pyside_source
    assert "Multilíngue" in flet_source
    assert "Multilíngue" in pyside_source
    assert "Inglês → Português" not in flet_source
    assert "Ingles para Portugues" not in pyside_source
    assert "--source-language" not in app_flet._build_command(
        {
            "model": "qwen2.5:14b",
            "batch_size": "10",
            "timeout": "60",
        }
    )
    assert "--source-language" not in app_gui._build_command(
        {
            "input_dir": "entrada",
            "output_dir": "saida",
            "model": "qwen2.5:14b",
            "batch_size": "10",
            "timeout": "60",
        }
    )
