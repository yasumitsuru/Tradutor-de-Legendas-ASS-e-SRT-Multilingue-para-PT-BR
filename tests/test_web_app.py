from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import web_app
from run_manifest import initialize_run_manifest, record_run_output


@pytest.fixture
def web_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(web_app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(web_app, "CACHE_PATH", tmp_path / "translation_cache.json")
    monkeypatch.setattr(web_app, "WEB_CONFIG_PATH", tmp_path / "web_config.json")
    web_app.app.config.update(TESTING=True, SECRET_KEY="test-only-secret", SESSION_COOKIE_SECURE=False)
    with web_app._state_lock:
        web_app._state.update(
            running=False,
            started_at=None,
            finished_at=None,
            return_code=None,
            command=None,
            logs=[],
        )
    client = web_app.app.test_client()
    with client.session_transaction() as session:
        session[web_app._CSRF_FIELD] = "csrf-for-tests"
    return client


def test_upload_list_and_zip_support_ass_and_srt_case_insensitively(
    web_client, tmp_path: Path
) -> None:
    response = web_client.post(
        "/upload",
        data={
            "csrf_token": "csrf-for-tests",
            "input_dir": "./entrada",
            "subtitle_files": [
                (io.BytesIO(b"srt"), "Episode One.SRT"),
                (io.BytesIO(b"ass"), "Episode Two.ASS"),
                (io.BytesIO(b"ignored"), "notes.txt"),
            ],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["saved"] == ["Episode One.SRT", "Episode Two.ASS"]
    assert response.get_json()["skipped"] == ["notes.txt"]

    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    (output_dir / "Episode One.pt.SRT").write_text("translated srt", encoding="utf-8")
    (output_dir / "Episode Two.pt.ASS").write_text("translated ass", encoding="utf-8")
    (output_dir / "debug.log").write_text("ignored", encoding="utf-8")
    initialize_run_manifest(web_app._run_manifest_path())
    record_run_output(
        web_app._run_manifest_path(), output_dir, output_dir / "Episode One.pt.SRT"
    )
    record_run_output(
        web_app._run_manifest_path(), output_dir, output_dir / "Episode Two.pt.ASS"
    )

    listing = web_client.get(
        "/files", query_string={"input_dir": "./entrada", "output_dir": "./saida"}
    )
    payload = listing.get_json()
    assert payload["input_files"] == ["Episode One.SRT", "Episode Two.ASS"]
    assert payload["output_files"] == ["Episode One.pt.SRT", "Episode Two.pt.ASS"]
    assert payload["input_counts"] == {"ass": 1, "srt": 1}
    assert payload["output_counts"] == {"ass": 1, "srt": 1}

    archive_response = web_client.get(
        "/download-output", query_string={"output_dir": "./saida"}
    )
    assert archive_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive_response.data)) as archive:
        assert set(archive.namelist()) == {"Episode One.pt.SRT", "Episode Two.pt.ASS"}


def test_upload_rejects_request_without_supported_subtitles(web_client) -> None:
    response = web_client.post(
        "/upload",
        data={
            "csrf_token": "csrf-for-tests",
            "input_dir": "./entrada",
            "subtitle_files": [(io.BytesIO(b"text"), "notes.txt")],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "ASS ou SRT" in response.get_json()["error"]


def test_start_rejects_empty_input_before_contacting_ollama(
    web_client, monkeypatch
) -> None:
    def unexpected_model_check(*_args, **_kwargs):
        raise AssertionError("Ollama must not be contacted without subtitle files")

    monkeypatch.setattr(web_app, "_ensure_ollama_model_available", unexpected_model_check)
    response = web_client.post(
        "/start",
        data={
            "csrf_token": "csrf-for-tests",
            "input_dir": "./entrada",
            "output_dir": "./saida",
            "model": "qwen2.5:14b",
            "batch_size": "10",
            "timeout": "60",
            "ollama_mode": "local",
            "ollama_endpoint": "",
        },
    )

    assert response.status_code == 400
    assert "ASS ou SRT" in response.get_json()["error"]


def test_web_progress_understands_generic_subtitle_logs() -> None:
    logs = [
        "📂 Encontrados 2 arquivos:",
        "✅ Arquivo ASS concluído: first.ass",
    ]

    percent, label = web_app._compute_progress(logs, running=True, return_code=None)

    assert percent == 50
    assert label == "50% (1/2 arquivos)"


def test_web_fallback_option_is_visible_unchecked_and_propagated(web_client) -> None:
    page = web_client.get("/")
    html = page.get_data(as_text=True)

    assert 'name="allow_original_fallback"' in html
    assert "Permitir manter texto original quando a tradução falhar" in html
    assert "pode gerar legendas misturando PT-BR com o idioma original" in html
    checkbox = html.split('name="allow_original_fallback"', 1)[0].rsplit("<input", 1)[1]
    assert "checked" not in checkbox

    base_form = {
        "input_dir": "entrada",
        "output_dir": "saida",
        "model": "qwen2.5:14b",
        "batch_size": "10",
        "timeout": "60",
    }
    disabled = web_app._build_command({**base_form, "allow_original_fallback": "off"})
    enabled = web_app._build_command({**base_form, "allow_original_fallback": "on"})

    assert "--allow-original-fallback" not in disabled
    assert "--allow-original-fallback" in enabled
    assert "--result-manifest" in disabled


def test_web_copy_presents_automatic_multilingual_translation(web_client) -> None:
    html = web_client.get("/").get_data(as_text=True)

    assert "Multilíngue" in html
    assert "PT-BR" in html
    assert "do Inglês para Português" not in html
    assert "--source-language" not in web_app._build_command(
        {
            "input_dir": "entrada",
            "output_dir": "saida",
            "model": "qwen2.5:14b",
            "batch_size": "10",
            "timeout": "60",
        }
    )


def test_web_does_not_present_or_download_old_output_without_current_manifest(
    web_client, tmp_path: Path
) -> None:
    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    (output_dir / "episode.pt.ass").write_text("resultado antigo", encoding="utf-8")

    listing = web_client.get(
        "/files", query_string={"input_dir": "./entrada", "output_dir": "./saida"}
    )
    download = web_client.get(
        "/download-output", query_string={"output_dir": "./saida"}
    )

    assert listing.get_json()["output_files"] == []
    assert download.status_code == 404
    assert "execução atual" in download.get_json()["error"]
