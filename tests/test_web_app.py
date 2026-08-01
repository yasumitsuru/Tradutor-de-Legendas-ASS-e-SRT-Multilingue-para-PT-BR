from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import web_app


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
