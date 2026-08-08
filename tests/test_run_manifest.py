from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_manifest import initialize_run_manifest, load_run_outputs, record_run_output


def test_run_manifest_contains_only_recorded_existing_subtitle_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    produced = output_dir / "episode.pt.ass"
    produced.write_text("translated", encoding="utf-8")
    unrelated = output_dir / "old.pt.srt"
    unrelated.write_text("old", encoding="utf-8")
    manifest = tmp_path / "run.json"

    initialize_run_manifest(manifest)
    record_run_output(manifest, output_dir, produced)

    assert load_run_outputs(manifest, output_dir) == [produced.resolve()]
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "outputs": ["episode.pt.ass"],
    }


def test_run_manifest_rejects_output_outside_configured_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    outside = tmp_path / "outside.ass"
    outside.write_text("translated", encoding="utf-8")
    manifest = tmp_path / "run.json"
    initialize_run_manifest(manifest)

    with pytest.raises(ValueError, match="diretório de saída"):
        record_run_output(manifest, output_dir, outside)


def test_missing_invalid_or_empty_manifest_never_discovers_old_directory_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    (output_dir / "old.pt.ass").write_text("old", encoding="utf-8")
    manifest = tmp_path / "run.json"

    assert load_run_outputs(manifest, output_dir) == []
    manifest.write_text("not json", encoding="utf-8")
    assert load_run_outputs(manifest, output_dir) == []
    initialize_run_manifest(manifest)
    assert load_run_outputs(manifest, output_dir) == []
