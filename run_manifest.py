"""Atomic manifest of subtitle outputs produced by the current translation run."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from subtitle_formats import is_supported_subtitle


RUN_MANIFEST_SCHEMA_VERSION = 1


def _atomic_write_manifest(path: Path, outputs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "outputs": outputs,
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest_outputs(path: Path) -> list[str]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        return []
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return []
    return [value for value in outputs if isinstance(value, str) and value]


def initialize_run_manifest(path: str | Path) -> None:
    """Start a run with no outputs, replacing any stale manifest atomically."""

    _atomic_write_manifest(Path(path), [])


def _validated_relative_output(output_root: Path, output: Path) -> tuple[Path, str]:
    root = output_root.resolve()
    resolved = output.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Output fora do diretório de saída configurado.") from exc
    if not resolved.is_file() or not is_supported_subtitle(resolved):
        raise ValueError("Output registrado deve ser um arquivo ASS ou SRT existente.")
    return resolved, relative.as_posix()


def record_run_output(
    path: str | Path,
    output_root: str | Path,
    output: str | Path,
) -> None:
    """Add one validated output to the current run manifest."""

    manifest_path = Path(path)
    _resolved, relative = _validated_relative_output(Path(output_root), Path(output))
    outputs = _read_manifest_outputs(manifest_path)
    if relative not in outputs:
        outputs.append(relative)
    _atomic_write_manifest(manifest_path, outputs)


def load_run_outputs(path: str | Path, output_root: str | Path) -> list[Path]:
    """Return only existing safe subtitle files explicitly recorded for the run."""

    manifest_path = Path(path)
    root = Path(output_root).resolve()
    loaded: list[Path] = []
    seen: set[Path] = set()
    for relative in _read_manifest_outputs(manifest_path):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate in seen or not candidate.is_file() or not is_supported_subtitle(candidate):
            continue
        seen.add(candidate)
        loaded.append(candidate)
    return loaded
