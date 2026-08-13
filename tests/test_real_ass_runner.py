from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from pathlib import Path

import tools.run_ass_regression as runner
from subtitle_formats import ASSFormatHandler, preserve_extension_output_path
from translation_engine import CONFIG, ModelUnavailableError


FIXTURES = Path(__file__).parent / "fixtures"


def _arguments(tmp_path: Path, input_dir: Path, *extra: str) -> argparse.Namespace:
    return runner.build_argument_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "test_results"),
            "--run-id",
            "20260813_120000",
            "--experiment",
            "baseline",
            *extra,
        ]
    )


def test_runner_defaults_match_safe_real_baseline() -> None:
    args = runner.build_argument_parser().parse_args([])

    assert args.input_dir == Path("entrada")
    assert args.output_dir == Path("test_results")
    assert args.model == CONFIG["model"]
    assert args.batch_size == CONFIG["batch_size"]
    assert args.no_cache is True
    assert args.turbo is False
    assert args.strict_semantic is False
    assert args.experiment == "baseline"


def test_runner_uses_production_cli_and_writes_isolated_reproducible_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    source = input_dir / "Episode.ASS"
    source.write_bytes((FIXTURES / "regression_ass_cases.ass").read_bytes())
    source_before = source.read_bytes()
    received_argv: list[str] = []

    async def fake_translation_main(argv, *, trace_hook=None, cache_file=None) -> int:
        received_argv.extend(argv)
        input_path = Path(argv[argv.index("--input-dir") + 1])
        output_path = Path(argv[argv.index("--output-dir") + 1])
        handler = ASSFormatHandler()
        for item in input_path.iterdir():
            if item.suffix.casefold() != ".ass":
                continue
            subs = handler.prepare_document(handler.load(item))
            subs.events[0].text = subs.events[0].text.replace("Hello.", "Olá.").replace(
                "Hi.", "Oi."
            )
            destination = preserve_extension_output_path(item, output_path)
            handler.save(subs, destination)
        if trace_hook is not None:
            trace_hook(
                {
                    "event": "batch_attempt_started",
                    "item_ids": [1],
                    "attempt": 1,
                    "prompt": "<<<ITEM_0001>>>fixture<<<END_ITEM_0001>>>",
                }
            )
            trace_hook(
                {
                    "event": "model_response",
                    "item_ids": [1],
                    "response": "<<<ITEM_0001>>>Olá<<<END_ITEM_0001>>>",
                }
            )
            trace_hook({"event": "item_accepted", "item_id": 1, "mode": "batch"})
        return 0

    monkeypatch.setattr(runner, "translation_cli_main", fake_translation_main)
    monkeypatch.setattr(runner, "ensure_ollama_model_available", lambda _model: None)

    exit_code = asyncio.run(runner.run_experiment(_arguments(tmp_path, input_dir)))

    experiment = tmp_path / "test_results" / "20260813_120000" / "baseline"
    assert exit_code == 0
    assert source.read_bytes() == source_before
    assert (experiment / "sources" / "Episode.ASS").read_bytes() == source_before
    assert (experiment / "outputs" / "Episode.pt.ASS").exists()
    assert (experiment / "report.json").exists()
    assert (experiment / "report.md").exists()
    assert (experiment / "context.json").exists()
    assert (experiment / "sources.json").exists()
    assert (experiment / "run.log").exists()
    assert (experiment / "trace.jsonl.gz").exists()
    assert "--format" in received_argv and "ass" in received_argv
    assert "--no-cache" in received_argv
    assert "--allow-original-fallback" not in received_argv

    context = json.loads((experiment / "context.json").read_text(encoding="utf-8"))
    assert context["model"] == CONFIG["model"]
    assert context["batch_size"] == CONFIG["batch_size"]
    assert context["cache_enabled"] is False
    assert context["turbo"] is False
    assert context["source_integrity_preserved"] is True
    assert context["started_at"]
    assert context["completed_at"]
    assert context["trace_metrics"]["model_responses"] == 1
    with gzip.open(experiment / "trace.jsonl.gz", "rt", encoding="utf-8") as trace_file:
        assert [json.loads(line)["event"] for line in trace_file] == [
            "batch_attempt_started",
            "model_response",
            "item_accepted",
        ]


def test_with_cache_uses_only_an_experiment_local_cache(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_bytes(
        (FIXTURES / "regression_ass_cases.ass").read_bytes()
    )
    project_cache = tmp_path / "translation_cache.json"
    project_cache.write_text("user cache", encoding="utf-8")
    received_cache: list[Path | None] = []

    async def fake_translation_main(argv, *, trace_hook=None, cache_file=None) -> int:
        received_cache.append(Path(cache_file) if cache_file is not None else None)
        Path(cache_file).write_text("test cache", encoding="utf-8")
        return 1

    monkeypatch.setattr(runner, "translation_cli_main", fake_translation_main)
    monkeypatch.setattr(runner, "ensure_ollama_model_available", lambda _model: None)

    result = asyncio.run(
        runner.run_experiment(_arguments(tmp_path, input_dir, "--with-cache"))
    )

    expected = (
        tmp_path
        / "test_results"
        / "20260813_120000"
        / "baseline"
        / "cache"
        / "translation_cache.json"
    )
    assert result == runner.EXIT_CRITICAL_FAILURE
    assert received_cache == [expected]
    assert expected.read_text(encoding="utf-8") == "test cache"
    assert project_cache.read_text(encoding="utf-8") == "user cache"
    context = json.loads(
        (expected.parents[1] / "context.json").read_text(encoding="utf-8")
    )
    assert context["cache_file"] == str(expected.resolve())


def test_runner_never_overwrites_an_existing_experiment(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_bytes(
        (FIXTURES / "regression_ass_cases.ass").read_bytes()
    )
    monkeypatch.setattr(runner, "ensure_ollama_model_available", lambda _model: None)

    async def no_op_translation(_argv, *, trace_hook=None, cache_file=None) -> int:
        return 0

    monkeypatch.setattr(runner, "translation_cli_main", no_op_translation)
    args = _arguments(tmp_path, input_dir)

    assert asyncio.run(runner.run_experiment(args)) == 1
    marker = tmp_path / "test_results" / "20260813_120000" / "baseline" / "marker.txt"
    marker.write_text("evidence", encoding="utf-8")
    assert asyncio.run(runner.run_experiment(args)) == runner.EXIT_CONFIGURATION_ERROR
    assert marker.read_text(encoding="utf-8") == "evidence"


def test_runner_reports_unavailable_ollama_as_not_executed(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_bytes(
        (FIXTURES / "regression_ass_cases.ass").read_bytes()
    )

    def unavailable(_model: str) -> None:
        raise ModelUnavailableError("model unavailable")

    monkeypatch.setattr(runner, "ensure_ollama_model_available", unavailable)

    exit_code = asyncio.run(runner.run_experiment(_arguments(tmp_path, input_dir)))

    experiment = tmp_path / "test_results" / "20260813_120000" / "baseline"
    assert exit_code == runner.EXIT_OLLAMA_UNAVAILABLE
    failure = json.loads((experiment / "preflight_error.json").read_text(encoding="utf-8"))
    assert failure["status"] == "NOT_EXECUTED"
    assert "model unavailable" in failure["reason"]


def test_runner_reports_late_model_unavailability_as_not_executed(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    (input_dir / "episode.ass").write_bytes(
        (FIXTURES / "regression_ass_cases.ass").read_bytes()
    )

    async def late_unavailable(_argv, *, trace_hook=None, cache_file=None) -> int:
        return runner.EXIT_OLLAMA_UNAVAILABLE

    monkeypatch.setattr(runner, "translation_cli_main", late_unavailable)
    monkeypatch.setattr(runner, "ensure_ollama_model_available", lambda _model: None)

    exit_code = asyncio.run(runner.run_experiment(_arguments(tmp_path, input_dir)))

    experiment = tmp_path / "test_results" / "20260813_120000" / "baseline"
    context = json.loads((experiment / "context.json").read_text(encoding="utf-8"))
    failure = json.loads((experiment / "preflight_error.json").read_text(encoding="utf-8"))
    assert exit_code == runner.EXIT_OLLAMA_UNAVAILABLE
    assert context["status"] == "NOT_EXECUTED"
    assert failure["status"] == "NOT_EXECUTED"


def test_source_removed_during_run_is_reported_instead_of_aborting(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    source = input_dir / "episode.ass"
    source.write_bytes((FIXTURES / "regression_ass_cases.ass").read_bytes())

    async def remove_source(argv, *, trace_hook=None, cache_file=None) -> int:
        source.unlink()
        return 1

    monkeypatch.setattr(runner, "translation_cli_main", remove_source)
    monkeypatch.setattr(runner, "ensure_ollama_model_available", lambda _model: None)

    exit_code = asyncio.run(runner.run_experiment(_arguments(tmp_path, input_dir)))

    experiment = tmp_path / "test_results" / "20260813_120000" / "baseline"
    report = json.loads((experiment / "report.json").read_text(encoding="utf-8"))
    context = json.loads((experiment / "context.json").read_text(encoding="utf-8"))
    categories = {
        diagnostic["category"]
        for file_result in report["files"]
        for diagnostic in file_result.get("diagnostics", [])
    }
    assert exit_code == runner.EXIT_CRITICAL_FAILURE
    assert "SOURCE_MUTATED" in categories
    assert context["source_integrity_preserved"] is False
    assert context["source_integrity_errors"]
