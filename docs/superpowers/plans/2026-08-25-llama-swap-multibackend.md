# Llama-swap Multi-backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar suporte multi-backend com `OllamaBackend` preservado, `LlamaSwapBackend` HTTP stdlib, perfis de modelo e isolamento de cache/provenance sem alterar o protocolo estrutural ASS/SRT.

**Architecture:** O engine consumirá `InferenceBackend` através de `GenerationRequest`/`GenerationResult`. `OllamaBackend` encapsulará a implementação existente; `LlamaSwapBackend` usará a API OpenAI-compatible com `api_base` e `service_root` separados; `ModelProfile` ficará separado dos transportes. Configuração compartilhada atenderá CLI, Flask, Flet, PySide6 e regression runner.

**Tech Stack:** Python 3.11 no CI, Python asyncio, `urllib.request`, `json`, Ollama Python client, pytest, Flask, Flet, PySide6, PyInstaller, ASS/SRT via `pysubs2`.

**Spec:** `docs/superpowers/specs/2026-08-25-llama-swap-multibackend-design.md`

## Global Constraints

- Default obrigatório: `backend=ollama`, `model=qwen2.5:14b`.
- API llama-swap: `http://127.0.0.1:9292/v1`; service root separado de `api_base`.
- Transporte llama-swap usa somente biblioteca padrão Python; OpenAI SDK não será adicionado.
- Transporte executa zero retries ocultos; retries continuam no pipeline existente.
- `LlamaSwapBackend.close()` não chama load, swap ou unload; `unload_on_exit=false`.
- Qwen3.6 permanece `EXPERIMENTAL`, `UNVALIDATED`, `NOT_PRODUCTION_APPROVED` e nunca é default.
- Nenhuma alteração pode relaxar ITEM, placeholders, tags, recoveries, fallback, publicação atômica ou regressão estrutural.
- Nenhuma chamada real de geração é permitida antes de infraestrutura, CLI, regression runner, suíte completa e review estarem verdes.
- Alterações preexistentes `entrada/.gitkeep`, `saida/.gitkeep` e `.understory-project-id` nunca entram no stage.
- Nunca usar `git add .`; todo stage é explícito.
- Subagents trabalham serialmente, não criam subagents, não fazem commit, push, merge ou escrita no Understory.
- Cada task só fica GREEN após RED comprovado, testes focados, testes relacionados, suíte completa, compileall, diff checks, review, Understory e stage auditado.
- Branch única: `feat/llama-swap-backend-qwen36`; `main` não recebe commits da feature.

## File Structure

| Arquivo | Responsabilidade planejada |
|---|---|
| `inference_backend.py` | Tipos de request/result, `InferenceBackend` e erros normalizados. |
| `ollama_backend.py` | Adapter do cliente Ollama e lifecycle legado. |
| `llama_swap_backend.py` | Transporte HTTP stdlib, health, models e chat completions. |
| `backend_config.py` | Configuração canônica, normalização de URLs e migração do `web_config.json`. |
| `backend_factory.py` | Construção do backend selecionado e injeção de fakes. |
| `model_profiles.py` | `ModelProfile`, estados de compatibilidade e registro Qwen3.6. |
| `translation_engine.py` | Consumo do contrato comum e cache schema 5; invariantes permanecem. |
| `translate_ass_fast.py` | Flags CLI, preflight, backend lifecycle e compatibilidade de aliases. |
| `tools/run_ass_regression.py` | Contexto e preflight backend-aware, mantendo artefatos isolados. |
| `web_app.py` | Integração Flask com configuração compartilhada. |
| `app_flet.py` | Integração Flet com configuração compartilhada. |
| `app_gui.py` | Integração PySide6 com configuração compartilhada. |
| `tests/test_inference_backend.py` | Contratos e erros. |
| `tests/test_ollama_backend.py` | Compatibilidade do adapter Ollama. |
| `tests/test_backend_config.py` | Normalização, migração e factory. |
| `tests/test_llama_swap_backend.py` | HTTP fake, discovery, geração e erros. |
| `tests/test_model_profiles.py` | Registro e estados do Qwen3.6. |
| `tests/test_translation_engine.py` | Integração do engine e isolamento do cache. |
| `tests/test_cli.py` | Flags e default Ollama. |
| `tests/test_real_ass_runner.py` | Contexto e preflight do runner. |
| `tests/test_web_app.py` | Flask e migração de configuração. |
| `tests/test_desktop_interfaces.py` | Flet/PySide command propagation. |
| `tools/run_model_compatibility_gate.py` | Harness posterior sem execução automática. |

### Task 1: Backend contract and normalized result types

**Files:**
- Create: `inference_backend.py`
- Create: `tests/test_inference_backend.py`

**Interfaces:**
- Produces `GenerationRequest`, `GenerationResult`, `ModelInfo` and `InferenceBackend`.
- Produces `BackendError`, `BackendConfigurationError`, `BackendUnavailableError`, `BackendModelNotFoundError`, `BackendTransportError`, `BackendTimeoutError` and `BackendResponseError`.
- `GenerationResult.metadata` is immutable-by-convention and optional fields remain `None` when providers do not report usage.

- [ ] **Step 1: Write the failing tests**

Add tests asserting that `GenerationRequest` stores `model`, `system_prompt`, `prompt`, `temperature`, `top_p`, `max_tokens` and `timeout`; `GenerationResult` stores normalized text/backend/model; each error subclass derives from `BackendError`; and a fake object satisfying the protocol can be passed to a type-checking helper without provider-specific fields.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_inference_backend.py -q`

Expected: collection fails because `inference_backend.py` and the named types do not exist.

- [ ] **Step 3: Implement the minimal contract**

Use frozen dataclasses and `typing.Protocol`. Keep the protocol limited to:

```python
def ensure_available(self, model: str) -> None: ...
def generate(self, request: GenerationRequest) -> GenerationResult: ...
def close(self) -> None: ...
```

Do not import Ollama, Flask, `pysubs2` or HTTP libraries in this module.

- [ ] **Step 4: Run focused GREEN and related checks**

Run: `python -m pytest tests/test_inference_backend.py -q`

Expected: all new contract tests pass.

- [ ] **Step 5: Review and commit**

Review the diff for provider leakage, run `python -m compileall -q .` and `git diff --check`, update Understory with the task and test result, then stage only `inference_backend.py` and `tests/test_inference_backend.py`.

Commit: `refactor: introduce inference backend contract`

### Task 2: Extract the Ollama backend

**Files:**
- Create: `ollama_backend.py`
- Create: `tests/test_ollama_backend.py`
- Modify: `translation_engine.py`
- Modify: `translate_ass_fast.py`
- Modify: `tests/test_translation_engine.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes `inference_backend.py` types.
- Produces `OllamaBackend(client=None, host=None)` with `backend_id == "ollama"`, `ensure_available()`, `generate()` and idempotent `close()`.
- Keeps compatibility functions `is_model_not_found_error()` and `ensure_ollama_model_available()` available from their existing import locations.

- [ ] **Step 1: Write RED adapter tests**

Add fake Ollama client tests for successful `show`, model-not-found classification, `generate` option mapping (`temperature`, `num_predict`, `top_p`), normalized `GenerationResult`, malformed provider text and an idempotent close path. Add a test that the legacy validation aliases still call the adapter.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_ollama_backend.py -q`

Expected: import failure for `OllamaBackend`.

- [ ] **Step 3: Implement the adapter and migrate engine injection**

Move direct Ollama operations into `ollama_backend.py`. Change `FixedASSTranslator` to accept `backend=None` and construct an Ollama adapter only when no backend is supplied. Preserve `ollama_client` injection by wrapping the supplied fake client. Move shutdown behavior behind `OllamaBackend.close()` while leaving compatibility wrappers in the CLI.

- [ ] **Step 4: Run focused and related GREEN**

Run: `python -m pytest tests/test_ollama_backend.py tests/test_translation_engine.py tests/test_cli.py -q`

Expected: new adapter tests and all existing engine/CLI tests pass without changing ITEM assertions.

- [ ] **Step 5: Review and commit**

Check that `translation_engine.py` no longer owns provider calls, that default config is unchanged, and that no shutdown path can be called for an unselected backend. Update Understory and stage only the listed adapter/engine/CLI/test files.

Commit: `refactor: isolate Ollama backend`

### Task 3: Canonical backend configuration and factory

**Files:**
- Create: `backend_config.py`
- Create: `backend_factory.py`
- Create: `tests/test_backend_config.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces `BackendSettings(backend, api_base, model)` plus existing batch/cache/timeout fields as needed by callers.
- Produces `normalize_backend_endpoint(backend, raw) -> NormalizedEndpoint` with `api_base` and `service_root`.
- Produces `load_backend_settings(path)`, `save_backend_settings(path, settings)` and `create_backend(settings, *, ollama_client=None, http_opener=None)`.

- [ ] **Step 1: Write RED normalization and migration tests**

Cover `http://127.0.0.1:9292`, `http://127.0.0.1:9292/`, and `http://127.0.0.1:9292/v1` normalizing to one llama-swap API base and service root; reject `/v2`, credentials, query, fragment and non-HTTP schemes; preserve old Ollama local/remote behavior; migrate legacy `ollama_mode`/`ollama_endpoint`; and keep `qwen2.5:14b`/Ollama as defaults.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_backend_config.py -q`

Expected: import failure for the new settings and normalization functions.

- [ ] **Step 3: Implement configuration and factory**

Keep endpoint normalization provider-specific. The factory selects `OllamaBackend` or `LlamaSwapBackend` by exact backend ID and rejects unknown IDs before any network call. Do not import Flask, Flet or PySide6 here.

- [ ] **Step 4: Run focused and related GREEN**

Run: `python -m pytest tests/test_backend_config.py tests/test_cli.py -q`

Expected: migration, default and factory tests pass; no real network access occurs.

- [ ] **Step 5: Review and commit**

Verify no legacy configuration key is silently discarded and no endpoint creates `/v1/v1` or `/v1/health`. Update Understory and stage only the two new modules and their tests.

Commit: `feat: add multi-backend configuration factory`

### Task 4: Integrate the engine and isolate the translation cache

**Files:**
- Modify: `translation_engine.py`
- Modify: `tests/test_translation_engine.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes `InferenceBackend`, `GenerationRequest`, `GenerationResult` and `BackendSettings`.
- `FixedASSTranslator(config, ollama_client=None, backend=None)` remains backward-compatible.
- `_generate()` becomes a backend-independent conversion from config to `GenerationRequest` and from `GenerationResult` to response text.

- [ ] **Step 1: Write RED engine adapter tests**

Add a fake backend that records `GenerationRequest` and returns `GenerationResult`. Assert that the system prompt, user prompt, temperature, top-p, max tokens, timeout and model reach the backend; the parser still receives only result text; backend metadata does not affect ITEM validation; and an Ollama fake-client construction remains compatible.

Add cache tests asserting schema 5 and distinct keys for `backend="ollama"` versus `backend="llama-swap"` with identical model/prompt/language/format/structure.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_translation_engine.py -q`

Expected: new backend request and schema-isolation assertions fail while existing tests remain the control set.

- [ ] **Step 3: Implement minimal integration**

Replace the direct `.generate()` call with `backend.generate(GenerationRequest(...))`. Add backend identity and a generation-profile fingerprint to the cache payload. Increment `CACHE_SCHEMA_VERSION` from 4 to 5. Preserve all existing parsing, retry, recovery, trace, rebuild and atomic save paths.

- [ ] **Step 4: Run focused and full related GREEN**

Run: `python -m pytest tests/test_translation_engine.py tests/test_cli.py tests/test_subtitle_formats.py tests/test_run_manifest.py -q --basetemp _task4_basetemp`

Expected: all existing structural tests and new backend/cache tests pass. Remove `_task4_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Confirm no `subtitle_formats.py` behavior changed, no ITEM delimiter changed, no fallback default changed and cache schema 4 is ignored. Update Understory and stage only engine and related tests.

Commit: `refactor: route translation through backend contract`

### Task 5: Implement the llama-swap HTTP backend

**Files:**
- Create: `llama_swap_backend.py`
- Create: `tests/test_llama_swap_backend.py`
- Modify: `backend_factory.py`

**Interfaces:**
- Produces `LlamaSwapBackend(api_base, *, opener=...)`.
- Produces `health() -> None`, `list_models() -> tuple[ModelInfo, ...]`, `ensure_available(model) -> None`, `generate(request) -> GenerationResult` and `close() -> None`.
- Consumes the standard-library opener seam so every test is local and deterministic.

- [ ] **Step 1: Write RED HTTP tests**

Use a fake opener/response to cover health 200, health error, valid `/v1/models`, malformed models JSON, exact target present, exact target missing, valid chat completion, missing choices, missing content, empty content, malformed JSON, HTTP 404/500, timeout, connection error, usage metadata and zero retry count. Assert paths are `/health`, `/v1/models`, `/running` and `/v1/chat/completions` as applicable.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_llama_swap_backend.py -q`

Expected: import failure for `LlamaSwapBackend`.

- [ ] **Step 3: Implement one-attempt stdlib transport**

Normalize the supplied API base once, derive service root from its origin, encode JSON UTF-8, set `Content-Type: application/json`, use `stream=false`, extract only `choices[0].message.content`, map status/timeout/connection/JSON/shape errors to backend errors, and make `close()` a no-op. Do not call `/running` during generation and do not call any load/unload endpoint.

- [ ] **Step 4: Run focused and factory GREEN**

Run: `python -m pytest tests/test_llama_swap_backend.py tests/test_backend_config.py -q`

Expected: all fake HTTP and factory tests pass with zero real network calls.

- [ ] **Step 5: Review and commit**

Review URL construction, exact model matching, error redaction, no hidden retry, no `think=false`, no lifecycle mutation and no new dependency. Update Understory and stage only the backend, factory adjustment and focused tests.

Commit: `feat: add llama-swap HTTP backend`

### Task 6: Add model profiles and experimental Qwen3.6 registration

**Files:**
- Create: `model_profiles.py`
- Create: `tests/test_model_profiles.py`
- Modify: `backend_config.py`
- Modify: `tests/test_backend_config.py`

**Interfaces:**
- Produces `ModelProfile(model_id, display_name, backend, status, generation_defaults, max_parallel, provenance)`.
- Produces status constants or enum values for `BACKEND_SUPPORTED`, `MODEL_DISCOVERED`, `MODEL_PROFILE_KNOWN`, `STRUCTURALLY_VALIDATED` and `PRODUCTION_APPROVED`.
- Produces `get_model_profile(backend, model) -> ModelProfile | None` and a static Qwen3.6 profile.

- [ ] **Step 1: Write RED profile tests**

Assert the exact Qwen3.6 ID, backend `llama-swap`, status `EXPERIMENTAL/UNVALIDATED/NOT_PRODUCTION_APPROVED`, default false, no unverified context/`np`/temperature/top-p/TTL values, and no backend class named after the model. Assert unknown llama-swap models can be represented as discovered without a fabricated profile.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_model_profiles.py -q`

Expected: import or symbol failure.

- [ ] **Step 3: Implement the small registry**

Register only model identity, display name, backend and compatibility state for Qwen3.6. Keep optional concurrency unset until runtime configuration verifies it. Do not send `think=false` or infer server parameters.

- [ ] **Step 4: Run focused GREEN**

Run: `python -m pytest tests/test_model_profiles.py tests/test_backend_config.py -q`

Expected: profile and unknown-model tests pass.

- [ ] **Step 5: Review and commit**

Confirm the model remains opt-in and no user-provided unverified parameter is represented as runtime fact. Update Understory and stage only profile/config files and tests.

Commit: `feat: add experimental Qwen3.6 model profile`

### Task 7: Add multi-backend CLI support

**Files:**
- Modify: `translate_ass_fast.py`
- Modify: `tests/test_cli.py`
- Modify: `backend_config.py`

**Interfaces:**
- CLI adds `--backend` with default `ollama` and `--api-base` with llama-swap default `http://127.0.0.1:9292/v1`.
- CLI builds exactly one selected backend, calls `ensure_available()` before files, passes the backend to `FixedASSTranslator`, and calls `close()` in `finally`.
- Legacy aliases remain importable and preserve existing Ollama tests.

- [ ] **Step 1: Write RED CLI tests**

Add tests asserting no-argument parser defaults, explicit llama-swap argument propagation, backend preflight before file processing, model-not-found exit behavior, and that llama-swap close does not invoke Ollama shutdown. Keep the existing tests that monkeypatch legacy Ollama helpers passing through compatibility seams.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: new flag and propagation assertions fail.

- [ ] **Step 3: Implement CLI selection**

Parse the two flags, create settings/factory output, pass the backend into the translator, keep summary/manifest behavior unchanged, and close only the selected backend. Do not contact llama-swap when backend remains Ollama.

- [ ] **Step 4: Run focused and full CLI-related GREEN**

Run: `python -m pytest tests/test_cli.py tests/test_translation_engine.py tests/test_run_manifest.py -q --basetemp _task7_basetemp`

Expected: legacy default tests and explicit backend tests pass. Remove `_task7_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Check the exact legacy invocation and default model, no accidental real request in unit tests, and no Ollama lifecycle call for llama-swap. Update Understory and stage only CLI/config/test files.

Commit: `feat: add multi-backend CLI support`

### Task 8: Make the real regression runner backend-aware

**Files:**
- Modify: `tools/run_ass_regression.py`
- Modify: `tests/test_real_ass_runner.py`
- Modify: `ass_regression.py` if context metadata requires a typed field

**Interfaces:**
- Runner arguments add `--backend` and `--api-base` while retaining existing model, cache, run-id and isolation flags.
- Experiment `context.json` records backend, normalized API base without credentials, model, profile status/provenance, generation parameters and cache path.
- Preflight calls selected backend `ensure_available()` and returns `NOT_EXECUTED` for unavailable service/model.

- [ ] **Step 1: Write RED runner tests**

Add tests asserting backend/api-base appear in context, selected backend preflight is used, cache remains experiment-local, existing output is never overwritten, source hashes remain checked, and llama-swap context never requests unload. Keep unavailable Ollama tests and production CLI delegation tests.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_real_ass_runner.py -q`

Expected: new context and backend assertions fail.

- [ ] **Step 3: Implement backend-aware context/preflight**

Use the factory for preflight, pass backend/api-base to the production CLI argv, preserve `trace.jsonl.gz`, source snapshots, output mapping, manifest and non-overwrite behavior, and redact credentials from serialized context.

- [ ] **Step 4: Run focused and related GREEN**

Run: `python -m pytest tests/test_real_ass_runner.py tests/test_ass_regression.py tests/test_cli.py -q --basetemp _task8_basetemp`

Expected: all runner and regression analyzer tests pass without a real model call. Remove `_task8_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Verify that the runner still invokes the production CLI route, source integrity remains fail-closed, and model responses are not written to Understory. Update Understory and stage only runner/analyzer tests.

Commit: `feat: make ASS regression runner backend-aware`

### Task 9: Centralize shared UI configuration and migrate Flask

**Files:**
- Modify: `backend_config.py`
- Modify: `web_app.py`
- Modify: `tests/test_backend_config.py`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- Shared loader/saver owns canonical `backend`, `api_base`, `model` and legacy migration.
- Flask form/config uses the shared normalizer and emits `--backend`/`--api-base`.
- `OLLAMA_HOST` is set only for Ollama subprocesses.

- [ ] **Step 1: Write RED Flask/config tests**

Add tests for loading old local and remote Ollama config, saving canonical config without losing legacy values, selecting llama-swap, rejecting invalid URLs, showing backend/api-base/model controls, and building the exact subprocess command/environment for each backend.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_backend_config.py tests/test_web_app.py -q`

Expected: new canonical config and Flask propagation assertions fail.

- [ ] **Step 3: Implement shared migration and Flask integration**

Remove only duplicated backend-specific normalization/validation from Flask. Preserve upload, progress, CSRF, manifest and output cleanup behavior. Keep UI labels and layout simple; no frontend redesign.

- [ ] **Step 4: Run focused GREEN**

Run: `python -m pytest tests/test_backend_config.py tests/test_web_app.py -q --basetemp _task9_basetemp`

Expected: legacy and llama-swap web tests pass. Remove `_task9_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Check that existing `web_config.json` loads, invalid configuration fails visibly, no credentials are logged, and the Flask default remains Ollama. Update Understory and stage only shared config, Flask and tests.

Commit: `feat: add backend configuration to Flask`

### Task 10: Migrate Flet and PySide6 interfaces

**Files:**
- Modify: `app_flet.py`
- Modify: `app_gui.py`
- Modify: `tests/test_desktop_interfaces.py`

**Interfaces:**
- Both interfaces expose backend, API base/endpoint and model while retaining existing batch, timeout, cache and fallback controls.
- Both use the shared config normalizer and pass `--backend`/`--api-base` to the CLI.
- Existing progress, file discovery, manifest and packaging entrypoints remain unchanged.

- [ ] **Step 1: Write RED desktop tests**

Add source/delegation tests asserting both interfaces contain backend selection, use shared configuration, propagate API base/model, preserve Ollama defaults and do not call Ollama shutdown for llama-swap.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_desktop_interfaces.py -q`

Expected: new backend field/command assertions fail.

- [ ] **Step 3: Implement the minimal desktop migration**

Replace only duplicated backend configuration paths. Preserve UI behavior unrelated to backend selection and keep `web_config.json` migration in the shared module.

- [ ] **Step 4: Run focused and related GREEN**

Run: `python -m pytest tests/test_desktop_interfaces.py tests/test_cli.py tests/test_backend_config.py -q --basetemp _task10_basetemp`

Expected: desktop delegation, CLI and shared config tests pass. Remove `_task10_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Check PySide6/Flet frozen and unfrozen command paths, no accidental UI-wide refactor, and no regression in default Ollama behavior. Update Understory and stage only desktop files and tests.

Commit: `feat: add backend selection to desktop interfaces`

### Task 11: Update packaging, documentation and build imports

**Files:**
- Modify: `requirements.txt` only if the final import graph proves a change is necessary
- Modify: `TradutorASS-GUI.spec`
- Modify: `TradutorASS-PySide.spec`
- Modify: `build_gui_exe.bat`
- Modify: `build_flet_exe.bat`
- Modify: `build_pyside_exe.bat`
- Modify: `README.md`
- Modify: `README_FLET.md`
- Modify: `tests/test_cli.py`

**Interfaces:**
- No new runtime dependency is expected.
- Documentation describes Ollama default, llama-swap opt-in, API base normalization, experimental Qwen3.6 status and no automatic unload.
- Frozen builds import the new local backend/config/profile modules.

- [ ] **Step 1: Write RED documentation/build assertions**

Add tests asserting README/help keeps `qwen2.5:14b` and Ollama default, documents `--backend llama-swap`, documents the exact model ID and experimental status, and does not claim production validation.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: new documentation assertions fail.

- [ ] **Step 3: Implement documentation and packaging updates**

Add hidden imports only for modules not found by the existing import graph. Keep dependency list unchanged if stdlib transport is sufficient. Do not claim runtime Qwen3.6 success or add a load/unload command.

- [ ] **Step 4: Run focused GREEN and import checks**

Run: `python -m pytest tests/test_cli.py tests/test_desktop_interfaces.py -q --basetemp _task11_basetemp`

Run: `python -c "import inference_backend, ollama_backend, llama_swap_backend, backend_config, backend_factory, model_profiles, translation_engine"`

Expected: tests and imports pass. Remove `_task11_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Review that every documented command is opt-in and accurate, no runtime dependency was added without evidence, and build specs do not include secrets. Update Understory and stage only docs/build files and tests.

Commit: `docs: document multi-backend packaging and usage`

### Task 12: Add a model-compatibility gate harness without real inference

**Files:**
- Create: `tools/run_model_compatibility_gate.py`
- Create: `tests/test_model_compatibility_gate.py`
- Modify: `README.md` only if the gate command needs user-facing documentation

**Interfaces:**
- The harness accepts backend, API base, model, prompt/input fixture and output paths, plus an explicit opt-in flag for any live execution.
- Default behavior is dry-run/preflight only and never performs real translation or generation.
- Gate output records backend, normalized service root, exact model identity, profile status, request provenance and pass/fail criteria without secrets.
- Qwen3.6 remains experimental/unvalidated/not production approved until an operator runs the gate against approved infrastructure and reviews the artifacts.

- [ ] **Step 1: Write RED harness tests**

Add tests for dry-run behavior, exact profile/status reporting, URL redaction, selected-backend preflight delegation, failure on unavailable service/model, and the prohibition on automatic unload or hidden retries. Use fakes; do not call a live endpoint.

- [ ] **Step 2: Run focused RED**

Run: `python -m pytest tests/test_model_compatibility_gate.py -q`

Expected: the new harness tests fail because the command and result schema do not exist.

- [ ] **Step 3: Implement the dry-run gate**

Build the harness on the shared profile/config/factory/backend contracts. Keep live execution explicitly disabled by default, make any live path opt-in and operator-visible, and ensure the implementation cannot silently translate, unload or retry.

- [ ] **Step 4: Run focused GREEN**

Run: `python -m pytest tests/test_model_compatibility_gate.py tests/test_model_profiles.py tests/test_backend_config.py -q --basetemp _task12_basetemp`

Expected: gate, profile and configuration tests pass. Remove `_task12_basetemp` after verification.

- [ ] **Step 5: Review and commit**

Review that the harness is a compatibility gate rather than a production benchmark, has no credentials in artifacts, and keeps Qwen3.6 opt-in. Update Understory and stage only gate/docs/tests.

Commit: `feat: add model compatibility gate harness`

## Final branch review and delivery gates

- [ ] Confirm every approved spec section has an implementation task and every task names exact files, tests, focused commands, review checks and a commit.
- [ ] Read the complete diff against `origin/main`; exclude the preexisting `D entrada/.gitkeep`, `D saida/.gitkeep` and untracked `.understory-project-id` from scope.
- [ ] Use `superpowers:requesting-code-review` for an independent review of architecture, compatibility, security, test quality and spec compliance.
- [ ] Use `superpowers:receiving-code-review` to evaluate every review finding against current code and tests before applying any changes.
- [ ] Run focused tests for each task, related tests, the full suite with a controlled basetemp, `python -m compileall -q .`, `git diff --check` and staged-diff checks.
- [ ] Verify no real llama-swap inference, Qwen3.6 production claim, hidden retry, credential logging or unintended unload path was introduced by the default test/verification flow.
- [ ] Verify the branch is `feat/llama-swap-backend-qwen36`, its remote is synchronized, and no commit contains `.understory-project-id` or the preexisting main-checkout deletions.
- [ ] Update Understory only with confirmed durable decisions/results, then use `superpowers:finishing-a-development-branch` for the final handoff. Do not merge `main` and do not create a PR unless separately requested.
