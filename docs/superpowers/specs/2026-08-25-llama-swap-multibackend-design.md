# Design: suporte multi-backend com llama-swap

**Data:** 2026-08-25
**Projeto:** Tradutor de Legendas ASS e SRT Multilíngue para PT-BR
**Base Git:** `758d70894e3a3bc93e95c6c2d67677dc54a5bb51`
**Status:** aprovado pelo usuário para implementação
**Project ID Understory:** `project-6d781075-e9cf-4903-b5cd-5c244188a1ac`

## Goals

- Introduzir uma abstração pequena e explícita de backend de inferência.
- Preservar Ollama, `qwen2.5:14b` e o comportamento atual como default.
- Adicionar `LlamaSwapBackend` usando HTTP da biblioteca padrão Python.
- Usar a API OpenAI-compatible do llama-swap em `http://127.0.0.1:9292/v1`.
- Preparar suporte a múltiplos modelos sem criar um backend por modelo.
- Adicionar o perfil experimental `Qwen3.6-28B-REAP20-A3B-Q4_K_M`.
- Isolar cache e provenance por backend, modelo e perfil de geração.
- Preservar protocolo ITEM, validações, retries, recoveries, tracing, publicação atômica e regressão real.

## Non-goals

- Não alterar o protocolo ITEM ou relaxar qualquer validação estrutural.
- Não tornar Qwen3.6 o modelo padrão.
- Não aprovar Qwen3.6 para produção nesta mudança.
- Não executar benchmark completo, avaliação semântica ou gate estrutural nesta etapa de infraestrutura.
- Não adicionar o OpenAI SDK nesta primeira implementação.
- Não criar um framework genérico de plugins ou uma nova camada de orquestração.
- Não refatorar `subtitle_formats.py` nem redesenhar as interfaces gráficas.
- Não controlar diretamente o processo `llama-server` ou o lifecycle do llama-swap.

## Current architecture

`translation_engine.py` contém o pipeline de tradução, cache, prompts, protocolo ITEM, retries, recuperação individual, recuperação segmentada de quebras ASS, tracing, reconstrução, validação e publicação atômica. A classe pública legada é `FixedASSTranslator`, com alias `SubtitleTranslator`.

O engine importa Ollama diretamente e pressupõe que o cliente ofereça `show(model)` e `generate(model, prompt, system, options)`. `translate_ass_fast.py` valida o modelo e executa o shutdown específico do Ollama. `web_app.py`, `app_flet.py` e `app_gui.py` duplicam normalização de endpoint, configuração, persistência, validação de modelo e construção do subprocesso.

`subtitle_formats.py` é a fronteira estável de ASS/SRT e permanece fora da abstração de inferência. `run_manifest.py` registra somente outputs explicitamente publicados e existentes.

## Target architecture

```text
translation_engine.py
        |
        v
InferenceBackend
   +----+----+
   |         |
OllamaBackend  LlamaSwapBackend
   |         |
Ollama API   HTTP stdlib / OpenAI-compatible API

ModelProfile registry
   |
   +-- Qwen3.6-28B-REAP20-A3B-Q4_K_M (experimental)
```

O backend representa o transporte e as capacidades do provedor. `ModelProfile` representa o modelo, seus defaults conhecidos e seu estado de compatibilidade. Um novo modelo servido pelo llama-swap usa um novo perfil, não uma nova classe de backend.

O contrato não conhece ASS, SRT, placeholders ou ITEM. O engine continua sendo o único responsável por preparar prompts, interpretar respostas e validar estrutura.

## Backend abstraction

O módulo `inference_backend.py` deverá conter tipos e contrato mínimos, com nomes finais consistentes com o projeto:

```python
@dataclass(frozen=True)
class GenerationRequest:
    model: str
    system_prompt: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout: float

@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    backend: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

class InferenceBackend(Protocol):
    backend_id: str

    def ensure_available(self, model: str) -> None: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def close(self) -> None: ...
```

The contract may expose read-only health/model discovery through a small capability interface or concrete backend methods, but generation must remain the only operation consumed by the translation engine. `close()` is idempotent; the llama-swap implementation is a no-op.

Errors are normalized into backend-specific subclasses of a common backend error: configuration/URL errors, unavailable service, model not found, timeout/transport error and malformed response. The engine receives a normalized `GenerationResult`, never a provider-specific response object.

## OllamaBackend

`OllamaBackend` moves the current Ollama calls out of `translation_engine.py` and `translate_ass_fast.py` without changing their observable behavior.

It preserves:

- `show(model)` validation and model-not-found classification;
- `ollama.generate()` prompt, system prompt and options mapping;
- current `temperature`, `num_predict` and `top_p` values;
- current `OLLAMA_HOST` local/remote semantics;
- existing error messages and compatibility aliases where tests or callers depend on them;
- current best-effort shutdown behavior for Ollama only.

Existing fake clients used by unit tests remain usable through an explicit injection seam. No direct `import ollama` remains in the engine after extraction, except compatibility code whose behavior is covered by tests.

## LlamaSwapBackend

`LlamaSwapBackend` uses `urllib.request` and `json` from the Python standard library. It does not add an SDK dependency and does not use hidden retries.

### API roots

The configured API base and service root are separate values:

```text
api_base:     http://127.0.0.1:9292/v1
service_root: http://127.0.0.1:9292
```

The backend uses:

```text
GET  service_root/health
GET  api_base/models
GET  service_root/running
POST api_base/chat/completions
```

It must never construct `/v1/health` or append `/v1` twice.

### Model validation and discovery

`ensure_available(model)` first checks service health and then lists `/v1/models`. The configured model must match an advertised `id` exactly; fuzzy matching is forbidden. A missing model is a model-not-found error. Discovery is read-only and must not intentionally load, unload or swap a model.

### Generation

The request body is:

```json
{
  "model": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stream": false
}
```

The response must contain a non-empty string at `choices[0].message.content`. Missing choices, missing content, empty content, malformed JSON and non-2xx responses produce normalized backend errors. Usage values are optional metadata. The request must not add `think=false` or other Ollama-only parameters.

### Transport policy

Each request makes one HTTP attempt. There are zero hidden transport retries. The existing engine remains responsible for selective batch retry, individual recovery and segmented recovery. A single configured timeout is used in the first implementation for connection and response waiting; the implementation must not abort a legitimate first model load by adding an unrelated short startup timeout.

## Backend configuration and endpoint normalization

`backend_config.py` will centralize only the configuration needed by all consumers:

- backend identifier;
- API base;
- model identifier;
- endpoint normalization and validation;
- persistence and migration of the existing UI configuration.

The canonical configuration uses `backend`, `api_base`, `model`, batch size, timeout and cache settings. The loader continues to read legacy `ollama_mode` and `ollama_endpoint` values. For an Ollama configuration, old local/remote semantics and `OLLAMA_HOST` behavior remain unchanged. The writer preserves legacy keys when needed for backward-compatible rollback, while canonical fields are authoritative.

URL validation accepts only `http` and `https`. It rejects credentials, query strings, fragments and arbitrary schemes. For llama-swap, an origin or an origin ending in `/v1` is normalized to one `api_base` ending in `/v1` and one `service_root` without that suffix. Other paths are rejected. Ollama normalization preserves its existing accepted forms and does not silently change its endpoint semantics.

## ModelProfile

`model_profiles.py` provides a deliberately small registry and discovery model. A profile contains:

- `model_id`;
- display name;
- backend identifier;
- compatibility status;
- known generation defaults;
- optional `max_parallel`;
- provenance indicating whether values are registered, user-provided or runtime-verified.

Profiles must not contain provider transport logic.

### Qwen3.6 profile

The registered profile is:

```text
model_id: Qwen3.6-28B-REAP20-A3B-Q4_K_M
backend: llama-swap
status: EXPERIMENTAL / UNVALIDATED / NOT_PRODUCTION_APPROVED
default: false
```

The Fase 0 investigation found the exact model in `/v1/models`, but it was unloaded and its profile parameters were not runtime-verified. Therefore the implementation must not record context, `np`, temperature, top-p, or llama-swap TTL values as verified facts. Unverified user-provided values remain explicitly marked as such. Until a runtime profile confirms concurrency, the scheduler keeps the existing sequential batch behavior and does not invent a provider parallelism guarantee.

`--reasoning off` remains a server-side policy. The client does not send `think=false`.

## Translation engine integration

`FixedASSTranslator` receives or constructs an `InferenceBackend`. Its `_generate()` method creates `GenerationRequest`, invokes `backend.generate()`, and consumes only `GenerationResult.text`.

The following behavior is unchanged:

- prompt version and ITEM envelopes;
- marker protection/restoration;
- parsing and validation of responses;
- retry count and retry delay;
- selective retry of pending items;
- individual and segmented recovery;
- fail-closed behavior when fallback is disabled;
- trace events and failure isolation;
- candidate serialization, reload validation and atomic publication.

The legacy constructor injection used by tests and callers remains supported through a compatibility adapter around an injected Ollama client.

## Cache and provenance

`CACHE_SCHEMA_VERSION` increments from 4 to 5. Old schema entries are ignored rather than silently reused.

The cache key includes, at minimum:

```text
backend
model
prompt_version
source_language
target_language
subtitle format
cleaned text
line structure
protected markings
generation profile fingerprint
```

Timeout and other operational-only values are excluded unless their change can alter generated content. Backend identity is mandatory so an Ollama result cannot silently satisfy a llama-swap lookup. Experimental profiles use the same isolation rules as production defaults.

## CLI

`translate_ass_fast.py` adds:

```text
--backend   default: ollama
--api-base  optional; required for explicit llama-swap configuration unless its default is used
```

The legacy invocation remains valid:

```powershell
python translate_ass_fast.py
```

It continues to use Ollama and `qwen2.5:14b`.

The new opt-in form is:

```powershell
python translate_ass_fast.py `
  --backend llama-swap `
  --api-base http://127.0.0.1:9292/v1 `
  --model Qwen3.6-28B-REAP20-A3B-Q4_K_M
```

The CLI performs backend-specific preflight before processing files. Llama-swap preflight is health/model discovery only and never invokes chat completions during configuration validation.

## Flask, Flet and PySide6

Each interface exposes the minimal fields:

- backend;
- API base or Ollama endpoint according to backend;
- model;
- existing batch, timeout, cache and fallback controls.

The three interfaces consume shared configuration normalization, migration and validation. Their unrelated layout, progress reporting and file management remain unchanged.

Subprocess construction passes `--backend` and `--api-base`. `OLLAMA_HOST` is set only for Ollama. For llama-swap, the API base is passed explicitly and no Ollama shutdown command is executed.

Existing `web_config.json` files continue to load. Invalid or ambiguous legacy endpoint data fails with a user-visible configuration error instead of being silently rewritten.

## Real regression

`tools/run_ass_regression.py` and its context become backend-aware. Each experiment records:

- backend;
- normalized API base without credentials;
- model;
- profile status and provenance;
- generation parameters relevant to semantics;
- cache isolation path;
- CLI route used;
- run identifier and source hashes.

The runner continues to snapshot original files, use isolated outputs/cache, protect against overwrite, execute the production CLI path, collect compressed traces, verify source integrity and generate structural/semantic reports. Llama-swap runs never call an unload endpoint. Raw model responses remain in experiment artifacts, not Understory.

## Testing

The existing 169-test suite must remain green. New tests are added before implementation for each behavior:

- backend request/result types and error taxonomy;
- Ollama adapter behavior with fake clients;
- URL normalization and legacy configuration migration;
- llama-swap health success/failure;
- valid and malformed `/v1/models` responses;
- exact target discovery and missing-model handling;
- valid chat completion extraction;
- missing choices/content and empty content;
- malformed JSON, HTTP 404/500, timeout and connection errors;
- zero transport retry behavior;
- lifecycle no-op for llama-swap;
- cache schema and backend isolation;
- CLI default compatibility and explicit llama-swap selection;
- Flask/Flet/PySide configuration and command propagation;
- backend-aware regression context without real inference.

Tests use local fakes and mocked HTTP transport. No real model call is permitted while the infrastructure tasks are being implemented.

The MODEL_COMPATIBILITY_GATE is a separate later execution path. It must use a reduced diagnostic corpus, preferably the existing 30-case methodology when artifacts are available, and require zero definitive structural failures, ambiguous associations, fallback, placeholder corruption, tag corruption, line-break corruption and protocol leakage. A gate failure is recorded as model failure; validators are not relaxed.

## Packaging

No new runtime dependency is required. PyInstaller/Flet specifications and build scripts must include any newly added local modules through normal imports or explicit hidden imports where required. Build smoke checks must continue to import `translation_engine`, `subtitle_formats`, the CLI and the backend modules.

## Migration and backward compatibility

- Ollama remains the default backend.
- `qwen2.5:14b` remains the default model.
- The original CLI invocation remains valid.
- Existing fake Ollama clients remain injectable in tests.
- Existing Ollama endpoint and `OLLAMA_HOST` behavior remains valid.
- Existing `web_config.json` values are migrated without silent data loss.
- Existing cache schema 4 is safely ignored after the schema bump.
- ASS/SRT output and all structural guarantees remain unchanged.

## Error handling and security

- Reject unsupported URL schemes, credentials, query strings and fragments.
- Never log API credentials or full response bodies when they may contain secrets.
- Report backend, model and endpoint class in errors without leaking sensitive values.
- Treat malformed or incomplete provider responses as backend failures.
- Treat model-not-found and service-unavailable states as preflight failures.
- Do not retry below the pipeline layer.
- Do not call load, unload or swap endpoints as a side effect of validation or close.
- Preserve fail-closed output behavior when translation items remain unresolved.

## Rollout

1. Add contract, normalized request/result types and error taxonomy.
2. Extract Ollama into `OllamaBackend` with compatibility tests.
3. Add backend configuration/factory and legacy migration.
4. Add standard-library llama-swap transport, health and exact model discovery.
5. Add chat completions generation and normalized response handling.
6. Add `ModelProfile` and the experimental Qwen3.6 registration.
7. Add backend-aware cache/provenance.
8. Add CLI and real-regression integration.
9. Add shared UI configuration and update Flask, Flet and PySide6.
10. Update packaging and documentation.
11. Add the MODEL_COMPATIBILITY_GATE harness without executing it in the infrastructure rollout.

At every step, the default path remains Ollama and the structural pipeline is unchanged. Real Qwen3.6 inference is allowed only after the infrastructure, focused tests, full suite, compile check and review are green.

## Approval and implementation gates

This spec records the design approved in Fase 0. Before implementation, it must pass self-review for placeholders, contradictions, ambiguous values, unverified runtime claims, hidden dependencies and scope drift. The implementation plan is a separate artifact with task-level RED/GREEN, review, Understory and commit gates.

The feature branch is `feat/llama-swap-backend-qwen36`. Preexisting changes in `entrada/.gitkeep`, `saida/.gitkeep` and `.understory-project-id` are never staged as feature changes.
