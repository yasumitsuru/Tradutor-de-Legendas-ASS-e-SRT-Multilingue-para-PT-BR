# Translation Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir parsing de respostas ITEM, sanitização de comentários ASS e salvamento parcial, mantendo a mesma semântica segura na CLI, Flask, Flet e PySide.

**Architecture:** O handler ASS cria uma única cópia sanitizada do documento antes de qualquer decisão de tradução. O motor centraliza retries seletivos, recuperação individual e bloqueio de salvamento; um manifesto compartilhado registra somente outputs produzidos na execução atual, e as interfaces apenas propagam opções e consomem esse manifesto.

**Tech Stack:** Python 3.11+, asyncio, pysubs2, Ollama client, pytest, Flask, Flet e PySide6.

## Global Constraints

- Preservar horários, ordem, estilos, tags ASS válidas, quebras e compatibilidade SRT.
- `allow_original_fallback` deve ser `False` por padrão em todas as entradas.
- A recuperação individual usa `temperature=0.0` por chamada, sem mutar `self.config`.
- A sanitização ASS ocorre uma vez e deve ser idempotente.
- Outputs finais só substituem o destino depois de serialização, reabertura e validação completas.
- Não tocar nas remoções preexistentes de `entrada/.gitkeep` e `saida/.gitkeep`.
- Não fazer commit nem push sem autorização do usuário; os passos de commit usuais desta skill foram deliberadamente omitidos.

---

### Task 1: Scanner e cópia sanitizada ASS

**Files:**
- Modify: `subtitle_formats.py`
- Modify: `tests/test_subtitle_formats.py`

**Interfaces:**
- Produces: `sanitize_ass_text(text: str) -> str`
- Produces: `SubtitleFormatHandler.prepare_document(subs: pysubs2.SSAFile) -> pysubs2.SSAFile`
- `ASSFormatHandler.prepare_document()` sanitiza somente eventos não `Comment` na cópia.

- [ ] **Step 1: escrever os testes de regressão do sanitizer**

Adicionar testes com expectativas literais:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Warte, Oogami! {Oogami! Wait, Oogami!}", "Warte, Oogami! "),
        (r"{\i1}Hello{\i0}", r"{\i1}Hello{\i0}"),
        (
            r"{\i1}Der Schlüssel,{The key,\Nif needed.\i0}",
            r"{\i1}Der Schlüssel,{\i0}",
        ),
        (
            r"{\i1}Text {English,\Ncomment}{note}{\i0}",
            r"{\i1}Text {\i0}",
        ),
    ],
)
def test_ass_text_comments_are_removed_without_losing_real_overrides(source, expected):
    assert sanitize_ass_text(source) == expected


@pytest.mark.parametrize(
    "tag",
    [
        r"{\pos(100,200)}",
        r"{\move(10,20,100,200)}",
        r"{\clip(0,0,1920,1080)}",
        r"{\t(0,500,\fs40\bord3)}",
        r"{\fad(200,300)}",
        r"{\fade(0,255,0,0,200,700,900)}",
        r"{\c&HFFFFFF&}",
        r"{\1c&HFFFFFF&}",
        r"{\alpha&H80&}",
    ],
)
def test_ass_complex_override_commands_are_preserved_exactly(tag):
    assert sanitize_ass_text(f"{tag}Hello") == f"{tag}Hello"


def test_ass_sanitizer_is_idempotent():
    source = r"{\i1}Text {English,\Ncomment\i0}{\pos(100,200)}"
    once = sanitize_ass_text(source)
    assert sanitize_ass_text(once) == once
```

Adicionar também um teste de documento garantindo que `Dialogue` usa texto sanitizado e
`Comment` permanece byte/textualmente idêntico.

- [ ] **Step 2: executar somente os novos testes e confirmar RED**

Run: `python -m pytest tests/test_subtitle_formats.py -q -p no:cacheprovider`

Expected: FAIL porque `sanitize_ass_text` e `prepare_document` ainda não existem.

- [ ] **Step 3: implementar scanner pequeno e explícito**

Em `subtitle_formats.py`, definir conjuntos de comandos reconhecidos e um scanner com as
assinaturas abaixo:

```python
_ASS_PAREN_COMMANDS = frozenset({"pos", "move", "org", "clip", "iclip", "fad", "fade", "t"})
_ASS_NUMERIC_COMMANDS = frozenset({
    "i", "b", "u", "s", "an", "a", "q", "p", "pbo", "be", "blur",
    "bord", "xbord", "ybord", "shad", "xshad", "yshad", "fs", "fscx",
    "fscy", "fsp", "fr", "frx", "fry", "frz", "fax", "fay", "fe",
    "k", "kf", "ko", "kt",
})
_ASS_COLOR_COMMANDS = frozenset({"c", "1c", "2c", "3c", "4c"})
_ASS_ALPHA_COMMANDS = frozenset({"alpha", "1a", "2a", "3a", "4a"})
_ASS_TEXT_COMMANDS = frozenset({"fn", "r"})


def _scan_balanced_parentheses(value: str, start: int) -> int | None:
    """Return the first index after the balanced group, or None."""


def _parse_ass_override_at(value: str, start: int) -> int | None:
    """Return the exact command end when a recognized ASS command starts at start."""


def _scan_ass_override_spans(value: str) -> tuple[tuple[int, int], ...]:
    """Find recognized command spans while leaving natural-language residue visible."""


def sanitize_ass_text(text: str) -> str:
    def sanitize_block(match: re.Match[str]) -> str:
        content = match.group(1)
        spans = _scan_ass_override_spans(content)
        if not spans:
            return ""
        residue_parts: list[str] = []
        cursor = 0
        for start, end in spans:
            residue_parts.append(content[cursor:start])
            cursor = end
        residue_parts.append(content[cursor:])
        residue = "".join(residue_parts)
        if not residue.strip():
            return match.group(0)
        commands = "".join(content[start:end] for start, end in spans)
        return "{" + commands + "}" if commands else ""

    return _ASS_BRACE_BLOCK_RE.sub(sanitize_block, text)
```

O código final deve implementar a coleta de gaps diretamente ou por helper definido no
mesmo task. `\N`, `\n` e `\h` não entram em nenhum conjunto. Para `t(...)`, validar
parênteses balanceados e que a parte de modifiers contém uma sequência reconhecida de
commands; preservar sempre o slice original, sem reconstruir argumentos.

Adicionar `prepare_document()` na base como `copy.deepcopy(subs)` e sobrescrever no
handler ASS para sanitizar uma única vez os eventos cuja propriedade `type != "Comment"`.
Não alterar `rebuild()`.

- [ ] **Step 4: executar os testes do handler e confirmar GREEN**

Run: `python -m pytest tests/test_subtitle_formats.py -q -p no:cacheprovider`

Expected: todos passam, inclusive idempotência e comandos complexos.

---

### Task 2: Warning externo, retry seletivo e recuperação individual

**Files:**
- Modify: `translation_engine.py`
- Modify: `tests/test_translation_engine.py`

**Interfaces:**
- Consumes: `handler.prepare_document(subs)` do Task 1.
- Produces: `_generate(prompt: str, *, temperature: float | None = None) -> str`.
- Produces: `_build_individual_prompt(item_id: int, prepared: PreparedSubtitleText) -> str`.

- [ ] **Step 1: escrever testes RED do protocolo e recuperação**

Adicionar testes que:

```python
def test_external_response_residue_does_not_invalidate_valid_item(tmp_path):
    # Fake returns "Here is the translation:" plus one valid ITEM.
    # Assert outcome.text == "Olá" and used_fallback is False.


def test_valid_items_survive_residue_and_only_missing_item_is_retried(tmp_path):
    # First response contains external residue, ITEM_0001 valid, ITEM_0002 absent.
    # Assert prompt 2 lacks ITEM_0001 and contains only ITEM_0002.


def test_individual_recovery_uses_zero_temperature_without_mutating_config(tmp_path):
    # Fake fails every normal batch attempt, then succeeds for minimal individual prompt.
    # Capture options for each call and assert final temperature == 0.0 while
    # translator.config["temperature"] remains the original value.
```

Usar respostas Ollama completas `{"response": "..."}` e assertar o resultado real, não
a existência do fake.

- [ ] **Step 2: executar os novos testes e confirmar RED**

Run: `python -m pytest tests/test_translation_engine.py -q -p no:cacheprovider`

Expected: o resíduo invalida o item e a recuperação individual não existe.

- [ ] **Step 3: aceitar warning global sem relaxar validação por item**

Manter `_parse_item_response()` retornando `(parsed, warning)` e remover somente o ramo
que copia `warning` para `errors[item_id]`. Logar uma vez:

```python
parsed, response_warning = self._parse_item_response(response_text)
if response_warning:
    print(f"   ⚠️ {response_warning} O resíduo externo foi descartado.")
```

Itens ausentes/duplicados continuam sem body; bodies presentes passam individualmente
por `_validate_model_body()`.

- [ ] **Step 4: implementar override de geração e recuperação individual**

Alterar `_generate` sem mutar config:

```python
async def _generate(self, prompt: str, *, temperature: float | None = None) -> str:
    effective_temperature = (
        float(self.config["temperature"]) if temperature is None else float(temperature)
    )
    options = {
        "temperature": effective_temperature,
        "num_predict": self.config["max_tokens"],
        "top_p": 0.9,
    }
```

Depois do loop normal, iterar sobre uma cópia dos pendentes, chamar um prompt individual
mínimo com `temperature=0.0`, fazer `_parse_item_response()` e `_validate_model_body()`
normalmente e remover apenas os itens recuperados.

- [ ] **Step 5: usar a cópia sanitizada no pipeline**

Em `parse_subtitle_file()`, separar carregamento e cópia de trabalho:

```python
loaded = handler.load(source)
subs = handler.prepare_document(loaded)
```

Todos os eventos, cache keys, skips e o retorno passam a usar `subs`.

- [ ] **Step 6: executar testes do motor e handlers**

Run: `python -m pytest tests/test_translation_engine.py tests/test_subtitle_formats.py -q -p no:cacheprovider`

Expected: todos passam.

---

### Task 3: Falha segura, cache novo e publicação atômica validada

**Files:**
- Modify: `translation_engine.py`
- Modify: `tests/test_translation_engine.py`

**Interfaces:**
- Produces: `IncompleteTranslationError(SubtitleValidationError)` com `failed_count: int`.
- Produces: config `allow_original_fallback: bool = False`.

- [ ] **Step 1: escrever testes RED de falha definitiva e destino antigo**

Adicionar testes que criem uma entrada real e um destino com bytes conhecidos. Um fake
Ollama sempre retorna estrutura inválida.

```python
with pytest.raises(IncompleteTranslationError, match="Arquivo não será salvo"):
    asyncio.run(translator.translate_file(source, destination))
assert destination.read_bytes() == old_bytes

legacy = FixedASSTranslator(
    translator_config(tmp_path, allow_original_fallback=True), broken_client
)
stats = asyncio.run(legacy.translate_file(source, destination))
assert stats["failed"] > 0
assert destination.exists()
```

Adicionar teste que monkeypatcha a validação após serialização para falhar e confirma que
o destino antigo permanece intacto.

- [ ] **Step 2: executar os testes e confirmar RED**

Run: `python -m pytest tests/test_translation_engine.py -q -p no:cacheprovider`

Expected: modo padrão ainda salva fallback e sobrescreve o destino.

- [ ] **Step 3: bloquear salvamento parcial no motor**

Definir:

```python
class IncompleteTranslationError(SubtitleValidationError):
    def __init__(self, failed_count: int):
        self.failed_count = failed_count
        super().__init__(
            f"Falha definitiva em {failed_count} item(ns). Arquivo não será salvo como "
            "tradução concluída para evitar mistura de idiomas."
        )
```

Adicionar `allow_original_fallback=False` a `CONFIG`. Em `translate_file()`, depois de
`process_lines_optimized()` e antes de reconstruir, levantar a exceção quando
`self.stats["failed"] > 0` e a opção for falsa. Emitir log explícito de que nenhum novo
output foi produzido.

- [ ] **Step 4: validar candidato temporário antes da substituição final**

Salvar em sibling temporário único, reabrir e validar, e somente então substituir:

```python
candidate = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.candidate")
try:
    handler.save(rebuilt, candidate)
    saved = handler.load(candidate)
    self._validate_rebuilt_document(rebuilt, saved, handler)
    candidate.replace(destination)
finally:
    candidate.unlink(missing_ok=True)
```

O suffix do candidato precisa continuar sendo `.ass` ou `.srt`; construir o nome como
`.{destination.stem}.{uuid}.candidate{destination.suffix}`.

- [ ] **Step 5: invalidar cache antigo**

Incrementar `PROMPT_VERSION` para `subtitle-items-v2` e `CACHE_SCHEMA_VERSION` para `3`.
Na validação de cache ASS, rejeitar um candidato cujo `handler.prepare_text(cached).original_text`
seja diferente de `cached`, impedindo comentários textuais mesmo em payloads adulterados.

- [ ] **Step 6: confirmar GREEN**

Run: `python -m pytest tests/test_translation_engine.py tests/test_subtitle_formats.py -q -p no:cacheprovider`

Expected: todos passam.

---

### Task 4: Manifesto de outputs da execução e CLI

**Files:**
- Create: `run_manifest.py`
- Modify: `translate_ass_fast.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_run_manifest.py`

**Interfaces:**
- Produces: `initialize_run_manifest(path: Path) -> None`.
- Produces: `record_run_output(path: Path, output_root: Path, output: Path) -> None`.
- Produces: `load_run_outputs(path: Path, output_root: Path) -> list[Path]`.
- Produces: CLI `--result-manifest PATH` e `--allow-original-fallback`.

- [ ] **Step 1: escrever testes RED do manifesto e CLI**

Cobrir escrita/leitura atômica, rejeição de caminhos fora do output root, manifesto vazio
após falha, `allow_original_fallback=False` por padrão e propagação `True` ao translator.
O teste de saída antiga deve:

```python
old_output.write_text("resultado antigo", encoding="utf-8")
exit_code = asyncio.run(
    main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--result-manifest", str(manifest),
        ]
    )
)
assert exit_code == 1
assert old_output.read_text(encoding="utf-8") == "resultado antigo"
assert load_run_outputs(manifest, output_dir) == []
```

Monkeypatchar somente validação Ollama e a classe do translator; manter `main()` e o
manifesto reais.

- [ ] **Step 2: executar testes e confirmar RED**

Run: `python -m pytest tests/test_cli.py tests/test_run_manifest.py -q -p no:cacheprovider`

Expected: módulo, opções e manifesto ainda ausentes.

- [ ] **Step 3: implementar manifesto compartilhado**

Formato JSON literal:

```json
{"schema_version": 1, "outputs": ["episodio.pt.ass"]}
```

Normalizar e validar cada saída contra `output_root.resolve()`, aceitar somente arquivos
ASS/SRT existentes e escrever via arquivo temporário + `replace`. Retornar lista vazia
para manifesto inexistente/inválido, sem incluir arquivos do diretório por descoberta.

- [ ] **Step 4: integrar opções no CLI**

Adicionar argumentos:

```python
parser.add_argument("--allow-original-fallback", action="store_true", default=False)
parser.add_argument("--result-manifest", type=Path)
```

Inicializar manifesto vazio antes do loop, copiar o booleano para `config` e registrar o
output somente no ramo `else` posterior a `translate_file()` bem-sucedido. Falhas devem
ser adicionadas a `failed_files` e manter o manifesto sem o destino antigo.

- [ ] **Step 5: confirmar GREEN**

Run: `python -m pytest tests/test_cli.py tests/test_run_manifest.py -q -p no:cacheprovider`

Expected: todos passam.

---

### Task 5: Propagação e apresentação em Flask, Flet e PySide

**Files:**
- Modify: `web_app.py`
- Modify: `templates/index.html`
- Modify: `static/style.css`
- Modify: `app_flet.py`
- Modify: `app_gui.py`
- Modify: `tests/test_web_app.py`
- Modify: `tests/test_desktop_interfaces.py`

**Interfaces:**
- Consumes: `load_run_outputs()` e CLI flags do Task 4.
- Cada `_build_command(form)` acrescenta `--allow-original-fallback` somente para valor ativo.

- [ ] **Step 1: escrever testes RED de defaults e propagação**

Testar comportamento real dos command builders:

```python
assert "--allow-original-fallback" not in _build_command({**form, "allow_original_fallback": "off"})
assert "--allow-original-fallback" in _build_command({**form, "allow_original_fallback": "on"})
```

Para Flet, aceitar booleano `True`; para PySide/Flask, usar o formato já adotado pelo
respectivo formulário. Testar GET Flask contendo checkbox desmarcado, label e aviso.
Testar manifesto vazio com arquivo antigo existente: `/files` não o retorna e
`/download-output` não o inclui/retorna erro de ausência de outputs da execução.

- [ ] **Step 2: executar testes e confirmar RED**

Run: `python -m pytest tests/test_web_app.py tests/test_desktop_interfaces.py -q -p no:cacheprovider`

Expected: flags, controles e filtragem por manifesto ainda ausentes.

- [ ] **Step 3: implementar Flask**

Adicionar checkbox desmarcado com label literal e aviso próximo. Em `/start`, propagar o
campo e sempre passar `--result-manifest` com o caminho da execução. Alterar listagem e
download para `load_run_outputs(RUN_MANIFEST_PATH, safe_output)`; nunca fazer fallback
para listar todo o diretório quando o manifesto está vazio.

- [ ] **Step 4: implementar Flet**

Adicionar `ft.Checkbox(value=False)` com o label literal e um `ft.Text` de aviso. Coletar
o booleano, repassar a flag somente quando ativo, passar manifesto e usar o leitor
compartilhado em lista e ZIP.

- [ ] **Step 5: implementar PySide**

Adicionar `QCheckBox` desmarcado com o label literal e `QLabel` de aviso. Coletar como
`"on"`/`"off"`, repassar flag/manifesto e usar o leitor compartilhado em lista e ZIP.

- [ ] **Step 6: confirmar GREEN**

Run: `python -m pytest tests/test_web_app.py tests/test_desktop_interfaces.py tests/test_cli.py tests/test_run_manifest.py -q -p no:cacheprovider`

Expected: todos passam.

---

### Task 6: README, exemplos exatos e validação completa

**Files:**
- Modify: `README.md`
- Modify: `tests/test_translation_engine.py`
- Modify: `tests/test_subtitle_formats.py`

**Interfaces:**
- Documents: protocolo ITEM, retry seletivo, recovery individual, sanitizer ASS, modo seguro, flag e UIs.

- [ ] **Step 1: atualizar README**

Substituir a descrição do fallback silencioso pelo modo seguro. Documentar:

- resíduo fora de ITEM vira warning descartado;
- somente itens inválidos são reenviados;
- há uma tentativa individual com temperatura zero;
- comentários ASS textuais são removidos e overrides reais preservados;
- `--allow-original-fallback` e o risco de mistura de idiomas;
- checkbox equivalente em Flask/Flet/PySide;
- manifesto da execução impede outputs antigos em ZIP/download;
- cache schema `3`.

- [ ] **Step 2: executar testes direcionados dos sete casos obrigatórios**

Run: `python -m pytest tests/test_translation_engine.py tests/test_subtitle_formats.py -q -p no:cacheprovider`

Expected: todos passam.

- [ ] **Step 3: executar a suíte completa exigida**

Run: `python -m pytest tests -q`

Expected: exit code `0`, zero falhas.

- [ ] **Step 4: revisar diff e requisitos**

Run: `git diff --check`

Run: `git status --short`

Confirmar que somente os arquivos planejados e as duas remoções preexistentes aparecem;
reler os seis exemplos ASS e os comportamentos com/sem fallback contra os testes.
