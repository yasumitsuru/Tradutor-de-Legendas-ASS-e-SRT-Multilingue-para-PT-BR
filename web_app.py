from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import tempfile
import webbrowser
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import time
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest

import ollama
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
)


SOURCE_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else SOURCE_DIR
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR)).resolve()
SCRIPT_PATH = SOURCE_DIR / "translate_ass_fast.py"
CACHE_PATH = BASE_DIR / "translation_cache.json"
WEB_CONFIG_PATH = BASE_DIR / "web_config.json"
TEMPLATES_DIR = RESOURCE_DIR / "templates"
STATIC_DIR = RESOURCE_DIR / "static"
APP_HOST = "127.0.0.1"
APP_PORT = 7860
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"
UI_HEARTBEAT_TIMEOUT_SECONDS = 8.0
DEFAULT_FORM_VALUES: dict[str, str] = {
    "input_dir": "./entrada",
    "output_dir": "./saida",
    "model": "qwen2.5:14b",
    "batch_size": "15",
    "timeout": "300",
    "ollama_mode": "local",
    "ollama_endpoint": "",
}
CONFIG_KEYS = ("model", "batch_size", "timeout", "ollama_mode", "ollama_endpoint")

_ALLOWED_MODEL_RE = re.compile(r"^[a-z0-9._:/@-]+$", re.IGNORECASE)

_PATH_SANITIZER = re.compile(
    r"[A-Z]:\\[^:\"<>|?*\r\n]*|"                  # C:\foo\bar
    r"(?:(?:\.\.|\.)[\\/][^\s:<>|?*\r\n]+)"       # ./entrada, ..\saida
)
_SEASON_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_TOTAL_FILES_RE = re.compile(r"Encontrados\s+(\d+)\s+arquivos\s+\.ass", re.IGNORECASE)
_EPISODE_DONE_RE = re.compile(r"epis.{0,3}dio.+conclu", re.IGNORECASE)
_BATCH_PROGRESS_RE = re.compile(r"Batch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _sanitize_logs(text: str) -> str:
    """Remove caminhos de arquivos dos logs para evitar exposição de diretórios locais."""
    return _PATH_SANITIZER.sub("[redacted]", text)


def _safe_directory(raw: str, *, fallback: Path) -> Path:
    """Resolve um diretório enviado pelo usuário e limita o resultado à pasta base permitida.

    Caminhos absolutos e tentativas de escapar da raiz esperada retornam o diretório fallback.
    """
    # Block absolute paths early (handles both Unix and Windows roots).
    raw_stripped = raw.strip()
    if raw_stripped.startswith(("/", "\\")) or (
        len(raw_stripped) >= 2 and raw_stripped[1] == ":"
    ):
        return fallback

    candidate = (fallback.parent / raw).resolve()
    if not str(candidate).startswith(str(fallback.parent.resolve())):
        return fallback
    return candidate


def _validate_model(name: str) -> bool:
    """Valida o identificador do modelo Ollama com regex de caracteres permitidos."""
    return bool(name) and bool(_ALLOWED_MODEL_RE.match(name))


def _normalize_ollama_connection(
    mode_raw: str,
    endpoint_raw: str,
) -> tuple[bool, str, str, str | None, str]:
    """Normaliza e valida modo/endpoint de conexão com Ollama (local ou remoto)."""
    mode = (mode_raw or "local").strip().lower()
    if mode not in {"local", "remote"}:
        return False, "", "", None, "Tipo de conexão Ollama inválido."

    endpoint = (endpoint_raw or "").strip()
    if mode == "local":
        return True, "local", "", None, ""

    if not endpoint:
        return (
            False,
            "",
            "",
            None,
            "Informe IP:porta ou URL/DNS para o Ollama remoto.",
        )

    candidate = endpoint
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", candidate):
        candidate = f"http://{candidate}"

    try:
        parsed = urlparse.urlparse(candidate)
    except Exception:
        return False, "", "", None, "Endpoint Ollama remoto inválido."

    if parsed.scheme not in {"http", "https"}:
        return False, "", "", None, "Use apenas endpoints http:// ou https://."
    if not parsed.hostname:
        return False, "", "", None, "Endpoint Ollama remoto inválido."
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        return False, "", "", None, "Use apenas host e porta (sem path ou query)."

    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return True, "remote", normalized_endpoint, normalized_endpoint, ""


def _normalize_config(form_like: dict[str, str]) -> tuple[bool, dict[str, str], str]:
    """Valida os campos de configuração do formulário e devolve valores normalizados."""
    model = form_like.get("model", "").strip()
    batch_size = form_like.get("batch_size", "").strip()
    timeout = form_like.get("timeout", "").strip()
    ollama_mode = form_like.get("ollama_mode", "local").strip()
    ollama_endpoint = form_like.get("ollama_endpoint", "").strip()

    if not _validate_model(model):
        return False, {}, "Nome de modelo inválido."

    try:
        batch_num = int(batch_size)
    except ValueError:
        return False, {}, "Batch size deve ser um número inteiro."
    if batch_num < 1:
        return False, {}, "Batch size deve ser maior ou igual a 1."

    try:
        timeout_num = int(timeout)
    except ValueError:
        return False, {}, "Timeout deve ser um número inteiro."
    if timeout_num < 10:
        return False, {}, "Timeout deve ser maior ou igual a 10 segundos."

    conn_ok, normalized_mode, normalized_endpoint, _, conn_error = _normalize_ollama_connection(
        ollama_mode, ollama_endpoint
    )
    if not conn_ok:
        return False, {}, conn_error

    normalized = {
        "model": model,
        "batch_size": str(batch_num),
        "timeout": str(timeout_num),
        "ollama_mode": normalized_mode,
        "ollama_endpoint": normalized_endpoint,
    }
    return True, normalized, ""


def _load_user_config() -> dict[str, str]:
    """Lê a configuração persistida no disco e retorna somente valores válidos."""
    if not WEB_CONFIG_PATH.exists():
        return {}
    try:
        with WEB_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return {}
        raw_config = {key: str(raw.get(key, "")).strip() for key in CONFIG_KEYS}
        ok, normalized, _ = _normalize_config(raw_config)
        return normalized if ok else {}
    except Exception:
        return {}


def _save_user_config(config: dict[str, str]) -> None:
    """Salva no arquivo local os campos de configuração permitidos pela aplicação."""
    payload = {key: config[key] for key in CONFIG_KEYS}
    with WEB_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _effective_defaults() -> dict[str, str]:
    """Mescla os padrões da aplicação com a configuração do usuário, quando existir."""
    defaults = dict(DEFAULT_FORM_VALUES)
    defaults.update(_load_user_config())
    return defaults


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detecta erros típicos de modelo inexistente retornados pelo Ollama."""
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status_code == 404:
        return True
    return "model" in text and "not found" in text


def _ensure_ollama_model_available(model_name: str, ollama_host: str | None = None) -> tuple[bool, str]:
    """Confere se o modelo informado está disponível no endpoint Ollama selecionado."""
    candidate_hosts: list[str] = []
    if ollama_host:
        candidate_hosts.append(ollama_host)
    else:
        # Em modo local, não depender de OLLAMA_HOST do ambiente evita falhas
        # quando a variável está com valor de bind (ex.: 0.0.0.0:11434).
        candidate_hosts.append(LOCAL_OLLAMA_HOST)
        env_host_raw = (os.environ.get("OLLAMA_HOST") or "").strip()
        if env_host_raw:
            env_host = env_host_raw
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", env_host):
                env_host = f"http://{env_host}"
            if env_host not in candidate_hosts:
                candidate_hosts.append(env_host)

    last_exc: Exception | None = None
    last_host: str | None = None

    for host in candidate_hosts:
        try:
            client = ollama.Client(host=host)
            client.show(model_name)
            return True, ""
        except Exception as exc:
            if _is_model_not_found_error(exc):
                endpoint_hint = f' no endpoint "{host}"' if host else ""
                return (
                    False,
                    f'Modelo "{model_name}" não está disponível no Ollama{endpoint_hint}. '
                    f'Instale antes com: ollama pull {model_name}',
                )
            last_exc = exc
            last_host = host

    where = f' no endpoint "{last_host}"' if last_host else ""
    return False, f"Não foi possível validar o modelo no Ollama{where}: {last_exc}"


def _secure_filename(name: str) -> str:
    """Sanitiza o nome do arquivo enviado para reduzir risco com caracteres perigosos."""
    name = Path(name).name
    # Keep apostrophes in file names (e.g. "Deviant's"), while still removing
    # unsafe path/control characters.
    name = name.replace("\u2019", "'")
    name = re.sub(r"[^\w\.\-\[\]()@,;+=! ']", "", name)
    name = name.strip(" .")
    return name or "arquivo"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = secrets.token_hex(32)
app.config["WTF_CSRF_ENABLED"] = False  # Manual CSRF, no Flask-WTF dependency.

# ---------------------------------------------------------------------------
# CSRF protection (token-based, stored in session + hidden form field)
# ---------------------------------------------------------------------------
_CSRF_FIELD = "csrf_token"


def _generate_csrf() -> str:
    """Gera e armazena um token CSRF na sessão atual quando necessário."""
    if _CSRF_FIELD not in session:
        session[_CSRF_FIELD] = secrets.token_hex(32)
    return session[_CSRF_FIELD]


def _validate_csrf(silent: bool = False) -> bool:
    """Valida se o token CSRF enviado na requisição corresponde ao token da sessão."""
    form_token = (
        request.form.get(_CSRF_FIELD)
        or request.headers.get("X-CSRF-Token", "")
    )
    session_token = session.get(_CSRF_FIELD, "")
    if not session_token or not secrets.compare_digest(form_token, session_token):
        if not silent:
            return False
    return True


def _require_csrf():
    """Interrompe a requisição com 403 em JSON quando a validação CSRF falha."""
    if not _validate_csrf():
        return jsonify({"ok": False, "error": "Token CSRF inválido ou ausente."}), 403
    return None

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "command": None,
    "logs": [],
}

_ui_watchdog_lock = threading.Lock()
_ui_watchdog_state: dict[str, Any] = {
    "enabled": False,
    "seen_heartbeat": False,
    "last_heartbeat": 0.0,
}


def _append_log(line: str) -> None:
    """Adiciona uma linha ao buffer de logs em memória com sanitização e limite de tamanho."""
    line = line.rstrip("\r\n")
    if not line:
        return
    # Sanitize paths at the source so raw paths are never stored in memory.
    line = _sanitize_logs(line)
    with _state_lock:
        _state["logs"].append(line)
        # Keep memory bounded.
        if len(_state["logs"]) > 4000:
            _state["logs"] = _state["logs"][-4000:]


def _register_ui_heartbeat() -> None:
    """Registra batimento da interface para indicar que a janela web ainda está ativa."""
    with _ui_watchdog_lock:
        _ui_watchdog_state["seen_heartbeat"] = True
        _ui_watchdog_state["last_heartbeat"] = time.monotonic()


def _start_ui_watchdog_if_needed() -> None:
    """Inicia monitor que encerra o app empacotado se a interface ficar inativa."""
    if not IS_FROZEN:
        return
    if os.environ.get("ASS_TRANSLATOR_KEEP_RUNNING") == "1":
        return

    with _ui_watchdog_lock:
        if _ui_watchdog_state["enabled"]:
            return
        _ui_watchdog_state["enabled"] = True
        _ui_watchdog_state["seen_heartbeat"] = False
        _ui_watchdog_state["last_heartbeat"] = 0.0

    def _watchdog_loop() -> None:
        """Loop em background que verifica timeout do heartbeat da interface."""
        while True:
            threading.Event().wait(1.0)
            with _ui_watchdog_lock:
                enabled = _ui_watchdog_state["enabled"]
                seen_heartbeat = _ui_watchdog_state["seen_heartbeat"]
                last_heartbeat = _ui_watchdog_state["last_heartbeat"]

            if not enabled:
                return
            if not seen_heartbeat:
                continue

            if (time.monotonic() - last_heartbeat) > UI_HEARTBEAT_TIMEOUT_SECONDS:
                _append_log(
                    "[WEB] Janela da interface foi fechada. Encerrando aplicativo."
                )
                os._exit(0)

    threading.Thread(target=_watchdog_loop, daemon=True).start()


def _stream_process_output(process: subprocess.Popen[str]) -> None:
    """Lê a saída do subprocesso em tempo real e publica cada atualização no log."""
    if process.stdout is None:
        return

    buffer_chars: list[str] = []
    while True:
        # Read 1 char so tqdm-style '\r' updates are emitted immediately.
        char = process.stdout.read(1)
        if char == "":
            if buffer_chars:
                _append_log("".join(buffer_chars))
            break

        if char in ("\n", "\r"):
            if buffer_chars:
                _append_log("".join(buffer_chars))
                buffer_chars = []
            continue

        buffer_chars.append(char)


def _logs_indicate_model_not_found() -> bool:
    """Verifica nos logs se houve erro de modelo inexistente para mensagem amigável."""
    with _state_lock:
        combined_logs = "\n".join(_state["logs"]).lower()
    patterns = (
        "model not found",
        "status code: 404",
        "não está disponível no ollama",
    )
    return any(pattern in combined_logs for pattern in patterns)


def _compute_progress(
    log_lines: list[str],
    running: bool,
    return_code: int | None,
) -> tuple[int, str]:
    """Calcula progresso percentual e rótulo com base nos logs da execução atual."""
    if not log_lines:
        if running:
            return 0, "0%"
        if return_code == 0:
            return 100, "100% (concluído)"
        return 0, "0%"

    # 1) Prefer explicit tqdm season percentage when present.
    tqdm_percent: int | None = None
    for line in log_lines:
        if "Processando Temporada" not in line:
            continue
        match = _SEASON_TQDM_PERCENT_RE.search(line)
        if match:
            try:
                tqdm_percent = max(0, min(100, int(match.group(1))))
            except ValueError:
                continue
    if tqdm_percent is not None:
        percent = tqdm_percent
        if running:
            percent = min(percent, 99)
            return percent, f"{percent}%"
        if return_code == 0:
            return 100, "100% (concluído)"
        return percent, f"{percent}% (finalizado com erro)"

    # 2) Fallback by episodes completed / total episodes.
    total_files: int | None = None
    completed_files = 0
    for line in log_lines:
        total_match = _TOTAL_FILES_RE.search(line)
        if total_match:
            try:
                total_files = int(total_match.group(1))
            except ValueError:
                total_files = None
        if _EPISODE_DONE_RE.search(line):
            completed_files += 1

    if total_files and total_files > 0:
        completed_clamped = min(completed_files, total_files)
        percent = round((completed_clamped / total_files) * 100)
        percent = max(0, min(100, int(percent)))
        if running:
            percent = min(percent, 99)
        elif return_code == 0:
            return 100, "100% (concluído)"

        suffix = ""
        if not running and return_code not in (None, 0):
            suffix = " (finalizado com erro)"
        label = f"{percent}% ({completed_clamped}/{total_files} episódios){suffix}"
        return percent, label

    # 3) Last fallback: batch progress for single-file runs.
    last_batch = (0, 0)
    for line in log_lines:
        batch_match = _BATCH_PROGRESS_RE.search(line)
        if not batch_match:
            continue
        try:
            current = int(batch_match.group(1))
            total = int(batch_match.group(2))
        except ValueError:
            continue
        if total > 0:
            last_batch = (current, total)

    current, total = last_batch
    if total > 0:
        current = max(0, min(current, total))
        percent = int(round((current / total) * 100))
        percent = max(0, min(100, percent))
        if running:
            percent = min(percent, 99)
        elif return_code == 0:
            percent = 100
        return percent, f"{percent}% (batch {current}/{total})"

    if running:
        return 0, "0%"
    if return_code == 0:
        return 100, "100% (concluído)"
    return 0, "0%"


def _status_payload_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Converte o estado interno em payload JSON pronto para endpoints de status."""
    logs_list = snapshot["logs"]
    progress_percent, progress_label = _compute_progress(
        logs_list,
        bool(snapshot["running"]),
        snapshot["return_code"],
    )
    return {
        "running": snapshot["running"],
        "started_at": snapshot["started_at"],
        "finished_at": snapshot["finished_at"],
        "return_code": snapshot["return_code"],
        "command": snapshot["command"],
        "logs": "\n".join(logs_list),
        "progress_percent": progress_percent,
        "progress_label": progress_label,
    }


def _run_translation(cmd: list[str], ollama_host: str | None) -> None:
    """Executa o backend de tradução em thread separada e sincroniza estado/logs."""
    try:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if ollama_host:
            env["OLLAMA_HOST"] = ollama_host
        else:
            env.pop("OLLAMA_HOST", None)
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        _stream_process_output(process)

        return_code = process.wait()
        with _state_lock:
            _state["running"] = False
            _state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _state["return_code"] = return_code

        if return_code == 0:
            _append_log("\n[WEB] Tradução finalizada com sucesso.")
        else:
            if _logs_indicate_model_not_found():
                _append_log(
                    "\n[WEB] Modelo informado não está disponível no Ollama. "
                    "Processo encerrado para nova execução com modelo válido."
                )
            else:
                _append_log(f"\n[WEB] Processo finalizou com erro (código {return_code}).")

    except Exception as exc:  # pragma: no cover
        with _state_lock:
            _state["running"] = False
            _state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _state["return_code"] = -1
        _append_log(f"[WEB] Erro ao executar tradução: {exc}")


def _build_command(form: dict[str, str]) -> list[str]:
    """Monta a linha de comando do tradutor com base nas opções recebidas do formulário."""
    cmd = [sys.executable]
    if IS_FROZEN:
        cmd.append("--run-backend")
    else:
        cmd.extend(["-u", str(SCRIPT_PATH)])

    cmd.extend(
        [
            "--input-dir",
            form["input_dir"],
            "--output-dir",
            form["output_dir"],
            "-m",
            form["model"],
            "--batch-size",
            form["batch_size"],
            "--timeout",
            form["timeout"],
        ]
    )

    if form.get("turbo") == "on":
        cmd.append("--turbo")
    if form.get("clear_cache") == "on":
        cmd.append("--clear-cache")
    if form.get("no_cache") == "on":
        cmd.append("--no-cache")

    return cmd


def _list_ass_files(directory: Path) -> list[str]:
    """Lista arquivos `.ass` de uma pasta para exibição na interface web."""
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.ass"))


def _clear_directory_contents(directory: Path) -> int:
    """Remove todo o conteúdo de uma pasta e retorna quantos itens foram apagados."""
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
            removed += 1
        elif item.is_dir():
            shutil.rmtree(item)
            removed += 1
    return removed


@app.route("/", methods=["GET"])
def index() -> str:
    """Renderiza a página inicial com defaults efetivos e token CSRF."""
    defaults = _effective_defaults()
    csrf_token = _generate_csrf()
    return render_template("index.html", defaults=defaults, csrf_token=csrf_token)


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Responde o favicon com 204 para evitar ruído de requisição no log."""
    return "", 204


@app.route("/start", methods=["POST"])
def start_translation():
    """Valida entrada, prepara estado e inicia a tradução assíncrona."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    if not IS_FROZEN and not SCRIPT_PATH.exists():
        return jsonify({"ok": False, "error": "Arquivo translate_ass_fast.py não encontrado."}), 400

    input_dir = request.form.get("input_dir", DEFAULT_FORM_VALUES["input_dir"]).strip()
    output_dir = request.form.get("output_dir", DEFAULT_FORM_VALUES["output_dir"]).strip()
    form_settings = {
        "model": request.form.get("model", DEFAULT_FORM_VALUES["model"]),
        "batch_size": request.form.get("batch_size", DEFAULT_FORM_VALUES["batch_size"]),
        "timeout": request.form.get("timeout", DEFAULT_FORM_VALUES["timeout"]),
        "ollama_mode": request.form.get("ollama_mode", DEFAULT_FORM_VALUES["ollama_mode"]),
        "ollama_endpoint": request.form.get("ollama_endpoint", DEFAULT_FORM_VALUES["ollama_endpoint"]),
    }
    config_ok, normalized_settings, config_error = _normalize_config(form_settings)
    if not config_ok:
        return jsonify({"ok": False, "error": config_error}), 400

    ollama_host = (
        normalized_settings["ollama_endpoint"]
        if normalized_settings["ollama_mode"] == "remote"
        else None
    )

    model_available, model_error = _ensure_ollama_model_available(
        normalized_settings["model"], ollama_host
    )
    if not model_available:
        return jsonify({"ok": False, "error": model_error}), 400

    safe_input = _safe_directory(input_dir, fallback=BASE_DIR / "entrada")
    safe_output = _safe_directory(output_dir, fallback=BASE_DIR / "saida")

    form = {
        "input_dir": str(safe_input),
        "output_dir": str(safe_output),
        "model": normalized_settings["model"],
        "batch_size": normalized_settings["batch_size"],
        "timeout": normalized_settings["timeout"],
        "ollama_mode": normalized_settings["ollama_mode"],
        "ollama_endpoint": normalized_settings["ollama_endpoint"],
        "turbo": request.form.get("turbo", "off"),
        "clear_cache": request.form.get("clear_cache", "off"),
        "no_cache": request.form.get("no_cache", "off"),
    }

    with _state_lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Já existe uma tradução em execução."}), 409

    cmd = _build_command(form)

    with _state_lock:
        _state["running"] = True
        _state["started_at"] = datetime.now().isoformat(timespec="seconds")
        _state["finished_at"] = None
        _state["return_code"] = None
        sanitized_cmd = _sanitize_logs(" ".join(cmd))
        ollama_info = (
            "local"
            if form["ollama_mode"] == "local"
            else form["ollama_endpoint"]
        )
        _state["command"] = sanitized_cmd
        _state["logs"] = [
            "[WEB] Iniciando processo...",
            f"[WEB] Ollama: {ollama_info}",
            f"[WEB] Comando: {sanitized_cmd}",
        ]

    worker = threading.Thread(target=_run_translation, args=(cmd, ollama_host), daemon=True)
    worker.start()

    return jsonify({"ok": True})


@app.route("/upload", methods=["POST"])
def upload_files():
    """Recebe upload de arquivos `.ass`, valida nomes e salva na pasta de entrada."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    safe_input = _safe_directory(
        request.form.get("input_dir", "./entrada"), fallback=BASE_DIR / "entrada"
    )
    uploaded_files = request.files.getlist("ass_files")

    if not uploaded_files:
        return jsonify({"ok": False, "error": "Nenhum arquivo enviado."}), 400

    try:
        safe_input.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Falha ao criar pasta de entrada: {exc}"}), 400

    saved = []
    skipped = []
    for file in uploaded_files:
        filename = _secure_filename(file.filename or "")
        if not filename:
            skipped.append("(sem nome)")
            continue
        if not filename.lower().endswith(".ass"):
            skipped.append(filename)
            continue

        destination = safe_input / filename
        file.save(destination)
        saved.append(filename)

    if not saved:
        return jsonify(
            {
                "ok": False,
                "error": "Nenhum arquivo .ass válido foi enviado.",
                "skipped": skipped,
            }
        ), 400

    return jsonify({"ok": True, "saved": saved, "skipped": skipped})


@app.route("/download-output", methods=["GET"])
def download_output():
    """Compacta os `.ass` processados em memória e entrega um ZIP para download."""
    output_dir = _safe_directory(
        request.args.get("output_dir", "./saida"), fallback=BASE_DIR / "saida"
    )
    if not output_dir.exists() or not output_dir.is_dir():
        return jsonify({"ok": False, "error": "Pasta de saída não encontrada."}), 404

    ass_files = sorted(output_dir.glob("*.ass"))
    if not ass_files:
        return jsonify({"ok": False, "error": "Nenhum arquivo .ass processado encontrado."}), 404

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in ass_files:
            archive.write(path, arcname=path.name)

    memory_file.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"arquivos_processados_{timestamp}.zip",
    )


@app.route("/files", methods=["GET"])
def list_files():
    """Retorna em JSON os arquivos `.ass` encontrados nas pastas de entrada e saída."""
    input_dir = _safe_directory(
        request.args.get("input_dir", "./entrada"), fallback=BASE_DIR / "entrada"
    )
    output_dir = _safe_directory(
        request.args.get("output_dir", "./saida"), fallback=BASE_DIR / "saida"
    )

    payload = {
        "ok": True,
        "input_files": _list_ass_files(input_dir),
        "output_files": _list_ass_files(output_dir),
    }
    return jsonify(payload)


@app.route("/cleanup", methods=["POST"])
def cleanup_previous_work():
    """Limpa entradas, saídas, cache e estado interno quando não há tradução em andamento."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    with _state_lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Não é possível limpar durante uma tradução em execução."}), 409

    input_dir = (BASE_DIR / "entrada").resolve()
    output_dir = (BASE_DIR / "saida").resolve()

    try:
        removed_input = _clear_directory_contents(input_dir)
        removed_output = _clear_directory_contents(output_dir)
        cache_cleared = False
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            cache_cleared = True
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Falha ao limpar pastas: {exc}"}), 500

    with _state_lock:
        _state["running"] = False
        _state["started_at"] = None
        _state["finished_at"] = None
        _state["return_code"] = None
        _state["command"] = None
        _state["logs"] = []

    return jsonify(
        {
            "ok": True,
            "removed_input": removed_input,
            "removed_output": removed_output,
            "cache_cleared": cache_cleared,
        }
    )


@app.route("/status", methods=["GET"])
def status():
    """Exibe o estado atual da tradução em formato JSON para polling tradicional."""
    with _state_lock:
        snapshot = {
            "running": _state["running"],
            "started_at": _state["started_at"],
            "finished_at": _state["finished_at"],
            "return_code": _state["return_code"],
            "command": _state["command"],
            "logs": _state["logs"][:],
        }
    payload = _status_payload_from_snapshot(snapshot)
    return jsonify(payload)


@app.route("/status/stream", methods=["GET"])
def status_stream():
    """Abre stream SSE para enviar atualizações de status em tempo real ao frontend."""
    def _event_stream():
        """Gera eventos SSE quando o payload muda e envia keepalive periódico."""
        last_sent = ""
        keepalive_counter = 0
        while True:
            with _state_lock:
                snapshot = {
                    "running": _state["running"],
                    "started_at": _state["started_at"],
                    "finished_at": _state["finished_at"],
                    "return_code": _state["return_code"],
                    "command": _state["command"],
                    "logs": _state["logs"][:],
                }
            payload = _status_payload_from_snapshot(snapshot)
            serialized = json.dumps(payload, ensure_ascii=False)
            if serialized != last_sent:
                last_sent = serialized
                yield f"data: {serialized}\n\n"
                keepalive_counter = 0
            else:
                keepalive_counter += 1
                if keepalive_counter >= 25:
                    keepalive_counter = 0
                    yield ": keepalive\n\n"
            time.sleep(0.2)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(
        stream_with_context(_event_stream()),
        headers=headers,
        mimetype="text/event-stream",
    )


@app.route("/config/save", methods=["POST"])
def save_config():
    """Valida e persiste as configurações do usuário para reutilização futura."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    incoming = {
        "model": request.form.get("model", ""),
        "batch_size": request.form.get("batch_size", ""),
        "timeout": request.form.get("timeout", ""),
        "ollama_mode": request.form.get("ollama_mode", ""),
        "ollama_endpoint": request.form.get("ollama_endpoint", ""),
    }
    ok, normalized, error = _normalize_config(incoming)
    if not ok:
        return jsonify({"ok": False, "error": error}), 400

    try:
        _save_user_config(normalized)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Falha ao salvar configuração: {exc}"}), 500

    return jsonify(
        {
            "ok": True,
            "message": "Configuração salva com sucesso.",
            "config": normalized,
        }
    )


@app.route("/config/reset", methods=["POST"])
def reset_config():
    """Remove o arquivo de configuração do usuário e devolve os padrões da aplicação."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    try:
        if WEB_CONFIG_PATH.exists():
            WEB_CONFIG_PATH.unlink()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Falha ao resetar configuração: {exc}"}), 500

    default_config = {
        "model": DEFAULT_FORM_VALUES["model"],
        "batch_size": DEFAULT_FORM_VALUES["batch_size"],
        "timeout": DEFAULT_FORM_VALUES["timeout"],
        "ollama_mode": DEFAULT_FORM_VALUES["ollama_mode"],
        "ollama_endpoint": DEFAULT_FORM_VALUES["ollama_endpoint"],
    }
    return jsonify(
        {
            "ok": True,
            "message": "Configuração resetada para os valores padrão.",
            "config": default_config,
        }
    )


@app.route("/ui-heartbeat", methods=["POST"])
def ui_heartbeat():
    """Endpoint chamado pela UI para informar atividade ao watchdog do app empacotado."""
    csrf_error = _require_csrf()
    if csrf_error:
        return csrf_error

    _register_ui_heartbeat()
    return jsonify({"ok": True})


def _run_backend_entrypoint() -> int:
    """Executa o backend de tradução no modo empacotado e devolve código de saída."""
    import asyncio
    import translate_ass_fast

    backend_args = [arg for arg in sys.argv[1:] if arg != "--run-backend"]
    sys.argv = ["translate_ass_fast.py", *backend_args]
    try:
        asyncio.run(translate_ass_fast.main())
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _find_edge_executable() -> str | None:
    """Localiza o executável do Microsoft Edge via PATH ou caminhos padrão do Windows."""
    edge = shutil.which("msedge")
    if edge:
        return edge

    candidates = [
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _wait_for_server_ready(timeout_seconds: float = 12.0) -> bool:
    """Aguarda o servidor Flask responder para só então abrir a interface no navegador."""
    deadline = datetime.now().timestamp() + timeout_seconds
    while datetime.now().timestamp() < deadline:
        try:
            with urlrequest.urlopen(APP_URL, timeout=1.0):
                return True
        except Exception:
            pass
        threading.Event().wait(0.2)
    return False


def _create_edge_temp_profile_dir() -> Path | None:
    """Cria perfil temporário do Edge e remove perfis antigos para evitar acúmulo."""
    try:
        root = Path(tempfile.gettempdir()) / "tradutor_ass_edge_profiles"
        root.mkdir(parents=True, exist_ok=True)

        # Best-effort cleanup of stale folders from older sessions.
        cutoff = time.time() - (3 * 24 * 60 * 60)
        for item in root.glob("profile-*"):
            if not item.is_dir():
                continue
            try:
                if item.stat().st_mtime < cutoff:
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                continue

        return Path(tempfile.mkdtemp(prefix="profile-", dir=str(root)))
    except Exception:
        return None


def _open_ui_on_start() -> None:
    """Abre a interface web ao iniciar a aplicação, priorizando modo app no Edge."""
    if os.environ.get("ASS_TRANSLATOR_NO_BROWSER") == "1":
        return

    def _open():
        """Thread auxiliar que espera o servidor e tenta abrir a URL no navegador."""
        if not _wait_for_server_ready():
            webbrowser.open(APP_URL)
            return

        edge = _find_edge_executable()
        if edge:
            try:
                edge_profile_dir = _create_edge_temp_profile_dir()
                edge_cmd = [
                    edge,
                    f"--app={APP_URL}",
                    "--new-window",
                    "--window-size=1280,860",
                    "--disable-extensions",
                    "--disable-component-extensions-with-background-pages",
                    "--disable-sync",
                    "--no-first-run",
                    "--no-default-browser-check",
                ]
                if edge_profile_dir:
                    edge_cmd.append(f"--user-data-dir={edge_profile_dir}")

                subprocess.Popen(
                    edge_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass

        webbrowser.open(APP_URL)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    if "--run-backend" in sys.argv:
        raise SystemExit(_run_backend_entrypoint())

    # Cleanup leftover work from previous sessions.
    _clear_directory_contents(BASE_DIR / "entrada")
    _clear_directory_contents(BASE_DIR / "saida")
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    _append_log("[WEB] Limpeza do trabalho anterior concluída ao iniciar.")

    _start_ui_watchdog_if_needed()
    _open_ui_on_start()
    app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True)
