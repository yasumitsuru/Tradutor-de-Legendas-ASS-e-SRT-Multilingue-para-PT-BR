from __future__ import annotations

import os
import re
import sys
import json
import shutil
import hashlib
import subprocess
import threading
import zipfile
import time
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from run_manifest import initialize_run_manifest, load_run_outputs
from subtitle_formats import is_supported_subtitle, iter_subtitle_files

try:
    from PySide6.QtCore import (
        Qt,
        QObject,
        QThread,
        Signal,
        QTimer,
        QSaveFile,
        QFileInfo,
    )
    from PySide6.QtGui import (
        QAction,
        QFont,
        QIcon,
        QPalette,
        QColor,
        QDesktopServices,
        QDragEnterEvent,
        QDropEvent,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QPushButton,
        QLineEdit,
        QSpinBox,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QVBoxLayout,
        QGridLayout,
        QFormLayout,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPlainTextEdit,
        QMessageBox,
        QTabWidget,
        QSplitter,
        QFrame,
        QScrollArea,
        QSizePolicy,
        QStyle,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 nao encontrado. Instale com: pip install PySide6\n"
        f"Detalhe: {exc}"
    ) from exc


SOURCE_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False))
if IS_FROZEN:
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = SOURCE_DIR
SCRIPT_PATH = SOURCE_DIR / "translate_ass_fast.py"
CACHE_PATH = BASE_DIR / "translation_cache.json"
WEB_CONFIG_PATH = BASE_DIR / "web_config.json"
ENTRY_DIR = BASE_DIR / "entrada"
OUTPUT_DIR = BASE_DIR / "saida"
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"

DEFAULT_FORM_VALUES: dict[str, str] = {
    "input_dir": str(ENTRY_DIR),
    "output_dir": str(OUTPUT_DIR),
    "model": "qwen2.5:14b",
    "batch_size": "15",
    "timeout": "300",
    "ollama_mode": "local",
    "ollama_endpoint": "",
}
CONFIG_KEYS = ("model", "batch_size", "timeout", "ollama_mode", "ollama_endpoint")


def _run_manifest_path() -> Path:
    return BASE_DIR / ".translation_run_outputs.json"

_ALLOWED_MODEL_RE = re.compile(r"^[a-z0-9._:/@-]+$", re.IGNORECASE)
_PATH_SANITIZER = re.compile(
    r"[A-Z]:\\[^:\"<>|?*\r\n]*|"
    r"(?:(?:\.\.|\.)[\\/][^\s:<>|?*\r\n]+)"
)
_SEASON_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_TOTAL_FILES_RE = re.compile(r"Encontrados\s+(\d+)\s+arquivos\b", re.IGNORECASE)
_FILE_DONE_RE = re.compile(r"Arquivo\s+(?:ASS|SRT)\s+conclu[ií]do", re.IGNORECASE)
_BATCH_PROGRESS_RE = re.compile(r"Batch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _sanitize_logs(text: str) -> str:
    return _PATH_SANITIZER.sub("[redacted]", text)


def _validate_model(name: str) -> bool:
    return bool(name) and bool(_ALLOWED_MODEL_RE.match(name))


def _normalize_ollama_connection(
    mode_raw: str,
    endpoint_raw: str,
) -> tuple[bool, str, str, str | None, str]:
    from urllib import parse as urlparse

    mode = (mode_raw or "local").strip().lower()
    if mode not in {"local", "remote"}:
        return False, "", "", None, "Tipo de conexao Ollama invalido."

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
        return False, "", "", None, "Endpoint Ollama remoto invalido."

    if parsed.scheme not in {"http", "https"}:
        return False, "", "", None, "Use apenas endpoints http:// ou https://."
    if not parsed.hostname:
        return False, "", "", None, "Endpoint Ollama remoto invalido."
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        return False, "", "", None, "Use apenas host e porta (sem path ou query)."

    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return True, "remote", normalized_endpoint, normalized_endpoint, ""


def _normalize_config(form_like: dict[str, str]) -> tuple[bool, dict[str, str], str]:
    model = form_like.get("model", "").strip()
    batch_size = form_like.get("batch_size", "").strip()
    timeout = form_like.get("timeout", "").strip()
    ollama_mode = form_like.get("ollama_mode", "local").strip()
    ollama_endpoint = form_like.get("ollama_endpoint", "").strip()

    if not _validate_model(model):
        return False, {}, "Nome de modelo invalido."

    try:
        batch_num = int(batch_size)
    except ValueError:
        return False, {}, "Batch size deve ser um numero inteiro."
    if batch_num < 1:
        return False, {}, "Batch size deve ser maior ou igual a 1."

    try:
        timeout_num = int(timeout)
    except ValueError:
        return False, {}, "Timeout deve ser um numero inteiro."
    if timeout_num < 10:
        return False, {}, "Timeout deve ser maior ou igual a 10 segundos."

    conn_ok, normalized_mode, normalized_endpoint, _, conn_error = (
        _normalize_ollama_connection(ollama_mode, ollama_endpoint)
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
    payload = {key: config[key] for key in CONFIG_KEYS}
    with WEB_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _effective_defaults() -> dict[str, str]:
    defaults = dict(DEFAULT_FORM_VALUES)
    defaults.update(_load_user_config())
    return defaults


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status_code == 404:
        return True
    return "model" in text and "not found" in text


def _ensure_ollama_model_available(
    model_name: str, ollama_host: str | None = None
) -> tuple[bool, str]:
    try:
        import ollama
    except Exception as exc:
        return False, f"Nao foi possivel importar o modulo ollama: {exc}"

    candidate_hosts: list[str] = []
    if ollama_host:
        candidate_hosts.append(ollama_host)
    else:
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
                    f'Modelo "{model_name}" nao esta disponivel no Ollama{endpoint_hint}. '
                    f'Instale antes com: ollama pull {model_name}',
                )
            last_exc = exc
            last_host = host

    where = f' no endpoint "{last_host}"' if last_host else ""
    return False, f"Nao foi possivel validar o modelo no Ollama{where}: {last_exc}"


def _clear_directory_contents(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in directory.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed += 1
        except OSError as exc:
            raise OSError(f"Falha ao remover {item}: {exc}") from exc
    return removed


def _build_command(form: dict[str, str]) -> list[str]:
    cmd: list[str] = []
    if IS_FROZEN:
        cmd.extend([sys.executable, "--run-backend"])
    else:
        cmd.extend([sys.executable, "-u", str(SCRIPT_PATH)])

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
    if form.get("allow_original_fallback") == "on":
        cmd.append("--allow-original-fallback")
    cmd.extend(["--result-manifest", str(_run_manifest_path())])

    return cmd


def _compute_progress(
    log_lines: list[str],
    running: bool,
    return_code: int | None,
) -> tuple[int, str]:
    if not log_lines:
        if running:
            return 0, "0%"
        if return_code == 0:
            return 100, "100% (concluido)"
        return 0, "0%"

    tqdm_percent: int | None = None
    for line in log_lines:
        if "Processando legendas" not in line and "Processando Temporada" not in line:
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
            return 100, "100% (concluido)"
        return percent, f"{percent}% (finalizado com erro)"

    total_files: int | None = None
    completed_files = 0
    for line in log_lines:
        total_match = _TOTAL_FILES_RE.search(line)
        if total_match:
            try:
                total_files = int(total_match.group(1))
            except ValueError:
                total_files = None
        if _FILE_DONE_RE.search(line):
            completed_files += 1

    if total_files and total_files > 0:
        completed_clamped = min(completed_files, total_files)
        percent = round((completed_clamped / total_files) * 100)
        percent = max(0, min(100, int(percent)))
        if running:
            percent = min(percent, 99)
        elif return_code == 0:
            return 100, "100% (concluido)"

        suffix = ""
        if not running and return_code not in (None, 0):
            suffix = " (finalizado com erro)"
        label = f"{percent}% ({completed_clamped}/{total_files} arquivos){suffix}"
        return percent, label

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
        return 100, "100% (concluido)"
    return 0, "0%"


class _StreamSignal(QObject):
    line = Signal(str)
    finished = Signal(int)


class TranslationWorker(QThread):
    line = Signal(str)
    finished_run = Signal(int)

    def __init__(self, cmd: list[str], ollama_host: str | None, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._ollama_host = ollama_host
        self._process: subprocess.Popen[str] | None = None

    def run(self) -> None:
        try:
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if self._ollama_host:
                env["OLLAMA_HOST"] = self._ollama_host
            else:
                env.pop("OLLAMA_HOST", None)

            self._process = subprocess.Popen(
                self._cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            buffer_chars: list[str] = []
            assert self._process.stdout is not None
            while True:
                char = self._process.stdout.read(1)
                if char == "":
                    if buffer_chars:
                        self.line.emit("".join(buffer_chars))
                    break
                if char in ("\n", "\r"):
                    if buffer_chars:
                        self.line.emit("".join(buffer_chars))
                        buffer_chars = []
                    continue
                buffer_chars.append(char)

            return_code = self._process.wait()
            self.finished_run.emit(return_code)
        except Exception as exc:
            self.line.emit(f"[GUI] Erro ao executar traducao: {exc}")
            self.finished_run.emit(-1)

    def terminate_process(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tradutor de legendas ASS/SRT - Ingles para Portugues (by Yasu)")
        self.resize(1180, 760)

        self._worker: Optional[TranslationWorker] = None
        self._state_lock = threading.Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "logs": [],
        }

        ENTRY_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._apply_stylesheet()
        self.load_defaults_into_form()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start()

        self._files_timer = QTimer(self)
        self._files_timer.setInterval(3000)
        self._files_timer.timeout.connect(self.refresh_files)
        self._files_timer.start()

        self.refresh_files()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        title = QLabel("Tradutor de legendas ASS/SRT do Ingles para o Portugues Brasil")
        title.setObjectName("title")
        root.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        upload_group = QGroupBox("Legendas de entrada (ASS/SRT)")
        upload_layout = QGridLayout(upload_group)
        upload_layout.setContentsMargins(12, 14, 12, 12)
        upload_layout.setSpacing(8)

        self.btn_upload = QPushButton("Selecionar arquivos ASS/SRT")
        self.btn_upload.setObjectName("upload")
        self.btn_upload.clicked.connect(self.on_upload_clicked)
        upload_layout.addWidget(self.btn_upload, 0, 0)

        self.btn_clear_input = QPushButton("Limpar pasta de entrada")
        self.btn_clear_input.setObjectName("dangerSoft")
        self.btn_clear_input.clicked.connect(self.on_clear_input_clicked)
        upload_layout.addWidget(self.btn_clear_input, 0, 1)

        self.upload_meta_label = QLabel("Nenhum upload feito.")
        self.upload_meta_label.setObjectName("meta")
        upload_layout.addWidget(self.upload_meta_label, 1, 0, 1, 2)

        self.input_files_list = QListWidget()
        self.input_files_list.setObjectName("fileList")
        upload_layout.addWidget(QLabel("Arquivos de entrada:"), 2, 0, 1, 2)
        upload_layout.addWidget(self.input_files_list, 3, 0, 1, 2)
        left_layout.addWidget(upload_group)

        config_group = QGroupBox("Configuracao")
        config_form = QFormLayout(config_group)
        config_form.setContentsMargins(12, 14, 12, 12)
        config_form.setSpacing(8)

        self.model_input = QLineEdit(DEFAULT_FORM_VALUES["model"])
        config_form.addRow("Modelo Ollama", self.model_input)

        connection_row = QHBoxLayout()
        self.ollama_mode_combo = QComboBox()
        self.ollama_mode_combo.addItem("Local", "local")
        self.ollama_mode_combo.addItem("IP/URL", "remote")
        self.ollama_mode_combo.currentIndexChanged.connect(self._update_endpoint_visibility)
        connection_row.addWidget(QLabel("Conexao Ollama"))
        connection_row.addWidget(self.ollama_mode_combo, 1)
        config_form.addRow(connection_row)

        self.ollama_endpoint_input = QLineEdit()
        self.ollama_endpoint_input.setPlaceholderText("192.168.1.10:11434 ou https://ollama.seudns.com")
        config_form.addRow("IP:Porta ou URL/DNS", self.ollama_endpoint_input)

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1000)
        self.batch_input.setValue(int(DEFAULT_FORM_VALUES["batch_size"]))
        config_form.addRow("Batch size", self.batch_input)

        self.timeout_input = QSpinBox()
        self.timeout_input.setRange(10, 100000)
        self.timeout_input.setSingleStep(10)
        self.timeout_input.setValue(int(DEFAULT_FORM_VALUES["timeout"]))
        config_form.addRow("Timeout (segundos)", self.timeout_input)

        checks_row = QHBoxLayout()
        self.turbo_check = QCheckBox("Modo turbo")
        self.clear_cache_check = QCheckBox("Limpar cache antes")
        self.no_cache_check = QCheckBox("Desativar cache")
        checks_row.addWidget(self.turbo_check)
        checks_row.addWidget(self.clear_cache_check)
        checks_row.addWidget(self.no_cache_check)
        checks_row.addStretch(1)
        config_form.addRow(checks_row)

        self.allow_original_fallback_check = QCheckBox(
            "Permitir manter texto original quando a tradução falhar"
        )
        self.allow_original_fallback_check.setChecked(False)
        config_form.addRow(self.allow_original_fallback_check)
        fallback_warning = QLabel(
            "Atenção: isso pode gerar legendas misturando PT-BR com o idioma original."
        )
        fallback_warning.setWordWrap(True)
        fallback_warning.setStyleSheet("color:#8a4b08; font-size:12px;")
        config_form.addRow(fallback_warning)

        config_btns = QHBoxLayout()
        self.btn_save_config = QPushButton("Salvar Configuracao")
        self.btn_save_config.setObjectName("secondary")
        self.btn_save_config.clicked.connect(self.on_save_config_clicked)
        config_btns.addWidget(self.btn_save_config)
        self.btn_reset_config = QPushButton("Resetar Configuracao")
        self.btn_reset_config.setObjectName("secondary")
        self.btn_reset_config.clicked.connect(self.on_reset_config_clicked)
        config_btns.addWidget(self.btn_reset_config)
        config_form.addRow(config_btns)

        left_layout.addWidget(config_group)

        actions_row = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar Traducao")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.on_start_clicked)
        actions_row.addWidget(self.btn_start, 2)
        self.btn_stop = QPushButton("Cancelar")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        actions_row.addWidget(self.btn_stop, 1)
        left_layout.addLayout(actions_row)

        left_output_group = QGroupBox("Legendas processadas (ASS/SRT)")
        left_output_layout = QVBoxLayout(left_output_group)
        left_output_layout.setContentsMargins(12, 14, 12, 12)
        left_output_layout.setSpacing(8)
        self.output_files_list = QListWidget()
        self.output_files_list.setObjectName("fileList")
        left_output_layout.addWidget(self.output_files_list)

        download_row = QHBoxLayout()
        self.btn_download = QPushButton("Baixar processados (ZIP)")
        self.btn_download.setObjectName("secondary")
        self.btn_download.clicked.connect(self.on_download_clicked)
        download_row.addWidget(self.btn_download)
        self.btn_open_output = QPushButton("Abrir pasta de saida")
        self.btn_open_output.setObjectName("secondary")
        self.btn_open_output.clicked.connect(self.on_open_output_clicked)
        download_row.addWidget(self.btn_open_output)
        left_output_layout.addLayout(download_row)
        left_layout.addWidget(left_output_group)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(12, 14, 12, 12)
        status_layout.setSpacing(8)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("0%")
        self.progress_label.setMinimumWidth(140)
        progress_row.addWidget(self.progress_label)
        status_layout.addLayout(progress_row)

        badge_row = QHBoxLayout()
        self.status_badge = QLabel("Parado")
        self.status_badge.setObjectName("badgeIdle")
        badge_row.addWidget(self.status_badge)
        badge_row.addStretch(1)
        self.status_meta = QLabel("Sem execucao.")
        self.status_meta.setObjectName("meta")
        badge_row.addWidget(self.status_meta, 1)
        status_layout.addLayout(badge_row)

        right_layout.addWidget(status_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 14, 12, 12)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        self.log_box.setMaximumBlockCount(4000)
        log_layout.addWidget(self.log_box)

        log_btns = QHBoxLayout()
        self.btn_clear_logs = QPushButton("Limpar log")
        self.btn_clear_logs.setObjectName("secondary")
        self.btn_clear_logs.clicked.connect(self.log_box.clear)
        log_btns.addWidget(self.btn_clear_logs)
        self.btn_save_logs = QPushButton("Salvar log")
        self.btn_save_logs.setObjectName("secondary")
        self.btn_save_logs.clicked.connect(self.on_save_logs_clicked)
        log_btns.addWidget(self.btn_save_logs)
        log_btns.addStretch(1)
        log_layout.addLayout(log_btns)

        right_layout.addWidget(log_group, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)

        menu = self.menuBar()
        file_menu = menu.addMenu("Arquivo")
        act_cleanup = QAction("Limpeza do trabalho anterior", self)
        act_cleanup.triggered.connect(self.on_cleanup_clicked)
        file_menu.addAction(act_cleanup)
        file_menu.addSeparator()
        act_quit = QAction("Sair", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = menu.addMenu("Ajuda")
        act_about = QAction("Sobre", self)
        act_about.triggered.connect(self.on_about_clicked)
        help_menu.addAction(act_about)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #f3f7f5; color: #18231f;
                font-family: "Segoe UI","Trebuchet MS",sans-serif; font-size: 14px; }
            QLabel#title { font-size: 18px; font-weight: 700; color: #0f1d19; }
            QGroupBox { background: #ffffff; border: 1px solid #d5e1db; border-radius: 12px;
                margin-top: 12px; padding: 10px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #29483d; }
            QLabel#meta { color: #5f726a; font-size: 12px; }
            QLineEdit, QSpinBox, QComboBox { background: #fff; border: 1px solid #d5e1db;
                border-radius: 8px; padding: 7px 10px; selection-background-color: #0e8b61; }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #0e8b61; }
            QListWidget#fileList { background: #fff; border: 1px solid #d5e1db; border-radius: 8px;
                padding: 4px; }
            QListWidget#fileList::item { padding: 4px 2px; border-bottom: 1px dashed #edf2ef; }
            QPushButton { border: none; border-radius: 10px; padding: 9px 14px; color: #fff;
                font-weight: 700; background: #0e8b61; }
            QPushButton:hover { background: #0a6a49; }
            QPushButton:disabled { background: #b9c7c1; color: #ecf2ef; }
            QPushButton#primary { background: #0e8b61; }
            QPushButton#secondary { background: #1f6d9c; }
            QPushButton#secondary:hover { background: #185880; }
            QPushButton#upload { background: #f57c00; }
            QPushButton#upload:hover { background: #d96d00; }
            QPushButton#danger { background: #b93838; }
            QPushButton#danger:hover { background: #9c2e2e; }
            QPushButton#dangerSoft { background: #d9848a; color: #2a1416; }
            QPushButton#dangerSoft:hover { background: #c46c73; }
            QProgressBar { background: #edf4f1; border: 1px solid #cfe0d7; border-radius: 6px;
                height: 14px; text-align: center; }
            QProgressBar::chunk { background: #1f6dbe; border-radius: 6px; }
            QLabel#badgeIdle { background: #d9f4e8; color: #235a45; padding: 3px 10px;
                border-radius: 999px; font-weight: 700; }
            QPlainTextEdit#logBox { background: #0f1d19; color: #d7f0e5; border: 1px solid #14342a;
                border-radius: 10px; font-family: "Cascadia Mono",Consolas,monospace;
                font-size: 12px; padding: 8px; }
            QMenuBar { background: #0f1d19; color: #d7f0e5; }
            QMenuBar::item:selected { background: #14342a; }
            QMenu { background: #ffffff; color: #18231f; border: 1px solid #d5e1db; }
            QMenu::item:selected { background: #e6f3ee; }
            """
        )

    def load_defaults_into_form(self) -> None:
        defaults = _effective_defaults()
        self.model_input.setText(defaults.get("model", DEFAULT_FORM_VALUES["model"]))
        self.batch_input.setValue(int(defaults.get("batch_size", DEFAULT_FORM_VALUES["batch_size"])))
        self.timeout_input.setValue(int(defaults.get("timeout", DEFAULT_FORM_VALUES["timeout"])))
        mode = defaults.get("ollama_mode", "local")
        idx = 0 if mode != "remote" else 1
        self.ollama_mode_combo.setCurrentIndex(idx)
        self.ollama_endpoint_input.setText(defaults.get("ollama_endpoint", ""))
        self._update_endpoint_visibility()

    def _update_endpoint_visibility(self) -> None:
        is_remote = self.ollama_mode_combo.currentData() == "remote"
        self.ollama_endpoint_input.setEnabled(is_remote)
        if not is_remote:
            self.ollama_endpoint_input.clear()

    def _append_log(self, line: str) -> None:
        line = _sanitize_logs(line)
        if not line.strip():
            return
        self.log_box.appendPlainText(line)

    def _set_running_style(self, running: bool) -> None:
        if running:
            self.status_badge.setText("Executando")
            self.status_badge.setObjectName("badgeIdle")
            self.status_badge.setStyleSheet(
                "background:#ffe3c7; color:#5f2b00; padding:3px 10px; border-radius:999px; font-weight:700;"
            )
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background:#1f6dbe; border-radius:6px; }")
        else:
            self.status_badge.setText("Parado")
            self.status_badge.setStyleSheet(
                "background:#d9f4e8; color:#235a45; padding:3px 10px; border-radius:999px; font-weight:700;"
            )
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _collect_form(self) -> dict[str, str]:
        return {
            "model": self.model_input.text().strip(),
            "batch_size": str(self.batch_input.value()),
            "timeout": str(self.timeout_input.value()),
            "ollama_mode": self.ollama_mode_combo.currentData(),
            "ollama_endpoint": self.ollama_endpoint_input.text().strip(),
        }

    def on_upload_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar arquivos ASS ou SRT",
            str(ENTRY_DIR),
            "Legendas suportadas (*.ass *.ASS *.srt *.SRT)",
        )
        if not files:
            return
        try:
            ENTRY_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Erro", f"Falha ao criar pasta de entrada: {exc}")
            return

        saved = []
        skipped = []
        for src in files:
            path = Path(src)
            if not is_supported_subtitle(path):
                skipped.append(path.name)
                continue
            try:
                shutil.copy2(path, ENTRY_DIR / path.name)
                saved.append(path.name)
            except Exception as exc:
                skipped.append(f"{path.name} ({exc})")

        self.upload_meta_label.setText(
            f"Upload concluido. Enviados: {len(saved)} | Ignorados: {len(skipped)}"
        )
        self.refresh_files()

    def on_clear_input_clicked(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Limpar entrada",
            "Deseja remover todos os arquivos da pasta de entrada?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        removed = _clear_directory_contents(ENTRY_DIR)
        self.upload_meta_label.setText(f"Pasta de entrada limpa. Itens removidos: {removed}")
        self.refresh_files()

    def on_start_clicked(self) -> None:
        if not IS_FROZEN and not SCRIPT_PATH.exists():
            QMessageBox.critical(
                self, "Erro", "Arquivo translate_ass_fast.py nao encontrado."
            )
            return

        with self._state_lock:
            if self._state["running"]:
                QMessageBox.warning(self, "Aviso", "Ja existe uma traducao em execucao.")
                return

        if not iter_subtitle_files(ENTRY_DIR):
            QMessageBox.information(
                self, "Nenhum arquivo", "Adicione pelo menos um arquivo ASS ou SRT antes de iniciar."
            )
            return

        form = self._collect_form()
        ok, normalized, error = _normalize_config(form)
        if not ok:
            QMessageBox.warning(self, "Configuracao invalida", error)
            return

        ollama_host = (
            normalized["ollama_endpoint"]
            if normalized["ollama_mode"] == "remote"
            else None
        )

        self._append_log("[GUI] Validando modelo no Ollama...")
        QApplication.processEvents()
        available, model_error = _ensure_ollama_model_available(normalized["model"], ollama_host)
        if not available:
            QMessageBox.critical(self, "Modelo indisponivel", model_error)
            self._append_log(f"[GUI] {model_error}")
            return

        full_form = {
            "input_dir": str(ENTRY_DIR),
            "output_dir": str(OUTPUT_DIR),
            "model": normalized["model"],
            "batch_size": normalized["batch_size"],
            "timeout": normalized["timeout"],
            "ollama_mode": normalized["ollama_mode"],
            "ollama_endpoint": normalized["ollama_endpoint"],
            "turbo": "on" if self.turbo_check.isChecked() else "off",
            "clear_cache": "on" if self.clear_cache_check.isChecked() else "off",
            "no_cache": "on" if self.no_cache_check.isChecked() else "off",
            "allow_original_fallback": (
                "on" if self.allow_original_fallback_check.isChecked() else "off"
            ),
        }

        try:
            initialize_run_manifest(_run_manifest_path())
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Falha ao iniciar execução",
                f"Não foi possível criar o manifesto de outputs: {exc}",
            )
            return
        cmd = _build_command(full_form)

        with self._state_lock:
            self._state["running"] = True
            self._state["started_at"] = datetime.now().isoformat(timespec="seconds")
            self._state["finished_at"] = None
            self._state["return_code"] = None
            self._state["logs"] = [
                "[GUI] Iniciando processo...",
                f"[GUI] Ollama: {normalized['ollama_endpoint'] if normalized['ollama_mode'] == 'remote' else 'local'}",
                f"[GUI] Comando: {_sanitize_logs(' '.join(cmd))}",
            ]

        self.log_box.clear()
        for initial in self._state["logs"]:
            self._append_log(initial)
        self._set_running_style(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")

        self._worker = TranslationWorker(cmd, ollama_host, self)
        self._worker.line.connect(self._on_worker_line)
        self._worker.finished_run.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_line(self, line: str) -> None:
        with self._state_lock:
            self._state["logs"].append(line)
            if len(self._state["logs"]) > 4000:
                self._state["logs"] = self._state["logs"][-4000:]
        self._append_log(line)

    def _on_worker_finished(self, return_code: int) -> None:
        with self._state_lock:
            self._state["running"] = False
            self._state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self._state["return_code"] = return_code

        if return_code == 0:
            self._append_log("\n[GUI] Traducao finalizada com sucesso.")
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background:#0e8b61; border-radius:6px; }")
            self.progress_bar.setValue(100)
            self.progress_label.setText("100% (concluido)")
        else:
            self._append_log(f"\n[GUI] Processo finalizou com erro (codigo {return_code}).")
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background:#b93838; border-radius:6px; }")

        self._set_running_style(False)
        self.refresh_files()

    def on_stop_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            confirm = QMessageBox.question(
                self,
                "Cancelar traducao",
                "Deseja cancelar a traducao em execucao? O processo sera encerrado.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            try:
                self._worker.terminate_process()
            except OSError as exc:
                QMessageBox.warning(self, "Cancelar traducao", f"Falha ao encerrar processo: {exc}")

    def on_download_clicked(self) -> None:
        subtitle_files = load_run_outputs(_run_manifest_path(), OUTPUT_DIR)
        if not subtitle_files:
            QMessageBox.information(
                self, "Download", "Nenhum arquivo ASS ou SRT processado encontrado."
            )
            return

        default_name = f"arquivos_processados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        dest, _ = QFileDialog.getSaveFileName(
            self, "Salvar ZIP", default_name, "Arquivo ZIP (*.zip)"
        )
        if not dest:
            return

        try:
            with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in subtitle_files:
                    archive.write(path, arcname=path.name)
            QMessageBox.information(self, "Download", f"ZIP salvo em:\n{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao gerar ZIP: {exc}")

    def on_open_output_clicked(self) -> None:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl("file:///" + str(OUTPUT_DIR.resolve()).replace("\\", "/"))
        except Exception as exc:
            QMessageBox.warning(self, "Abrir pasta", f"Nao foi possivel abrir a pasta: {exc}")

    def on_save_logs_clicked(self) -> None:
        default_name = f"log_traducao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        dest, _ = QFileDialog.getSaveFileName(self, "Salvar log", default_name, "Texto (*.txt)")
        if not dest:
            return
        try:
            Path(dest).write_text(self.log_box.toPlainText(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao salvar log: {exc}")

    def on_save_config_clicked(self) -> None:
        form = self._collect_form()
        ok, normalized, error = _normalize_config(form)
        if not ok:
            QMessageBox.warning(self, "Configuracao invalida", error)
            return
        try:
            _save_user_config(normalized)
            QMessageBox.information(self, "Configuracao", "Configuracao salva com sucesso.")
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao salvar configuracao: {exc}")

    def on_reset_config_clicked(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Resetar configuracao",
            "Deseja resetar os campos de configuracao para o padrao?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            if WEB_CONFIG_PATH.exists():
                WEB_CONFIG_PATH.unlink()
        except Exception as exc:
            QMessageBox.warning(self, "Erro", f"Falha ao resetar configuracao: {exc}")
            return
        self.load_defaults_into_form()
        QMessageBox.information(self, "Configuracao", "Configuracao resetada para o padrao.")

    def on_cleanup_clicked(self) -> None:
        with self._state_lock:
            if self._state["running"]:
                QMessageBox.warning(
                    self, "Aviso", "Nao e possivel limpar durante uma traducao em execucao."
                )
                return

        confirm = QMessageBox.question(
            self,
            "Limpeza",
            "Deseja limpar as pastas de entrada, saida e o cache?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            removed_input = _clear_directory_contents(ENTRY_DIR)
            removed_output = _clear_directory_contents(OUTPUT_DIR)
            cache_cleared = False
            if CACHE_PATH.exists():
                CACHE_PATH.unlink()
                cache_cleared = True
            _run_manifest_path().unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao limpar pastas: {exc}")
            return

        with self._state_lock:
            self._state["running"] = False
            self._state["started_at"] = None
            self._state["finished_at"] = None
            self._state["return_code"] = None
            self._state["logs"] = []

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        cache_txt = "sim" if cache_cleared else "nao"
        QMessageBox.information(
            self,
            "Limpeza",
            f"Entrada removidos: {removed_input}\nSaida removidos: {removed_output}\nCache limpo: {cache_txt}",
        )
        self.refresh_files()

    def on_about_clicked(self) -> None:
        QMessageBox.about(
            self,
            "Sobre",
            "Tradutor de legendas ASS/SRT do Ingles para o Portugues Brasil\n"
            "Desenvolvido por Yasu\n\n"
            "Interface grafica (PySide6) para o backend translate_ass_fast.py,\n"
            "usando modelos locais via Ollama.",
        )

    def refresh_files(self) -> None:
        self.input_files_list.clear()
        self.output_files_list.clear()
        if ENTRY_DIR.exists():
            for name in (path.name for path in iter_subtitle_files(ENTRY_DIR)):
                self.input_files_list.addItem(QListWidgetItem(name))
        if OUTPUT_DIR.exists():
            for name in (
                path.name for path in load_run_outputs(_run_manifest_path(), OUTPUT_DIR)
            ):
                self.output_files_list.addItem(QListWidgetItem(name))
        if self.input_files_list.count() == 0:
            self.input_files_list.addItem(QListWidgetItem("Nenhum arquivo encontrado."))
        if self.output_files_list.count() == 0:
            self.output_files_list.addItem(QListWidgetItem("Nenhum arquivo processado."))

    def _refresh_status(self) -> None:
        with self._state_lock:
            running = self._state["running"]
            started_at = self._state["started_at"]
            finished_at = self._state["finished_at"]
            return_code = self._state["return_code"]
            logs = list(self._state["logs"])

        percent, label = _compute_progress(logs, running, return_code)
        if running:
            self.progress_bar.setValue(percent)
            self.progress_label.setText(label)
        elif return_code is None:
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")

        if running:
            self.status_meta.setText(f"Iniciado: {started_at or '-'} | em andamento")
        elif finished_at:
            code = "-" if return_code is None else return_code
            self.status_meta.setText(
                f"Iniciado: {started_at or '-'} | Finalizado: {finished_at} | Codigo: {code}"
            )
        else:
            self.status_meta.setText("Sem execucao.")

    def closeEvent(self, event) -> None:
        with self._state_lock:
            running = self._state["running"]
        if running:
            confirm = QMessageBox.question(
                self,
                "Sair",
                "Uma traducao esta em execucao. Encerrar o aplicativo pode interromper o processo. Continuar?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                event.ignore()
                return
        if self._worker and self._worker.isRunning():
            try:
                self._worker.requestInterruption()
                self._worker.terminate()
            except RuntimeError as exc:
                self._append_log(f"[GUI] Falha ao encerrar worker durante a saída: {exc}")
        event.accept()


def main() -> int:
    if "--run-backend" in sys.argv:
        import asyncio
        import translate_ass_fast

        backend_args = [arg for arg in sys.argv[1:] if arg != "--run-backend"]
        sys.argv = ["translate_ass_fast.py", *backend_args]
        try:
            asyncio.run(translate_ass_fast.main())
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("TradutorASS-GUI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
