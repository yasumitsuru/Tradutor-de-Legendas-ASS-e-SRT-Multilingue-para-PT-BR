from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import parse as urlparse

import flet as ft


APP_TITLE = "Tradutor de arquivos .ass - Inglês para Português"
ACCENT = "#0F766E"
ACCENT_DARK = "#115E59"
BLUE = "#2563EB"
ORANGE = "#EA580C"
RED = "#B91C1C"
BG = "#F4F7F6"
CARD_BG = "#FFFFFF"
TEXT = "#17211D"
MUTED = "#64748B"
BORDER = "#DDE7E2"
LOG_BG = "#0B1713"
LOG_TEXT = "#D8F3E8"
COMPACT_BREAKPOINT = 1100
MAX_CONTENT_WIDTH = 1920

SOURCE_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else SOURCE_DIR
SCRIPT_PATH = SOURCE_DIR / "translate_ass_fast.py"
ENTRY_DIR = BASE_DIR / "entrada"
OUTPUT_DIR = BASE_DIR / "saida"
CACHE_PATH = BASE_DIR / "translation_cache.json"
CONFIG_PATH = BASE_DIR / "web_config.json"
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"

DEFAULT_FORM_VALUES: dict[str, str] = {
    "model": "qwen2.5:14b",
    "batch_size": "15",
    "timeout": "300",
    "ollama_mode": "local",
    "ollama_endpoint": "",
}
CONFIG_KEYS = ("model", "batch_size", "timeout", "ollama_mode", "ollama_endpoint")

_ALLOWED_MODEL_RE = re.compile(r"^[a-z0-9._:/@-]+$", re.IGNORECASE)
_PATH_SANITIZER = re.compile(
    r"[A-Z]:\\[^:\"<>|?*\r\n]*|"
    r"(?:(?:\.\.|\.)[\\/][^\s:<>|?*\r\n]+)"
)
_SEASON_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_TOTAL_FILES_RE = re.compile(r"Encontrados\s+(\d+)\s+arquivos\s+\.ass", re.IGNORECASE)
_EPISODE_DONE_RE = re.compile(r"epis.{0,3}dio.+conclu", re.IGNORECASE)
_BATCH_PROGRESS_RE = re.compile(r"Batch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _sanitize_logs(text: str) -> str:
    return _PATH_SANITIZER.sub("[caminho oculto]", text)


def _normalize_ollama_connection(
    mode_raw: str,
    endpoint_raw: str,
) -> tuple[bool, str, str, str | None, str]:
    mode = (mode_raw or "local").strip().lower()
    if mode not in {"local", "remote"}:
        return False, "", "", None, "Tipo de conexão Ollama inválido."

    endpoint = (endpoint_raw or "").strip()
    if mode == "local":
        return True, "local", "", None, ""
    if not endpoint:
        return False, "", "", None, "Informe IP:porta ou URL do Ollama remoto."

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
        return False, "", "", None, "Use somente host e porta, sem caminho ou consulta."

    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return True, "remote", normalized_endpoint, normalized_endpoint, ""


def _normalize_config(form: dict[str, str]) -> tuple[bool, dict[str, str], str]:
    model = form.get("model", "").strip()
    if not model or not _ALLOWED_MODEL_RE.match(model):
        return False, {}, "Nome de modelo inválido."

    try:
        batch_size = int(form.get("batch_size", ""))
    except ValueError:
        return False, {}, "Batch size deve ser um número inteiro."
    if not 1 <= batch_size <= 1000:
        return False, {}, "Batch size deve ficar entre 1 e 1000."

    try:
        timeout = int(form.get("timeout", ""))
    except ValueError:
        return False, {}, "Timeout deve ser um número inteiro."
    if not 10 <= timeout <= 100000:
        return False, {}, "Timeout deve ficar entre 10 e 100000 segundos."

    ok, mode, endpoint, _, error = _normalize_ollama_connection(
        form.get("ollama_mode", "local"),
        form.get("ollama_endpoint", ""),
    )
    if not ok:
        return False, {}, error

    return True, {
        "model": model,
        "batch_size": str(batch_size),
        "timeout": str(timeout),
        "ollama_mode": mode,
        "ollama_endpoint": endpoint,
    }, ""


def _load_user_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        values = {key: str(raw.get(key, "")).strip() for key in CONFIG_KEYS}
        ok, normalized, _ = _normalize_config(values)
        return normalized if ok else {}
    except Exception:
        return {}


def _save_user_config(config: dict[str, str]) -> None:
    payload = {key: config[key] for key in CONFIG_KEYS}
    CONFIG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _effective_defaults() -> dict[str, str]:
    values = dict(DEFAULT_FORM_VALUES)
    values.update(_load_user_config())
    return values


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 404 or ("model" in message and "not found" in message)


def _ensure_ollama_model_available(
    model_name: str,
    ollama_host: str | None = None,
) -> tuple[bool, str]:
    try:
        import ollama
    except Exception as exc:
        return False, f"Não foi possível carregar o cliente Ollama: {exc}"

    candidate_hosts: list[str] = [ollama_host or LOCAL_OLLAMA_HOST]
    if not ollama_host:
        env_host = (os.environ.get("OLLAMA_HOST") or "").strip()
        if env_host and not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", env_host):
            env_host = f"http://{env_host}"
        if env_host and env_host not in candidate_hosts:
            candidate_hosts.append(env_host)

    last_exc: Exception | None = None
    last_host = candidate_hosts[-1]
    for host in candidate_hosts:
        try:
            ollama.Client(host=host).show(model_name)
            return True, ""
        except Exception as exc:
            if _is_model_not_found_error(exc):
                return False, (
                    f'O modelo "{model_name}" não está disponível em {host}. '
                    f"Instale-o com: ollama pull {model_name}"
                )
            last_exc = exc
            last_host = host
    return False, f"Não foi possível acessar o Ollama em {last_host}: {last_exc}"


def _clear_directory_contents(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        removed += 1
    return removed


def _build_command(form: dict[str, str]) -> list[str]:
    if IS_FROZEN:
        cmd = [sys.executable, "--run-backend"]
    else:
        cmd = [sys.executable, "-u", str(SCRIPT_PATH)]

    cmd.extend(
        [
            "--input-dir",
            str(ENTRY_DIR),
            "--output-dir",
            str(OUTPUT_DIR),
            "--model",
            form["model"],
            "--batch-size",
            form["batch_size"],
            "--timeout",
            form["timeout"],
        ]
    )
    if form.get("turbo"):
        cmd.append("--turbo")
    if form.get("clear_cache"):
        cmd.append("--clear-cache")
    if form.get("no_cache"):
        cmd.append("--no-cache")
    return cmd


def _compute_progress(
    log_lines: list[str],
    running: bool,
    return_code: int | None,
) -> tuple[int, str]:
    if not log_lines:
        return (0, "0%") if return_code != 0 else (100, "100% — concluído")

    season_percent: int | None = None
    for line in log_lines:
        if "Processando Temporada" in line:
            match = _SEASON_TQDM_PERCENT_RE.search(line)
            if match:
                season_percent = max(0, min(100, int(match.group(1))))
    if season_percent is not None:
        percent = min(season_percent, 99) if running else season_percent
        if return_code == 0:
            return 100, "100% — concluído"
        suffix = " — finalizado com erro" if not running else ""
        return percent, f"{percent}%{suffix}"

    total_files: int | None = None
    completed_files = 0
    for line in log_lines:
        total_match = _TOTAL_FILES_RE.search(line)
        if total_match:
            total_files = int(total_match.group(1))
        if _EPISODE_DONE_RE.search(line):
            completed_files += 1
    if total_files:
        completed = min(completed_files, total_files)
        percent = round(completed / total_files * 100)
        if running:
            percent = min(percent, 99)
        if return_code == 0:
            return 100, "100% — concluído"
        suffix = " — finalizado com erro" if not running else ""
        return percent, f"{percent}% — {completed}/{total_files} episódios{suffix}"

    current = total = 0
    for line in log_lines:
        batch_match = _BATCH_PROGRESS_RE.search(line)
        if batch_match:
            current, total = int(batch_match.group(1)), int(batch_match.group(2))
    if total:
        percent = max(0, min(100, round(min(current, total) / total * 100)))
        if running:
            percent = min(percent, 99)
        if return_code == 0:
            percent = 100
        return percent, f"{percent}% — batch {current}/{total}"

    if return_code == 0:
        return 100, "100% — concluído"
    return 0, "0%"


def _uses_compact_layout(width: float) -> bool:
    return width < COMPACT_BREAKPOINT


def _responsive_content_width(width: float, compact: bool) -> float:
    horizontal_margin = 20 if compact else 36
    return max(0, min(width - horizontal_margin, MAX_CONTENT_WIDTH))


class TranslatorFletApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._running = False
        self._cancel_requested = False
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._return_code: int | None = None
        self._logs: list[str] = []
        self._layout_mode: str | None = None

        ENTRY_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)
        self._configure_page()
        self._build_controls()
        self._build_layout()
        self._load_defaults_into_form()
        self.page.on_resize = self._on_page_resize
        self._apply_responsive_layout(force=True)
        self._refresh_files()
        self.page.on_close = self._on_page_close
        self.page.update()

    def _configure_page(self) -> None:
        self.page.title = APP_TITLE
        self.page.bgcolor = BG
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.theme = ft.Theme(
            color_scheme_seed=ACCENT,
            font_family="Segoe UI",
            use_material3=True,
            visual_density=ft.VisualDensity.COMFORTABLE,
        )
        self.page.window.width = 1280
        self.page.window.height = 840
        self.page.window.min_width = 800
        self.page.window.min_height = 600
        self.page.window.center()

    def _build_controls(self) -> None:
        defaults = _effective_defaults()
        field_style = dict(
            border_color=BORDER,
            focused_border_color=ACCENT,
            border_radius=10,
            dense=True,
            bgcolor=CARD_BG,
        )
        self.model_input = ft.TextField(
            label="Modelo Ollama",
            value=defaults["model"],
            prefix_icon=ft.Icons.MEMORY_ROUNDED,
            **field_style,
        )
        self.connection_input = ft.Dropdown(
            label="Conexão Ollama",
            value=defaults["ollama_mode"],
            options=[
                ft.DropdownOption(key="local", text="Local — neste computador"),
                ft.DropdownOption(key="remote", text="Remota — IP ou URL"),
            ],
            on_select=self._on_connection_change,
            leading_icon=ft.Icons.LAN_ROUNDED,
            **field_style,
        )
        self.endpoint_input = ft.TextField(
            label="IP:porta ou URL/DNS",
            value=defaults["ollama_endpoint"],
            hint_text="192.168.1.10:11434",
            prefix_icon=ft.Icons.LINK_ROUNDED,
            disabled=defaults["ollama_mode"] != "remote",
            **field_style,
        )
        self.batch_input = ft.TextField(
            label="Batch size",
            value=defaults["batch_size"],
            keyboard_type=ft.KeyboardType.NUMBER,
            prefix_icon=ft.Icons.STACKED_BAR_CHART_ROUNDED,
            col=6,
            **field_style,
        )
        self.timeout_input = ft.TextField(
            label="Timeout (segundos)",
            value=defaults["timeout"],
            keyboard_type=ft.KeyboardType.NUMBER,
            prefix_icon=ft.Icons.TIMER_OUTLINED,
            col=6,
            **field_style,
        )
        self.turbo_check = ft.Checkbox(label="Modo turbo", value=False)
        self.clear_cache_check = ft.Checkbox(label="Limpar cache antes", value=False)
        self.no_cache_check = ft.Checkbox(
            label="Desativar cache",
            value=False,
            on_change=self._on_no_cache_change,
        )

        self.input_files = ft.ListView(height=116, spacing=2, padding=0)
        self.output_files = ft.ListView(height=105, spacing=2, padding=0)
        self.upload_meta = ft.Text(
            "Selecione um ou mais arquivos .ass.",
            size=12,
            color=MUTED,
        )

        self.status_badge = ft.Container(
            content=ft.Text("Parado", size=12, weight=ft.FontWeight.W_700, color="#166534"),
            bgcolor="#DCFCE7",
            padding=ft.Padding.symmetric(horizontal=12, vertical=5),
            border_radius=999,
        )
        self.status_meta = ft.Text("Sem execução.", size=12, color=MUTED, expand=True)
        self.progress = ft.ProgressBar(
            value=0,
            color=BLUE,
            bgcolor="#E2E8F0",
            bar_height=10,
            border_radius=999,
            expand=True,
        )
        self.progress_label = ft.Text("0%", size=13, weight=ft.FontWeight.W_600, width=190)
        self.log_view = ft.ListView(
            expand=True,
            auto_scroll=True,
            spacing=1,
            padding=12,
        )

        self.start_button = ft.FilledButton(
            "Iniciar tradução",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor=ACCENT,
            color=ft.Colors.WHITE,
            height=46,
            expand=2,
            on_click=self._on_start_clicked,
        )
        self.stop_button = ft.FilledButton(
            "Cancelar",
            icon=ft.Icons.STOP_ROUNDED,
            bgcolor=RED,
            color=ft.Colors.WHITE,
            height=46,
            expand=1,
            disabled=True,
            on_click=self._on_stop_clicked,
        )

    def _card(self, title: str, icon: Any, content: ft.Control, *, expand: bool = False) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(icon, color=ACCENT, size=20),
                            ft.Text(title, size=15, weight=ft.FontWeight.W_700, color=TEXT),
                        ],
                        spacing=8,
                    ),
                    ft.Divider(height=1, color="#EDF2F0"),
                    content,
                ],
                spacing=12,
                expand=expand,
            ),
            bgcolor=CARD_BG,
            border=ft.Border.all(1, BORDER),
            border_radius=16,
            padding=16,
            shadow=ft.BoxShadow(
                blur_radius=16,
                spread_radius=0,
                color="#120F2A22",
                offset=ft.Offset(0, 4),
            ),
            expand=expand,
        )

    def _secondary_button(
        self,
        text: str,
        icon: Any,
        handler: Callable[..., Any],
        *,
        color: str = BLUE,
        expand: bool | int | None = None,
    ) -> ft.OutlinedButton:
        return ft.OutlinedButton(
            text,
            icon=icon,
            on_click=handler,
            expand=expand,
            style=ft.ButtonStyle(
                color=color,
                side=ft.BorderSide(1, "#CBD5E1"),
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            ),
        )

    def _build_layout(self) -> None:
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Icon(ft.Icons.SUBTITLES_ROUNDED, size=30, color=ft.Colors.WHITE),
                        bgcolor=ACCENT,
                        width=50,
                        height=50,
                        border_radius=14,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                "Tradutor de Legendas ASS",
                                size=23,
                                weight=ft.FontWeight.W_700,
                                color=TEXT,
                            ),
                            ft.Text(
                                "Inglês → Português do Brasil • processamento local com Ollama",
                                size=13,
                                color=MUTED,
                            ),
                        ],
                        spacing=1,
                        expand=True,
                    ),
                    ft.PopupMenuButton(
                        icon=ft.Icons.MORE_VERT_ROUNDED,
                        tooltip="Mais opções",
                        items=[
                            ft.PopupMenuItem(
                                content="Limpar trabalho anterior",
                                icon=ft.Icons.CLEANING_SERVICES_ROUNDED,
                                on_click=self._on_cleanup_clicked,
                            ),
                            ft.PopupMenuItem(
                                content="Sobre",
                                icon=ft.Icons.INFO_OUTLINE_ROUNDED,
                                on_click=self._on_about_clicked,
                            ),
                        ],
                    ),
                ],
                spacing=14,
            ),
            bgcolor=CARD_BG,
            border=ft.Border(bottom=ft.BorderSide(1, BORDER)),
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
        )

        input_card = self._card(
            "Arquivos de entrada",
            ft.Icons.UPLOAD_FILE_ROUNDED,
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.FilledButton(
                                "Selecionar .ass",
                                icon=ft.Icons.ADD_ROUNDED,
                                bgcolor=ORANGE,
                                color=ft.Colors.WHITE,
                                on_click=self._on_upload_clicked,
                                expand=True,
                            ),
                            self._secondary_button(
                                "Limpar",
                                ft.Icons.DELETE_SWEEP_OUTLINED,
                                self._on_clear_input_clicked,
                                color=RED,
                            ),
                        ]
                    ),
                    self.upload_meta,
                    ft.Container(
                        content=self.input_files,
                        bgcolor="#F8FAF9",
                        border=ft.Border.all(1, BORDER),
                        border_radius=10,
                        padding=10,
                    ),
                ],
                spacing=10,
            ),
        )

        config_card = self._card(
            "Configuração",
            ft.Icons.TUNE_ROUNDED,
            ft.Column(
                [
                    self.model_input,
                    self.connection_input,
                    self.endpoint_input,
                    ft.ResponsiveRow([self.batch_input, self.timeout_input], spacing=10, run_spacing=10),
                    ft.Row(
                        [self.turbo_check, self.clear_cache_check, self.no_cache_check],
                        wrap=True,
                        spacing=6,
                        run_spacing=0,
                    ),
                    ft.Row(
                        [
                            self._secondary_button(
                                "Salvar configuração",
                                ft.Icons.SAVE_OUTLINED,
                                self._on_save_config_clicked,
                                expand=True,
                            ),
                            self._secondary_button(
                                "Restaurar padrão",
                                ft.Icons.RESTART_ALT_ROUNDED,
                                self._on_reset_config_clicked,
                                expand=True,
                            ),
                        ]
                    ),
                ],
                spacing=10,
            ),
        )

        output_card = self._card(
            "Arquivos processados",
            ft.Icons.TASK_ALT_ROUNDED,
            ft.Column(
                [
                    ft.Container(
                        content=self.output_files,
                        bgcolor="#F8FAF9",
                        border=ft.Border.all(1, BORDER),
                        border_radius=10,
                        padding=10,
                    ),
                    ft.Row(
                        [
                            self._secondary_button(
                                "Salvar ZIP",
                                ft.Icons.ARCHIVE_OUTLINED,
                                self._on_download_clicked,
                                expand=True,
                            ),
                            self._secondary_button(
                                "Abrir pasta",
                                ft.Icons.FOLDER_OPEN_ROUNDED,
                                self._on_open_output_clicked,
                                expand=True,
                            ),
                        ]
                    ),
                ]
            ),
        )

        status_card = self._card(
            "Status da tradução",
            ft.Icons.MONITOR_HEART_OUTLINED,
            ft.Column(
                [
                    ft.Row([self.progress, self.progress_label], spacing=14),
                    ft.Row([self.status_badge, self.status_meta], spacing=12),
                ]
            ),
        )

        self.log_card = self._card(
            "Log de atividade",
            ft.Icons.TERMINAL_ROUNDED,
            ft.Column(
                [
                    ft.Container(
                        content=self.log_view,
                        bgcolor=LOG_BG,
                        border=ft.Border.all(1, "#17382E"),
                        border_radius=12,
                        expand=True,
                    ),
                    ft.Row(
                        [
                            self._secondary_button(
                                "Limpar log",
                                ft.Icons.CLEAR_ALL_ROUNDED,
                                self._on_clear_logs_clicked,
                            ),
                            self._secondary_button(
                                "Salvar log",
                                ft.Icons.DOWNLOAD_ROUNDED,
                                self._on_save_logs_clicked,
                            ),
                        ]
                    ),
                ],
                expand=True,
            ),
            expand=True,
        )

        self.left_panel = ft.Column(
            [input_card, config_card, ft.Row([self.start_button, self.stop_button]), output_card],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
        )
        self.right_panel = ft.Column(
            [status_card, self.log_card],
            spacing=14,
            expand=True,
        )
        self.workspace_host = ft.Container()
        self.workspace_row = ft.Row(
            [self.workspace_host],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
        )
        self.body_container = ft.Container(
            content=self.workspace_row,
            padding=18,
            expand=True,
        )
        self.page.add(
            ft.Column(
                [
                    header,
                    self.body_container,
                ],
                spacing=0,
                expand=True,
            )
        )

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        width = float(self.page.width or self.page.window.width or 1280)
        height = float(self.page.height or self.page.window.height or 840)
        compact = _uses_compact_layout(width)
        mode = "compact" if compact else "wide"

        self.workspace_host.width = _responsive_content_width(width, compact)
        self.body_container.padding = 10 if compact else 18

        if force or mode != self._layout_mode:
            if compact:
                self.left_panel.expand = False
                self.left_panel.scroll = None
                self.right_panel.expand = False
                self.log_card.expand = False
                self.log_card.height = max(320, min(520, height * 0.62))
                self.workspace_host.content = ft.Column(
                    [self.left_panel, self.right_panel],
                    spacing=12,
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                )
            else:
                self.left_panel.expand = 5
                self.left_panel.scroll = ft.ScrollMode.AUTO
                self.right_panel.expand = 7
                self.log_card.expand = True
                self.log_card.height = None
                self.workspace_host.content = ft.Row(
                    [self.left_panel, self.right_panel],
                    spacing=14,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    expand=True,
                )
            self._layout_mode = mode
        elif compact:
            self.log_card.height = max(320, min(520, height * 0.62))

    def _on_page_resize(self, _: Any) -> None:
        self._apply_responsive_layout()
        self.page.update()

    def _load_defaults_into_form(self) -> None:
        values = _effective_defaults()
        self.model_input.value = values["model"]
        self.batch_input.value = values["batch_size"]
        self.timeout_input.value = values["timeout"]
        self.connection_input.value = values["ollama_mode"]
        self.endpoint_input.value = values["ollama_endpoint"]
        self.endpoint_input.disabled = values["ollama_mode"] != "remote"

    def _collect_form(self) -> dict[str, str]:
        return {
            "model": self.model_input.value or "",
            "batch_size": self.batch_input.value or "",
            "timeout": self.timeout_input.value or "",
            "ollama_mode": self.connection_input.value or "local",
            "ollama_endpoint": self.endpoint_input.value or "",
        }

    def _file_rows(self, files: list[Path], empty_text: str, icon: Any) -> list[ft.Control]:
        if not files:
            return [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.INBOX_OUTLINED, size=17, color="#94A3B8"),
                        ft.Text(empty_text, size=12, color="#94A3B8", italic=True),
                    ],
                    spacing=8,
                )
            ]
        rows: list[ft.Control] = []
        for path in files:
            rows.append(
                ft.Row(
                    [
                        ft.Icon(icon, size=17, color=ACCENT),
                        ft.Text(path.name, size=12, color=TEXT, tooltip=str(path)),
                    ],
                    spacing=8,
                )
            )
        return rows

    def _refresh_files(self) -> None:
        input_paths = sorted(ENTRY_DIR.glob("*.ass"), key=lambda path: path.name.lower())
        output_paths = sorted(OUTPUT_DIR.glob("*.ass"), key=lambda path: path.name.lower())
        self.input_files.controls = self._file_rows(
            input_paths,
            "Nenhum arquivo na pasta de entrada.",
            ft.Icons.SUBTITLES_OUTLINED,
        )
        self.output_files.controls = self._file_rows(
            output_paths,
            "Nenhum arquivo traduzido ainda.",
            ft.Icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
        )
        self.page.update()

    def _append_log(self, line: str, *, store: bool = True) -> None:
        clean = _sanitize_logs(line).strip()
        if not clean:
            return
        with self._lock:
            if store:
                self._logs.append(clean)
                if len(self._logs) > 4000:
                    self._logs = self._logs[-4000:]
            self.log_view.controls.append(
                ft.Text(clean, size=11, color=LOG_TEXT, font_family="Consolas", selectable=True)
            )
            if len(self.log_view.controls) > 800:
                self.log_view.controls = self.log_view.controls[-800:]
        self.page.update()

    def _set_running_ui(self, running: bool, status: str | None = None) -> None:
        self.start_button.disabled = running
        self.stop_button.disabled = not running
        if running:
            self.status_badge.content.value = status or "Executando"
            self.status_badge.content.color = "#9A3412"
            self.status_badge.bgcolor = "#FFEDD5"
            self.progress.color = BLUE
        else:
            self.status_badge.content.value = status or "Parado"
            self.status_badge.content.color = "#166534"
            self.status_badge.bgcolor = "#DCFCE7"
        self.page.update()

    def _update_progress(self) -> None:
        with self._lock:
            percent, label = _compute_progress(
                list(self._logs), self._running, self._return_code
            )
            started_at = self._started_at
            finished_at = self._finished_at
            return_code = self._return_code
            running = self._running
        self.progress.value = percent / 100
        self.progress_label.value = label
        self.page.window.progress_bar = percent / 100 if running else None
        if running:
            self.status_meta.value = f"Iniciado em {started_at or '-'} • em andamento"
        elif finished_at:
            self.status_meta.value = (
                f"Início {started_at or '-'} • fim {finished_at} • código {return_code}"
            )
        else:
            self.status_meta.value = "Sem execução."
        self.page.update()

    def _show_message(self, title: str, message: str, *, error: bool = False) -> None:
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        ft.Icons.ERROR_OUTLINE_ROUNDED if error else ft.Icons.INFO_OUTLINE_ROUNDED,
                        color=RED if error else ACCENT,
                    ),
                    ft.Text(title, weight=ft.FontWeight.W_700),
                ]
            ),
            content=ft.Text(message, selectable=True),
            actions=[ft.TextButton("OK", on_click=lambda _: self.page.pop_dialog())],
        )
        self.page.show_dialog(dialog)

    def _confirm(self, title: str, message: str, on_confirm: Callable[[], None]) -> None:
        def yes(_: Any) -> None:
            self.page.pop_dialog()
            on_confirm()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.W_700),
            content=ft.Text(message),
            actions=[
                ft.TextButton("Cancelar", on_click=lambda _: self.page.pop_dialog()),
                ft.FilledButton("Confirmar", bgcolor=RED, color=ft.Colors.WHITE, on_click=yes),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    def _on_connection_change(self, _: Any) -> None:
        remote = self.connection_input.value == "remote"
        self.endpoint_input.disabled = not remote
        if not remote:
            self.endpoint_input.value = ""
        self.page.update()

    def _on_no_cache_change(self, _: Any) -> None:
        self.clear_cache_check.disabled = bool(self.no_cache_check.value)
        if self.no_cache_check.value:
            self.clear_cache_check.value = False
        self.page.update()

    async def _on_upload_clicked(self, _: Any) -> None:
        with self._lock:
            if self._running:
                self._show_message("Tradução em andamento", "Aguarde ou cancele antes de alterar a entrada.")
                return
        try:
            selected = await self.file_picker.pick_files(
                dialog_title="Selecionar legendas ASS",
                initial_directory=str(ENTRY_DIR),
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["ass"],
                allow_multiple=True,
            )
            if not selected:
                return
            saved: list[str] = []
            skipped: list[str] = []
            for picked in selected:
                path_text = getattr(picked, "path", None)
                if not path_text:
                    skipped.append(getattr(picked, "name", "arquivo sem caminho"))
                    continue
                source = Path(path_text)
                if source.suffix.lower() != ".ass":
                    skipped.append(source.name)
                    continue
                try:
                    destination = ENTRY_DIR / source.name
                    if source.resolve() != destination.resolve():
                        shutil.copy2(source, destination)
                    saved.append(source.name)
                except Exception as exc:
                    skipped.append(f"{source.name}: {exc}")
            self.upload_meta.value = (
                f"Adicionados: {len(saved)} • ignorados: {len(skipped)}"
            )
            self._refresh_files()
        except Exception as exc:
            self._show_message("Falha ao selecionar arquivos", str(exc), error=True)

    def _on_clear_input_clicked(self, _: Any) -> None:
        with self._lock:
            if self._running:
                self._show_message("Tradução em andamento", "A entrada não pode ser limpa agora.")
                return

        def clear() -> None:
            try:
                removed = _clear_directory_contents(ENTRY_DIR)
                self.upload_meta.value = f"Pasta limpa • {removed} item(ns) removido(s)."
                self._refresh_files()
            except Exception as exc:
                self._show_message("Falha ao limpar entrada", str(exc), error=True)

        self._confirm(
            "Limpar pasta de entrada",
            "Todos os arquivos da pasta de entrada serão removidos.",
            clear,
        )

    def _on_start_clicked(self, _: Any) -> None:
        if not IS_FROZEN and not SCRIPT_PATH.exists():
            self._show_message("Backend não encontrado", "translate_ass_fast.py não foi localizado.", error=True)
            return
        if not any(ENTRY_DIR.glob("*.ass")):
            self._show_message("Nenhum arquivo", "Adicione pelo menos um arquivo .ass antes de iniciar.")
            return
        with self._lock:
            if self._running:
                self._show_message("Tradução em andamento", "Já existe uma tradução em execução.")
                return

        ok, normalized, error = _normalize_config(self._collect_form())
        if not ok:
            self._show_message("Configuração inválida", error, error=True)
            return

        full_form: dict[str, Any] = {
            **normalized,
            "turbo": bool(self.turbo_check.value),
            "clear_cache": bool(self.clear_cache_check.value),
            "no_cache": bool(self.no_cache_check.value),
        }
        ollama_host = normalized["ollama_endpoint"] if normalized["ollama_mode"] == "remote" else None

        with self._lock:
            self._running = True
            self._cancel_requested = False
            self._started_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self._finished_at = None
            self._return_code = None
            self._logs = []
            self.log_view.controls = []
        self.progress.value = 0
        self.progress_label.value = "0%"
        self._set_running_ui(True, "Validando")
        self._append_log("[Flet] Validando conexão e modelo no Ollama...")
        self._update_progress()
        self.page.run_thread(self._run_translation, full_form, ollama_host)

    def _run_translation(self, form: dict[str, Any], ollama_host: str | None) -> None:
        return_code = -1
        try:
            available, model_error = _ensure_ollama_model_available(form["model"], ollama_host)
            if not available:
                self._append_log(f"[Flet] {model_error}")
                self._finish_translation(2, model_error)
                return
            with self._lock:
                if self._cancel_requested:
                    self._finish_translation(1)
                    return

            self._set_running_ui(True, "Executando")
            command = _build_command(form)
            endpoint_label = ollama_host or "local"
            self._append_log(f"[Flet] Ollama: {endpoint_label}")
            self._append_log(f"[Flet] Comando: {_sanitize_logs(' '.join(command))}")

            env = dict(os.environ)
            env.update(
                {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            if ollama_host:
                env["OLLAMA_HOST"] = ollama_host
            else:
                env.pop("OLLAMA_HOST", None)

            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            with self._lock:
                self._process = process

            chars: list[str] = []
            assert process.stdout is not None
            while True:
                char = process.stdout.read(1)
                if char == "":
                    if chars:
                        self._append_log("".join(chars))
                    break
                if char in {"\r", "\n"}:
                    if chars:
                        self._append_log("".join(chars))
                        chars = []
                        self._update_progress()
                    continue
                chars.append(char)
            return_code = process.wait()
        except Exception as exc:
            self._append_log(f"[Flet] Erro ao executar a tradução: {exc}")
            self._append_log(traceback.format_exc())
        finally:
            with self._lock:
                self._process = None
            self._finish_translation(return_code)

    def _finish_translation(self, return_code: int, error_message: str | None = None) -> None:
        with self._lock:
            if not self._running and self._return_code is not None:
                return
            cancelled = self._cancel_requested
            self._running = False
            self._return_code = return_code
            self._finished_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if cancelled:
            self._append_log("[Flet] Tradução cancelada pelo usuário.")
            self.progress.color = ORANGE
            self._set_running_ui(False, "Cancelado")
        elif return_code == 0:
            self._append_log("[Flet] Tradução finalizada com sucesso.")
            self.progress.color = ACCENT
            self._set_running_ui(False, "Concluído")
        else:
            self._append_log(f"[Flet] Processo finalizado com erro (código {return_code}).")
            self.progress.color = RED
            self._set_running_ui(False, "Erro")
        self._update_progress()
        self._refresh_files()
        if error_message:
            self._show_message("Ollama indisponível", error_message, error=True)

    def _on_stop_clicked(self, _: Any) -> None:
        def stop() -> None:
            with self._lock:
                self._cancel_requested = True
                process = self._process
            if process and process.poll() is None:
                try:
                    process.terminate()
                except Exception as exc:
                    self._show_message("Falha ao cancelar", str(exc), error=True)
            else:
                self._append_log("[Flet] Cancelamento solicitado.")

        self._confirm(
            "Cancelar tradução",
            "O processo atual será encerrado. Arquivos já concluídos serão mantidos.",
            stop,
        )

    async def _on_download_clicked(self, _: Any) -> None:
        files = sorted(OUTPUT_DIR.glob("*.ass"))
        if not files:
            self._show_message("Nenhum arquivo", "Ainda não há legendas processadas para compactar.")
            return
        default_name = f"legendas_traduzidas_{datetime.now():%Y%m%d_%H%M%S}.zip"
        try:
            destination = await self.file_picker.save_file(
                dialog_title="Salvar legendas traduzidas",
                file_name=default_name,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["zip"],
            )
            if not destination:
                return
            dest_path = Path(destination)
            if dest_path.suffix.lower() != ".zip":
                dest_path = dest_path.with_suffix(".zip")
            with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, arcname=path.name)
            self._show_message("ZIP salvo", f"Arquivo criado em:\n{dest_path}")
        except Exception as exc:
            self._show_message("Falha ao criar ZIP", str(exc), error=True)

    def _on_open_output_clicked(self, _: Any) -> None:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(OUTPUT_DIR.resolve()))
            else:
                self.page.launch_url(OUTPUT_DIR.resolve().as_uri())
        except Exception as exc:
            self._show_message("Falha ao abrir pasta", str(exc), error=True)

    def _on_clear_logs_clicked(self, _: Any) -> None:
        with self._lock:
            self._logs = []
            self.log_view.controls = []
        self._update_progress()

    async def _on_save_logs_clicked(self, _: Any) -> None:
        with self._lock:
            log_text = "\n".join(self._logs)
        if not log_text:
            self._show_message("Log vazio", "Não há mensagens para salvar.")
            return
        try:
            destination = await self.file_picker.save_file(
                dialog_title="Salvar log da tradução",
                file_name=f"traducao_{datetime.now():%Y%m%d_%H%M%S}.log",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["log", "txt"],
            )
            if destination:
                Path(destination).write_text(log_text, encoding="utf-8")
                self._show_message("Log salvo", f"Arquivo criado em:\n{destination}")
        except Exception as exc:
            self._show_message("Falha ao salvar log", str(exc), error=True)

    def _on_save_config_clicked(self, _: Any) -> None:
        ok, normalized, error = _normalize_config(self._collect_form())
        if not ok:
            self._show_message("Configuração inválida", error, error=True)
            return
        try:
            _save_user_config(normalized)
            self._show_message("Configuração salva", "As preferências serão carregadas na próxima abertura.")
        except Exception as exc:
            self._show_message("Falha ao salvar configuração", str(exc), error=True)

    def _on_reset_config_clicked(self, _: Any) -> None:
        def reset() -> None:
            try:
                if CONFIG_PATH.exists():
                    CONFIG_PATH.unlink()
                self._load_defaults_into_form()
                self.page.update()
            except Exception as exc:
                self._show_message("Falha ao restaurar configuração", str(exc), error=True)

        self._confirm(
            "Restaurar configuração",
            "Os campos voltarão aos valores padrão.",
            reset,
        )

    def _on_cleanup_clicked(self, _: Any) -> None:
        with self._lock:
            if self._running:
                self._show_message("Tradução em andamento", "A limpeza não pode ser executada agora.")
                return

        def cleanup() -> None:
            try:
                removed_input = _clear_directory_contents(ENTRY_DIR)
                removed_output = _clear_directory_contents(OUTPUT_DIR)
                cache_cleared = CACHE_PATH.exists()
                if cache_cleared:
                    CACHE_PATH.unlink()
                with self._lock:
                    self._logs = []
                    self.log_view.controls = []
                    self._started_at = None
                    self._finished_at = None
                    self._return_code = None
                self.progress.value = 0
                self.progress_label.value = "0%"
                self.status_meta.value = "Sem execução."
                self._set_running_ui(False)
                self._refresh_files()
                self._show_message(
                    "Limpeza concluída",
                    f"Entrada: {removed_input} item(ns)\n"
                    f"Saída: {removed_output} item(ns)\n"
                    f"Cache removido: {'sim' if cache_cleared else 'não'}",
                )
            except Exception as exc:
                self._show_message("Falha durante a limpeza", str(exc), error=True)

        self._confirm(
            "Limpar trabalho anterior",
            "Entrada, saída e cache de tradução serão removidos.",
            cleanup,
        )

    def _on_about_clicked(self, _: Any) -> None:
        self._show_message(
            "Sobre",
            "Tradutor de arquivos .ass do Inglês para Português do Brasil\n"
            "Interface Flet para Windows • desenvolvido por Yasu\n\n"
            "O executável inclui Python, Flet e as bibliotecas do projeto. "
            "O Ollama e o modelo escolhido devem estar disponíveis localmente ou pela rede.",
        )

    def _on_page_close(self, _: Any) -> None:
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass


def _run_backend() -> int:
    import translate_ass_fast

    sys.argv = ["translate_ass_fast.py", *[arg for arg in sys.argv[1:] if arg != "--run-backend"]]
    try:
        asyncio.run(translate_ass_fast.main())
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def main(page: ft.Page) -> None:
    TranslatorFletApp(page)


if __name__ == "__main__":
    if "--run-backend" in sys.argv:
        raise SystemExit(_run_backend())
    ft.run(main)
