@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Validando ambiente de build...
python -c "import flet, ollama, pysubs2, PyInstaller" >nul 2>&1
if errorlevel 1 (
  echo [ERRO] Dependencias de build ausentes.
  echo Execute: python -m pip install -r requirements.txt
  exit /b 1
)

echo [2/3] Gerando executavel Flet autonomo para Windows...
python -m flet.cli pack ^
  --yes ^
  --name TradutorASS-Flet ^
  --distpath dist_flet ^
  --hidden-import translate_ass_fast ^
  --product-name "Tradutor ASS - Portugues Brasil" ^
  --file-description "Tradutor de legendas ASS com Ollama" ^
  --product-version 1.0.0 ^
  --file-version 1.0.0.0 ^
  --company-name "Yasu" ^
  app_flet.py

if errorlevel 1 (
  echo [ERRO] Falha ao gerar o executavel.
  exit /b 1
)

echo [3/3] Build concluido.
echo Executavel: dist_flet\TradutorASS-Flet.exe
echo.
echo O arquivo .exe inclui Python, Flet e as bibliotecas Python.
echo O Ollama e o modelo escolhido precisam estar disponiveis na maquina ou rede.

endlocal
