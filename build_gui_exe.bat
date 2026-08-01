@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Limpando artefatos antigos...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist TradutorASS-GUI.spec del /q TradutorASS-GUI.spec

echo [2/3] Gerando pacote GUI...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name TradutorASS-GUI ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --hidden-import translate_ass_fast ^
  --hidden-import translation_engine ^
  --hidden-import subtitle_formats ^
  web_app.py

if errorlevel 1 (
  echo [ERRO] Falha ao gerar o executavel.
  exit /b 1
)

echo [3/3] Build concluido.
echo Pasta: dist\TradutorASS-GUI
echo Executavel: dist\TradutorASS-GUI\TradutorASS-GUI.exe

endlocal
