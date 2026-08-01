@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Limpando artefatos antigos...
if exist build_pyside rmdir /s /q build_pyside
if exist dist_pyside rmdir /s /q dist_pyside

echo [2/3] Gerando pacote GUI (PySide6) - modo onefile...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --distpath dist_pyside ^
  --workpath build_pyside ^
  TradutorASS-PySide.spec

if errorlevel 1 (
  echo [ERRO] Falha ao gerar o executavel.
  exit /b 1
)

echo [3/3] Build concluido.
echo Executavel: dist_pyside\TradutorASS-PySide.exe
echo.
echo OBS: O backend translate_ass_fast.py e embutido no .exe (importado em runtime).
echo      Para que a traducao funcione, tenha o Ollama instalado na maquina.

endlocal
