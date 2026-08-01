@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Validando ambiente de build...
python -c "import PySide6, ollama, pysubs2, PyInstaller" >nul 2>&1
if errorlevel 1 (
  echo [ERRO] Dependencias de build ausentes.
  echo Execute: python -m pip install -r requirements.txt
  exit /b 1
)

echo [2/3] Gerando executavel GUI PySide6 em modo onefile...
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
