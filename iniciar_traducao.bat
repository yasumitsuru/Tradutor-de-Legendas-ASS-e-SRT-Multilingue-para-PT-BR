@echo off
setlocal
pushd "%~dp0" || (echo ERRO: Nao foi possivel acessar a pasta do script. & pause & exit /b 1)
chcp 65001 >nul
cls

echo ============================================================
echo   PIPELINE DE TRADUCAO - MODO DIRETO
echo ============================================================
echo.

:: Cria pastas (2>nul ignora erro se ja existirem)
if not exist "entrada" mkdir entrada
if not exist "saida" mkdir saida
echo [OK] Pastas prontas.
echo.

:: 1. MATAR OLLAMA
echo [1/3] Encerrando processos Ollama...
taskkill /F /IM "ollama.exe" /T >nul 2>&1
taskkill /F /IM "ollama app.exe" /T >nul 2>&1
timeout /t 3 /nobreak >nul
echo   [OK] Processos encerrados.
echo.

:: 2. INICIAR OLLAMA
echo [2/3] Iniciando servidor Ollama...
start /b "" cmd /c "ollama serve > ollama_daemon.log 2>&1"
timeout /t 5 /nobreak >nul
echo   [OK] Ollama iniciado. Aguardando estabilizacao...
echo.

:: 3. EXECUTAR PYTHON
echo [3/3] Iniciando tradutor...
echo ============================================================
python translate_ass_fast.py --input-dir .\entrada --output-dir .\saida -m qwen2.5:14b --batch-size 20 --turbo
echo ============================================================
echo.
echo [CONCLUIDO] Processo finalizado!
pause