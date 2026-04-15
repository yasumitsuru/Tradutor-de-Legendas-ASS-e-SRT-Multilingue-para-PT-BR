@echo off
setlocal
pushd "%~dp0" || (echo ERRO: Nao foi possivel acessar a pasta do script. & pause & exit /b 1)
chcp 65001 >nul
cls
set "MODEL=qwen2.5:14b"

echo ============================================================
echo   PIPELINE DE TRADUCAO - MODO DIRETO
echo ============================================================
echo.

:: Cria pastas (2>nul ignora erro se ja existirem)
if not exist "entrada" mkdir entrada
if not exist "saida" mkdir saida
echo [OK] Pastas prontas.
echo.

where ollama >nul 2>&1 || (echo [ERRO] Ollama nao encontrado no PATH. Instale em https://ollama.com/download & pause & exit /b 1)
where python >nul 2>&1 || (echo [ERRO] Python nao encontrado no PATH. & pause & exit /b 1)

:: 1. MATAR OLLAMA
echo [1/4] Encerrando processos Ollama...
taskkill /F /IM "ollama.exe" /T >nul 2>&1
taskkill /F /IM "ollama app.exe" /T >nul 2>&1
timeout /t 3 /nobreak >nul
echo   [OK] Processos encerrados.
echo.

:: 2. INICIAR OLLAMA
echo [2/4] Iniciando servidor Ollama...
start /b "" cmd /c "ollama serve > ollama_daemon.log 2>&1"
timeout /t 5 /nobreak >nul
echo   [OK] Ollama iniciado. Aguardando estabilizacao...
echo.

:: 3. GARANTIR MODELO
echo [3/4] Validando modelo %MODEL%...
ollama show "%MODEL%" >nul 2>&1
if errorlevel 1 (
  echo   [INFO] Modelo %MODEL% nao encontrado. Baixando...
  ollama pull "%MODEL%"
  if errorlevel 1 (
    echo   [ERRO] Falha ao baixar o modelo %MODEL%.
    pause
    exit /b 1
  )
)
echo   [OK] Modelo pronto.
echo.

:: 4. EXECUTAR PYTHON
echo [4/4] Iniciando tradutor...
echo ============================================================
python translate_ass_fast.py --input-dir .\entrada --output-dir .\saida -m %MODEL% --batch-size 20 --turbo
echo ============================================================
echo.
echo [CONCLUIDO] Processo finalizado!
pause
