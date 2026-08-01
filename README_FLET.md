# Interfaces desktop para Windows

A interface principal está em `app_flet.py`. Ela oferece:

- seleção de vários arquivos `.ass`;
- configuração de Ollama local ou remoto;
- progresso, log em tempo real e cancelamento;
- cache opcional e modo turbo;
- exportação das legendas processadas em ZIP;
- salvamento de configurações e logs.
- layout responsivo de 800×600 até 4K, com modo compacto e rolagem automática.

Uma interface alternativa baseada em PySide6 está disponível em `app_gui.py`.
As duas interfaces usam o mesmo backend `translate_ass_fast.py`.

## Executar em desenvolvimento

```powershell
python -m pip install -r requirements.txt
python app_flet.py
# ou
python app_gui.py
```

## Gerar os executáveis autônomos

Execute no Windows:

```powershell
.\build_flet_exe.bat
.\build_pyside_exe.bat
```

Os arquivos finais serão criados em:

```text
dist_flet\TradutorASS-Flet.exe
dist_pyside\TradutorASS-PySide.exe
```

O computador de destino não precisa ter Python, Flet ou os pacotes Python
instalados. O Ollama continua necessário como motor de IA, seja instalado na
mesma máquina ou acessível por um endpoint remoto. O modelo padrão pode ser
baixado com:

```powershell
ollama pull qwen2.5:14b
```

O workflow manual `Release GUI` publica o ZIP da interface Web e os executáveis
Flet e PySide6, acompanhados por um arquivo com hashes SHA-256.
