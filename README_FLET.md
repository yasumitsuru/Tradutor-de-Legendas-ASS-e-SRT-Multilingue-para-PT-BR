# Interface Flet para Windows

A interface principal está em `app_flet.py`. Ela oferece:

- seleção de vários arquivos `.ass`;
- configuração de Ollama local ou remoto;
- progresso, log em tempo real e cancelamento;
- cache opcional e modo turbo;
- exportação das legendas processadas em ZIP;
- salvamento de configurações e logs.
- layout responsivo de 800×600 até 4K, com modo compacto e rolagem automática.

## Executar em desenvolvimento

```powershell
python -m pip install -r requirements.txt
python app_flet.py
```

## Gerar o executável autônomo

Execute no Windows:

```powershell
.\build_flet_exe.bat
```

O arquivo final será criado em:

```text
dist_flet\TradutorASS-Flet.exe
```

O computador de destino não precisa ter Python, Flet ou os pacotes Python
instalados. O Ollama continua necessário como motor de IA, seja instalado na
mesma máquina ou acessível por um endpoint remoto. O modelo padrão pode ser
baixado com:

```powershell
ollama pull qwen2.5:14b
```
