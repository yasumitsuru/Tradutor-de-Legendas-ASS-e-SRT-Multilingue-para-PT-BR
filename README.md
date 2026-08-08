# Tradutor de legendas ASS/SRT — Inglês para Português Brasil

Aplicação local para traduzir legendas `.ass` e `.srt` com modelos executados pelo
[Ollama](https://ollama.com/). O projeto oferece a mesma base de tradução por quatro
formas de uso: linha de comando, interface web Flask, aplicativo Flet e aplicativo
PySide6.

O pipeline preserva horários, ordem dos eventos, estrutura de linhas e marcações já
existentes. Respostas incompletas ou estruturalmente inseguras repetem somente os itens
afetados e recebem uma última tentativa individual. Por padrão, uma falha definitiva
impede o salvamento do arquivo para evitar legendas misturando idiomas.

## Formatos suportados

As extensões são reconhecidas sem distinção entre maiúsculas e minúsculas:

- `.ass` e `.ASS`;
- `.srt` e `.SRT`.

| Recurso | ASS | SRT |
| --- | --- | --- |
| Horários e ordem | Preservados | Preservados |
| Estilos | Estilos, margens, efeitos e comandos ASS preservados | Tags existentes `<i>`, `<b>`, `<u>` e `<font>` preservadas |
| Quebras | `\N`, `\n` e `\h` preservados | Quantidade de linhas do bloco preservada com quebras reais no arquivo |
| Posicionamento | Recursos nativos do formato preservados | Somente marcações que já existiam são preservadas literalmente |
| Códigos novos | O modelo não pode inventar comandos | Nenhum comando ASS ou posicionamento experimental é inserido |

SRT não possui suporte universal a estilos ou posicionamento avançado. O tradutor não
converte comandos ASS para SRT, não cria `{\an8}`, `{\pos(...)}` ou `{\i1}`, e
não tenta simular recursos que o formato não representa de forma portátil. Marcações
desconhecidas já presentes no SRT são protegidas e restauradas sem serem enviadas ao
modelo.

## Requisitos

- Windows 10/11 para os aplicativos e scripts de build fornecidos;
- Python 3.11 ou mais recente para execução pelo código-fonte;
- Ollama local ou acessível por rede;
- um modelo instalado, por padrão `qwen2.5:14b`.

Instale o Ollama e prepare o modelo:

```powershell
ollama pull qwen2.5:14b
```

Instale as dependências do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para desenvolvimento e testes, use `requirements-dev.txt` no lugar de
`requirements.txt`.

## Uso pela CLI

Coloque as legendas em `entrada` e execute:

```powershell
python translate_ass_fast.py --input-dir .\entrada --output-dir .\saida
```

O padrão processa ASS e SRT. Para limitar o formato:

```powershell
python translate_ass_fast.py -i .\entrada -o .\saida --format srt
python translate_ass_fast.py -i .\entrada -o .\saida --format ass
```

Exemplo com opções de desempenho:

```powershell
python translate_ass_fast.py `
  -i .\entrada `
  -o .\saida `
  --model qwen2.5:14b `
  --batch-size 15 `
  --timeout 300 `
  --turbo
```

Opções úteis:

- `--format all|ass|srt`: filtra o formato de entrada;
- `--clear-cache`: remove o cache antes do processamento;
- `--no-cache`: não lê nem grava traduções em cache;
- `--allow-original-fallback`: permite explicitamente manter o texto original nos itens
  que falharem definitivamente. Atenção: isso pode gerar legendas misturando PT-BR com
  o idioma original;
- `--turbo`: reduz temperatura e limita o lote a dez itens;
- `--batch-size N`: define quantos blocos seguem em cada requisição;
- `--timeout SEGUNDOS`: define o limite de cada chamada ao modelo.

A saída mantém a extensão original e usa o padrão `nome.pt.ext`, por exemplo
`episodio.pt.ass` e `episodio.pt.SRT`. Arquivos não suportados são informados como
ignorados.

Para um Ollama remoto pela CLI, defina `OLLAMA_HOST` antes da execução:

```powershell
$env:OLLAMA_HOST = "http://192.168.1.10:11434"
python translate_ass_fast.py -i .\entrada -o .\saida
```

## Interfaces

### Flask

```powershell
python web_app.py
```

A interface abre em `http://127.0.0.1:7860/` e permite upload múltiplo de ASS/SRT,
configuração de Ollama local ou remoto, acompanhamento de progresso, limpeza do
trabalho e download das saídas em ZIP.

### Flet

```powershell
python app_flet.py
```

### PySide6

```powershell
python app_gui.py
```

As duas interfaces desktop selecionam múltiplas legendas ASS/SRT, exibem entrada e
saída, oferecem cancelamento e empacotam os resultados em ZIP. Flask, Flet e PySide
oferecem a opção desativada por padrão **Permitir manter texto original quando a
tradução falhar**, acompanhada do aviso sobre mistura de idiomas. As interfaces apenas
repassam essa escolha ao mesmo backend e usam as mesmas validações.

Listas e downloads usam um manifesto da execução atual. Um arquivo antigo que já esteja
na pasta de saída não é apresentado nem incluído no ZIP quando a nova execução falha
antes de produzir e validar uma substituição.

## Preservação e validação

Antes da tradução, o handler do formato substitui por tokens opacos:

- tags e comandos de formatação;
- quebras de linha e espaços rígidos;
- prefixos de diálogo, como `- `;
- URLs, e-mails e variáveis comuns;
- override tags ASS reconhecidas e metadados de posicionamento existentes.

Em ASS, blocos entre chaves são classificados por um scanner de comandos. Override tags
reais, inclusive `\pos`, `\move`, `\clip`, `\t`, cores e alpha, são preservadas
textualmente. Blocos de linguagem natural como `{English annotation}` são comentários
textuais e são removidos antes da tradução. Em blocos mistos, o texto é descartado e
comandos válidos necessários, como `\i0`, são preservados. `\N`, `\n` e `\h` dentro de
um comentário misto não são promovidos a estrutura visível.

O protocolo de lote usa blocos `<<<ITEM_0001>>>`/`<<<END_ITEM_0001>>>`. Cada resposta
é validada individualmente. Marcadores ausentes, duplicados, reordenados ou deformados,
Markdown, numeração inventada, mensagens explicativas e comandos ASS novos invalidam
somente o item afetado. Texto ou Markdown fora dos delimitadores é descartado e
registrado como warning, sem invalidar itens corretos nem entrar na legenda. Retries
contêm somente itens ainda inválidos. Depois dos retries normais, cada item pendente tem
uma tentativa individual com temperatura zero e a mesma validação estrutural.

Se algum item continuar inválido, o modo padrão informa a quantidade de falhas e não
publica um novo arquivo. A opção `--allow-original-fallback` (ou seu checkbox equivalente)
habilita conscientemente o comportamento legado de copiar o original nesses itens.

Antes de salvar, o projeto valida horários, quantidade de eventos, estrutura de linhas,
ordem e contagem das marcações, tags HTML simples e ausência de placeholders internos.
O candidato serializado é reaberto para detectar conversões ou vazamentos específicos
do formato. Somente depois dessa validação ele substitui atomicamente o destino final.

## Cache

O cache local fica em `translation_cache.json` e não é versionado pelo Git. O schema
atual é `3`. A chave inclui:

- formato da legenda;
- texto limpo;
- estrutura de quebras;
- marcações protegidas;
- idiomas de origem e destino;
- modelo Ollama;
- versão do prompt.

Isso impede que traduções ASS com escapes sejam reutilizadas em SRT ou entre contextos
incompatíveis. Caches de schema antigo são ignorados e recriados com segurança.

## Testes

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

A suíte não precisa de um Ollama real. Ela usa um cliente simulado e cobre pipeline
completo, cache, warning externo, retries por item, recuperação individual, falha segura,
manifesto de outputs, BOM UTF-8, CRLF/LF, tags SRT, posicionamento já existente, blocos
vazios, quebras, hifens de diálogo, scanner ASS, Flask, Flet/PySide6 e regressões ASS/SRT.

O workflow `Tests` executa a suíte no Windows em pushes e pull requests. O workflow
manual `Release GUI` também exige os testes antes de empacotar e publicar artefatos.

## Gerar executáveis no Windows

Com as dependências instaladas:

```powershell
.\build_gui_exe.bat
.\build_flet_exe.bat
.\build_pyside_exe.bat
```

Saídas esperadas:

- `dist\TradutorASS-GUI\TradutorASS-GUI.exe` — interface Flask empacotada;
- `dist_flet\TradutorASS-Flet.exe` — interface Flet;
- `dist_pyside\TradutorASS-PySide.exe` — interface PySide6.

Os nomes históricos dos executáveis continuam usando `TradutorASS` por compatibilidade,
mas todos processam ASS e SRT. Python e as bibliotecas são embutidos; o Ollama e o
modelo escolhido continuam necessários no computador ou endpoint remoto.

## Solução de problemas

- **Modelo não encontrado:** execute `ollama pull NOME_DO_MODELO` e confira o nome na
  interface ou em `--model`.
- **Ollama remoto indisponível:** confirme URL, porta, firewall e se o endpoint responde
  à API do Ollama. Use uma URL completa, como `http://host:11434`.
- **Nenhum arquivo encontrado:** confirme se a legenda está diretamente na pasta de
  entrada e termina em `.ass` ou `.srt`.
- **Falha definitiva em itens:** consulte o log. Por padrão nenhum novo arquivo é
  publicado. Corrija a causa e execute novamente; use `--allow-original-fallback`
  somente se aceitar conscientemente uma possível mistura de idiomas.
- **Tag SRT rejeitada:** corrija tags simples sem fechamento ou aninhamento inválido no
  arquivo original. Marcações desconhecidas são preservadas, mas não são convertidas.
- **Build falhou:** instale `requirements.txt`, confirme Python compatível e execute o
  script a partir da raiz do repositório.

As legendas são enviadas somente ao endpoint Ollama configurado. Evite apontar a
aplicação para serviços que você não controla quando o conteúdo for sensível.
