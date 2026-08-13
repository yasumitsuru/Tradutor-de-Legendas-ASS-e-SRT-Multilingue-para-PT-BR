# Tradutor de Legendas ASS e SRT Multilíngue para PT-BR

Aplicação local para traduzir legendas `.ass` e `.srt` de um ou mais idiomas para
português do Brasil com modelos executados pelo [Ollama](https://ollama.com/). A
detecção automática por item acontece semanticamente em cada legenda: uma mesma fala
pode combinar vários idiomas e partes que já estão em português. O projeto oferece a
mesma base de tradução por quatro formas de uso: linha de comando, interface web Flask,
aplicativo Flet e aplicativo PySide6.

O pipeline preserva horários, ordem dos eventos, estrutura de linhas e marcações já
existentes. Respostas incompletas ou estruturalmente inseguras repetem somente os itens
afetados e recebem uma última tentativa individual. Por padrão, uma falha definitiva
impede o salvamento do arquivo para evitar legendas misturando idiomas.

## Detecção multilíngue

O modo padrão é `auto`. O texto natural visível de cada item completo é enviado ao
modelo, que detecta o idioma ou os idiomas presentes e produz uma única fala natural em
PT-BR. O pipeline não separa palavras ou fragmentos para traduzi-los isoladamente e não
usa uma biblioteca externa de detecção. Isso mantém o contexto de exemplos mistos como
`Ich brauche the key para abrir a porta`.

Conteúdo já natural em português deve ser preservado, enquanto nomes próprios,
honoríficos, siglas e termos técnicos permanecem protegidos pelas instruções do modelo.
A detecção é baseada na capacidade multilíngue do modelo configurado e pode cometer
erros; o projeto não promete detecção perfeita. O modelo padrão é `qwen2.5:14b`.

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
- `--source-language IDIOMA`: define a origem. O padrão `auto` detecta um ou mais
  idiomas por item; também são aceitas strings livres como `English` ou `German` para
  uso manual;
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
usam detecção automática multilíngue por item e mantêm PT-BR como destino. A decisão
semântica permanece exclusivamente no backend compartilhado. As interfaces também
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
atual é `4`. A chave inclui:

- formato da legenda;
- texto limpo;
- estrutura de quebras;
- marcações protegidas;
- modo/idioma de origem (`auto` ou valor manual) e idioma de destino;
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
completo, prompts multilíngues, conteúdo de idiomas mistos, cache, warning externo,
retries por item, recuperação individual, falha segura, manifesto de outputs, BOM UTF-8,
CRLF/LF, tags SRT, posicionamento já existente, blocos vazios, quebras, hifens de
diálogo, scanner ASS, Flask, Flet/PySide6 e regressões ASS/SRT.

### Regressão ASS com Ollama real

A bateria real é manual e permanece fora do CI. Ela descobre `.ass`/`.ASS`, chama a
mesma CLI de produção, desativa cache por padrão, valida estrutura/tags/quebras e grava
relatórios reproduzíveis em `test_results/<run-id>/<experimento>/`. Essa pasta nunca é
versionada e um experimento existente não é sobrescrito.

Baseline completa:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
python tools/run_ass_regression.py --experiment baseline --no-cache
```

Para selecionar qualquer outro modelo disponível sem alterar a configuração de
produção, acrescente `--model NOME_DO_MODELO` e use um nome de experimento distinto.

Um único arquivo:

```powershell
python tools/run_ass_regression.py `
  --file ".\entrada\episodio.ass" `
  --experiment single_episode `
  --no-cache
```

Comparação de batch no mesmo agrupamento de execução:

```powershell
python tools/run_ass_regression.py --run-id investigacao_01 --experiment batch_1 --batch-size 1
python tools/run_ass_regression.py --run-id investigacao_01 --experiment batch_5 --batch-size 5
python tools/run_ass_regression.py --run-id investigacao_01 --experiment turbo --batch-size 15 --turbo
```

Relatório detalhado e promoção de warnings semânticos de alta confiança:

```powershell
python tools/run_ass_regression.py --experiment revisao --verbose --strict-semantic
```

O exit code é diferente de zero para corrupção objetiva, output ausente/incompleto ou
falha da rota de produção. Warnings de qualidade sem tradução humana de referência não
falham por padrão. Ollama/modelo indisponível retorna estado `NOT_EXECUTED`, nunca PASS.

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
