# Bateria de regressão ASS com Ollama real

## Objetivo

O projeto ganhará uma bateria manual e reproduzível para observar regressões de
tradução em todos os arquivos `.ass` ou `.ASS` encontrados em `entrada/`. A primeira
fase não mudará prompt, parser, batching, retry, cache, regras multilíngues nem
reconstrução: ela apenas executará o caminho de produção e medirá seus resultados.

## Arquitetura

A solução terá três camadas:

1. `ass_regression.py` conterá descoberta do corpus, hashes, IDs estáveis de eventos,
   comparação estrutural, heurísticas semânticas, métricas e renderização dos
   relatórios JSON e Markdown.
2. `translation_engine.py` aceitará um observador opcional de eventos de diagnóstico.
   Com ele desligado, o fluxo continuará idêntico. O observador receberá apenas fatos
   do caminho existente: prompts, respostas, IDs analisados, itens aceitos/rejeitados,
   retries e recuperação individual.
3. `tools/run_ass_regression.py` será o comando manual. Ele criará um snapshot dos
   originais, coletará contexto reprodutível e chamará a própria
   `translate_ass_fast.main()` com um observador, sem reimplementar a tradução.

Cada execução usará `test_results/<timestamp>/<experimento>/`, com subdiretórios
separados para fontes, outputs e traços brutos comprimidos. Relatórios resumidos não
repetirão eventos aprovados; detalhes completos ficarão restritos aos diagnósticos.
`test_results/` será ignorado pelo Git.

## Identidade e contexto

Cada arquivo será identificado por caminho relativo e SHA-256. Cada evento terá um ID
estável derivado de arquivo, índice, início, fim, estilo e hash do texto original.

O contexto registrará data/hora, commit, branch, estado Git, hash do estado do código,
Python, sistema operacional, endpoint/modelo Ollama, parâmetros efetivos, batch,
turbo, cache, idiomas, versão e hashes dos prompts, além das quantidades do corpus.
Mudanças locais preexistentes continuarão intactas e aparecerão apenas como contexto.

## Validação objetiva

A saída será comparada com a cópia de trabalho produzida por
`ASSFormatHandler.prepare_document()`, pois essa é a base sanitizada usada pela
tradução. Eventos `Comment` serão comparados integralmente com o original e nunca
serão avaliados como tradução pendente.

Serão falhas críticas, com exit code diferente de zero:

- output ausente, ilegível, truncado ou ASS inválido;
- seções, estilos, metadados, ordem, quantidade ou campos não textuais alterados;
- tags ASS, prefixos, URLs, e-mails, variáveis, `\\N`, `\\n` ou `\\h` alterados;
- placeholder, Markdown ou artefato de protocolo no resultado;
- evento traduzível vazio;
- âncora exclusiva de outro evento, duplicação/mapeamento comprovado ou falha
  definitiva reportada pela rota de produção.

O validador não relaxará nenhuma validação do backend e não usará fallback original
para tornar a execução verde.

## Alertas semânticos

Sem tradução humana de referência, heurísticas produzirão `WARNING`, não `FAIL`.
Cada alerta terá categoria, confiança, justificativa, arquivo, evento, original e
tradução. As heurísticas procurarão resíduo provável do idioma de origem, mistura
parcial, boilerplate do modelo, repetição, possível truncamento, mudança de números e
entidades, e outros desvios de alta utilidade diagnóstica com limites conservadores.

`--strict-semantic` poderá promover somente warnings de alta confiança durante uma
investigação explícita.

## Baseline e experimentos

A baseline usará os 14 arquivos atuais, `qwen2.5:14b`, configuração normal vigente,
cache desativado e fallback original desativado. Nenhum comportamento de tradução será
alterado antes dela.

Experimentos posteriores terão diretórios próprios: `batch_1`, `batch_2`, `batch_5`,
`batch_10`, `batch_15`, `turbo` e `repeat`. A amostra será selecionada por cobertura
de recursos ASS e IDs estáveis, permitindo comparar o mesmo evento em todas as
configurações. Diferenças serão classificadas como determinísticas, dependentes de
batch/posição, não determinísticas ou não reproduzidas.

## Testes e fases

Fixtures sintéticas pequenas cobrirão tags complexas, comentários, quebras, hífens,
campos não textuais, placeholders e contaminação. Testes simulados validarão também a
instrumentação e os relatórios, sem tornar Ollama obrigatório no pytest comum.

Após a baseline, cada bug confirmado seguirá RED → correção mínima → GREEN → suíte
completa → repetição real afetada. A fase 1 terminará com relatório, hipóteses por
evidência e a escolha do primeiro bug; mudanças de produção ficam para a fase 2.
