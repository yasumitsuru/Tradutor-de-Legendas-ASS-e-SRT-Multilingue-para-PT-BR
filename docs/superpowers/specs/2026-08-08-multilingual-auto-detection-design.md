# Detecção automática multilíngue por item

## Objetivo

O tradutor deixará de assumir inglês como origem. No modo padrão, cada ITEM completo
será interpretado semanticamente pelo modelo Ollama, que poderá reconhecer um ou mais
idiomas na mesma fala e devolver uma única versão natural em português do Brasil.

## Arquitetura

`source_language` terá `auto` como padrão. O texto visível preparado pelos handlers
ASS/SRT seguirá inteiro ao modelo, sem detector externo e sem fragmentação por idioma.
Tags, quebras e demais marcações protegidas continuarão opacas durante a tradução.

O system prompt será multilíngue e os prompts de lote e recuperação individual terão
uma instrução de origem produzida pelo backend:

- em `auto`, detectar idioma ou idiomas por ITEM, traduzir todo conteúdo estrangeiro,
  preservar conteúdo já natural em PT-BR e integrar a fala completa;
- em modo manual, considerar a string informada como idioma de origem, sem limitar a
  configuração a uma enumeração rígida.

Ambos os modos preservarão nomes próprios, honoríficos, siglas, termos técnicos e o
protocolo ITEM. Respostas explicativas, Markdown e identificações de idioma continuarão
inválidas pelas validações existentes.

As correções específicas de contrações inglesas deixarão de modificar o texto antes ou
depois da chamada ao modelo. A confiabilidade de `I`, `I'm`, `I've`, `I'll` e `I'd`
passará a ser responsabilidade explícita do prompt, coberta por regressões simuladas e
por smoke real. O skip de texto curto será generalizado apenas para aceitar palavras
alfabéticas significativas; símbolos, SFX e texto vazio manterão a semântica atual.

## Cache e compatibilidade

A versão do prompt e o schema do cache serão incrementados. A chave continuará
incluindo `source_language`, portanto `auto` não compartilhará traduções com `English`,
`German` ou outra origem manual. Caches anteriores serão ignorados.

A CLI oferecerá `--source-language`, com `auto` como padrão. Flask, Flet e PySide não
ganharão seletores adicionais: continuarão delegando ao mesmo backend e, ao omitir a
opção, usarão o modo automático. Seus títulos e textos deixarão de anunciar uma origem
exclusivamente inglesa.

## Garantias preservadas

Sanitização ASS, tags e escapes protegidos, timestamps, estilos, ordem, retry seletivo,
recuperação individual a temperatura zero, fallback original opt-in, manifesto da
execução, validação serializada e substituição atômica não serão enfraquecidos.

## Validação

Testes simulados cobrirão prompts automáticos e manuais, falas multilíngues, conteúdo
PT-BR, nomes e honoríficos, contrações inglesas, palavras isoladas, skip, cache, CLI e
textos públicos. A suíte completa será seguida por um smoke temporário com
`qwen2.5:14b`, incluindo ASS/SRT, idiomas mistos, palavras isoladas, tags e `\N`, com
`allow_original_fallback=False` e varredura contra artefatos internos.
