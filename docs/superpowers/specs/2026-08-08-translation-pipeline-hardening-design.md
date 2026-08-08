# Endurecimento do pipeline de tradução ASS/SRT

## Objetivo

Corrigir três falhas relacionadas: aceitar itens válidos mesmo quando a resposta do
modelo contém resíduo fora do protocolo, distinguir comandos ASS reais de comentários
textuais entre chaves e impedir que uma tradução parcialmente original seja salva como
concluída sem autorização explícita.

O resultado deve preservar horários, ordem dos eventos, estilos, comandos ASS válidos,
quebras e compatibilidade SRT. Nenhuma lógica de fallback será duplicada nas interfaces.

## Abordagens consideradas

### 1. Correções locais em cada interface

Cada interface poderia bloquear o download ou apagar uma saída quando encontrasse
falhas. Essa abordagem foi rejeitada porque duplicaria regras, permitiria diferenças de
semântica entre CLI, Flask, Flet e PySide e não protegeria consumidores diretos do motor.

### 2. Salvar um arquivo parcial com aviso mais forte

O motor poderia continuar copiando o original e apenas destacar o risco no log. Essa
abordagem foi rejeitada porque não atende ao requisito central: uma saída misturando
idiomas continuaria existindo e poderia ser confundida com uma tradução concluída.

### 3. Decisão centralizada no backend, com compatibilidade explícita

Esta é a abordagem escolhida. O motor executará batch, retries seletivos e recuperação
individual. Se ainda houver falhas, uma exceção específica impedirá o salvamento por
padrão. Somente `allow_original_fallback=True`, propagado por
`--allow-original-fallback`, permitirá reconstruir e salvar usando o texto original nos
itens definitivamente inválidos. Todas as interfaces apenas exibem e repassam a opção.

## Desenho do protocolo ITEM

`_parse_item_response()` continuará extraindo blocos completos e eliminando IDs
duplicados. O segundo resultado será tratado como warning global, não como erro de cada
item. Texto ou Markdown fora dos blocos será descartado e registrado no log, sem jamais
entrar na legenda.

Cada item pendente continuará sendo validado isoladamente. Permanecem pendentes itens
ausentes, duplicados, vazios, estruturalmente inválidos, com marcadores corrompidos,
Markdown no corpo ou comandos ASS inventados. Assim que um item for validado, ele sai de
`pending` e não aparece nos prompts seguintes.

Depois dos retries normais, cada item ainda pendente terá uma última tentativa em uma
requisição própria, com prompt mínimo, os mesmos delimitadores, validação normal e
temperatura `0.0`. Resíduo externo nessa resposta também será apenas warning.

## Sanitização ASS

O handler ASS ganhará uma função explícita e idempotente para classificar e sanitizar
blocos `{...}`. O classificador será um pequeno scanner, não uma regex genérica nem um
teste baseado apenas na presença de `\`. Ele reconhecerá nomes de comandos ASS
suportados, consumirá seus argumentos e validará parênteses balanceados, inclusive em
comandos complexos e transforms com comandos aninhados.

Uma cadeia formada integralmente por override commands reconhecidos será preservada
byte/textualmente como recebida. Isso inclui, entre outros, `\i`, `\b`, `\an`, cores,
alpha, posicionamento, movimento, clip, bordas, sombras, karaoke, `\fad`, `\fade` e
`\t(...)`. Exemplos obrigatórios incluem `\pos(100,200)`,
`\move(10,20,100,200)`, `\clip(0,0,1920,1080)`,
`\t(0,500,\fs40\bord3)`, `\c&HFFFFFF&`, `\1c&HFFFFFF&` e
`\alpha&H80&`.

Blocos formados apenas por linguagem natural ou anotações serão removidos. Em blocos
mistos, o texto natural será descartado e somente comandos ASS reconhecidos e
estruturalmente válidos serão reconstruídos em um bloco limpo. Escapes `\N`, `\n` e
`\h` não fazem parte da lista de override commands do scanner e, quando encontrados
dentro de um comentário misto, não serão promovidos a estrutura visível.
Por exemplo:

```text
{The key is in the basket,\Nif you wouldn’t mind.\i0}
```

será sanitizado para:

```text
{\i0}
```

A sanitização acontecerá uma única vez, logo após o carregamento. O handler criará uma
cópia de trabalho do documento e sanitizará somente os eventos ASS `Dialogue`,
preservando eventos `Comment` intactos. Preparação para o modelo, skip, chaves de cache,
cache hits, retries, validação e reconstrução usarão essa mesma cópia sanitizada.
`handler.rebuild()` apenas aplicará traduções sobre essa base e não sanitizará novamente.
Assim, linhas puladas, linhas vindas do cache e linhas traduzidas obedecerão à mesma
regra. A idempotência será garantida por teste:

```python
assert sanitize_ass_text(sanitize_ass_text(text)) == sanitize_ass_text(text)
```

As validações estruturais compararão a forma sanitizada da entrada com a saída. Tags
reais continuam protegidas por tokens opacos e devem voltar na mesma ordem e grafia.

## Falha definitiva e salvamento

O config compartilhado terá `allow_original_fallback=False`. Resultados definitivamente
inválidos continuarão carregando erro e texto original internamente para diagnóstico e
para o modo legado, mas serão contabilizados como falhas.

Antes de reconstruir ou salvar, `translate_file()` verificará se existem itens com falha.
Quando o fallback não estiver autorizado, lançará uma exceção específica com mensagem
clara, incluindo a quantidade de itens, e não chamará `handler.save()`. Uma saída
preexistente não será apagada nem sobrescrita. O CLI marcará o arquivo como falho e
retornará código diferente de zero, registrando explicitamente que nenhum novo output
foi produzido para o arquivo.

Com `allow_original_fallback=True`, o fluxo antigo será permitido explicitamente: o
arquivo será salvo, mantendo o texto original somente nos itens definitivamente falhos,
e o log continuará mostrando a contagem de falhas.

Mesmo no caminho de sucesso, a serialização será feita em um candidato temporário. O
candidato será reaberto e validado integralmente; somente então uma substituição atômica
atualizará o destino final. Se qualquer etapa falhar, o destino anterior permanece
inalterado.

O backend manterá um manifesto da execução contendo apenas arquivos efetivamente
produzidos e validados naquela invocação. As interfaces passarão um caminho de manifesto
ao CLI e usarão o mesmo leitor compartilhado para listas e ZIP/download. O manifesto é
inicializado vazio no começo da execução e atualizado atomicamente após cada sucesso.
Assim, uma saída antiga que permaneça no diretório após uma nova falha não será
apresentada como resultado novo nem incluída no download da execução falha.

## CLI e interfaces

O parser da CLI adicionará `--allow-original-fallback`, `store_true`, desativado por
padrão. Esse valor será copiado para o config único do tradutor.

Flask, Flet e PySide exibirão uma opção desmarcada com o texto:

> Permitir manter texto original quando a tradução falhar

Próximo a ela aparecerá o aviso:

> Atenção: isso pode gerar legendas misturando PT-BR com o idioma original.

Cada interface acrescentará `--allow-original-fallback` ao comando somente quando a
opção estiver ativa e passará seu manifesto de execução ao backend. A decisão de
fallback e salvamento permanecerá exclusivamente em `translation_engine.py`; leitura e
filtragem do manifesto usarão um módulo compartilhado.

## Cache

`PROMPT_VERSION` será incrementada porque os prompts e a preparação mudam.
`CACHE_SCHEMA_VERSION` também será incrementada para invalidar explicitamente entradas
anteriores. Além disso, a validação de cache ASS não aceitará como valor final uma string
que mude ao ser sanitizada. `--clear-cache` continuará removendo o mesmo arquivo de
cache.

## Testes

Os testes serão escritos antes do código e executados inicialmente para confirmar as
falhas esperadas. A cobertura incluirá:

1. resíduo externo descartado sem invalidar um ITEM válido;
2. retry contendo somente o item ausente ou inválido;
3. remoção de comentário ASS simples;
4. preservação exata de tags ASS verdadeiras;
5. bloco misto que remove inglês e preserva `\i0`;
6. múltiplos comentários removidos com `\i1`/`\i0` preservados;
7. idempotência da sanitização ASS;
8. preservação textual de `\pos`, `\move`, `\clip`, `\t`, `\fad`, `\fade`, cores e
   alpha em blocos válidos;
9. recuperação individual depois dos retries normais, com temperatura zero passada
   somente à chamada, sem mutar `self.config`;
10. falha definitiva sem criar ou sobrescrever saída no modo seguro;
11. saída antiga preservada, execução marcada como falha e arquivo antigo ausente do
    manifesto/lista/ZIP da nova execução;
12. comportamento legado quando o fallback é explicitamente permitido;
13. padrão `False` e propagação correta na CLI, Flask, Flet e PySide;
14. invalidação do cache anterior e funcionamento de `--clear-cache`.

Depois dos testes direcionados, a validação final obrigatória será:

```powershell
python -m pytest tests -q
```

## Limites

Não haverá refatoração não relacionada. Arquivos de legenda, horários, estilos e ordem
dos eventos não serão modificados fora das substituições de texto descritas. As
remoções locais preexistentes de `entrada/.gitkeep` e `saida/.gitkeep` não serão tocadas.
Nenhum commit ou push será realizado sem autorização do usuário.
