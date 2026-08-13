# Plano de implementação da regressão ASS real

## Fase 1 — observabilidade e baseline

1. Adicionar fixtures ASS determinísticas com comentários, campos não textuais, tags
   simples/complexas, quebras, prefixos, Unicode e âncoras de contaminação.
2. Adicionar testes RED para descoberta case-insensitive, hashes, IDs estáveis,
   comparação estrutural, preservação de `Comment`, tags/quebras/placeholders,
   outputs ausentes/truncados e warnings semânticos.
3. Implementar `ass_regression.py` com dataclasses de diagnóstico/relatório, análise de
   pares ASS e serialização JSON/Markdown.
4. Adicionar testes RED para o observador opcional: tentativas de batch, resposta,
   itens aceitos/rejeitados, retry seletivo e recuperação individual, garantindo que
   a ausência de observador preserve resultados e chamadas.
5. Implementar eventos de diagnóstico mínimos em `translation_engine.py` e passagem
   opcional do observador por `translate_ass_fast.main()`.
6. Adicionar testes RED do runner para argumentos, diretórios por experimento,
   snapshot/hashes, contexto Git/Python/Ollama, trace comprimido, exit codes e modo de
   análise sem Ollama.
7. Implementar `tools/run_ass_regression.py`, usando a CLI de produção para traduzir e
   `ass_regression.py` para analisar, sem duplicar lógica do backend.
8. Adicionar `test_results/` ao `.gitignore`, documentar o comando manual e executar
   testes direcionados seguidos de `python -m pytest tests -q`.
9. Confirmar Ollama e `qwen2.5:14b`; registrar hashes dos 14 originais e executar a
   baseline completa, normal, sem cache e sem fallback.
10. Verificar novamente os hashes do corpus, analisar criticidades/warnings, revisar o
    relatório e encerrar a fase 1 com hipóteses ordenadas por evidência.

## Fase 2 — correções orientadas por evidência

11. Escolher a falha real mais reproduzível e criar um teste RED permanente.
12. Determinar a causa raiz com comparação de trace, batch 1 e batch padrão.
13. Aplicar a menor correção robusta no componente responsável e executar o teste
    direcionado, a suíte completa e a amostra real afetada.
14. Repetir para cada bug confirmado, sem combinar sintomas independentes em uma
    alteração ampla.
15. Executar a matriz representativa de batches, normal/turbo, repetição controlada,
    cache separado e uma rota CLI end-to-end.
16. Gerar comparação antes/depois, revisar o diff e entregar causas, arquivos,
    testes, métricas, limitações e comandos de repetição.
