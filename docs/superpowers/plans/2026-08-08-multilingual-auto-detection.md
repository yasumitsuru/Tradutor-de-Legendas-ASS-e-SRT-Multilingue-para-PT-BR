# Plano de implementação multilíngue

1. Adicionar testes RED em `tests/test_translation_engine.py` para:
   - configuração `auto` e prompts multilíngues;
   - conteúdo misto e PT-BR já existente;
   - nomes/honoríficos e marcadores protegidos;
   - `I`, `I'm`, `I've`, `I'll` e `I'd`;
   - palavras isoladas significativas sem regressão de símbolos, SFX e vazio;
   - invalidação do schema anterior e distinção `auto`/manual no cache.
2. Adicionar testes RED de CLI e textos públicos para o padrão e a propagação de
   `--source-language`, além do novo nome nas interfaces e no README.
3. Executar somente os testes novos e confirmar falhas pela ausência da funcionalidade.
4. Alterar `translation_engine.py`:
   - padrão `source_language=auto`;
   - system prompt e prompts ITEM multilíngues;
   - origem manual opcional;
   - remoção das transformações específicas de inglês no caminho ativo;
   - generalização segura de texto curto;
   - novas versões de prompt e cache.
5. Alterar `translate_ass_fast.py` para aceitar e propagar `--source-language`.
6. Atualizar `app_flet.py`, `app_gui.py`, `templates/index.html` e `README.md` sem
   duplicar detecção nem adicionar um seletor complexo.
7. Executar testes direcionados e `python -m pytest tests -q`.
8. Criar fixtures temporárias fora do versionamento e executar smoke real com
   `qwen2.5:14b`, `OLLAMA_HOST=http://127.0.0.1:11434` somente no processo e fallback
   original desativado.
9. Inspecionar resultados e varrer tokens ITEM, placeholders, mensagens do modelo e
   blocos de raciocínio.
10. Reexecutar suíte completa e `git diff --check`; revisar integralmente o diff.
11. Fazer stage explícito, revisar o índice e criar o commit convencional.
12. Buscar alterações remotas, integrar sem force se necessário e publicar `main`.
13. Renomear o repositório GitHub, aplicar a descrição exata, atualizar `origin` e
    confirmar metadados, hashes e funcionamento local com uma última suíte completa.
