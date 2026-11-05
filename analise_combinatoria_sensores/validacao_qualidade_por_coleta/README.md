# LOSO com controle de qualidade por coleta

Esta analise mantem **os dez vasos** e o conjunto fixo de entradas MQ2, MQ3, MQ8, MQ135, MQ138, indice de solo, temperatura e pressao. O modelo e Extra Trees (900 arvores, `max_features=sqrt`, `min_samples_leaf=10`, semente 42). Cada fold do LOSO treina em nove vasos e testa o decimo; a metrica e acuracia por leitura crua.

Como teste exploratorio de qualidade do sinal, foram sinalizadas as coletas nas quais pelo menos 5% das leituras continham saltos abruptos em dois ou mais canais MQ, segundo a auditoria de eventos com limiares por canal no percentil 99,5. A regra sinalizou apenas duas coletas: `dia 19 - Soja Heterodera Vaso 1` (362/1.261; 28,71%) e `dia 19 - Soja Saudavel Vaso 1` (86/1.271; 6,77%). As outras 34 coletas, inclusive as do dia 18 desses vasos, permaneceram. Esta regra foi formulada depois da inspecao dos dados e serve como analise de sensibilidade.

| Cenario | Vasos | Coletas | Acertos / leituras | Acuracia |
|---|---:|---:|---:|---:|
| LOSO original | 10 | 36 | 55.916 / 63.492 | 88,07% |
| Previsoes originais nas 34 coletas restantes | 10 | 34 | 54.675 / 60.960 | 89,69% |
| LOSO refeito nas 34 coletas restantes | 10 | 34 | 54.655 / 60.960 | 89,66% |

Comparar as **mesmas 34 coletas** mostra que o novo treino nao elevou a acuracia: foram 20 acertos a menos. A diferenca entre 88,07% e 89,66% se deve sobretudo a retirar duas coletas dificeis da avaliacao. Nao e correto interpretar 89,66% como estimativa validada de desempenho futuro; o filtro foi definido apos examinar esta base.

O ensaio anterior que alcancou 96,29% retirou **tres vasos inteiros por concentrarem erros de classificacao**. Essa exclusao removeu tambem coletas boas e mudou a populacao avaliada. Ele permanece documentado em `../validacao_sem_vasos_criticos/`, mas nao representa a limpeza de qualidade por coleta. Erros de classificacao, por si, nao sao criterio de exclusao: as coletas `dia 12 - Soja Saudavel Vaso 2`, `dia 19 - Soja Saudavel Vaso 2` e `dia 19 - Soja Heterodera Vaso 9` tiveram muitos erros no LOSO original, embora nao apresentassem eventos multissensor na auditoria.

Para reproduzir: `.\.venv\Scripts\python.exe analise_combinatoria_sensores\avaliar_loso_qualidade_por_coleta.py` a partir da raiz do projeto. Os arquivos `comparacao_metricas.csv`, `coletas_sinalizadas.csv`, `predicoes_loso.csv` e `metodo.json` guardam a comparacao, os registros filtrados, as previsoes e os parametros.
