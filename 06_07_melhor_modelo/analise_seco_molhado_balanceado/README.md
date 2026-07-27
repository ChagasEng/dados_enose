# Avaliacao balanceada por condicao de solo

O modelo nao foi retreinado. Foram usadas somente as predicoes do conjunto Teste ja separado por Coleta, preservando a independencia em relacao ao treino.

## Como foi balanceado

- `seco`: 6507 doentes + 6507 saudaveis; 3678 saudaveis restantes foram para validacao.
- `molhado`: 1271 doentes + 1271 saudaveis; 2881 doentes restantes foram para validacao.
- Amostragem aleatoria reprodutivel com `random_state=42`. Nenhuma linha aparece nos dois conjuntos.

## Resultado

| Conjunto | Doentes | Saudaveis | Acertos | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Avaliacao balanceada | 7778 | 7778 | 13955 / 15556 | 89.71% | 89.71% |
| Validacao com sobra | 2881 | 3678 | 6132 / 6559 | 93.49% | 92.59% |

A validacao com sobra combina os excedentes das duas faixas; por isso ela volta a conter as duas classes. Ainda assim, os resultados devem ser lidos junto com os resultados por faixa, pois cada sobra isolada contem apenas a classe que era majoritaria naquela faixa.

## Arquivos

- `avaliacao_balanceada.csv`: conjunto equilibrado para comparar doente e saudavel.
- `validacao_sobra.csv`: linhas nao usadas na avaliacao balanceada.
- `resumo_metricas.csv`: metricas dos dois conjuntos.
- `grafico_metricas_acuracia_f1_recall.png`: painel de acuracia, F1 e recall por classe.
- `grafico_metricas_seco_molhado_balanceado.png`: comparacao direta seco x molhado, com as classes balanceadas em cada faixa.
- `resumo_metricas_por_faixa_balanceada.csv`: valores exibidos no grafico seco x molhado.
- `matriz_confusao_*.csv/png`: erros por classe.
