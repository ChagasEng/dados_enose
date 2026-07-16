# Acertos por condicao do solo

Esta analise usa o mesmo ExtraTrees e o mesmo conjunto de teste por coleta do melhor modelo. Ela nao retreina um modelo para solo seco ou molhado: apenas separa as predicoes do teste para comparar a generalizacao em cada condicao.

## Regra operacional

- `seco`: `Soil_indice_0_1 <= 0.4`
- `molhado`: `Soil_indice_0_1 > 0.4`

O indice e uma normalizacao min-max da leitura analogica `Soil`. A convencao desta rodada foi definida como indice ate 0,4 = seco e acima de 0,4 = molhado. Portanto, os nomes representam faixas operacionais do experimento, nao percentual fisico de agua no solo.

## Resultado no teste

| Faixa | Linhas | Acertos | Accuracy | Balanced accuracy | Acerto classe 0 | Acerto classe 1 |
|---|---:|---:|---:|---:|---:|---:|
| Molhado | 5423 | 4038 | 74.46% | 62.96% | 84.61% | 41.31% |
| Seco | 16692 | 16049 | 96.15% | 95.06% | 90.12% | 100.00% |

A accuracy no solo molhado ficou -21.69 pontos percentuais acima da faixa seca. Como as faixas possuem proporcoes diferentes das classes, a balanced accuracy e indispensavel para comparar o desempenho sem deixar a classe majoritaria mascarar o resultado.

## Arquivos

- `resumo_acertos_seco_molhado.csv`: comparacao consolidada.
- `predicoes_teste_com_faixa_soil.csv`: cada predicao do teste, com o indice e a faixa.
- `matriz_confusao_molhado.*` e `matriz_confusao_seco.*`: erros por classe em cada faixa.
- `comparacao_acertos_seco_molhado.png`: grafico de acertos.
