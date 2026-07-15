# Dia 20+ - reproducao completa do fluxo 06_07

## Bases

- Base sem ambiente: `dados/dataset_dia_20_mais.csv` (27195 linhas)
- Base com ambiente: `dados/dataset_dia_20_mais_com_ambiente.csv` (27195 linhas)
- Base apos corte estrito por pressao: 27195 linhas

As duas bases de entrada foram validadas: possuem as mesmas linhas, coletas,
classes e leituras MQ. A segunda apenas acrescenta `Tempo`, `Soil`, `Temp.` e
`Pres.`.

## Cenarios

1. baseline MQ + ambiente;
2. corte estrito por pressao, somente MQ;
3. corte estrito por pressao, MQ + ambiente;
4. base polida final, somente MQ.

Todos usam split 70/30 por grupos de `Coleta`, dentro de cada classe.

## Melhor resultado

- Cenario: `01_baseline_dia_20_mais_mq_ambiente`
- Modelo: ExtraTrees
- Accuracy: 0.8443

Os resultados de cada cenario estao nas pastas numeradas; o comparativo geral
esta em `modelagem_comparativa/`.
