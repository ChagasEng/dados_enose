# Modelagem - 02_filtrado_pressao_dia_20_mais_mq

Dia_20_mais apos corte estrito por pressao, somente MQ.

Dataset: `comparacao\pressao_filtrada\dia_20_mais_pressao_filtrada_estrito.csv`
Linhas: `27195`
Coletas: `17`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138`

## ExtraTrees

- accuracy: 0.8187
- balanced accuracy: 0.8083
- f1 macro: 0.8080

## Rede neural MLP

- accuracy: 0.7969
- balanced accuracy: 0.7862
- f1 macro: 0.7844

## Importancia

Foram salvas tres leituras:

- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
