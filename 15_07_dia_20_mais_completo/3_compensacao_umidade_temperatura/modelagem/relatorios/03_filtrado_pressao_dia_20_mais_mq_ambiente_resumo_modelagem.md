# Modelagem - 03_filtrado_pressao_dia_20_mais_mq_ambiente

Dia_20_mais apos corte estrito por pressao, com MQ + ambiente.

Dataset: `comparacao\pressao_filtrada\dia_20_mais_pressao_filtrada_estrito.csv`
Linhas: `27195`
Coletas: `17`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138, Soil, Temp., Pres.`

## ExtraTrees

- accuracy: 0.8443
- balanced accuracy: 0.8354
- f1 macro: 0.8372

## Rede neural MLP

- accuracy: 0.8166
- balanced accuracy: 0.8062
- f1 macro: 0.8057

## Importancia

Foram salvas tres leituras:

- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
