# Modelagem - 04_polido_final_mq

Base polida final, usando somente sensores MQ.

Dataset: `4_polimento_inicial_modelagem\datasets_limpos\antes_dia_20_pressao_filtrada_estrito.csv`
Linhas: `63373`
Coletas: `36`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138`

## ExtraTrees

- accuracy: 0.8996
- balanced accuracy: 0.9000
- f1 macro: 0.8996

## Rede neural MLP

- accuracy: 0.7912
- balanced accuracy: 0.7912
- f1 macro: 0.7910

## Importancia

Foram salvas tres leituras:

- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
