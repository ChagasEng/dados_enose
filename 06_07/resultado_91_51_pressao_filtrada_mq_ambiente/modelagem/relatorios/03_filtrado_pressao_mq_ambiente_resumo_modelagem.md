# Modelagem - 03_filtrado_pressao_mq_ambiente

Apos corte estrito por pressao, usando MQ + Soil + Temp. + Pres.

Dataset: `3_compensacao_umidade_temperatura\dados_base\antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv`
Linhas: `63373`
Coletas: `36`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138, Soil, Temp., Pres.`

## ExtraTrees

- accuracy: 0.9151
- balanced accuracy: 0.9128
- f1 macro: 0.9143

## Rede neural MLP

- accuracy: 0.8994
- balanced accuracy: 0.9012
- f1 macro: 0.8994

## Importancia

Foram salvas tres leituras:

- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
