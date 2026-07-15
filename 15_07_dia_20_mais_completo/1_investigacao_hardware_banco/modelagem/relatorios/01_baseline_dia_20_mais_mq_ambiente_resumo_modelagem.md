# Modelagem - 01_baseline_dia_20_mais_mq_ambiente

Baseline do dia_20_mais completo, com MQ + ambiente.

Dataset: `dados\dataset_dia_20_mais_com_ambiente.csv`
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
