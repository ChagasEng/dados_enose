# Modelagem - 01_baseline_antes_corte_mq_ambiente

Baseline antes do corte por pressao, usando MQ + ambiente.

Dataset: `1_investigacao_hardware_banco\dados_base\antes_dia_20_com_ambiente_baseline.csv`
Linhas: `70894`
Coletas: `36`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138, Soil, Temp., Pres.`

## ExtraTrees

- accuracy: 0.8758
- balanced accuracy: 0.8692
- f1 macro: 0.8728

## Rede neural MLP

- accuracy: 0.8274
- balanced accuracy: 0.8353
- f1 macro: 0.8270

## Importancia

Foram salvas tres leituras:

- importancia nativa do ExtraTrees;
- importancia por permutacao do ExtraTrees;
- importancia por permutacao da rede neural.
