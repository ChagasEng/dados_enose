# ExtraTrees com sensores ambientais confirmados

Rodada criada depois de confirmar o hardware ambiental:

- `Temp.` e `Pres.`: BMP280.
- `Soil`: Capacitive Soil Moisture Sensor V2.0.

## Base usada

`3_compensacao_umidade_temperatura\dados_base\antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv`

A base ja estava com corte estrito por pressao. Nesta rodada foram criadas colunas interpretadas:

- `Temp_C`: temperatura do BMP280.
- `Pres_kPa`: pressao em kPa, mantendo a escala 93.x do dataset.
- `Pres_hPa`: pressao convertida para hPa.
- `Soil_indice_0_1`: normalizacao operacional do sensor capacitivo de solo.

## Correcao aplicada

Como o BMP280 nao mede umidade relativa do ar, nao foi aplicada uma correcao completa de datasheet por RH. A correcao feita aqui e estatistica: para cada MQ, ajustei no treino um HuberRegressor usando `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`, removendo do MQ a componente linear associada ao ambiente. Isso reduz efeito ambiental sem fingir que `Soil` e RH.

## Resultados ExtraTrees

| cenario | accuracy | balanced_accuracy | f1_macro | top1_nativo | top1_permutacao |
|---|---:|---:|---:|---|---|
| 01_mq_cru | 0.8986 | 0.8990 | 0.8986 | MQ8 | MQ8 |
| 02_mq_ambiente_confirmado | 0.8793 | 0.8789 | 0.8791 | MQ8 | Pres_kPa |
| 03_mq_corrigido_ambiente | 0.8648 | 0.8637 | 0.8643 | MQ138_corrigido_env | MQ135_corrigido_env |
| 04_mq_corrigido_ambiente_com_contexto | 0.9083 | 0.9073 | 0.9080 | Pres_kPa | Pres_kPa |

Melhor cenario por accuracy: `04_mq_corrigido_ambiente_com_contexto` com accuracy `0.9083`.

## Arquivos principais

- `dados_processados/dataset_sensores_confirmados_com_correcoes.csv`
- `modelagem/metricas/resumo_extratrees_sensores_confirmados.csv`
- `modelagem/importancias/`
- `modelagem/matrizes/`
- `graficos/01_coletas_por_nematoide_sinais_crus_confirmados.png`
- `graficos/02_coletas_por_nematoide_sinais_corrigidos_overlay.png`
- `graficos/03_coletas_por_nematoide_mq_corrigidos_zscore.png`
- `graficos/04_correlacao_ambiente_antes_depois_correcao.png`
- `graficos/05_comparacao_metricas_extratrees.png`
