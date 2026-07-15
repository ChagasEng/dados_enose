# 06/07 - Melhor modelo ExtraTrees

## Resultado principal

Melhor cenario:

```text
MQ corrigido + ambiente confirmado
```

Metricas no teste:

```text
Accuracy: 90,83%
Balanced accuracy: 90,73%
F1 macro: 90,80%
```

Split usado:

```text
70/30 por grupos de Coleta dentro de cada classe
```

## Atencao: correcao por datasheet

Os sensores foram tratados com base nos datasheets e na identificacao fisica confirmada:

- `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`: datasheets MQ indicam dependencia por temperatura/umidade.
- `Temp_C` e `Pres_kPa`: confirmados como BMP280.
- `Soil_indice_0_1`: confirmado como Capacitive Soil Moisture Sensor V2.0.

Como o BMP280 nao mede umidade relativa do ar, a correcao completa fisica `Rs/R0` + RH ainda nao foi possivel. Portanto, a correcao usada no modelo e:

```text
MQ_corrigido_env = MQ_cru - efeito_estimado_de(Soil_indice_0_1, Temp_C, Pres_kPa)
```

Essa correcao e orientada pelos datasheets e pelos sensores ambientais confirmados, mas ainda deve ser apresentada como compensacao estatistica ambiental, nao como calibracao fisica final completa.

## Features usadas no melhor modelo

```text
MQ2_corrigido_env
MQ3_corrigido_env
MQ7_corrigido_env
MQ8_corrigido_env
MQ135_corrigido_env
MQ138_corrigido_env
Soil_indice_0_1
Temp_C
Pres_kPa
```

## Arquivos principais

- `graficos/painel_resumo_melhor_modelo.png`: resumo geral do melhor modelo.
- `graficos/comparacao_metricas_extratrees.png`: comparacao dos cenarios testados.
- `graficos/coletas_por_nematoide_sinais_corrigidos.png`: grafico das coletas com sinais corrigidos.
- `graficos/coletas_por_nematoide_atualizado_estilo_original.png`: grafico atualizado no mesmo estilo do grafico antigo de coletas por nematoide.
- `graficos/diagnostico_visual_C13_C17_C28.png`: zoom visual em C13-C17 e C28.
- `graficos/ranking_ruido_por_coleta.png`: ranking de coletas com mais saltos abruptos.
- `graficos/mq_corrigidos_zscore_por_coleta.png`: MQ corrigidos normalizados para visualizar curvas.
- `graficos/correlacao_ambiente_antes_depois_correcao.png`: correlacao ambiente x MQ antes/depois da correcao.
- `matriz_confusao/matriz_confusao_melhor_modelo.png`: matriz de confusao.
- `importancia_sensores/grafico_importancia_nativa_melhor_modelo.png`: importancia nativa do ExtraTrees.
- `importancia_sensores/grafico_importancia_permutacao_melhor_modelo.png`: importancia por permutacao.
- `codigo/rodar_extratrees_melhor_modelo.py`: codigo local para reproduzir o melhor modelo.
- `dados/dataset_melhor_modelo_sensores_corrigidos.csv`: dataset usado com colunas corrigidas.

## Leitura cientifica

O modelo melhorou quando usamos os MQ corrigidos junto com o contexto ambiental interno da camara. Porem, `Pres_kPa` aparece como variavel muito importante. Como essa pressao e medida dentro da camara de gases, ela pode carregar informacao real do ensaio, mas tambem pode carregar efeito fisico da caixa, vedacao, bomba ou manuseio.

Conclusao honesta:

```text
Depois da substituicao da C16 duplicada pelos dados corretos, o melhor modelo chegou a 90,83%. A interpretacao cientifica ainda exige validar se a pressao interna representa sinal biologico/respiratorio ou artefato fisico da camara.
```

## Diagnostico C13-C17 e C28

Foi adicionada uma analise especifica em `analise_C13_C17_C28/diagnostico_C13_C17_C28.md`.

Achados principais:

- `C28` nao e a coleta mais ruidosa em Pres.+MQ; ela tem poucos saltos, mas degraus grandes em alguns MQ e muito ruido no `Soil_indice_0_1`.
- `C17` e a coleta mais critica por ruido nos MQ, com 1484 saltos abruptos em Pres.+MQ, sendo 1480 nos MQ.
- A duplicacao entre `C15` e `C16` foi resolvida: a C16 foi substituida pelos dados corretos, filtrada e reprocessada como classe 0. As duas coletas agora possuem sinais diferentes.
