# 4. Polimento inicial e modelagem

## Objetivo

Guardar a base ja filtrada e os resultados do modelo treinado somente apos remover trechos suspeitos de variacao abrupta de pressao.

## Dataset usado

`datasets_limpos/antes_dia_20_pressao_filtrada_estrito.csv`

Esse dataset possui 63.373 linhas apos remover:

- janelas ao redor de variacoes abruptas de `Pres.`;
- pontos fora da faixa estavel da pressao na versao estrita.

## Modelo rodado

ExtraTrees com as features MQ:

`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`

Split usado:

- 70/30 por grupos de `Coleta`;
- validacao interna tambem por grupos de `Coleta`.

## Resultado principal

Com ExtraTrees MQ-only:

- accuracy: 89.96%
- balanced accuracy: 90.00%
- f1 macro: 89.96%

Com rede neural MLP MQ-only:

- accuracy: 79.12%
- balanced accuracy: 79.12%
- f1 macro: 79.10%

Feature mais importante no ExtraTrees: `MQ8`.

O limiar ajustado `0.72` ficou pior no teste anterior, com accuracy de 87.98%. Portanto, neste dataset filtrado, o limiar padrao `0.50` continua sendo a melhor leitura operacional para o ExtraTrees MQ-only.

## Arquivos nesta pasta

- `datasets_limpos/`: dataset limpo e separado por classe.
- `graficos/`: split treino/teste e curvas por nematoide apos filtro.
- `resultados_modelo/`: metricas, matriz de confusao e importancia dos sensores.
- `scripts/`: scripts para reproduzir treino e grafico.
- `modelagem/`: nova rodada completa com ExtraTrees, rede neural e importancia dos sensores.

## Cuidado metodologico

Nao extrair area sob curva, media por coleta ou features novas antes de limpar os trechos fisicamente suspeitos. A ordem correta e limpar, validar visualmente, preservar mapa das coletas e so depois treinar ou extrair caracteristicas.
