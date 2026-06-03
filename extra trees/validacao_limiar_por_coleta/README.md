# Validacao do limiar por Coleta

Esta subpasta valida se o limiar `0.57` do ExtraTrees e estavel quando o dataset e dividido de formas diferentes por `Coleta`.

## O que foi feito

- Foi usada validacao cruzada aninhada com `StratifiedGroupKFold`.
- O grupo usado foi `Coleta`, para evitar que amostras da mesma coleta aparecam em treino e teste ao mesmo tempo.
- Em cada fold externo, o limiar foi escolhido apenas usando folds internos do treino.
- O teste externo ficou isolado e foi usado somente para avaliar o resultado.

## Dataset usado

```text
sem pressao/dataset_sem_pressao.csv
```

## Features e alvo

- Features: `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`.
- Target: `Classe`, com `0 = doente` e `1 = saudavel`.
- `Coleta` e usada somente para separacao dos folds.

## Resultados

- Limiares por fold: `0.47`, `0.60`, `0.66`, `0.58`, `0.58`.
- Limiar medio: `0.578`.
- Limiar mediano: `0.58`.
- Acuracia media com limiar padrao `0.50`: `0.8684`.
- Acuracia media com limiar validado por fold: `0.8869`.
- Acuracia media com limiar fixo `0.57`: `0.8908`.

## Arquivos

- `scripts/validar_limiar_extra_trees_cv.py`: script da validacao cruzada.
- `resultados/comparacoes/validacao_cruzada_limiar_extra_trees.csv`: resultados por fold externo.
- `resultados/comparacoes/validacao_cruzada_limiar_extra_trees_detalhes_internos.csv`: busca de limiar nos folds internos.
- `resultados/comparacoes/limiares_extra_trees_cv.png`: grafico dos limiares por fold.
- `resultados/matrizes/matriz_confusao_extra_trees_cv_limiar_validado.csv`: matriz somada da validacao.
- `resultados/metricas/metricas_validacao_cruzada_limiar_extra_trees.json`: resumo das metricas.
