# XGBoost

Esta pasta concentra os experimentos com `XGBClassifier` para classificacao das amostras de soja.

## Estrutura

- `scripts/`: scripts de treinamento e comparacao do XGBoost.
- `modelos/`: modelos treinados em formato `.joblib`.
- `resultados/matrizes/`: matrizes de confusao em CSV e PNG.
- `resultados/metricas/`: metricas, resumo de split e configuracoes dos experimentos.
- `resultados/relatorios/`: relatorios de classificacao.
- `resultados/importancias/`: importancia das features.
- `resultados/comparacoes/`: comparacao entre configuracoes do XGBoost.

## Dataset usado

```text
sem pressao/dataset_sem_pressao.csv
```

As features usadas nos modelos sao:

```text
MQ2, MQ3, MQ7, MQ8, MQ135, MQ138
```

A coluna `Classe` e o alvo. A coluna `Coleta` e usada apenas para separar treino e teste sem vazamento.
