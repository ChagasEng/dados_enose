# Modelos de classificacao

Esta pasta concentra os experimentos de classificacao das amostras de soja usando apenas os sensores MQ como entrada.

## Estrutura

- `scripts/`: scripts de treinamento e comparacao dos modelos.
- `modelos/`: modelos treinados em formato `.joblib`.
- `resultados/matrizes/`: matrizes de confusao em CSV e PNG.
- `resultados/metricas/`: metricas, resumo de split e configuracoes dos experimentos.
- `resultados/relatorios/`: relatorios de classificacao.
- `resultados/importancias/`: importancia das features dos modelos.
- `resultados/comparacoes/`: comparacao entre tecnicas de machine learning.

## Dataset usado

O treinamento final usa:

```text
sem pressao/dataset_sem_pressao.csv
```

As features usadas nos modelos sao:

```text
MQ2, MQ3, MQ7, MQ8, MQ135, MQ138
```

A coluna `Classe` e o alvo. A coluna `Coleta` e usada apenas para separar treino e teste sem vazamento.
