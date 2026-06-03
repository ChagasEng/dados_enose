# Random Forest

Esta pasta contem somente o classificador RandomForest treinado com o dataset sem pressao.

## Estrutura

- `scripts/`: script de treinamento do RandomForest.
- `modelos/`: modelo treinado em formato `.joblib`.
- `resultados/matrizes/`: matrizes de confusao em CSV e PNG.
- `resultados/metricas/`: metricas, resumo de split e configuracoes.
- `resultados/relatorios/`: relatorios de classificacao.
- `resultados/importancias/`: importancia das features do RandomForest.

## Dataset usado

```text
sem pressao/dataset_sem_pressao.csv
```

## Configuracao

- Features usadas: `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`.
- Target: `Classe`, com `0 = doente` e `1 = saudavel`.
- Split: 70/30 por grupos de `Coleta` dentro de cada classe.
- A coluna `Coleta` nao entra no treinamento; ela e usada somente para separar treino e teste sem vazamento.

## Resultado principal

- Acuracia: `0.8328`.
- Matriz de confusao: `[[8792, 3033], [1228, 12435]]`.
