# Extra Trees

Esta pasta contem os experimentos do classificador ExtraTreesClassifier, separados da pasta de RandomForest.

## Estrutura

- `scripts/`: treinamento, ajuste de limiar e validacao cruzada do ExtraTrees.
- `modelos/`: modelos treinados em formato `.joblib`.
- `resultados/matrizes/`: matrizes de confusao em CSV e PNG.
- `resultados/metricas/`: metricas e configuracoes dos experimentos.
- `resultados/relatorios/`: relatorios de classificacao.
- `resultados/importancias/`: importancia das features.
- `resultados/comparacoes/`: comparacoes de tecnicas e validacoes do limiar.
- `validacao_limiar_por_coleta/`: validacao cruzada especifica para conferir se o limiar `0.57` e estavel em diferentes divisoes por `Coleta`.

## Dataset usado

```text
sem pressao/dataset_sem_pressao.csv
```

## Configuracao

- Features usadas: `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`.
- Target: `Classe`, com `0 = doente` e `1 = saudavel`.
- Split: 70/30 por grupos de `Coleta` dentro de cada classe.
- A coluna `Coleta` nao entra no treinamento; ela e usada somente para separar treino e teste sem vazamento.

## Resultados principais

- ExtraTrees sem ajuste de limiar: acuracia `0.8967`, balanced accuracy `0.8959`.
- ExtraTrees com limiar ajustado: acuracia `0.9265`, balanced accuracy `0.9280`.
- Limiar validado: `0.57`.
- Matriz de confusao com limiar ajustado: `[[11221, 604], [1269, 12394]]`.

## Validacao do limiar

O limiar `0.57` foi escolhido em validacao interna separada por `Coleta`, sem usar o teste final para decidir. Depois, foi conferido com validacao cruzada aninhada por `Coleta`.

Os arquivos dessa etapa ficam em:

```text
extra trees/validacao_limiar_por_coleta/
```

Resumo da validacao cruzada:

- Limiares por fold: `0.47`, `0.60`, `0.66`, `0.58`, `0.58`.
- Limiar medio: `0.578`.
- Limiar mediano: `0.58`.
- Acuracia media com limiar padrao `0.50`: `0.8684`.
- Acuracia media com limiar validado por fold: `0.8869`.
- Acuracia media com limiar fixo `0.57`: `0.8908`.
