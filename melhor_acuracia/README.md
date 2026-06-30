# Melhor acuracia - Extra Trees com limiar ajustado

Esta pasta reune o melhor modelo encontrado ate agora no projeto.

## Modelo escolhido

- Algoritmo: ExtraTreesClassifier
- Modelo salvo: `modelo/modelo_extra_trees_limiar_ajustado.joblib`
- Dataset usado nos experimentos: `sem pressao/dataset_sem_pressao.csv`
- Classes:
  - `0 = doente`
  - `1 = saudavel`
- Features usadas:
  - `MQ2`
  - `MQ3`
  - `MQ7`
  - `MQ8`
  - `MQ135`
  - `MQ138`
- Coluna usada apenas para separar treino/teste: `Coleta`
- Limiar final usado: `0.57`

## Melhor resultado de teste holdout

Arquivo:

`resultados/metricas/metricas_extra_trees_limiar_ajustado.json`

Metricas:

- Accuracy: `0.9265144381669805`
- Balanced accuracy: `0.9280215993596385`
- F1 macro: `0.9263584659179928`

Matriz de confusao:

`resultados/matrizes/matriz_confusao_extra_trees_limiar_ajustado.png`

## Validacao mais confiavel por Coleta

Arquivo de leitura principal:

`validacao_por_coleta/resultados/relatorios/relatorio_validacao_limiar_por_coleta.txt`

Arquivo com metricas em JSON:

`validacao_por_coleta/resultados/metricas/metricas_validacao_cruzada_limiar_extra_trees.json`

Metodo:

- Validacao cruzada aninhada com `StratifiedGroupKFold`
- Grupo: `Coleta`
- O limiar foi escolhido somente nos folds internos
- As coletas de teste ficaram isoladas

Metricas medias com limiar fixo `0.57`:

- Accuracy: `0.8907669047926194`
- Balanced accuracy: `0.891851407146605`
- F1 macro: `0.888489059423154`

## Organizacao da pasta

- `algoritmo/`: scripts usados para treinar, ajustar limiar, comparar tecnicas e analisar importancia dos sensores.
- `modelo/`: modelo final salvo em `.joblib`.
- `resultados/metricas/`: arquivos JSON com os numeros principais.
- `resultados/relatorios/`: relatorio textual do classificador.
- `resultados/matrizes/`: matriz de confusao em CSV e PNG.
- `resultados/comparacoes/`: comparacoes entre tecnicas e ajustes de limiar.
- `resultados/importancias/`: importancia dos sensores.
- `graficos/`: graficos gerais do modelo e do split.
- `validacao_por_coleta/`: validacao cruzada por Coleta, usada como evidencia mais honesta.

## Observacao

O resultado de holdout ficou mais alto, mas a validacao por Coleta e a evidencia mais forte porque reduz o risco de vazamento entre coletas parecidas.
