# Metodo do autor aplicado a base sem pressao

Esta pasta contem uma reproducao adaptada das tecnicas descritas no trabalho do autor, usando a base que vinha dando melhor resultado no projeto.

## Melhor modelo encontrado nesta rodada

- Modelo: `minmax_random_forest`
- Accuracy: `0.747256`
- Balanced accuracy: `0.734620`
- F1 macro: `0.736875`

## O que foi aplicado

- Filtro de media movel modificado nos sensores MQ.
- Compensacao linear com as colunas ambientais disponiveis (`Temp.` e `Pres.`), quando presentes.
- Normalizacao maximo-minimo.
- Selecao de atributos com chi-quadrado, V de Cramer, importancia por Random Forest e busca exaustiva por subconjuntos.
- PCA com 3 componentes.
- LDA com 1 componente, pois esta base possui apenas 2 classes.
- KNN, SVM linear e Random Forest.
- Validacao cruzada `StratifiedGroupKFold` por `Coleta` com 10 folds.

## Arquivos principais

- `scripts/rodar_metodo_autor.py`: script que gera toda a pasta.
- `resultados/relatorios/relatorio_metodo_autor.txt`: leitura principal dos resultados.
- `resultados/metricas/resumo_modelos.csv`: comparacao dos modelos.
- `resultados/matrizes/matriz_confusao_melhor_modelo.png`: matriz de confusao do melhor modelo.
- `resultados/selecao_atributos/feature_selection_scores.csv`: pontuacao dos sensores.
- `resultados/selecao_atributos/busca_exaustiva_subconjuntos_rf.csv`: busca de subconjuntos.
- `graficos/`: PCA, LDA, comparacao de modelos e selecao de atributos.
- `modelos/melhor_modelo_metodo_autor.joblib`: melhor pipeline treinado na base completa.

## Limitacoes

A base possui apenas os 6 sensores MQ atuais, entao nao foi possivel repetir exatamente a reducao de 13 sensores para 6. Tambem nao existe coluna explicita de umidade relativa; a compensacao ambiental usa apenas `Temp.` e `Pres.`.
