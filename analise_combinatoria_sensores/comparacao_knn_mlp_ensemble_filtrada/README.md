# KNN, MLP e ensemble na base filtrada de 31 coletas

Comparação por leitura com os mesmos cinco MQ da tabela LOSO do artigo. A base contém 55.946 leituras, 31 coletas e dez vasos. Em cada um dos dez folds LOSO, um vaso inteiro fica fora do treino. Foram avaliados dois conjuntos fixos de entradas: `MQ2`, `MQ3`, `MQ8`, `MQ135`, `MQ138`; e os mesmos cinco MQ com `Soil_indice_0_1`, `Temp_C`, `Pres_kPa`.

| Modelo | Cinco MQ: acertos / acurácia | Cinco MQ + ambiente: acertos / acurácia |
|---|---:|---:|
| k-NN | 49.767/55.946 (88,96%) | 53.183/55.946 (95,06%) |
| MLP (ANN) | 50.118/55.946 (89,58%) | 53.746/55.946 (96,07%) |
| SVM linear calibrada | 37.015/55.946 (66,16%) | 42.679/55.946 (76,29%) |
| LDA | 37.465/55.946 (66,97%) | 42.968/55.946 (76,80%) |
| Ensemble ANN + k-NN + SVM + LDA | 49.867/55.946 (89,13%) | 51.677/55.946 (92,37%) |

As acurácias balanceadas e os F1 macro estão em [metricas.csv](metricas.csv); as previsões e probabilidades por leitura estão em [predicoes_mq.csv](predicoes_mq.csv) e [predicoes_mq_ambiente.csv](predicoes_mq_ambiente.csv). A [tabela LaTeX](tabela_modelos_loso_expandida.tex) reúne esses resultados aos três modelos já apresentados no artigo.

O k-NN usa cinco vizinhos e distância euclidiana, com votos ponderados pela distância. A ANN é uma `MLPClassifier` de duas camadas ocultas (64 e 32 neurônios). A SVM é linear e tem probabilidades calibradas em três folds internos agrupados por vaso. A LDA usa shrinkage automático. Os quatro modelos recebem padronização ajustada somente no treino de cada fold LOSO. O ensemble usa a média não ponderada das probabilidades da classe 1, com limiar de 0,5. Essa forma foi escolhida após constatar que o voto simples tinha muitos empates, sem selecionar pesos pela acurácia do teste. A configuração completa está em [metodo.json](metodo.json), e o experimento pode ser repetido com [comparar_knn_mlp_ensemble_filtrada.py](../comparar_knn_mlp_ensemble_filtrada.py).

Os resultados são exploratórios: as cinco coletas excluídas e as entradas foram selecionadas após inspeção dos dados, usando os mesmos vasos avaliados. As leituras da mesma coleta são correlacionadas. A tabela não estima desempenho em vasos externos e não comprova superioridade geral de um classificador.
