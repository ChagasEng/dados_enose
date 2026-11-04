# X-TCN com MQ + Soil: resultado geral e por faixa de umidade

A X-TCN usa os seis MQ e `Soil` leitura a leitura, com 60 epocas. A faixa seco/umido nao e entrada do modelo. Depois do treino, cada coleta de teste e classificada pela **media de Soil da propria coleta**, expressa no indice min-max usado no projeto: ate 0,4 = seco; acima de 0,4 = umido. Neste dataset, o corte corresponde a `Soil = 1189`.

## Tabela geral

| Resultado | Coletas de teste | Janelas de teste | Acuracia por janela | Acuracia por coleta |
|---|---:|---:|---:|---:|
| X-TCN MQ + Soil, nova execucao | 12 | 339 | 84,66% | 83,33% |

O resultado **93,23%** e o valor informado para o ensaio anterior MQ + Soil, com 60 epocas e as mesmas coletas do estudo de ablacao. Nao foram encontrados as predicoes ou o checkpoint desse treino; por isso a tabela de seco/umido abaixo se refere a **nova execucao**, e nao pode ser apresentada como a decomposicao daquele 93,23%.

## Comparacao com os 96,15% da Tabela 22

O arquivo local que reproduz 96,15% e `06_07_melhor_modelo/analise_seco_molhado/resumo_acertos_seco_molhado.csv`: 16.692 leituras secas de 8 coletas no teste 70/30. O modelo salvo usado nessa analise e Extra Trees, embora a legenda da tabela apresentada pelo usuario diga Random Forest; a identificacao do modelo na tese deve ser conferida.

A nova X-TCN MQ + Soil usou exatamente as mesmas coletas de treino e teste. Para obter uma previsao por leitura, foi calculada a media das probabilidades das janelas que cobrem cada leitura. No solo seco, acertou 16.692/16.692 leituras: 100,00% de acuracia, acuracia balanceada e F1 macro. No solo umido desse mesmo teste, acertou apenas 36,38% por leitura; no total, 84,40%. Portanto, o 100% seco supera numericamente o valor da Tabela 22 naquele recorte, mas nao demonstra melhora geral nem estabilidade em outras coletas. A validacao em 4 folds abaixo e a verificacao de estabilidade.

## Tabela por umidade media da coleta: validacao em 4 folds

Cada uma das 36 coletas aparece uma vez no teste, sempre fora do respectivo treino. O modelo e retreinado em cada fold com a mesma arquitetura e 60 epocas. Esta e a tabela recomendada para comparar seco e umido:

| Faixa | Coletas | Acuracia por janela | F1 macro por janela | Recall macro por janela | Acuracia por coleta | F1 macro por coleta | Recall macro por coleta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seco | 17 | 90,50% | 90,38% | 91,05% | 94,12% | 94,04% | 95,00% |
| Umido | 19 | 77,70% | 77,61% | 77,78% | 78,95% | 78,89% | 78,89% |

O recall macro e a media do recall das classes 0 e 1; neste caso, tem o mesmo valor da acuracia balanceada. As medias de Soil das coletas sao 294,49 para seco e 2646,77 para umido.

No total, a validacao em 4 folds resultou em 85,03% por janela e 86,11% por coleta. Houve erro em uma coleta seca e quatro umidas. Essa validacao mostra que o 100% da divisao unica abaixo nao se sustenta para todas as coletas.

## Divisao unica 70/30: resultado inicial

| Faixa | Media de Soil das coletas | Media do indice | Coletas de teste | Janelas | Acuracia por janela | Acuracia por coleta |
|---|---:|---:|---:|---:|---:|---:|
| Seco | 307,17 | 0,103 | 8 | 257 | 100,00% | 100,00% |
| Umido | 2664,71 | 0,897 | 4 | 82 | 36,59% | 50,00% |

As medias da tabela sao medias nao ponderadas das medias por coleta no conjunto de teste. O modelo e unico para ambas as faixas. No teste, seco possui 3 coletas da classe 0 e 5 da classe 1; umido possui 3 da classe 0 e 1 da classe 1. As coletas secas do teste sao dos dias 11 a 13, e as umidas dos dias 18 e 19; portanto, dia e umidade estao misturados nesta divisao. A validacao em 4 folds tambem separa por coleta, mas nao por dia; a diferenca nao deve ser tratada como efeito isolado da umidade.

Arquivos: `tabela_validacao_cruzada_4_folds.csv`, `predicoes_janelas_cv.csv`, `predicoes_coletas_cv.csv`, `metodo_cv.json`, alem dos arquivos da divisao unica (`tabela_geral.csv`, `tabela_seco_umido.csv`, `medias_e_faixas_por_coleta.csv`, `predicoes_por_janela.csv`, `predicoes_por_coleta.csv`, `metodo.json`). Para reproduzir, executar `scripts/validar_xtcn_seco_umido_cv.py` a partir da raiz do projeto.
