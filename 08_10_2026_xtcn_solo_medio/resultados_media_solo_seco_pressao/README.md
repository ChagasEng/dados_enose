# Media de Soil nas coletas secas sob pressao forcada

Base: `06_07/4_polimento_inicial_modelagem/datasets_limpos/antes_dia_20_pressao_filtrada_estrito.csv`. As coletas secas foram selecionadas pela media de `Soil` de cada coleta, convertida ao indice min-max do conjunto: seco <= 0,4. Isso corresponde a `Soil` medio <= 1189 nesta base. As 17 coletas secas sao anteriores ao dia 20 e pertencem ao regime sob pressao forcada, apos filtragem de variacoes abruptas de pressao. `Pres.` nao foi usada como entrada.

## Medias descritivas do solo seco

| Condicao | Coletas | Media das medias de Soil por coleta | Menor media | Maior media |
|---|---:|---:|---:|---:|
| Com nematoide (classe 0) | 7 | 126,88 | 44,43 | 158,90 |
| Sem nematoide (classe 1) | 10 | 411,82 | 193,31 | 691,21 |

As faixas das medias por coleta nao se sobrepoem: a maior media da classe 0 e 158,90; a menor da classe 1 e 193,31. As medias por classe desta tabela servem para descrever os dados. Elas nao sao atribuídas às amostras de teste como entrada, pois isso exigiria conhecer a classe verdadeira.

## Validacao cruzada em 4 folds por coleta

Nos dois ensaios X-TCN, as janelas preservam os seis MQ e a rede treina por 60 epocas. O experimento com media substitui `Soil` leitura a leitura pela media de `Soil` **da propria coleta**, calculada sem usar o rotulo. Os folds sao identicos para a comparacao abaixo.

| Entrada | Acuracia por janela | F1 macro por janela | Acuracia por coleta | F1 macro por coleta |
|---|---:|---:|---:|---:|
| MQ + Soil bruto | 91,04% | 90,93% | 94,12% (16/17) | 94,04% |
| MQ + media de Soil da coleta | 93,19% | 93,12% | 94,12% (16/17) | 94,04% |
| Somente media de Soil, arvore com um corte | — | — | 100,00% (17/17) | 100,00% |

A arvore com um corte usa somente a media de `Soil` e tambem foi avaliada fora do treino em cada fold. O desempenho alto dela mostra que, nestas 17 coletas secas, `Soil` por si so identifica a classe. Nao e possivel atribuir esse poder preditivo exclusivamente ao efeito dos nematoides: umidade, dia, vaso ou condicoes de coleta podem estar associados a classe. O resultado por janela da X-TCN melhora 2,15 pontos percentuais com a media, mas o numero de coletas corretas nao muda.

Como controle adicional, RF e Extra Trees foram avaliados nos mesmos quatro folds e com as mesmas coletas secas. Por leitura, RF somente MQ obteve 85,94%; RF MQ + Soil bruto, 87,75%; RF MQ + media de Soil por coleta, 92,47%; Extra Trees MQ + Soil bruto, 92,36%. Assim, nenhum desses controles chegou a 96,15% na validacao em quatro folds. Os arquivos sao `comparacao_rf_extratrees_seco_cv.csv` e `predicoes_rf_extratrees_seco_cv.csv`.

Arquivos: `medias_solo_seco_por_classe.csv`, `metricas_validacao_cruzada.csv`, `predicoes_janelas.csv`, `predicoes_coletas.csv`, `predicoes_janelas_solo_bruto.csv`, `predicoes_coletas_solo_bruto.csv`, `controle_somente_solo.csv` e `metodo.json`.
