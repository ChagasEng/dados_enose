# Comparação final — Tabela 22 e X-TCN

**Batido o martelo:** estes são os números fechados desta comparação, calculados **por leitura** no conjunto de teste de cada cenário. Isso não significa que a X-TCN seja superior em todos os cenários ou em novas coletas.

| Condição | Referência: acurácia / acurácia balanceada / F1 macro | X-TCN: acurácia / acurácia balanceada / F1 macro |
|---|---|---|
| a) Solo seco, com pressão forçada | 96,15% / 95,06% / 95,87% | **98,47% / 98,03% / 98,38%** |
| b) Solo úmido, com pressão forçada | 74,46% / 62,96% / 63,33% | **63,58% / 41,52% / 38,87%** |
| c) Sem pressão forçada, sem estratificação por umidade | 84,43% / 83,54% / 83,72% | **83,06% / 82,10% / 82,17%** |

## Protocolo das condições a e b: pressão forçada

- Base: [`dataset_melhor_modelo_sensores_corrigidos.csv`](../06_07_melhor_modelo/dados/dataset_melhor_modelo_sensores_corrigidos.csv). O treino/teste já está identificado na coluna `Conjunto`: 24/12 coletas, com 41.377/22.115 leituras. Nenhuma coleta aparece nos dois conjuntos.
- Alvo: `Classe` (`0` = com nematoide; `1` = sem nematoide). As nove entradas são `MQ2_corrigido_env`, `MQ3_corrigido_env`, `MQ7_corrigido_env`, `MQ8_corrigido_env`, `MQ135_corrigido_env`, `MQ138_corrigido_env`, `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`.
- Pré-processamento da base: `Soil_indice_0_1` é a normalização min–max crescente do sinal `Soil`. Cada MQ corrigido é o sinal original menos a componente ambiental estimada com `HuberRegressor` a partir de `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`; o regressor é ajustado somente no treino e a componente é centralizada pela média estimada no treino. É **compensação estatística**, não calibração física por umidade relativa do ar. Ver [código de preparação](../06_07/5_extratrees_sensores_ambiente_confirmados/scripts/rodar_extratrees_sensores_confirmados.py).
- A referência para a/b usa **um único Extra Trees** treinado na base inteira de pressão forçada. A X-TCN também é treinada **uma única vez** nessa base. As condições a e b são recortes do mesmo teste, não modelos treinados separadamente: a) `Soil_indice_0_1 <= 0,4`, 16.692 leituras de 8 coletas; b) `> 0,4`, 5.423 leituras de 4 coletas.
- Configuração do Extra Trees de referência: 900 árvores, `max_features="sqrt"`, `min_samples_leaf=10`, sem *bootstrap* e semente 42. As previsões de referência são obtidas diretamente por leitura.
- No ensaio, foi mantida a convenção observada: sinal `Soil` menor = grupo denominado solo seco; sinal maior = grupo denominado solo úmido. O corte de 0,4 é **operacional**, não porcentagem calibrada de água no solo.

## Protocolo da condição c: sem pressão forçada

- Base: [`dataset_dia_20_mais_com_ambiente.csv`](../15_07_dia_20_mais_completo/dados/dataset_dia_20_mais_com_ambiente.csv), com 17 coletas e 27.195 leituras. Divisão 70/30 **por coleta dentro de cada classe**, semente 42: 11/6 coletas e 17.842/9.353 leituras em treino/teste. A X-TCN usa o mesmo recorte da referência.
- Alvo: `Classe`, com os mesmos códigos de a/b. As nove entradas são os seis MQ **brutos** (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`), `Soil`, `Temp.` e `Pres.`. Não se usam os MQ corrigidos da base a/b.
- “Sem pressão” significa **sem pressão forçada na coleta**, não ausência da variável medida `Pres.`: ela continua entre as entradas. Não há separação seco/úmido neste cenário. O modelo de referência local também é Extra Trees.
- O Extra Trees da referência c usa a mesma configuração de 900 árvores, `max_features="sqrt"`, `min_samples_leaf=10`, sem *bootstrap* e semente 42; confira o [código da modelagem de referência](../06_07/rodar_modelagem_extra_trees_rede_neural_importancia.py). A X-TCN é treinada separadamente da X-TCN de a/b, pois a base é outra.

## X-TCN e cálculo das métricas

- Mesma implementação de X-TCN nos dois protocolos: blocos convolucionais temporais residuais com 24, 24 e 32 canais, dilatações 1, 2 e 4, *global average pooling*, *dropout* de 0,20 e saída para duas classes. Veja a [implementação da arquitetura](../08_10_2026_xtcn_solo_medio/scripts/comparar_rf_xtcn_medias_ambiente.py).
- Leituras ordenadas por `Coleta` e `Tempo`; janelas de **128 leituras**, passo **64**, sem misturar coletas. `StandardScaler` ajustado somente no treino. Treinamento por **60 épocas**, lote 32, semente 42, AdamW (taxa `0,001`, *weight decay* `0,0001`) e entropia cruzada com pesos inversamente proporcionais à frequência das classes **nas janelas de treino**. Veja o [treinamento e a inferência](../08_10_2026_xtcn_solo_medio/scripts/avaliar_xtcn_original_por_umidade.py).
- A X-TCN prevê a probabilidade da classe `1` por janela. Para dar **uma previsão a cada leitura**, calcula-se a média das probabilidades de todas as janelas que cobrem aquela leitura; probabilidade `>= 0,5` resulta na classe `1`. A tabela acima usa essas previsões por leitura, **não** a acurácia por janela ou por coleta.
- Acurácia = fração de leituras corretas; acurácia balanceada = média dos *recalls* das classes 0 e 1; F1 macro = média simples dos F1 das duas classes. As métricas de a/b e c são calculadas no respectivo teste, sem reunir as bases.
- **Não foram usadas nesta tabela** médias de `Soil` ou temperatura por coleta como entradas, remoção de *outliers* por IQR, 200 épocas, nem votação por coleta. Esses são ensaios diferentes. A estratificação de a/b é feita pelo valor de `Soil_indice_0_1` **em cada leitura**.

## Rastreabilidade e ressalva sobre a referência

- Resultados, previsões e método da X-TCN: [pasta do ensaio](../08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/README.md), [métricas de a/b](../08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/metricas_xtcn_tabela22.csv), [métricas de c](../08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/metricas_xtcn_c_sem_pressao.csv), [script de a/b](../08_10_2026_xtcn_solo_medio/scripts/avaliar_xtcn_mesmo_protocolo_tabela22.py) e [script de c](../08_10_2026_xtcn_solo_medio/scripts/avaliar_xtcn_c_sem_pressao.py).
- Referência de a/b: [análise seco/úmido](../06_07_melhor_modelo/analise_seco_molhado/README.md) e [código do modelo](../06_07_melhor_modelo/codigo/rodar_extratrees_melhor_modelo.py). Referência de c: [métricas do cenário basal](../15_07_dia_20_mais_completo/1_investigacao_hardware_banco/modelagem/metricas/01_baseline_dia_20_mais_mq_ambiente_metricas.json).
- **Divergência de identificação:** a legenda da Tabela 22 fornecida chama o modelo de referência de *Random Forest*, mas o código e os resultados locais que reproduzem esses números identificam **Extra Trees**. Por isso, a coluna acima permanece como “Referência”; não se deve afirmar que um Random Forest foi reproduzido sem localizar seu protocolo original.
- Esta é uma comparação em *splits* específicos por coleta. Leituras da mesma coleta são correlacionadas; os percentuais por leitura não equivalem a validação em coletas independentes adicionais. O cenário a tem apenas 8 coletas no teste, e o b apenas 4.

Fonte dos resultados da X-TCN e descrição do protocolo: [`../08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/README.md`](../08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/README.md).
