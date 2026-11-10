# LOSO corrigido: cinco coletas excluidas, dez vasos mantidos

Este e o cenario corrigido apos a observacao do usuario de que a coleta inoculada ruidosa do dia 19 nao poderia voltar a analise. Foram excluidas as **quatro coletas completas com mais erros no Extra Trees original** e tambem a coleta inoculada ruidosa rotulada `Vaso 1` no dia 19. A coleta controle ruidosa do dia 19 ja esta entre as quatro. A lista exata, as classes, os motivos e as contagens estao em `coletas_excluidas.csv`.

As cinco coletas somam **7.546 leituras**. Restam **55.946 leituras em 31 coletas**, distribuidas pelos **dez vasos**. Nenhum vaso foi eliminado da base: cada um foi testado em uma das dez rodadas LOSO, ficando fora do treino somente na propria rodada. Os dados originais de 63.492 leituras permanecem intactos; a filtragem e aplicada nos scripts antes de cada treino.

Para comparar os modelos nas mesmas entradas, foram usados (a) `MQ2, MQ3, MQ8, MQ135, MQ138` e (b) os mesmos cinco MQ com `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`. O Random Forest tem 300 arvores e pesos de classe equilibrados. A rede temporal do projeto, chamada `XTCN` no codigo, foi avaliada como TCN com 60 epocas, janelas de 128 leituras e passo 64; cada leitura recebeu a media das probabilidades das janelas que a cobrem. Ambos foram treinados de novo em cada fold.

| Modelo, entradas fixas | Acertos / leituras | Acuracia por leitura |
|---|---:|---:|
| Random Forest, cinco MQ | 47.172 / 55.946 | 84,32% |
| TCN, cinco MQ | 49.352 / 55.946 | 88,21% |
| Extra Trees, cinco MQ | 53.015 / 55.946 | 94,76% |
| Random Forest, cinco MQ + ambiente | 53.385 / 55.946 | 95,42% |
| TCN, cinco MQ + ambiente | 49.789 / 55.946 | 88,99% |
| Extra Trees, cinco MQ + ambiente | 54.783 / 55.946 | 97,92% |

## Busca combinatoria de Extra Trees

Foram treinados os **63 subconjuntos nao vazios dos seis MQ**, sem ambiente e com os tres sinais ambientais, nas mesmas 31 coletas e dez folds LOSO. O modelo usa 900 arvores, `max_features=sqrt`, `min_samples_leaf=10` e semente 42, como na busca original.

| Cenario | Primeiro lugar | Acertos / leituras | Acuracia |
|---|---|---:|---:|
| So MQ | MQ2 + MQ3 + MQ7 + MQ8 + MQ135 + MQ138 | 53.388 / 55.946 | 95,43% |
| MQ + ambiente | MQ2 + MQ3 + MQ8 + MQ135 + MQ138 + ambiente | 54.783 / 55.946 | 97,92% |

O conjunto de cinco MQ pedido ficou em **segundo lugar** na busca so MQ (53.015/55.946; 94,76%) e em **primeiro lugar** com ambiente. No cenario com ambiente, o segundo lugar foi MQ3 + MQ135 + MQ138 + ambiente (54.704/55.946; 97,78%).

Uma busca posterior variou **solo, temperatura e pressao separadamente** entre todas as 511 combinacoes das nove variaveis. Ela confirmou o mesmo primeiro lugar; ver `../busca_nove_variaveis_loso/README.md`.

Os totais anteriormente indicados como 57.778/63.492 (91,32%) e 57.143/63.492 (90,26%) nao coincidem com os resultados salvos da base original; as fracoes correspondem, respectivamente, a 91,00% e 90,00%. A busca original de Extra Trees registrou 55.916/63.492 (88,07%) para os cinco MQ + ambiente e 50.308/63.492 (79,24%) para os mesmos cinco MQ sem ambiente. Os percentuais do presente cenario usam outra base, com 55.946 leituras.

**Limite:** tres das cinco coletas foram retiradas apenas por concentrarem erros do classificador original, sem ruido multissensor detectado na auditoria. A selecao foi feita apos observar o desempenho na mesma base. Portanto, as acuracias acima descrevem uma analise de sensibilidade condicionada a essa exclusao e tendem a ser otimistas como estimativas para vasos novos. A busca e a avaliacao das combinacoes tambem usam os mesmos dez vasos.

Reproducao a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\avaliar_rf_tcn_sem_quatro_coletas.py --somar-ruido
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_mq_loso_sem_quatro_coletas.py --somar-ruido
```

Arquivos: `coletas_excluidas.csv`, `metricas_rf_tcn.csv`, `ranking_63_mq.csv`, `ranking_63_mq_ambiente.csv`, `acertos_por_coleta_63_mq.csv`, `acertos_por_coleta_63_mq_ambiente.csv`, `predicoes_*.csv`, `metodo_rf_tcn.json` e `metodo_busca.json`.
