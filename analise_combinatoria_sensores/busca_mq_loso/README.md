# Busca combinatoria MQ sem ambiente por LOSO

**Nota posterior:** este ranking usa uma decisao por coleta. Para ranquear os MQ por acuracia de leituras cruas ou medias de quatro, use `../busca_mq_loso_leituras/README.md`.

Foram retirados `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`. Testamos as **63 combinacoes nao vazias** dos seis MQ brutos, mantendo o Extra Trees de 900 arvores e os mesmos hiperparametros. Em cada combinacao, o LOSO fez dez rodadas: nove vasos no treino e todas as coletas do decimo no teste. O vaso deixado de fora nunca aparece no treino daquela rodada.

A avaliacao e **exclusivamente por coleta**. A media das probabilidades das leituras gera uma previsao por coleta. A metrica de escolha e a **acuracia normal**, isto e, coletas corretas divididas por 36. Em empate, preferimos menos sensores MQ.

| Classificacao | Sensores MQ | Acertos | Acuracia normal por coleta |
|---:|---|---:|---:|
| 1 | **MQ2, MQ3, MQ135, MQ138** | **30/36** | **83,33%** |
| 2 | MQ2, MQ3 | 29/36 | 80,56% |
| 3 | MQ2, MQ3, MQ8 | 29/36 | 80,56% |
| 7 | MQ2, MQ3, MQ8, MQ135, MQ138 | 29/36 | 80,56% |
| 22 | MQ2, MQ3, MQ7, MQ8, MQ135, MQ138 | 28/36 | 77,78% |

A combinacao vencedora foi unica no numero de acertos. Ela classificou corretamente **15/17 coletas inoculadas** e **15/19 saudaveis**, com seis dos dez vasos sem erro em nenhuma de suas coletas.

Para comparacao contextual, o candidato anterior **MQ2 + MQ135 + MQ138 + ambiente** obteve **32/36 (88,89%)** no mesmo LOSO. Sem as tres entradas ambientais, esses mesmos tres MQ obtiveram **23/36 (63,89%)**. O melhor subconjunto puramente MQ, com quatro sensores, reduziu essa diferenca para duas coletas. Esses valores descrevem o experimento atual; o conjunto de quatro MQ foi escolhido depois de comparar 63 opcoes nos mesmos dez vasos, entao sua acuracia LOSO ainda precisa ser confirmada em vasos novos.

Arquivos: `ranking_63_combinacoes_mq_loso.csv` (uma linha por combinacao), `predicoes_63_combinacoes_mq_loso.csv` (36 previsoes por combinacao), `metodo.json` e `secao_resultados_mq_loso.tex`.

Para reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_mq_loso.py
```
