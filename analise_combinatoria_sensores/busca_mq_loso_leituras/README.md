# Busca MQ por LOSO com previsoes por leitura

**Resultado principal para o artigo: leituras cruas.** A media de quatro foi apenas um exemplo e permanece salva como analise exploratoria.

Esta busca incorpora a orientacao de avaliar **os dados dentro de cada coleta**. Foram testados os 63 subconjuntos nao vazios dos seis MQ brutos, sem solo, temperatura ou pressao, em duas representacoes:

- **Cru:** 63.492 leituras. Cada leitura recebe uma previsao.
- **Media de quatro:** 15.861 blocos, cada um formado pela media de quatro leituras consecutivas da mesma coleta. O modelo e treinado novamente com os blocos. Quarenta e oito leituras finais que nao completaram um bloco foram descartadas somente neste ensaio.

Em ambas, o LOSO deixa um **vaso biologico inteiro** fora do treino por rodada (dez rodadas). O Extra Trees tem 900 arvores e os hiperparametros usados nas analises anteriores. A metrica de escolha e **acuracia normal por leitura ou bloco**: numero de previsoes corretas dividido pelo total de amostras avaliadas. Nao ha voto unico por coleta. Em empate, ganha o conjunto com menos MQ.

| Representacao | Melhor conjunto apenas MQ | Acertos | Acuracia normal |
|---|---|---:|---:|
| Leituras cruas | **MQ2, MQ7, MQ8, MQ135, MQ138** | **51.180/63.492** | **80,61%** |
| Medias de quatro | **MQ2, MQ3, MQ8, MQ135, MQ138** | **13.182/15.861** | **83,11%** |

Para comparar nas mesmas representacoes, o conjunto **MQ2, MQ135, MQ138 + solo, temperatura e pressao** obteve **55.549/63.492 (87,49%)** nas leituras cruas e **13.959/15.861 (88,01%)** nas medias de quatro. O conjunto de quatro MQ escolhido antes pelo critério de *uma decisao por coleta* obteve apenas **74,32%** e **74,44%**, respectivamente. Assim, a escolha de sensores depende da unidade em que o desempenho e medido.

Arquivos: `ranking_63_mq_cru.csv`, `ranking_63_mq_media_4.csv`, `acertos_por_coleta_63_mq_cru.csv`, `acertos_por_coleta_63_mq_media_4.csv`, `predicoes_vencedor_cru.csv`, `predicoes_vencedor_media_4.csv` e `metodo.json`.

Os dez vasos sao as unidades biologicas independentes. Embora os denominadores tenham 63.492 leituras e 15.861 blocos, observacoes do mesmo vaso sao correlacionadas. Alem disso, o melhor subconjunto foi escolhido comparando 63 alternativas nesses mesmos vasos; o valor do vencedor precisa de confirmacao em novos vasos. As medias de quatro suavizam o sinal e geram outra tarefa de classificacao; seus percentuais nao devem ser comparados como se fossem as mesmas amostras cruas.

Para reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_mq_loso_leituras.py
```
