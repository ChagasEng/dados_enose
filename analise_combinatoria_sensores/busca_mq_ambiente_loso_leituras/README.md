# Busca MQ + ambiente por LOSO com previsoes por amostra

**Resultado principal para o artigo: leituras cruas.** A media de quatro foi apenas um exemplo e permanece salva como analise exploratoria.

Foram avaliados os **63 subconjuntos nao vazios dos seis MQ**, sempre acrescidos de `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`. Cada configuracao foi treinada e testada em dez rodadas LOSO por vaso biologico, usando o mesmo Extra Trees de 900 arvores. Em cada rodada, todas as leituras e coletas do vaso de teste ficam fora do treino.

A metrica e **acuracia normal por amostra**, sem agregar as previsoes em uma decisao por coleta. A busca foi executada separadamente em 63.492 leituras cruas e em 15.861 blocos formados pela media de quatro leituras consecutivas da mesma coleta. O modelo foi treinado novamente para cada representacao. No segundo ensaio, 48 leituras finais foram descartadas por nao completarem blocos de quatro.

| Representacao | Melhor subconjunto MQ, alem dos tres sensores ambientais | Acertos | Acuracia normal |
|---|---|---:|---:|
| Leituras cruas | **MQ2, MQ3, MQ8, MQ135, MQ138** | **55.916/63.492** | **88,07%** |
| Medias de quatro | **MQ3, MQ7** | **14.042/15.861** | **88,53%** |

O conjunto anterior, MQ2 + MQ135 + MQ138 + ambiente, obteve **55.549/63.492 (87,49%)** nas leituras cruas e **13.959/15.861 (88,01%)** nas medias de quatro. Ele ficou em terceiro e sexto lugar, respectivamente. Nos dados crus, **MQ2 + MQ7 + MQ138 + ambiente** ficou em segundo, com **55.764/63.492 (87,83%)**, usando dois MQ a menos que o primeiro lugar.

Os melhores conjuntos apenas MQ, avaliados exatamente nos mesmos protocolos, foram **MQ2 + MQ7 + MQ8 + MQ135 + MQ138** com **80,61%** em dados crus e **MQ2 + MQ3 + MQ8 + MQ135 + MQ138** com **83,11%** em medias de quatro. Nesta base, acrescentar solo, temperatura e pressao esteve associado a acuracias maiores, mas a representacao e o conjunto MQ tambem influenciaram o resultado.

Arquivos: `ranking_63_mq_cru.csv`, `ranking_63_mq_media_4.csv`, `acertos_por_coleta_63_mq_cru.csv`, `acertos_por_coleta_63_mq_media_4.csv`, `predicoes_vencedor_cru.csv`, `predicoes_vencedor_media_4.csv` e `metodo.json`. Os arquivos de previsao mostram acertos e erros individuais ao longo do tempo de cada coleta.

Os **dez vasos** sao as unidades biologicas; leituras e blocos dentro do mesmo vaso sao correlacionados. O melhor subconjunto foi selecionado comparando 63 opcoes nos mesmos vasos usados para calcular sua acuracia LOSO. Portanto, os percentuais dos vencedores sao exploratorios e precisam de confirmacao em vasos independentes.

## Onde o vencedor em dados crus errou

O conjunto MQ2, MQ3, MQ8, MQ135, MQ138 + ambiente errou **7.576 de 63.492 leituras**. Nenhum vaso teve maioria de leituras erradas; o pior foi o **saudavel vaso 2**, com 3.573 erros em 7.368 leituras (51,51% corretas). Quatro coletas tiveram maioria errada e concentraram 6.051 erros (79,87% de todos os erros):

| Coleta | Leituras erradas | Leituras totais |
|---|---:|---:|
| Saudavel vaso 2, dia 12 | 2.352 | 2.352 |
| Inoculado vaso 9, dia 19 | 1.441 | 1.441 |
| Saudavel vaso 2, dia 19 | 1.221 | 1.221 |
| Saudavel vaso 1, dia 19 | 1.037 | 1.271 |

Os arquivos `erros_por_vaso_cru.csv` e `erros_por_coleta_cru.csv` trazem todas as contagens; `predicoes_vencedor_cru.csv` permite localizar os erros leitura a leitura pelo tempo.

Para reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_mq_loso_leituras.py --com-ambiente
```
