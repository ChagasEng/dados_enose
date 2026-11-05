# Validacao por leitura crua e por media de quatro

**Protocolo adotado para o artigo: leituras cruas.** A media de quatro foi apenas um exemplo metodologico do orientador. Seu ensaio esta salvo abaixo como exploracao, sem integrar a conclusao principal nem a secao LaTeX do artigo.

**Etapa 70/30 corrigida:** a divisao pedida agora e por leitura crua: 44.444 leituras de treino e 19.048 de teste, estratificadas por classe. Para MQ2, MQ3, MQ8, MQ135 e MQ138, foram **18.586/19.048 (97,57%)** sem ambiente e **18.643/19.048 (97,87%)** com ambiente. Consulte `../validacao_70_30_por_leitura/README.md`. O LOSO continua separando vasos inteiros e testa generalizacao a vaso novo.

Esta analise responde a observacao do orientador sobre a unidade avaliada. Uma coleta contem muitas leituras, e cada leitura pode ser acertada ou errada. Foram avaliadas **63.492 leituras cruas** e, em um segundo ensaio, **15.861 medias de quatro leituras consecutivas da mesma coleta**. As 48 leituras que formariam blocos incompletos foram descartadas somente no segundo ensaio.

Em ambos os ensaios, os modelos sao treinados na representacao que sera testada. A acuracia normal e o numero de leituras ou blocos corretos dividido pelo total testado. As metricas nao agregam as previsoes para obter uma unica decisao por coleta. O arquivo `acertos_dentro_de_cada_coleta.csv` mostra quantos acertos e erros ocorreram em cada coleta.

**Nota sobre escolha de sensores:** a tabela abaixo compara configuracoes definidas antes desta correcao metodologica. A busca exaustiva com a nova metrica esta em `../busca_mq_loso_leituras/README.md` e `../busca_mq_ambiente_loso_leituras/README.md`; seus vencedores sao diferentes.

## LOSO por vaso: dados crus como resultado principal

### Comparacao direta da combinacao MQ2, MQ3, MQ8, MQ135 e MQ138

| Entradas | 70/30 por vaso | LOSO por vaso |
|---|---:|---:|
| Cinco MQ, sem ambiente | 15.509/17.030 (91,07%) | **50.308/63.492 (79,24%)** |
| Os mesmos cinco MQ + solo, temperatura e pressao | 16.305/17.030 (95,74%) | **55.916/63.492 (88,07%)** |

O mesmo subconjunto MQ foi treinado e testado nas duas formas. Os valores sao acuracia **normal por leitura crua**. Isso permite avaliar diretamente o efeito de incluir as tres entradas ambientais nesta combinacao, sem trocar os MQ.

| Sensores | Leituras cruas | Medias de quatro |
|---|---:|---:|
| MQ2, MQ135, MQ138 + solo, temperatura e pressao | **55.549/63.492 (87,49%)** | **13.959/15.861 (88,01%)** |
| MQ2, MQ3, MQ8, MQ135, MQ138 | 50.308/63.492 (79,24%) | 13.182/15.861 (83,11%) |
| Seis MQ | 50.041/63.492 (78,81%) | 13.110/15.861 (82,66%) |
| Seis MQ + ambiente | 55.136/63.492 (86,84%) | 13.825/15.861 (87,16%) |
| MQ7 + ambiente | 55.131/63.492 (86,83%) | 13.871/15.861 (87,45%) |
| MQ2, MQ3, MQ135, MQ138 (melhor anterior por coleta) | 47.190/63.492 (74,32%) | 11.807/15.861 (74,44%) |

O LOSO separa vasos inteiros: em cada uma das dez rodadas, todas as leituras de um vaso ficam no teste e o modelo treina nos outros nove. Portanto, o treino nunca ve leituras de outra coleta do mesmo vaso testado.

Tambem foram executados o split original por coleta e um 70/30 por vaso, ambos nas duas representacoes. Os valores completos estao em `metricas_por_amostra.csv`. O split original tem oito vasos presentes em treino e teste; a divisao por vaso usa sete vasos para treino e tres para teste.

Depois da busca exaustiva, os vencedores de cada cenario e representacao foram tambem avaliados nos dois protocolos 70/30. O arquivo `metricas_vencedores_70_30.csv` guarda os numeradores e denominadores. Comparacao resumida:

| Vencedor no LOSO | 70/30 por vaso | LOSO por vaso |
|---|---:|---:|
| MQ cru: MQ2, MQ7, MQ8, MQ135, MQ138 | 15.605/17.030 (91,63%) | 51.180/63.492 (80,61%) |
| MQ + ambiente cru: MQ2, MQ3, MQ8, MQ135, MQ138 + ambiente | 16.305/17.030 (95,74%) | 55.916/63.492 (88,07%) |
| MQ medio de 4: MQ2, MQ3, MQ8, MQ135, MQ138 | 3.885/4.254 (91,33%) | 13.182/15.861 (83,11%) |
| MQ + ambiente medio de 4: MQ3, MQ7 + ambiente | 4.029/4.254 (94,71%) | 14.042/15.861 (88,53%) |

O 70/30 por vaso testa somente tres vasos sorteados. Os vencedores foram identificados pelo LOSO usando os dez vasos, entao os percentuais 70/30 acima sao descritivos e nao um novo teste independente da selecao.

Os **10 vasos** continuam sendo as unidades biologicas independentes. As dezenas de milhares de leituras sao observacoes correlacionadas dentro de coletas e vasos. Assim, o denominador de acuracia por amostra e 63.492 ou 15.861, mas a evidencia de generalizacao biologica continua limitada a dez vasos. Os sensores acima foram selecionados em analises anteriores com os mesmos dados; a validacao e interna.

Arquivos: `metricas_por_amostra.csv`, `metricas_vencedores_70_30.csv`, `predicoes_por_amostra.csv`, `acertos_dentro_de_cada_coleta.csv` e `metodo.json`.
