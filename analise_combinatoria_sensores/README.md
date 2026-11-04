# Analise combinatoria por coleta

Esta analise usa a **coleta** como unica unidade de avaliacao. Cada modelo gera probabilidades para as leituras de uma coleta; a media dessas probabilidades produz **uma previsao por coleta** (limiar 0,5). Todas as metricas, selecoes e comparacoes usam essas previsoes agregadas. Nenhuma metrica por leitura entra no ranking.

Foram testados os 63 subconjuntos nao vazios dos seis MQ em dois cenarios: MQ brutos e os mesmos MQ brutos acrescidos de `Soil_indice_0_1`, `Temp_C` e `Pres_kPa` (126 configuracoes). Os sinais `MQ*_corrigido_env` nao sao usados porque ja incorporam informacao ambiental. O classificador foi Extra Trees, com 900 arvores, `max_features=sqrt`, `min_samples_leaf=10` e semente 42.

O split preexistente tem 24 coletas de treino e 12 de teste, sem coleta compartilhada. Dentro do treino, quatro folds estratificados por classe separam coletas inteiras. O criterio principal e a **acuracia balanceada por coleta**; acuracia e F1 macro por coleta tambem sao registrados.

## Resultados por coleta

| Cenario | Configuracao | CV no treino | Teste | Acertos no teste |
|---|---|---:|---:|---:|
| So MQ, melhor CV | MQ7 | 71,68% | 75,00% | 9/12 |
| So MQ, melhor ranking exploratorio | MQ2, MQ3, MQ8, MQ135, MQ138 | 66,43% | 91,67% | 11/12 |
| So MQ, seis sensores | Todos os seis MQ | 65,73% | 91,67% | 11/12 |
| MQ + ambiente, melhor CV | MQ2 + solo, temperatura e pressao | 74,83% | 83,33% | 10/12 |
| MQ + ambiente, melhor ranking exploratorio | MQ2, MQ135, MQ138 + solo, temperatura e pressao | 74,13% | 91,67% | 11/12 |
| MQ + ambiente, seis MQ | Todos os seis MQ + ambiente | 70,28% | 83,33% | 10/12 |

Os percentuais sao de acuracia balanceada **por coleta**. No teste, as classes tem seis coletas cada; por isso 11/12 corresponde a 91,67% nessa metrica.

O melhor candidato empirico geral e **MQ2 + MQ135 + MQ138 + solo, temperatura e pressao**. Trinta e quatro configuracoes empataram em 11/12 coletas no teste. Entre elas, esse conjunto teve a maior acuracia balanceada na CV do treino. Restrito a MQ, o melhor pelo mesmo criterio foi **MQ2 + MQ3 + MQ8 + MQ135 + MQ138**. O ranking exploratorio ordena por acuracia balanceada por coleta no teste, depois pela mesma metrica na CV, depois por menor numero total de sensores.

O ranking exploratorio consulta as 12 coletas de teste para escolher o conjunto. Logo, 91,67% descreve estas coletas e **nao e uma estimativa imparcial de desempenho futuro do vencedor**. A confirmacao de hardware exige novas coletas independentes. O teste tambem havia sido usado em outras analises do projeto.

## Reproducao e arquivos

Execute a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\rodar_combinacoes.py
.\.venv\Scripts\python.exe analise_combinatoria_sensores\explorar_todas_no_teste.py
```

- `validacao_cruzada_126_combinacoes.csv`: todas as combinacoes na CV, somente metricas por coleta.
- `avaliacao_teste_reservado.csv`: teste dos vencedores da CV e dos baselines de seis MQ.
- `exploracao_teste_126_combinacoes.csv`: teste das 126 configuracoes, somente metricas por coleta, em ranking exploratorio.
- `efeito_ambiente_mesmos_mq_cv.csv`: diferenca pareada por coleta entre os dois cenarios para cada subconjunto MQ.
- `predicoes_teste_por_coleta.csv` e `predicoes_candidato_exploratorio.csv`: previsoes agregadas por coleta.
- `metodo.json`: parametros e regras da validacao original.
- `resultados_para_artigo.tex`: texto e tabela para revisao no artigo.
- `LEMBRETE_RESULTADO.md`: anotacao curta do conjunto e dos percentuais, incluindo o valor 91,36% mencionado na conversa e a verificacao contra os resultados calculados.
