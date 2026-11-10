# LOSO exploratorio apos retirar quatro coletas com mais erros

**Cenario historico, corrigido pelo usuario:** esta versao reincluiu a coleta inoculada ruidosa do dia 19. Para os resultados com essa coleta tambem retirada, consultar `../validacao_sem_cinco_coletas/README.md` apos a conclusao da busca.

Foram retiradas **quatro coletas completas**, escolhidas depois de observar os erros do Extra Trees original: `dia 12 - Soja Saudavel Vaso 2` (2.352 leituras), `dia 19 - Soja Heterodera Vaso 9` (1.441), `dia 19 - Soja Saudavel Vaso 2` (1.221) e `dia 19 - Soja Saudavel Vaso 1` (1.271). Este cenario **substitui** a selecao anterior de duas coletas ruidosas: a coleta inoculada ruidosa rotulada `Vaso 1` no dia 19 permanece aqui. Restam **57.207 leituras, 32 coletas e dez vasos**. Nenhum vaso foi retirado permanentemente; cada um foi testado em um dos dez folds LOSO.

As duas configuracoes fixas foram `MQ2 + MQ3 + MQ8 + MQ135 + MQ138` e os mesmos cinco MQ acrescidos de indice operacional de solo, temperatura e pressao. Random Forest (300 arvores, pesos de classe equilibrados) e TCN (60 epocas, janelas de 128 leituras com passo 64) foram treinados de novo em cada fold. Para a TCN, cada leitura recebeu a media das probabilidades das janelas que a cobrem. A busca combinatoria de Extra Trees testou **todos os 63 subconjuntos MQ** em cada cenario, com 900 arvores e os mesmos hiperparametros da busca original.

| Modelo / conjunto | Acertos / leituras | Acuracia por leitura |
|---|---:|---:|
| Random Forest, cinco MQ | 49.775 / 57.207 | 87,01% |
| TCN, cinco MQ | 51.884 / 57.207 | 90,70% |
| Extra Trees, cinco MQ indicados | 51.571 / 57.207 | 90,15% |
| **Extra Trees, melhor so MQ: MQ2 + MQ7 + MQ8 + MQ135 + MQ138** | **53.268 / 57.207** | **93,11%** |
| Random Forest, cinco MQ + ambiente | 53.181 / 57.207 | 92,96% |
| TCN, cinco MQ + ambiente | 52.695 / 57.207 | 92,11% |
| **Extra Trees, melhor MQ + ambiente: cinco MQ indicados** | **55.679 / 57.207** | **97,33%** |

Na base original de 36 coletas, a busca de Extra Trees registrou 55.916/63.492 (88,07%) para os cinco MQ + ambiente e 50.308/63.492 (79,24%) para os mesmos cinco MQ sem ambiente. Os numeros informados depois como 57.778/63.492 (91,32%) e 57.143/63.492 (90,26%) nao coincidem com esses arquivos; alem disso, as duas fracoes correspondem aritmeticamente a 91,00% e 90,00%.

**Limite da interpretacao:** as quatro coletas foram selecionadas por terem sido classificadas incorretamente no LOSO original. Refazer o treino nao elimina esse vies de selecao. A acuracia deste subconjunto descreve um conjunto mais facil, escolhido depois de ver as respostas; nao estima o desempenho esperado em vasos independentes. As tres coletas `dia 12 - Soja Saudavel Vaso 2`, `dia 19 - Soja Saudavel Vaso 2` e `dia 19 - Soja Heterodera Vaso 9` nao apresentaram eventos abruptos multissensor na auditoria, portanto erro do modelo nao prova defeito do sinal.

Reproducao a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\avaliar_rf_tcn_sem_quatro_coletas.py
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_mq_loso_sem_quatro_coletas.py
```

Os arquivos `metricas_rf_tcn.csv`, `ranking_63_mq.csv`, `ranking_63_mq_ambiente.csv`, `acertos_por_coleta_63_mq.csv`, `acertos_por_coleta_63_mq_ambiente.csv` e `predicoes_*.csv` guardam metricas e previsoes. Os arquivos `metodo_*.json` registram os parametros.
