# RF e X-TCN: medias de solo e temperatura por coleta

## Resultado anterior informado na conversa

Na ablacao anterior da X-TCN, usando as mesmas coletas e 60 epocas, foram informados: MQ apenas 84,10%; MQ + `Soil` 93,23%; MQ + `Soil` + temperatura 88,09%; MQ + `Soil` + pressao 90,76%; MQ + `Soil` + temperatura + pressao 87,50%. O RF teve 84,45% com apenas MQ e 92,37% com as tres variaveis ambientais. Aumentar a X-TCN para 200 epocas nao melhorou a media.

Assim, o resultado de 93,23% corresponde a MQ + `Soil` no ensaio anterior, sem media por coleta nem exclusao de outliers proposta depois. A conversa confirma a combinacao de entradas, as 60 epocas e a manutencao das coletas; os arquivos atuais nao confirmam o split exato, a semente nem a unidade da metrica daquele numero. O ensaio abaixo e novo e seus percentuais nao substituem esse registro.

A avaliacao da X-TCN MQ + Soil separada pela media de umidade das coletas esta em `resultados_estratificados_mq_solo/README.md`.

A comparacao em solo seco sob pressao forcada, incluindo medias por classe apenas descritivas e o teste da media de Soil por coleta como entrada, esta em `resultados_media_solo_seco_pressao/README.md`.

A comparacao direta com a Tabela 22, usando na X-TCN as mesmas nove entradas, o mesmo split e as mesmas leituras do resultado de 96,15%, esta em `resultados_tabela22_xtcn_mesmas_features/README.md`.

Executar: `.venv\Scripts\python.exe 08_10_2026_xtcn_solo_medio\scripts\comparar_rf_xtcn_medias_ambiente.py`

Dados: `06_07/4_polimento_inicial_modelagem/datasets_limpos/antes_dia_20_pressao_filtrada_estrito.csv`, com 36 coletas. A divisao 70/30 por coleta dentro de cada classe (seed 42) deixa 24 coletas no treino e 12 no teste. Todos os cenarios usam exatamente as mesmas leituras, classes e coletas.

Para cada coleta e para cada uma das colunas `Soil` e `Temp.`, a media usa apenas valores entre Q1 - 1,5 IQR e Q3 + 1,5 IQR. Os valores descartados nao entram no calculo da respectiva media; nenhuma linha MQ e removida por esse filtro. A media resultante e repetida em todas as leituras daquela coleta. `Soil` e a medida do sensor de umidade do solo, nao a umidade relativa do ar. Pressao nao entra neste ensaio.

O RF usa 300 arvores e avalia cada leitura. A X-TCN usa janelas de 128 leituras, passo 64 e 60 epocas, com PyTorch em modo deterministico e uma thread de CPU. A acuracia por coleta usa a media das probabilidades das leituras (RF) ou janelas (X-TCN), com limiar 0,5.

| Entradas | RF por leitura | X-TCN por janela | RF por coleta | X-TCN por coleta |
|---|---:|---:|---:|---:|
| MQ | 83,23% | 80,24% | 83,33% | 83,33% |
| MQ + solo bruto | 90,48% | 84,66% | 91,67% | 83,33% |
| MQ + media do solo sem filtro | 89,63% | 85,25% | 91,67% | 83,33% |
| MQ + solo medio sem outliers | 89,63% | 84,66% | 91,67% | 83,33% |
| MQ + temperatura media | 68,92% | 69,62% | 66,67% | 66,67% |
| MQ + solo medio + temperatura media | 87,40% | 75,81% | 83,33% | 75,00% |

Foram excluidos 2.167 valores de `Soil` das medias (3,41% das leituras) e nenhum de `Temp.`. Na X-TCN, a media sem filtro ficou 0,59 ponto percentual acima da media filtrada; o solo bruto e a media filtrada empataram. No RF, o solo bruto ficou 0,85 ponto percentual acima das duas medias. Assim, esta comparacao nao encontrou ganho ao excluir os outliers de `Soil` pelo criterio usado. As metricas por coleta variam em passos de 8,33 pontos percentuais, pois ha apenas 12 coletas no teste. Os percentuais deste ensaio nao sao diretamente comparaveis aos 93,23% de outro protocolo sem verificar suas divisoes e parametros.

Arquivos em `resultados_medias_robustas/`: tabela completa de metricas, medias e contagens de outliers por coleta, divisao treino/teste e parametros do metodo.
