# X-TCN nas mesmas condicoes da Tabela 22

## Comparacao das letras a, b e c por leitura

| Condicao | Leituras de teste | Referencia: acuracia / balanceada / F1 macro | X-TCN: acuracia / balanceada / F1 macro |
|---|---:|---:|---:|
| a) Solo seco sob pressao forcada | 16.692 | 96,15% / 95,06% / 95,87% | **98,47% / 98,03% / 98,38%** |
| b) Solo umido sob pressao forcada | 5.423 | 74,46% / 62,96% / 63,33% | **63,58% / 41,52% / 38,87%** |
| c) Sem pressao forcada, sem estratificar umidade | 9.353 | 84,43% / 83,54% / 83,72% | **83,06% / 82,10% / 82,17%** |

As letras a e b compartilham uma X-TCN treinada na base sob pressao forcada; a separacao seco/umido ocorre somente no teste. A letra c usa outra X-TCN, treinada na base sem pressao forcada. Em todos os casos sao 60 epocas, a mesma arquitetura, as mesmas coletas de treino/teste e as mesmas entradas da respectiva referencia. A X-TCN produz probabilidades por janela; para obter previsoes por leitura, cada leitura recebe a media das probabilidades das janelas que a cobrem, com limiar 0,5.

Para c, as nove entradas sao MQ2, MQ3, MQ7, MQ8, MQ135, MQ138, `Soil`, `Temp.` e `Pres.`. A divisao foi conferida contra a funcao usada no projeto de referencia: sao as mesmas 17 coletas, com 11 no treino e 6 no teste (17.842 e 9.353 leituras). O modelo de referencia local para c e Extra Trees.

Foi usada a mesma base `06_07_melhor_modelo/dados/dataset_melhor_modelo_sensores_corrigidos.csv`, a mesma coluna `Conjunto` (24 coletas no treino, 12 no teste) e as mesmas nove entradas do modelo local que produz 96,15% em solo seco: seis MQ corrigidos ambientalmente, `Soil_indice_0_1`, `Temp_C` e `Pres_kPa`. O classificador foi trocado pela X-TCN de 60 epocas, janela 128 e passo 64.

Para avaliar por leitura, cada leitura recebe a media das probabilidades de todas as janelas que a cobrem; o limiar de classe e 0,5. A faixa seca usa `Soil_indice_0_1 <= 0,4`, como no arquivo de referencia. As 16.692 leituras secas e as oito coletas secas de teste coincidem com a Tabela 22.

| Modelo e condicao | Leituras | Acuracia | Acuracia balanceada | F1 macro |
|---|---:|---:|---:|---:|
| Referencia local, solo seco | 16.692 | 96,15% | 95,06% | 95,87% |
| X-TCN, solo seco | 16.692 | **98,47%** | **98,03%** | **98,38%** |
| X-TCN, solo umido | 5.423 | 63,58% | 41,52% | 38,87% |
| X-TCN, total | 22.115 | 89,91% | 89,95% | 89,91% |

O ganho pontual no solo seco e de 2,32 pontos percentuais de acuracia, 2,97 de acuracia balanceada e 2,51 de F1 macro. O recall para a classe com nematoide no solo seco e 96,07%; para a classe sem nematoide e 100%. No solo umido, o recall da classe sem nematoide e **0%**. Portanto, a melhora no seco nao representa melhora geral.

Ha somente oito coletas secas no teste, e as leituras dentro da mesma coleta sao correlacionadas. Alem disso, no conjunto seco a media de `Soil` separa as classes sem sobreposicao. O resultado de um unico split nao deve ser tratado como superioridade geral sem validacao adicional em coletas independentes. O modelo que reproduz os 96,15% nos arquivos locais e identificado como Extra Trees em `06_07_melhor_modelo/analise_seco_molhado/`, embora a legenda da tabela apresentada diga Random Forest.

Arquivos: `metricas_xtcn_tabela22.csv`, `predicoes_por_janela.csv`, `predicoes_por_leitura.csv`, `metodo.json`, `metricas_xtcn_c_sem_pressao.csv`, `predicoes_janelas_c_sem_pressao.csv`, `predicoes_leituras_c_sem_pressao.csv`, `predicoes_coletas_c_sem_pressao.csv` e `metodo_c_sem_pressao.json`. Para reproduzir: `scripts/avaliar_xtcn_mesmo_protocolo_tabela22.py` e `scripts/avaliar_xtcn_c_sem_pressao.py`.
