# Onde o Extra Trees com cinco MQ errou (52.486 leituras)

Fonte: previsões LOSO por vaso já salvas em `../loso_balanceado_52486_solo_binario/`. A classe 0 é inoculado; a classe 1 é controle. Os cinco MQ são MQ-2, MQ-3, MQ-8, MQ-135 e MQ-138. Esta é uma análise descritiva feita **depois** da escolha das entradas e da inspeção dos dez vasos.

## Localização

O modelo MQ classificou incorretamente **2.932/52.486 leituras (5,59%)**: **2.700** inoculadas foram previstas como controle e **232** controles foram previstos como inoculados. As 2.932 leituras erradas vieram de **9 das 31 coletas** e **6 dos 10 vasos**. Cinco coletas de vasos inoculados concentraram 2.700 erros (92,09% do total):

| Coleta | Erros/leituras | Taxa de erro |
|---|---:|---:|
| Dia 19, inoculado 8 | 754/1.317 | 57,25% |
| Dia 11, inoculado 7 | 627/2.048 | 30,62% |
| Dia 18, inoculado 9 | 544/1.327 | 40,99% |
| Dia 18, inoculado 6 | 479/1.609 | 29,77% |
| Dia 19, inoculado 7 | 296/1.330 | 22,26% |

As outras 232 leituras erradas ocorreram em quatro coletas de controles: dia 18 controle 5 (97), dia 11 controle 5 (73), dia 19 controle 3 (61) e dia 11 controle 3 (1). A tabela completa por coleta está em `erros_por_coleta.csv`.

Para as cinco coletas inoculadas da tabela, `leituras_certas_e_erradas_5_coletas.csv` marca cada uma das **7.631 leituras** como `ACERTO` ou `ERRO` no modelo MQ e no modelo MQ + ambiente; `cinco_coletas_por_quartil_tempo.csv` agrega esses resultados por quarto temporal de cada coleta. Nessas cinco coletas, todo erro do modelo MQ é uma leitura inoculada (classe 0) prevista como controle (classe 1).

As outras **22 coletas não tiveram erros** com esse modelo. Nos dias 12 e 13, não houve leituras classificadas incorretamente; 2.231 erros (76,09% do total) ocorreram nos dias 18 e 19. Como data, vaso e condição do solo variam conjuntamente neste experimento, essa concentração não permite atribuir os erros a um único fator.

| Vaso | Erros MQ | Taxa no próprio vaso | Erros MQ + ambiente |
|---|---:|---:|---:|
| Inoculado 1 | 0/3.527 | 0,00% | 0 |
| Inoculado 6 | 479/7.165 | 6,69% | 685 |
| Inoculado 7 | 923/7.153 | 12,90% | 0 |
| Inoculado 8 | 754/4.877 | 15,46% | 750 |
| Inoculado 9 | 544/3.521 | 15,45% | 0 |
| Controle 1 | 0/5.667 | 0,00% | 0 |
| Controle 2 | 0/3.364 | 0,00% | 0 |
| Controle 3 | 62/4.805 | 1,29% | 93 |
| Controle 4 | 0/5.905 | 0,00% | 0 |
| Controle 5 | 170/6.502 | 2,61% | 419 |

Portanto, **não é sempre o mesmo vaso**. O perfil também muda quando as entradas mudam. Com ambiente, o ET deixa de errar nos vasos inoculados 7 e 9, mas erra mais nos inoculados 6 e nos controles 3 e 5. Em nível de leitura, 1.726 erros do modelo MQ deixam de ser erros no modelo com ambiente, 741 leituras antes corretas passam a ser erradas, e 1.206 continuam erradas. Esses números descrevem duas predições de modelos retreinados; não isolam o efeito causal de uma variável.

Comparando os cinco classificadores individuais com apenas MQ, **1.401 leituras** foram classificadas incorretamente por todos eles (e também pelo ensemble). Todas pertencem a quatro coletas inoculadas: dia 18 vaso 9 (544), dia 18 vaso 6 (394), dia 11 vaso 7 (384) e dia 19 vaso 8 (79). Há, portanto, trechos difíceis compartilhados, embora cada modelo também tenha erros próprios. A tabela `erros_por_vaso_todos_modelos.csv` permite comparar os dez vasos em todas as configurações.

## Trechos temporais e escores

Os erros não estão espalhados uniformemente nas coletas. No dia 11, inoculado 7, **96,5% das leituras do último quarto da coleta** estão erradas; o maior bloco contíguo contém 477 leituras, entre 545,46 e 665,38 s. No dia 18, inoculado 9, **todo o último quarto** está errado; o maior bloco tem 541 leituras, de 252,60 a 381,96 s. No dia 19, inoculado 8, as taxas de erro nos dois primeiros quartos são 86,1% e 86,9%, caindo a zero no último quarto. Os demais padrões estão em `erros_por_coleta.csv` e na figura `probabilidades_ao_longo_das_cinco_coletas.png`.

Entre as 2.932 leituras erradas, **1.315 (44,85%)** receberam probabilidade da classe errada entre 0,50 e 0,60: são decisões próximas do limiar de 0,50. Outras **166 (5,66%)** receberam probabilidade de pelo menos 0,90 para a classe errada. Todas essas 166 vieram da coleta do inoculado 7 no dia 11, entre 591,89 e 665,38 s. Essa é uma indicação de escore extremo, **não uma medida calibrada de convicção**: as probabilidades do Extra Trees não foram calibradas nem validadas para esse uso.

Com cinco MQ, a média das probabilidades classifica corretamente **31/31 coletas e 10/10 vasos**, mas a coleta do inoculado 8 no dia 19 fica em 0,4984 para a classe 1, apenas 0,0016 abaixo do limiar. Com MQ e ambiente, essa média sobe para 0,5051 e a coleta fica incorreta: **30/31 coletas**, embora os erros por leitura caiam de 2.932 para 1.947. Os dez vasos continuam corretos após agregação.

## Pistas para investigar nas próximas coletas

As curvas de probabilidade mostram mudanças de regime dentro de algumas coletas. No inoculado 9 do dia 18, a probabilidade prevista cruza 0,50 aproximadamente aos 250 s e permanece acima desse limiar; no mesmo trecho, o sinal MQ-8 aumenta. No inoculado 7 do dia 11, os escores MQ crescem no final e chegam acima de 0,90 para a classe errada. Esses padrões são **hipóteses**, não causas demonstradas: podem refletir evolução do sinal durante a coleta, temperatura, umidade do solo, condições do equipamento ou diferenças entre dias e vasos.

Para a próxima campanha, vale registrar explicitamente a ordem e os horários das amostragens, temperatura/umidade, tempo de estabilização, calibração ou deriva dos MQ e condições de pressão/vácuo; repetir vasos e dias; e definir previamente qualquer janela temporal de análise. Uma janela escolhida agora por dar menos erros precisará ser testada em **novos vasos**. Também é útil confirmar a condição biológica por medição nematológica independente.

## Interpretação

Esta investigação torna os erros **rastreáveis** e ajuda a formular controles e hipóteses para novas coletas. Ela não transforma dez vasos em evidência definitiva: as leituras de uma mesma coleta são dependentes, e a escolha dos sensores e a exclusão prévia de cinco coletas foram feitas usando estes mesmos vasos. A evidência mais forte será verificar, com protocolo pré-definido, se os padrões e o desempenho se repetem em vasos e dias novos.
