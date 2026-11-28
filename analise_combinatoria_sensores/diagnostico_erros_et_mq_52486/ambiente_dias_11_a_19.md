# Temperatura, solo e pressão nas 31 coletas analisadas

Base: 52.486 leituras balanceadas, 31 coletas, dez vasos. Cada valor abaixo é a **mediana das medianas de cada coleta** no dia, para que uma coleta com mais leituras não domine o resumo. Os MQ são sinais brutos em unidades instrumentais; o MQ-7 foi registrado, mas não usado na configuração de cinco MQ do Extra Trees. O índice do solo varia de 0 a 1 e é **operacional**; não equivale a um percentual físico de umidade. Na codificação usada pelo estudo, índice acima de 0,4 corresponde a solo molhado.

| Dia | Coletas (inoc./controle) | MQ-2 | MQ-3 | MQ-7 | MQ-8 | MQ-135 | MQ-138 | Temp. (°C) | Índice do solo | Pressão (kPa) | Erros ET só MQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 7 (2/5) | 8.024,5 | 14.635 | 1.034 | 4.292 | 20.738 | 12.039 | 40,3 | 0,074 | 93,780 | 701/13.823 (5,07%) |
| 12 | 5 (5/0) | 6.365,5 | 14.348 | 222 | 2.547 | 18.680,5 | 10.179 | 39,4 | 0,052 | 93,700 | 0/10.801 |
| 13 | 4 (0/4) | 20.540,5 | 7.972 | 20.540,5 | 7.849 | 7.202,5 | 20.541 | 34,3 | 0,227 | 93,565 | 0/6.978 |
| 18 | 9 (5/4) | 11.546 | 18.548 | 345 | 3.604 | 21.101 | 8.983,5 | 39,0 | 0,958 | 93,540 | 1.120/13.883 (8,07%) |
| 19 | 6 (3/3) | 10.319,5 | 17.877,25 | 115,5 | 2.988 | 20.283 | 7.695 | 35,4 | 0,881 | 93,715 | 1.111/7.001 (15,87%) |

As medianas do índice por coleta ficaram entre 0,015 e 0,231 nos dias 11–13 e entre 0,728 e 0,967 nos dias 18–19. Todas as coletas dos dias 18 e 19 tinham mediana acima do limiar 0,4, enquanto as dos dias 11–13 ficaram abaixo. As medianas de temperatura das coletas foram de 25,4 a 45,6 °C nos dias 11–13 e de 26,7 a 41,8 °C nos dias 18–19: há ampla sobreposição. A pressão variou menos em termos absolutos: as medianas por coleta ficaram entre 93,48 e 93,86 kPa em todos os dias.

## Coletas inoculadas dos dias 18 e 19

| Dia | Vaso | Erros MQ | Temperatura mediana (°C) | Índice do solo mediano | Pressão mediana (kPa) |
|---|---:|---:|---:|---:|---:|
| 18 | 1 | 0/1.377 | 41,8 | 0,934 | 93,48 |
| 18 | 6 | 479/1.609 | 39,0 | 0,961 | 93,55 |
| 18 | 7 | 0/1.602 | 39,9 | 0,967 | 93,53 |
| 18 | 8 | 0/1.400 | 40,7 | 0,957 | 93,53 |
| 18 | 9 | 544/1.327 | 40,7 | 0,949 | 93,54 |
| 19 | 6 | 0/1.200 | 31,45 | 0,949 | 93,67 |
| 19 | 7 | 296/1.330 | 33,1 | 0,967 | 93,69 |
| 19 | 8 | 754/1.317 | 34,4 | 0,955 | 93,71 |

No dia 18, o vaso inoculado 9 teve a **mesma temperatura mediana (40,7 °C)** do inoculado 8, que não teve erros, e índices do solo e pressões próximos. Assim, nenhuma dessas três grandezas, considerada isoladamente no nível da coleta, identifica a falha. No dia 19, as três coletas inoculadas tiveram índice do solo alto; só os vasos 7 e 8 tiveram erros. Isso motiva verificar a evolução dos sinais MQ dentro de cada coleta e a ordem/condições de amostragem.

Os dias 12 e 13 contêm apenas uma das classes cada na base analisada. Data, grupo e condição do solo estão, portanto, parcialmente confundidos. A mudança acentuada no índice do solo depois da irrigação e a maior taxa de erros nos dias 18–19 são **associações descritivas**, não prova de que a umidade causou os erros. A coleta do vaso inoculado 7 no dia 11 teve 627 erros mesmo com índice do solo baixo.

Arquivos de apoio: `ambiente_31_coletas.csv`, `ambiente_por_dia.csv`, `ambiente_por_dia_e_classe.csv` e `ambiente_dias_11_a_19.png`.

As medianas MQ exatas por coleta e por dia estão em `medianas_mq_ambiente_por_coleta.csv` e `medianas_mq_ambiente_por_dia.csv`. O dia 13 apresenta medianas altas e semelhantes em MQ-2, MQ-7 e MQ-138; como as quatro coletas desse dia são controles, essa diferença também está confundida com a composição dos grupos e requer exame específico antes de qualquer interpretação biológica.
