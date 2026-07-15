# Diagnostico C13-C17 e C28

## Arquivos gerados

- `graficos/coletas_por_nematoide_atualizado_estilo_original.png`
- `graficos/diagnostico_visual_C13_C17_C28.png`
- `graficos/ranking_ruido_por_coleta.png`
- `analise_C13_C17_C28/estatisticas_por_coleta.csv`
- `analise_C13_C17_C28/eventos_ruido_C13_C17_C28.csv`
- `analise_C13_C17_C28/duplicatas_exatas_coletas.csv`

## Mapa

- `C13`: dia 19 - Soja Heterodera Vaso 6, com nematoide.
- `C14`: dia 19 - Soja Heterodera Vaso 7, com nematoide.
- `C15`: dia 19 - Soja Heterodera Vaso 8, com nematoide.
- `C16`: dia 19 - Soja Heterodera Vaso 9, com nematoide.
- `C17`: dia 19 - Soja Heterodera Vaso 1, com nematoide.
- `C28`: dia 13 - Soja Saudavel Vaso 1, sem nematoide.

## C28

C28 nao foi a coleta mais ruidosa pelo criterio de saltos abruptos em `Pres.` + MQ. Ela teve `14` eventos acima do limiar global, contra `1484` em C17. O que chama atencao em C28 e o tamanho dos degraus: `MQ2` teve amplitude de `8631` e maior salto de `6911`; `MQ138` teve amplitude de `3919` e maior salto de `3265`. A pressao em C28 ficou relativamente estavel, com amplitude de `0.04 kPa`.

Leitura provavel: C28 parece mais uma mudanca brusca de regime/saturacao/reacomodacao dos MQ do que ruido continuo causado por pressao. Como a pressao interna ja foi filtrada e ficou estavel, a causa mais provavel esta em transiente de gas, memoria/saturacao de sensor, fluxo interno, troca/manuseio ou efeito de contaminacao/residuo na camara.

Um detalhe adicional: em C28 o `Soil_indice_0_1` teve `233` saltos acima do limiar, com amplitude de `0.244` na escala 0-1. Portanto, parte do ruido visual nessa coleta vem do sensor capacitivo de umidade do solo, nao da pressao da camara. Isso pode indicar contato/posicionamento do sensor de solo, leitura analogica instavel ou mudanca real da condicao do solo naquele vaso.

## C13-C17

As coletas C13-C17 sao todas do dia 19 com nematoide, mas nao se comportam como repeticoes consistentes.

- `C15` (`dia 19 - Soja Heterodera Vaso 8`) e `C16` (`dia 19 - Soja Heterodera Vaso 9`) sao 100% identicas em 1322 linhas.

Essa duplicacao nao nasceu na correcao do melhor modelo. Ela ja aparece nas bases anteriores:

- `dataset_processado_por_dia_vaso_sem_vref0/dataset_unico_por_dia_vaso_sem_vref0.csv`: 1513 linhas identicas.
- `comparacao/datasets_com_ambiente/antes_dia_20_com_ambiente.csv`: 1513 linhas identicas.
- `06_07/2_filtragem_ruidos_anomalias/datasets_filtrados/antes_dia_20_pressao_filtrada_estrito.csv`: 1322 linhas identicas.
- `06_07_melhor_modelo/dados/dataset_melhor_modelo_sensores_corrigidos.csv`: 1322 linhas identicas.

Isso aponta para duplicacao na origem/montagem do dataset ou na planilha, nao para efeito biologico nem para erro do ExtraTrees.

`C17` e o ponto mais critico: ela teve `1484` saltos abruptos em Pres.+MQ, sendo `1480` nos MQ. Isso atingiu varios canais ao mesmo tempo (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135` e `MQ138`), enquanto a pressao variou pouco (`0.02 kPa`).

Leitura provavel: C17 tem cara de instabilidade de aquisicao/sinal dos MQ, saturacao ou transiente eletrico/ADC, nao de variacao fisica de pressao. Ja C15 e C16 devem ser auditadas antes de qualquer conclusao biologica, porque aparecem como vasos diferentes, mas com os mesmos valores linha a linha.

## Acao recomendada

1. Conferir no arquivo bruto/planilha se C15 e C16 realmente sao duas coletas diferentes ou se houve duplicacao de aba/dados.
2. Rodar uma versao do modelo removendo uma das duplicatas C15/C16.
3. Rodar outra versao removendo ou marcando C17 como coleta anomala.
4. Para C28, revisar o log experimental: troca de vaso, abertura/fechamento, tempo de estabilizacao, fluxo/bomba e possivel saturacao dos MQ.
5. Nao tratar esses pontos como resposta biologica sem validar a origem fisica/operacional.
