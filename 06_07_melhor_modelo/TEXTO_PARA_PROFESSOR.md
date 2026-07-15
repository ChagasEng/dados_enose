# Texto para apresentar ao professor

Professor, depois de confirmar o hardware, separamos corretamente as variaveis ambientais: `Temp.` e `Pres.` vieram do BMP280, e `Soil` veio do sensor capacitivo de umidade do solo V2.0. Com isso, tratamos os MQ com uma compensacao ambiental orientada pelos datasheets. Como o BMP280 nao mede umidade relativa do ar, ainda nao aplicamos a calibracao fisica completa `Rs/R0` com RH; em vez disso, removemos estatisticamente dos MQ a componente associada a `Soil`, `Temp.` e `Pres.` usando somente o conjunto de treino.

Depois de substituir a C16 duplicada pelos dados corretos e refazer todo o processamento, o melhor resultado foi com ExtraTrees usando `MQ corrigido + ambiente confirmado`, chegando a `90,83%` de acuracia, `90,73%` de balanced accuracy e `90,80%` de F1 macro. A matriz de confusao e as importancias mostram que o modelo classifica bem, mas `Pres_kPa` ainda aparece como variavel forte. Como essa pressao e interna a camara de gases, ela pode conter informacao real do processo respiratorio, mas tambem pode refletir vedacao, bomba, manuseio ou condicao fisica da camara.

O proximo passo cientifico e validar se a pressao interna esta associada a respiracao/atividade biologica ou se e artefato fisico. Para isso, precisamos de ensaios controle com a mesma camara, mesma vedacao e mesma rotina, alem de medir umidade relativa do ar para permitir a calibracao completa dos MQ por datasheet.

Tambem fizemos uma auditoria das coletas C13-C17 e C28. A C28 nao foi a coleta mais ruidosa em pressao/MQ; ela apresentou degraus grandes em alguns MQ e muito ruido no sensor capacitivo de solo, enquanto a pressao ficou relativamente estavel. Ja C17 foi a coleta mais critica por ruido nos MQ. A duplicacao entre C15 e C16 foi resolvida com a substituicao da C16 pelos dados corretos; elas agora apresentam sinais diferentes.

## Imagens para mostrar

1. `graficos/painel_resumo_melhor_modelo.png`
2. `graficos/comparacao_metricas_extratrees.png`
3. `matriz_confusao/matriz_confusao_melhor_modelo.png`
4. `importancia_sensores/grafico_importancia_nativa_melhor_modelo.png`
5. `graficos/coletas_por_nematoide_sinais_corrigidos.png`
6. `graficos/correlacao_ambiente_antes_depois_correcao.png`
7. `graficos/coletas_por_nematoide_atualizado_estilo_original.png`
8. `graficos/diagnostico_visual_C13_C17_C28.png`
9. `graficos/ranking_ruido_por_coleta.png`
