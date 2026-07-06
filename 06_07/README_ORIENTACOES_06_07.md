# Organizacao das orientacoes 06/07

Esta pasta organiza o trabalho em quatro frentes, seguindo as orientacoes do professor:

1. `1_investigacao_hardware_banco`
   - verifica quais colunas existem;
   - confirma ausencia de coluna direta de corrente/tensao/alimentacao;
   - deixa checklist para conversa com Artur.

2. `2_filtragem_ruidos_anomalias`
   - analisa C13-C17 e C28;
   - guarda graficos e tabelas de anomalias;
   - contem o algoritmo de corte por variacao abrupta de pressao.

3. `3_compensacao_umidade_temperatura`
   - organiza as duas abordagens propostas: modelo com ambiente e correcao matematica;
   - deixa base com `Soil`, `Temp.` e `Pres.`;
   - inclui template para preencher fatores vindos dos datasheets.

4. `4_polimento_inicial_modelagem`
   - guarda o dataset ja limpo por pressao;
   - guarda o ExtraTrees rodado apenas apos o corte;
   - contem graficos, metricas, matriz de confusao e importancia dos sensores.

## Modelagem rodada

Depois da organizacao, foi executado o script:

`rodar_modelagem_extra_trees_rede_neural_importancia.py`

Ele treinou ExtraTrees e rede neural MLP nas pastas 1, 2, 3 e 4, salvando os resultados dentro de cada subpasta `modelagem`.

Resumo:

- melhor ExtraTrees: 91.51% no cenario 3, com MQ + ambiente;
- melhor MQ-only: 89.96% nos cenarios 2 e 4, com pressao filtrada;
- melhor rede neural: 89.94% no cenario 3, com MQ + ambiente;
- sensor mais forte no ExtraTrees: `MQ8`.

Comparativo geral:

`modelagem_comparativa/comparativo_extra_trees_rede_neural_importancia.csv`

Leitura principal: primeiro confirmar falhas fisicas e alimentacao, depois remover anomalias de pressao, depois discutir compensacao ambiental, e so entao treinar/avaliar modelos.
