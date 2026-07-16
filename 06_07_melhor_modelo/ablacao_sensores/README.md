# Ablacao e combinacoes de sensores MQ

Todos os testes mantem o mesmo ExtraTrees, a mesma divisao Treino/Teste por coleta e as variaveis Soil_indice_0_1, Temp_C e Pres_kPa.
Assim, a comparacao mede somente o efeito de usar ou retirar MQs.

- Baseline (6 MQ): 90.73% de balanced accuracy.
- Melhor combinacao absoluta: MQ2 | MQ3 | MQ8 | MQ135 | MQ138 (5 MQ), 91.17%.
- Menor combinacao com queda de no maximo 1.0 p.p. frente ao baseline: MQ2 | MQ8 | MQ135 (3 MQ), 90.13%.

A recomendacao de reducao e operacional para este split. Antes de fechar hardware, confirme a combinacao em novas coletas/ensaios independentes.
