# Ablacao e combinacoes de sensores MQ

Todos os testes mantem o mesmo ExtraTrees, a mesma divisao Treino/Teste por coleta e as variaveis Soil_indice_0_1, Temp_C e Pres_kPa.
Assim, a comparacao mede somente o efeito de usar ou retirar MQs.
A coleta C16 (dia 19 - Soja Heterodera Vaso 9) foi removida somente do treino; a compensacao ambiental foi recalculada usando o treino restante.

- Baseline (6 MQ): 92.90% de balanced accuracy.
- Melhor combinacao absoluta: MQ3 | MQ135 (2 MQ), 93.91%.
- Menor combinacao com queda de no maximo 1.0 p.p. frente ao baseline: MQ2 (1 MQ), 92.58%.

A recomendacao de reducao e operacional para este split. Antes de fechar hardware, confirme a combinacao em novas coletas/ensaios independentes.
