# ExtraTrees pressao filtrada

Experimento com o mesmo classificador ExtraTrees, usando somente as linhas mantidas depois do corte por pressao.

Dataset: `comparacao\pressao_filtrada\antes_dia_20_pressao_filtrada_estrito.csv`
Linhas usadas: `63373`
Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138`

## Resultado

- Limiar validado: `0.72`
- Accuracy limiar 0.50: `0.8996`
- Balanced accuracy limiar 0.50: `0.9000`
- F1 macro limiar 0.50: `0.8996`
- Accuracy limiar ajustado: `0.8798`
- Balanced accuracy limiar ajustado: `0.8827`
- F1 macro limiar ajustado: `0.8795`

## Arquivos

- `resultados/metricas/metricas_extra_trees_pressao_filtrada.json`
- `resultados/matrizes/matriz_confusao_limiar_ajustado.png`
- `resultados/importancias/importancia_features_extra_trees_pressao_filtrada.csv`
- `modelo/modelo_extra_trees_pressao_filtrada.joblib`
