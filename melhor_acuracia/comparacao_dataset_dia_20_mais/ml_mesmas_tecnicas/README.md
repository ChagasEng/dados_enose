# Comparacao ML: sem_pressao vs dia_20_mais

Foram aplicadas as mesmas tecnicas do experimento `melhor_acuracia` no dataset `dia_20_mais`.

- Dataset original: `sem pressao\dataset_sem_pressao.csv`
- Dataset novo: `dia_20_mais\dia_20_mais\dataset_dia_20_mais.csv`
- Features: `MQ2, MQ3, MQ7, MQ8, MQ135, MQ138`
- Melhor holdout no novo dataset: `hist_gradient_boosting` com accuracy `0.8208`
- Melhor CV media no novo dataset: `logistic_regression_scaled` com accuracy `0.7623`

## Graficos

- `graficos/grafico_dataset_treino_teste_dia_20_mais.png`
- `graficos/holdout_comparacao_tecnicas_accuracy.png`
- `graficos/holdout_comparacao_tecnicas_balanced_accuracy.png`
- `graficos/holdout_comparacao_tecnicas_f1_macro.png`
- `graficos/cv_media_comparacao_tecnicas_accuracy.png`
- `graficos/cv_media_comparacao_tecnicas_balanced_accuracy.png`
- `graficos/cv_media_comparacao_tecnicas_f1_macro.png`
