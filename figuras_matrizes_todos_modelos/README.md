# Confusion matrices in the example layout

The requested images use three columns and two rows, blue 2×2 matrices, `WV` and `NO` on both axes, and one decimal place per cell. Each value is a percentage within the actual class. For this dataset, the displayed `WV` position corresponds to class 0 (inoculated) and `NO` to class 1 (non-inoculated control). These display labels came from the supplied example; they are not the original class names.

- `example_style_five_mq.png`: Extra Trees, Random Forest, MLP (ANN), k-NN, SVM, and LDA under LOSO by pot with five MQ sensors.
- `example_style_five_mq_environment.png`: the same six classifiers with soil, temperature, and pressure added.
- `example_style_five_mq_tcn.png` and `example_style_five_mq_environment_tcn.png`: TCN results for the corresponding LOSO scenarios.
- `example_style_historical.png`: TCN and Extra Trees on the separate historical dry/wet soil split by collection.

The ensemble panel was removed at the author's request. Its numerical results remain in `../analise_combinatoria_sensores/comparacao_knn_mlp_ensemble_filtrada/` for traceability. `confusion_matrix_counts.csv` retains exact counts for all computed models. The historical reference is Extra Trees in the local code, although some older figures label it Random Forest.

Run `gerar_estilo_exemplo.py` to regenerate the images and `gerar_matrizes.py` to regenerate the count CSV.

## Metric tables

`metricas_classificadores.csv` contains accuracy, balanced accuracy, Recall, and macro F1, all calculated per reading. The displayed Recall is for class 0 (inoculated). `tabela_metricas_loso.tex` and `tabela_metricas_historico.tex` are ready to insert into the Portuguese article. The ensemble is omitted from these tables at the author's request. Macro recall would repeat balanced accuracy for this binary classification task. Run `gerar_tabela_metricas.py` to regenerate them from the saved predictions and historical confusion matrices.
