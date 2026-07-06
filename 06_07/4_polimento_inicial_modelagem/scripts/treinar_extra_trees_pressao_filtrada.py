from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATASET_PATH = (
    ROOT_DIR
    / "comparacao"
    / "pressao_filtrada"
    / "antes_dia_20_pressao_filtrada_estrito.csv"
)

GROUP_COLUMN = "Coleta"
TARGET_COLUMN = "Classe"
MQ_COLUMNS = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
TRAIN_RATIO = 0.70
RANDOM_STATE = 42
VALIDATION_RANDOM_STATE = 43

RESULTS_DIR = PROJECT_DIR / "resultados"
METRICS_DIR = RESULTS_DIR / "metricas"
MATRICES_DIR = RESULTS_DIR / "matrizes"
REPORTS_DIR = RESULTS_DIR / "relatorios"
IMPORTANCES_DIR = RESULTS_DIR / "importancias"
COMPARISONS_DIR = RESULTS_DIR / "comparacoes"
GRAPHICS_DIR = PROJECT_DIR / "graficos"
MODELS_DIR = PROJECT_DIR / "modelo"


def ensure_dirs() -> None:
    for directory in [
        METRICS_DIR,
        MATRICES_DIR,
        REPORTS_DIR,
        IMPORTANCES_DIR,
        COMPARISONS_DIR,
        GRAPHICS_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    required = [GROUP_COLUMN, TARGET_COLUMN, *MQ_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    model_df = df[["Coleta", "Dia", "Vaso", *MQ_COLUMNS, TARGET_COLUMN]].copy()
    for column in ["Dia", "Vaso", *MQ_COLUMNS, TARGET_COLUMN]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=[GROUP_COLUMN, *MQ_COLUMNS, TARGET_COLUMN])
    model_df[TARGET_COLUMN] = model_df[TARGET_COLUMN].astype(int)
    return model_df


def split_by_group_inside_class(
    df: pd.DataFrame, train_ratio: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value, class_block in df.groupby(TARGET_COLUMN):
        groups = (
            pd.Series(class_block[GROUP_COLUMN].dropna().unique())
            .sample(frac=1, random_state=random_state)
            .tolist()
        )
        train_group_count = int(len(groups) * train_ratio)
        train_groups = set(groups[:train_group_count])
        test_groups = set(groups[train_group_count:])
        train_block = class_block[class_block[GROUP_COLUMN].isin(train_groups)]
        test_block = class_block[class_block[GROUP_COLUMN].isin(test_groups)]

        train_parts.append(train_block)
        test_parts.append(test_block)
        summary.append(
            {
                "classe": int(class_value),
                "total_linhas": int(len(class_block)),
                "linhas_treino": int(len(train_block)),
                "linhas_teste": int(len(test_block)),
                "grupos_total": int(len(groups)),
                "grupos_treino": int(len(train_groups)),
                "grupos_teste": int(len(test_groups)),
            }
        )

    train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    return train_df, test_df, summary


def build_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def metrics_from_prediction(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
    }


def choose_threshold(model: ExtraTreesClassifier, validation_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    proba = model.predict_proba(validation_df[MQ_COLUMNS])[:, 1]
    y_true = validation_df[TARGET_COLUMN].to_numpy()
    rows = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        prediction = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics_from_prediction(y_true, prediction),
            }
        )

    threshold_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1_macro", "accuracy"], ascending=False
    )
    best_threshold = float(threshold_df.iloc[0]["threshold"])
    return best_threshold, threshold_df


def save_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0 - doente", "1 - saudavel"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_split_plot(split_summary: list[dict[str, int]]) -> None:
    split_df = pd.DataFrame(split_summary).sort_values("classe")
    x = np.arange(len(split_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    train_bars = ax.bar(
        x - width / 2,
        split_df["linhas_treino"],
        width,
        label="Treino",
        color="#2d6cdf",
    )
    test_bars = ax.bar(
        x + width / 2,
        split_df["linhas_teste"],
        width,
        label="Teste",
        color="#ff7f0e",
    )
    ax.set_xticks(x, [f"Classe {int(c)}" for c in split_df["classe"]])
    ax.set_ylabel("Linhas")
    ax.set_title("ExtraTrees pressao filtrada - split 70/30 por Coleta")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.bar_label(train_bars, padding=3, fontsize=9)
    ax.bar_label(test_bars, padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "grafico_split_treino_teste_pressao_filtrada.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load_dataset()

    train_df, test_df, holdout_summary = split_by_group_inside_class(
        df, TRAIN_RATIO, RANDOM_STATE
    )
    inner_train_df, validation_df, validation_summary = split_by_group_inside_class(
        train_df, TRAIN_RATIO, VALIDATION_RANDOM_STATE
    )

    threshold_model = build_model()
    threshold_model.fit(inner_train_df[MQ_COLUMNS], inner_train_df[TARGET_COLUMN])
    best_threshold, threshold_df = choose_threshold(threshold_model, validation_df)
    threshold_df.to_csv(
        COMPARISONS_DIR / "ajuste_limiar_extra_trees_validacao_pressao_filtrada.csv",
        index=False,
    )

    model = build_model()
    model.fit(train_df[MQ_COLUMNS], train_df[TARGET_COLUMN])

    test_proba = model.predict_proba(test_df[MQ_COLUMNS])[:, 1]
    prediction_050 = (test_proba >= 0.50).astype(int)
    prediction_tuned = (test_proba >= best_threshold).astype(int)

    metrics_050 = metrics_from_prediction(test_df[TARGET_COLUMN], prediction_050)
    metrics_tuned = metrics_from_prediction(test_df[TARGET_COLUMN], prediction_tuned)
    matrix_050 = confusion_matrix(test_df[TARGET_COLUMN], prediction_050, labels=[0, 1])
    matrix_tuned = confusion_matrix(test_df[TARGET_COLUMN], prediction_tuned, labels=[0, 1])

    pd.DataFrame(
        matrix_050,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_limiar_050.csv")
    pd.DataFrame(
        matrix_tuned,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_limiar_ajustado.csv")

    save_confusion_matrix(
        matrix_050,
        MATRICES_DIR / "matriz_confusao_limiar_050.png",
        "ExtraTrees pressao filtrada - limiar 0.50",
    )
    save_confusion_matrix(
        matrix_tuned,
        MATRICES_DIR / "matriz_confusao_limiar_ajustado.png",
        f"ExtraTrees pressao filtrada - limiar {best_threshold:.2f}",
    )
    save_split_plot(holdout_summary)

    pd.DataFrame(
        {"feature": MQ_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        IMPORTANCES_DIR / "importancia_features_extra_trees_pressao_filtrada.csv",
        index=False,
    )

    report = classification_report(
        test_df[TARGET_COLUMN],
        prediction_tuned,
        target_names=["0 - doente", "1 - saudavel"],
        digits=4,
    )
    (REPORTS_DIR / "relatorio_classificacao_limiar_ajustado.txt").write_text(
        report,
        encoding="utf-8",
    )

    metrics = {
        "modelo": "ExtraTreesClassifier com limiar ajustado",
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "observacao_dataset": (
            "Dataset antes_dia_20 com linhas removidas quando a pressao variou "
            "abruptamente e fora da faixa estavel."
        ),
        "target": TARGET_COLUMN,
        "target_mapping": {"0": "doente", "1": "saudavel"},
        "features_usadas": MQ_COLUMNS,
        "coluna_usada_apenas_para_split": GROUP_COLUMN,
        "split_holdout": "70/30 por grupos de Coleta dentro de cada classe",
        "split_validacao": "70/30 dentro do treino, tambem por grupos de Coleta",
        "random_state": RANDOM_STATE,
        "validation_random_state": VALIDATION_RANDOM_STATE,
        "parametros": model.get_params(),
        "threshold_padrao": 0.50,
        "threshold_validado": best_threshold,
        "metricas_teste_limiar_050": metrics_050,
        "metricas_teste_limiar_validado": metrics_tuned,
        "matriz_confusao_limiar_050": matrix_050.astype(int).tolist(),
        "matriz_confusao_limiar_validado": matrix_tuned.astype(int).tolist(),
        "linhas_dataset": int(len(df)),
        "treino_total": int(len(train_df)),
        "validacao_total": int(len(validation_df)),
        "teste_total": int(len(test_df)),
        "split_holdout_por_classe": holdout_summary,
        "split_validacao_por_classe": validation_summary,
    }
    (METRICS_DIR / "metricas_extra_trees_pressao_filtrada.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "model": model,
            "features": MQ_COLUMNS,
            "threshold": best_threshold,
            "metrics": metrics,
        },
        MODELS_DIR / "modelo_extra_trees_pressao_filtrada.joblib",
    )

    readme = [
        "# ExtraTrees pressao filtrada",
        "",
        "Experimento com o mesmo classificador ExtraTrees, usando somente as linhas mantidas depois do corte por pressao.",
        "",
        f"Dataset: `{metrics['dataset']}`",
        f"Linhas usadas: `{len(df)}`",
        f"Features: `{', '.join(MQ_COLUMNS)}`",
        "",
        "## Resultado",
        "",
        f"- Limiar validado: `{best_threshold:.2f}`",
        f"- Accuracy limiar 0.50: `{metrics_050['accuracy']:.4f}`",
        f"- Balanced accuracy limiar 0.50: `{metrics_050['balanced_accuracy']:.4f}`",
        f"- F1 macro limiar 0.50: `{metrics_050['f1_macro']:.4f}`",
        f"- Accuracy limiar ajustado: `{metrics_tuned['accuracy']:.4f}`",
        f"- Balanced accuracy limiar ajustado: `{metrics_tuned['balanced_accuracy']:.4f}`",
        f"- F1 macro limiar ajustado: `{metrics_tuned['f1_macro']:.4f}`",
        "",
        "## Arquivos",
        "",
        "- `resultados/metricas/metricas_extra_trees_pressao_filtrada.json`",
        "- `resultados/matrizes/matriz_confusao_limiar_ajustado.png`",
        "- `resultados/importancias/importancia_features_extra_trees_pressao_filtrada.csv`",
        "- `modelo/modelo_extra_trees_pressao_filtrada.joblib`",
    ]
    (PROJECT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("ExtraTrees pressao filtrada concluido.")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Linhas: {len(df)}")
    print(f"Limiar validado: {best_threshold:.2f}")
    print(
        "Teste limiar 0.50: "
        f"acc={metrics_050['accuracy']:.4f} "
        f"bal={metrics_050['balanced_accuracy']:.4f} "
        f"f1={metrics_050['f1_macro']:.4f}"
    )
    print(
        "Teste limiar ajustado: "
        f"acc={metrics_tuned['accuracy']:.4f} "
        f"bal={metrics_tuned['balanced_accuracy']:.4f} "
        f"f1={metrics_tuned['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
