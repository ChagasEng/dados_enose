from pathlib import Path
import json

import joblib
import matplotlib
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


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
MATRICES_DIR = PROJECT_DIR / "resultados" / "matrizes"
METRICS_DIR = PROJECT_DIR / "resultados" / "metricas"
REPORTS_DIR = PROJECT_DIR / "resultados" / "relatorios"
IMPORTANCES_DIR = PROJECT_DIR / "resultados" / "importancias"
MODELS_DIR = PROJECT_DIR / "modelos"
COMPARISONS_DIR = PROJECT_DIR / "resultados" / "comparacoes"
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.70
RANDOM_STATE = 42
VALIDATION_RANDOM_STATE = 43


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def load_dataset() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(DATASET_PATH)
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]

    model_df = df[[GROUP_COLUMN] + mq_columns + [target_column]].copy()
    for column in mq_columns + [target_column]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column])
    model_df[target_column] = model_df[target_column].astype(int)
    return model_df, mq_columns, target_column


def split_by_group_inside_class(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value]
        groups = (
            pd.Series(class_block[GROUP_COLUMN].unique())
            .sample(frac=1, random_state=random_state)
            .tolist()
        )
        train_group_count = int(len(groups) * train_ratio)
        train_groups = set(groups[:train_group_count])

        train_part = class_block[class_block[GROUP_COLUMN].isin(train_groups)]
        test_part = class_block[~class_block[GROUP_COLUMN].isin(train_groups)]

        train_parts.append(train_part)
        test_parts.append(test_part)
        summary.append(
            {
                "classe": int(class_value),
                "total_linhas": int(len(class_block)),
                "linhas_treino": int(len(train_part)),
                "linhas_teste": int(len(test_part)),
                "grupos_total": int(len(groups)),
                "grupos_treino": int(len(train_groups)),
                "grupos_teste": int(len(groups) - len(train_groups)),
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


def tune_threshold(
    y_true: pd.Series, probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.30, 0.75, 46):
        prediction = (probabilities >= threshold).astype(int)
        metrics = metrics_from_prediction(y_true, prediction)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "f1_macro"], ascending=False
    )
    return float(threshold_df.iloc[0]["threshold"]), threshold_df


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0 - doente", "1 - saudavel"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title("Matriz de confusao - Extra Trees com limiar ajustado")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    for output_dir in [
        MATRICES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        IMPORTANCES_DIR,
        MODELS_DIR,
        COMPARISONS_DIR,
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)

    df, mq_columns, target_column = load_dataset()
    train_df, test_df, holdout_summary = split_by_group_inside_class(
        df, target_column, TRAIN_RATIO, RANDOM_STATE
    )
    fit_df, validation_df, validation_summary = split_by_group_inside_class(
        train_df, target_column, VALIDATION_RATIO, VALIDATION_RANDOM_STATE
    )

    validation_model = build_model()
    validation_model.fit(fit_df[mq_columns], fit_df[target_column])
    validation_probabilities = validation_model.predict_proba(validation_df[mq_columns])[:, 1]
    best_threshold, threshold_df = tune_threshold(
        validation_df[target_column], validation_probabilities
    )
    threshold_df.to_csv(
        COMPARISONS_DIR / "ajuste_limiar_extra_trees_validacao.csv", index=False
    )

    final_model = build_model()
    final_model.fit(train_df[mq_columns], train_df[target_column])
    test_probabilities = final_model.predict_proba(test_df[mq_columns])[:, 1]
    default_prediction = (test_probabilities >= 0.50).astype(int)
    tuned_prediction = (test_probabilities >= best_threshold).astype(int)

    labels = [0, 1]
    default_matrix = confusion_matrix(
        test_df[target_column], default_prediction, labels=labels
    )
    tuned_matrix = confusion_matrix(
        test_df[target_column], tuned_prediction, labels=labels
    )
    tuned_normalized_matrix = confusion_matrix(
        test_df[target_column], tuned_prediction, labels=labels, normalize="true"
    )

    pd.DataFrame(
        tuned_matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_extra_trees_limiar_ajustado.csv")

    pd.DataFrame(
        tuned_normalized_matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(
        MATRICES_DIR / "matriz_confusao_extra_trees_limiar_ajustado_normalizada.csv"
    )

    save_confusion_matrix(
        tuned_matrix,
        MATRICES_DIR / "matriz_confusao_extra_trees_limiar_ajustado.png",
    )

    report = classification_report(
        test_df[target_column],
        tuned_prediction,
        labels=labels,
        target_names=["0_doente", "1_saudavel"],
        digits=4,
    )
    (REPORTS_DIR / "relatorio_classificacao_extra_trees_limiar_ajustado.txt").write_text(
        report, encoding="utf-8"
    )

    pd.DataFrame(
        {
            "feature": mq_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        IMPORTANCES_DIR / "importancia_features_extra_trees_limiar_ajustado.csv",
        index=False,
    )

    default_metrics = metrics_from_prediction(test_df[target_column], default_prediction)
    tuned_metrics = metrics_from_prediction(test_df[target_column], tuned_prediction)
    metrics = {
        "modelo": "ExtraTreesClassifier com limiar ajustado",
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "target": target_column,
        "target_mapping": {"0": "doente", "1": "saudavel"},
        "features_usadas": mq_columns,
        "coluna_usada_apenas_para_split": GROUP_COLUMN,
        "split_holdout": "70/30 por grupos de Coleta dentro de cada classe",
        "split_validacao": "70/30 dentro do treino, tambem por grupos de Coleta",
        "random_state": RANDOM_STATE,
        "validation_random_state": VALIDATION_RANDOM_STATE,
        "parametros": {
            "n_estimators": 900,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": None,
            "bootstrap": False,
        },
        "threshold_padrao": 0.50,
        "threshold_validado": best_threshold,
        "metricas_teste_limiar_050": default_metrics,
        "metricas_teste_limiar_validado": tuned_metrics,
        "matriz_confusao_limiar_050": default_matrix.astype(int).tolist(),
        "matriz_confusao_limiar_validado": tuned_matrix.astype(int).tolist(),
        "matriz_confusao_normalizada_limiar_validado": tuned_normalized_matrix.tolist(),
        "treino_total": int(len(train_df)),
        "validacao_total": int(len(validation_df)),
        "teste_total": int(len(test_df)),
        "split_holdout_por_classe": holdout_summary,
        "split_validacao_por_classe": validation_summary,
        "observacao": (
            "O limiar de decisao foi selecionado em validacao interna separada por "
            "Coleta. O teste final permaneceu separado por Coleta."
        ),
    }
    (METRICS_DIR / "metricas_extra_trees_limiar_ajustado.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "model": final_model,
            "features": mq_columns,
            "threshold": best_threshold,
            "target": target_column,
            "group_column_used_only_for_split": GROUP_COLUMN,
            "metrics": metrics,
        },
        MODELS_DIR / "modelo_extra_trees_limiar_ajustado.joblib",
    )

    print("Extra Trees com limiar ajustado treinado com sucesso.")
    print(f"Limiar validado: {best_threshold:.2f}")
    print(
        "Teste limiar 0.50: "
        f"acc={default_metrics['accuracy']:.4f} "
        f"bal={default_metrics['balanced_accuracy']:.4f} "
        f"f1={default_metrics['f1_macro']:.4f}"
    )
    print(
        "Teste limiar validado: "
        f"acc={tuned_metrics['accuracy']:.4f} "
        f"bal={tuned_metrics['balanced_accuracy']:.4f} "
        f"f1={tuned_metrics['f1_macro']:.4f}"
    )
    print(tuned_matrix)


if __name__ == "__main__":
    main()
