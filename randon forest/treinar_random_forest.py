from pathlib import Path
import json

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_PATH = (
    ROOT_DIR
    / "dataset_processado_por_dia_vaso_sem_vref0_sem_tempo_soil_temp_pres"
    / "dataset_unico_por_dia_vaso_sem_vref0_sem_tempo_soil_temp_pres.csv"
)
TRAIN_RATIO = 0.70
RANDOM_STATE = 42
GROUP_COLUMN = "Coleta"


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.lower() == "classe":
            return column
    raise ValueError("Coluna alvo 'classe' nao encontrada.")


def split_70_30_by_group_inside_each_class_block(
    df: pd.DataFrame, target_column: str, group_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value].copy()
        groups = (
            pd.Series(class_block[group_column].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_group_count = int(len(groups) * TRAIN_RATIO)
        train_groups = set(groups[:train_group_count])

        train_part = class_block[class_block[group_column].isin(train_groups)]
        test_part = class_block[~class_block[group_column].isin(train_groups)]

        train_parts.append(train_part)
        test_parts.append(test_part)
        summary.append(
            {
                "classe": int(class_value),
                "total": int(len(class_block)),
                "treino": int(len(train_part)),
                "teste": int(len(test_part)),
                "grupos_total": int(len(groups)),
                "grupos_treino": int(len(train_groups)),
                "grupos_teste": int(len(groups) - len(train_groups)),
            }
        )

    train_df = pd.concat(train_parts, axis=0).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df, summary


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]

    if not mq_columns:
        raise ValueError("Nenhuma coluna MQ encontrada para o treinamento.")
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Coluna de agrupamento '{GROUP_COLUMN}' nao encontrada.")

    model_df = df[[GROUP_COLUMN] + mq_columns + [target_column]].copy()
    for column in mq_columns + [target_column]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    rows_before = len(model_df)
    model_df = model_df.dropna(subset=[GROUP_COLUMN] + mq_columns + [target_column]).copy()
    model_df[target_column] = model_df[target_column].astype(int)
    dropped_rows = rows_before - len(model_df)

    train_df, test_df, split_summary = split_70_30_by_group_inside_each_class_block(
        model_df, target_column, GROUP_COLUMN
    )

    x_train = train_df[mq_columns]
    y_train = train_df[target_column]
    x_test = test_df[mq_columns]
    y_test = test_df[target_column]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    labels = [0, 1]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    normalized_matrix = confusion_matrix(
        y_test, y_pred, labels=labels, normalize="true"
    )

    pd.DataFrame(
        matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(BASE_DIR / "matriz_confusao.csv")

    pd.DataFrame(
        normalized_matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(BASE_DIR / "matriz_confusao_normalizada.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0 - doente", "1 - saudavel"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Matriz de confusao - Random Forest")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "matriz_confusao.png", dpi=180)
    plt.close(fig)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=["0_doente", "1_saudavel"],
        digits=4,
    )
    (BASE_DIR / "relatorio_classificacao.txt").write_text(report, encoding="utf-8")

    pd.DataFrame(
        {
            "feature": mq_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        BASE_DIR / "importancia_features.csv", index=False
    )

    pd.DataFrame(split_summary).to_csv(
        BASE_DIR / "resumo_split_70_30_por_classe.csv", index=False
    )

    metrics = {
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "target": target_column,
        "target_mapping": {"0": "doente", "1": "saudavel"},
        "features_usadas": mq_columns,
        "coluna_usada_apenas_para_split": GROUP_COLUMN,
        "train_ratio_por_classe": TRAIN_RATIO,
        "split": "70/30 por grupos de Coleta dentro de cada classe",
        "random_state": RANDOM_STATE,
        "linhas_removidas_por_nan": int(dropped_rows),
        "treino_total": int(len(train_df)),
        "teste_total": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "matriz_confusao": matrix.astype(int).tolist(),
        "matriz_confusao_normalizada": normalized_matrix.tolist(),
        "split_por_classe": split_summary,
    }
    (BASE_DIR / "metricas.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(model, BASE_DIR / "modelo_random_forest.joblib")

    print("Random Forest treinado com sucesso.")
    print(f"Features usadas: {', '.join(mq_columns)}")
    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(matrix)


if __name__ == "__main__":
    main()
