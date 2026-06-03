from pathlib import Path
import json

import joblib
import matplotlib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
MATRICES_DIR = PROJECT_DIR / "resultados" / "matrizes"
METRICS_DIR = PROJECT_DIR / "resultados" / "metricas"
REPORTS_DIR = PROJECT_DIR / "resultados" / "relatorios"
IMPORTANCES_DIR = PROJECT_DIR / "resultados" / "importancias"
MODELS_DIR = PROJECT_DIR / "modelos"
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


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


def split_70_30_by_group_inside_class(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value]
        groups = (
            pd.Series(class_block[GROUP_COLUMN].unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_group_count = int(len(groups) * TRAIN_RATIO)
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

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df, summary


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="tanh",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=250,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=12,
                    random_state=RANDOM_STATE,
                    batch_size=2048,
                ),
            ),
        ]
    )


def main() -> None:
    for output_dir in [MATRICES_DIR, METRICS_DIR, REPORTS_DIR, IMPORTANCES_DIR, MODELS_DIR]:
        output_dir.mkdir(parents=True, exist_ok=True)

    df, mq_columns, target_column = load_dataset()
    train_df, test_df, split_summary = split_70_30_by_group_inside_class(
        df, target_column
    )

    model = build_model()
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_df[target_column])
    model.fit(train_df[mq_columns], train_df[target_column], model__sample_weight=sample_weight)

    prediction = model.predict(test_df[mq_columns])
    labels = [0, 1]
    matrix = confusion_matrix(test_df[target_column], prediction, labels=labels)
    normalized_matrix = confusion_matrix(
        test_df[target_column], prediction, labels=labels, normalize="true"
    )

    pd.DataFrame(
        matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_mlp.csv")

    pd.DataFrame(
        normalized_matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(MATRICES_DIR / "matriz_confusao_mlp_normalizada.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0 - doente", "1 - saudavel"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Purples", values_format="d", colorbar=False)
    ax.set_title("Matriz de confusao - MLP")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(MATRICES_DIR / "matriz_confusao_mlp.png", dpi=180)
    plt.close(fig)

    report = classification_report(
        test_df[target_column],
        prediction,
        labels=labels,
        target_names=["0_doente", "1_saudavel"],
        digits=4,
    )
    (REPORTS_DIR / "relatorio_classificacao_mlp.txt").write_text(
        report, encoding="utf-8"
    )

    importance_sample = test_df.sample(
        n=min(5000, len(test_df)), random_state=RANDOM_STATE
    )
    permutation = permutation_importance(
        model,
        importance_sample[mq_columns],
        importance_sample[target_column],
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pd.DataFrame(
        {
            "feature": mq_columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        IMPORTANCES_DIR / "importancia_features_mlp.csv", index=False
    )

    metrics = {
        "modelo": "MLPClassifier",
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "target": target_column,
        "target_mapping": {"0": "doente", "1": "saudavel"},
        "features_usadas": mq_columns,
        "coluna_usada_apenas_para_split": GROUP_COLUMN,
        "split": "70/30 por grupos de Coleta dentro de cada classe",
        "random_state": RANDOM_STATE,
        "tecnicas": [
            "StandardScaler",
            "early stopping",
            "regularizacao L2 via alpha",
            "pesos balanceados por classe",
            "permutation importance no conjunto de teste",
        ],
        "parametros": {
            "hidden_layer_sizes": [64, 32],
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "batch_size": 2048,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 12,
        },
        "iteracoes": int(model.named_steps["model"].n_iter_),
        "treino_total": int(len(train_df)),
        "teste_total": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df[target_column], prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_df[target_column], prediction)
        ),
        "f1_macro": float(f1_score(test_df[target_column], prediction, average="macro")),
        "matriz_confusao": matrix.astype(int).tolist(),
        "matriz_confusao_normalizada": normalized_matrix.tolist(),
        "split_por_classe": split_summary,
    }
    (METRICS_DIR / "metricas_mlp.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "model": model,
            "features": mq_columns,
            "target": target_column,
            "group_column_used_only_for_split": GROUP_COLUMN,
            "metrics": metrics,
        },
        MODELS_DIR / "modelo_mlp.joblib",
    )

    print("MLP treinado com sucesso.")
    print(f"Features usadas: {', '.join(mq_columns)}")
    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")
    print(f"Acuracia: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(matrix)


if __name__ == "__main__":
    main()
