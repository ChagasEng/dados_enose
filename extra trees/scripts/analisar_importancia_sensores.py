from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "resultados" / "analises"
DATASET_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_extra_trees_limiar_ajustado.joblib"

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


def split_by_group_inside_class(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for class_value in sorted(df[target_column].unique()):
        class_block = df[df[target_column] == class_value]
        groups = (
            pd.Series(class_block[GROUP_COLUMN].unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_groups = set(groups[: int(len(groups) * TRAIN_RATIO)])
        train_parts.append(class_block[class_block[GROUP_COLUMN].isin(train_groups)])
        test_parts.append(class_block[~class_block[GROUP_COLUMN].isin(train_groups)])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df


def cohen_d_abs(class_0: pd.Series, class_1: pd.Series) -> float:
    std_0 = class_0.std(ddof=1)
    std_1 = class_1.std(ddof=1)
    pooled_std = np.sqrt(
        ((len(class_0) - 1) * std_0**2 + (len(class_1) - 1) * std_1**2)
        / (len(class_0) + len(class_1) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float(abs((class_1.mean() - class_0.mean()) / pooled_std))


def build_analysis() -> tuple[pd.DataFrame, float]:
    df, mq_columns, target_column = load_dataset()
    train_df, test_df = split_by_group_inside_class(df, target_column)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    rows = []
    for column in mq_columns:
        class_0 = train_df.loc[train_df[target_column] == 0, column].astype(float)
        class_1 = train_df.loc[train_df[target_column] == 1, column].astype(float)
        values = train_df[column].astype(float).to_numpy()
        target = train_df[target_column].to_numpy()
        auc = roc_auc_score(target, values)

        rows.append(
            {
                "sensor": column,
                "media_doente_classe_0": float(class_0.mean()),
                "media_saudavel_classe_1": float(class_1.mean()),
                "diferenca_media_1_menos_0": float(class_1.mean() - class_0.mean()),
                "cohen_d_abs": cohen_d_abs(class_0, class_1),
                "auc_individual_melhor_direcao": float(max(auc, 1 - auc)),
                "correlacao_abs_classe": float(abs(np.corrcoef(values, target)[0, 1])),
            }
        )

    analysis_df = pd.DataFrame(rows)
    analysis_df["mutual_info"] = mutual_info_classif(
        train_df[mq_columns], train_df[target_column], random_state=RANDOM_STATE
    )

    importance_df = pd.DataFrame(
        {
            "sensor": bundle["features"],
            "extra_trees_importance": model.feature_importances_,
        }
    )
    permutation = permutation_importance(
        model,
        test_df[mq_columns],
        test_df[target_column],
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    permutation_df = pd.DataFrame(
        {
            "sensor": mq_columns,
            "queda_balanced_accuracy_por_permutacao": permutation.importances_mean,
            "desvio_permutacao": permutation.importances_std,
        }
    )

    result_df = (
        importance_df.merge(analysis_df, on="sensor")
        .merge(permutation_df, on="sensor")
        .sort_values("extra_trees_importance", ascending=False)
    )
    baseline_balanced_accuracy = balanced_accuracy_score(
        test_df[target_column], model.predict(test_df[mq_columns])
    )
    return result_df, float(baseline_balanced_accuracy)


def save_importance_plot(result_df: pd.DataFrame) -> Path:
    plot_df = result_df.sort_values("extra_trees_importance", ascending=True)
    y = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    fig.suptitle("Importancia dos sensores MQ - Extra Trees", fontsize=14)

    axes[0].barh(y, plot_df["extra_trees_importance"], color="#2f6fba")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["sensor"])
    axes[0].set_xlabel("Importancia interna do Extra Trees")
    axes[0].set_title("Reducao de impureza nas arvores")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(
        y,
        plot_df["queda_balanced_accuracy_por_permutacao"],
        xerr=plot_df["desvio_permutacao"],
        color="#d88725",
    )
    axes[1].set_xlabel("Queda na balanced accuracy")
    axes[1].set_title("Permutacao no conjunto de teste")
    axes[1].grid(axis="x", alpha=0.2)

    for ax, column in [
        (axes[0], "extra_trees_importance"),
        (axes[1], "queda_balanced_accuracy_por_permutacao"),
    ]:
        max_value = plot_df[column].max()
        ax.set_xlim(0, max_value * 1.18)
        for index, value in enumerate(plot_df[column]):
            ax.text(
                value + max_value * 0.02,
                index,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = OUTPUT_DIR / "grafico_importancia_sensores.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df, baseline_balanced_accuracy = build_analysis()

    csv_path = OUTPUT_DIR / "analise_importancia_sensores.csv"
    txt_path = OUTPUT_DIR / "analise_importancia_sensores.txt"
    result_df.to_csv(csv_path, index=False)
    plot_path = save_importance_plot(result_df)

    report = [
        "Analise de importancia dos sensores MQ - Extra Trees",
        "",
        f"Balanced accuracy original no teste: {baseline_balanced_accuracy:.6f}",
        "",
        "A ordem principal vem de extra_trees_importance.",
        "Essa metrica mede quanto cada sensor reduziu a impureza das classes nas arvores.",
        "",
        result_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "Leitura recomendada:",
        "- cohen_d_abs: tamanho de efeito entre classe 0 e classe 1.",
        "- auc_individual_melhor_direcao: poder do sensor sozinho separar as classes.",
        "- mutual_info: dependencia nao linear entre sensor e classe.",
        "- queda_balanced_accuracy_por_permutacao: perda ao embaralhar o sensor no teste.",
    ]
    txt_path.write_text("\n".join(report), encoding="utf-8")
    print(f"CSV salvo em: {csv_path}")
    print(f"Relatorio salvo em: {txt_path}")
    print(f"Grafico salvo em: {plot_path}")


if __name__ == "__main__":
    main()
