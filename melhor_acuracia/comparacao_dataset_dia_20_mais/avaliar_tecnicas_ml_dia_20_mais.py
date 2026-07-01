from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parents[1]
DATASET_ORIGINAL_PATH = ROOT_DIR / "sem pressao" / "dataset_sem_pressao.csv"
DATASET_NOVO_PATH = ROOT_DIR / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv"
ORIGINAL_COMPARISONS_DIR = ROOT_DIR / "melhor_acuracia" / "resultados" / "comparacoes"
OUTPUT_DIR = PROJECT_DIR / "ml_mesmas_tecnicas"
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
GROUP_COLUMN = "Coleta"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


def find_target_column(columns: list[str]) -> str:
    for column in columns:
        if column.strip().lower() == "classe":
            return column
    raise ValueError("Coluna Classe nao encontrada.")


def load_dataset(path: Path) -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(path)
    target_column = find_target_column(df.columns.tolist())
    mq_columns = [column for column in df.columns if column.upper().startswith("MQ")]
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Coluna {GROUP_COLUMN} nao encontrada em {path}.")
    return df, mq_columns, target_column


def build_models() -> dict[str, object]:
    return {
        "random_forest_baseline": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "random_forest_leaf10": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=10,
        ),
        "extra_trees_leaf5": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight=None,
            bootstrap=False,
        ),
        "extra_trees_leaf10": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight=None,
            bootstrap=False,
        ),
        "extra_trees_leaf50": ExtraTreesClassifier(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=None,
            min_samples_leaf=50,
            class_weight="balanced",
            bootstrap=True,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "logistic_regression_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_70_30_by_group_inside_class(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    train_parts = []
    test_parts = []
    summary = []

    for class_value, class_block in df.groupby(target_column):
        groups = (
            pd.Series(class_block[GROUP_COLUMN].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_group_count = int(len(groups) * TRAIN_RATIO)
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

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train_df, test_df, summary


def score_model(model: object, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro")),
    }


def evaluate_holdout(
    df: pd.DataFrame, mq_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    train_df, test_df, split_summary = split_70_30_by_group_inside_class(
        df, target_column
    )
    rows = []

    for model_name, model in build_models().items():
        fitted_model = clone(model)
        fitted_model.fit(train_df[mq_columns], train_df[target_column])
        scores = score_model(fitted_model, test_df[mq_columns], test_df[target_column])
        rows.append(
            {
                "avaliacao": "holdout_70_30_por_coleta",
                "modelo": model_name,
                "features_usadas": ",".join(mq_columns),
                "treino_linhas": int(len(train_df)),
                "teste_linhas": int(len(test_df)),
                **scores,
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False), split_summary


def evaluate_group_cv(
    df: pd.DataFrame, mq_columns: list[str], target_column: str
) -> pd.DataFrame:
    x = df[mq_columns]
    y = df[target_column]
    groups = df[GROUP_COLUMN]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for model_name, model in build_models().items():
        fold_scores = []
        for fold_index, (train_index, test_index) in enumerate(cv.split(x, y, groups), 1):
            fitted_model = clone(model)
            fitted_model.fit(x.iloc[train_index], y.iloc[train_index])
            scores = score_model(fitted_model, x.iloc[test_index], y.iloc[test_index])
            fold_scores.append(scores)
            rows.append(
                {
                    "avaliacao": "stratified_group_kfold",
                    "modelo": model_name,
                    "fold": fold_index,
                    **scores,
                }
            )

        rows.append(
            {
                "avaliacao": "stratified_group_kfold_media",
                "modelo": model_name,
                "fold": "media",
                "accuracy": float(pd.Series([s["accuracy"] for s in fold_scores]).mean()),
                "balanced_accuracy": float(
                    pd.Series([s["balanced_accuracy"] for s in fold_scores]).mean()
                ),
                "f1_macro": float(pd.Series([s["f1_macro"] for s in fold_scores]).mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_train_test_split(split_summary: list[dict[str, int]]) -> None:
    split_df = pd.DataFrame(split_summary).sort_values("classe")
    labels = [f"Classe {classe}" for classe in split_df["classe"]]
    x = range(len(split_df))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_train = ax.bar(
        [i - 0.18 for i in x],
        split_df["linhas_treino"],
        width=0.36,
        label="Treino",
        color="#2f6f73",
    )
    bars_test = ax.bar(
        [i + 0.18 for i in x],
        split_df["linhas_teste"],
        width=0.36,
        label="Teste",
        color="#c7503d",
    )
    ax.set_title("dia_20_mais - split treino/teste por Coleta")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Linhas")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    ax.bar_label(bars_train, padding=3, fontsize=9)
    ax.bar_label(bars_test, padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "grafico_dataset_treino_teste_dia_20_mais.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison(
    original_df: pd.DataFrame, novo_df: pd.DataFrame, prefix: str, title_suffix: str
) -> None:
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "f1_macro": "F1 macro",
    }
    original = original_df.assign(dataset="sem_pressao_melhor")
    novo = novo_df.assign(dataset="dia_20_mais")
    combined = pd.concat([original, novo], ignore_index=True)
    model_order = (
        combined.groupby("modelo")["accuracy"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )

    for metric, title in metric_labels.items():
        pivot = combined.pivot_table(index="modelo", columns="dataset", values=metric)
        pivot = pivot.reindex(model_order)
        fig, ax = plt.subplots(figsize=(10, 6))
        y = range(len(pivot))
        ax.barh(
            [i - 0.18 for i in y],
            pivot["sem_pressao_melhor"],
            height=0.36,
            label="sem_pressao (melhor)",
            color="#2f6f73",
        )
        ax.barh(
            [i + 0.18 for i in y],
            pivot["dia_20_mais"],
            height=0.36,
            label="dia_20_mais",
            color="#c7503d",
        )
        ax.set_title(f"{title_suffix} - {title}")
        ax.set_xlabel(title)
        ax.set_yticks(list(y), pivot.index)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.18)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(GRAFICOS_DIR / f"{prefix}_comparacao_tecnicas_{metric}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

    novo_df, mq_columns, target_column = load_dataset(DATASET_NOVO_PATH)
    holdout_novo, split_summary = evaluate_holdout(novo_df, mq_columns, target_column)
    cv_novo = evaluate_group_cv(novo_df, mq_columns, target_column)

    holdout_original = pd.read_csv(
        ORIGINAL_COMPARISONS_DIR / "comparacao_tecnicas_ml_holdout.csv"
    )
    cv_original = pd.read_csv(ORIGINAL_COMPARISONS_DIR / "comparacao_tecnicas_ml_cv.csv")
    cv_original_media = cv_original[cv_original["fold"].astype(str) == "media"]
    cv_novo_media = cv_novo[cv_novo["fold"].astype(str) == "media"]

    holdout_novo.to_csv(OUTPUT_DIR / "comparacao_tecnicas_ml_holdout_dia_20_mais.csv", index=False)
    cv_novo.to_csv(OUTPUT_DIR / "comparacao_tecnicas_ml_cv_dia_20_mais.csv", index=False)

    holdout_comparado = holdout_original.merge(
        holdout_novo,
        on=["avaliacao", "modelo", "features_usadas"],
        suffixes=("_sem_pressao", "_dia_20_mais"),
    )
    holdout_comparado.to_csv(
        OUTPUT_DIR / "comparacao_holdout_sem_pressao_vs_dia_20_mais.csv", index=False
    )

    cv_comparado = cv_original_media.merge(
        cv_novo_media,
        on=["avaliacao", "modelo", "fold"],
        suffixes=("_sem_pressao", "_dia_20_mais"),
    )
    cv_comparado.to_csv(
        OUTPUT_DIR / "comparacao_cv_media_sem_pressao_vs_dia_20_mais.csv", index=False
    )

    plot_train_test_split(split_summary)
    plot_metric_comparison(
        holdout_original,
        holdout_novo,
        "holdout",
        "Holdout 70/30 por Coleta",
    )
    plot_metric_comparison(
        cv_original_media,
        cv_novo_media,
        "cv_media",
        "Media StratifiedGroupKFold por Coleta",
    )

    summary = {
        "dataset_original_melhor_acuracia": str(DATASET_ORIGINAL_PATH.relative_to(ROOT_DIR)),
        "dataset_novo": str(DATASET_NOVO_PATH.relative_to(ROOT_DIR)),
        "features_usadas": mq_columns,
        "target": target_column,
        "split_novo": split_summary,
        "melhor_holdout_novo": holdout_novo.iloc[0].to_dict(),
        "melhor_cv_media_novo": cv_novo_media.sort_values("accuracy", ascending=False)
        .iloc[0]
        .to_dict(),
    }
    (OUTPUT_DIR / "resumo_comparacao_ml.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = [
        "# Comparacao ML: sem_pressao vs dia_20_mais",
        "",
        "Foram aplicadas as mesmas tecnicas do experimento `melhor_acuracia` no dataset `dia_20_mais`.",
        "",
        f"- Dataset original: `{summary['dataset_original_melhor_acuracia']}`",
        f"- Dataset novo: `{summary['dataset_novo']}`",
        f"- Features: `{', '.join(mq_columns)}`",
        f"- Melhor holdout no novo dataset: `{summary['melhor_holdout_novo']['modelo']}` com accuracy `{summary['melhor_holdout_novo']['accuracy']:.4f}`",
        f"- Melhor CV media no novo dataset: `{summary['melhor_cv_media_novo']['modelo']}` com accuracy `{summary['melhor_cv_media_novo']['accuracy']:.4f}`",
        "",
        "## Graficos",
        "",
        "- `graficos/grafico_dataset_treino_teste_dia_20_mais.png`",
        "- `graficos/holdout_comparacao_tecnicas_accuracy.png`",
        "- `graficos/holdout_comparacao_tecnicas_balanced_accuracy.png`",
        "- `graficos/holdout_comparacao_tecnicas_f1_macro.png`",
        "- `graficos/cv_media_comparacao_tecnicas_accuracy.png`",
        "- `graficos/cv_media_comparacao_tecnicas_balanced_accuracy.png`",
        "- `graficos/cv_media_comparacao_tecnicas_f1_macro.png`",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("Resultados gerados em:", OUTPUT_DIR)
    print("Holdout dia_20_mais:")
    print(holdout_novo[["modelo", "accuracy", "balanced_accuracy", "f1_macro"]].to_string(index=False))
    print()
    print("CV media dia_20_mais:")
    print(
        cv_novo_media.sort_values("accuracy", ascending=False)[
            ["modelo", "accuracy", "balanced_accuracy", "f1_macro"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
