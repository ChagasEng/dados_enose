from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_06 = Path(__file__).resolve().parent
TARGET = "Classe"
GROUP = "Coleta"
MQ_FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
AMBIENT_FEATURES = ["Soil", "Temp.", "Pres."]
RANDOM_STATE = 42
VALIDATION_SPLIT_STATE = 43
TRAIN_RATIO = 0.70
PERMUTATION_SAMPLE_SIZE = 8000


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "01_baseline_antes_corte_mq_ambiente",
        "folder": BASE_06 / "1_investigacao_hardware_banco",
        "dataset": BASE_06
        / "1_investigacao_hardware_banco"
        / "dados_base"
        / "antes_dia_20_com_ambiente_baseline.csv",
        "features": [*MQ_FEATURES, *AMBIENT_FEATURES],
        "descricao": "Baseline antes do corte por pressao, usando MQ + ambiente.",
    },
    {
        "id": "02_filtrado_pressao_mq",
        "folder": BASE_06 / "2_filtragem_ruidos_anomalias",
        "dataset": BASE_06
        / "2_filtragem_ruidos_anomalias"
        / "datasets_filtrados"
        / "antes_dia_20_pressao_filtrada_estrito.csv",
        "features": MQ_FEATURES,
        "descricao": "Apos corte estrito por pressao, usando somente sensores MQ.",
    },
    {
        "id": "03_filtrado_pressao_mq_ambiente",
        "folder": BASE_06 / "3_compensacao_umidade_temperatura",
        "dataset": BASE_06
        / "3_compensacao_umidade_temperatura"
        / "dados_base"
        / "antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv",
        "features": [*MQ_FEATURES, *AMBIENT_FEATURES],
        "descricao": "Apos corte estrito por pressao, usando MQ + Soil + Temp. + Pres.",
    },
    {
        "id": "04_polido_final_mq",
        "folder": BASE_06 / "4_polimento_inicial_modelagem",
        "dataset": BASE_06
        / "4_polimento_inicial_modelagem"
        / "datasets_limpos"
        / "antes_dia_20_pressao_filtrada_estrito.csv",
        "features": MQ_FEATURES,
        "descricao": "Base polida final, usando somente sensores MQ.",
    },
]


def ensure_model_dirs(folder: Path) -> dict[str, Path]:
    base = folder / "modelagem"
    dirs = {
        "base": base,
        "metricas": base / "metricas",
        "matrizes": base / "matrizes",
        "importancias": base / "importancias",
        "graficos": base / "graficos",
        "modelos": base / "modelos",
        "relatorios": base / "relatorios",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_model_data(path: Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [GROUP, TARGET, *features]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {missing}")

    keep_columns = [column for column in ["Coleta", "Dia", "Vaso", "Tempo"] if column in df.columns]
    model_df = df[[*keep_columns, *features, TARGET]].copy()

    for column in [*features, TARGET]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=[GROUP, *features, TARGET])
    model_df[TARGET] = model_df[TARGET].astype(int)
    return model_df.reset_index(drop=True)


def split_by_group_inside_class(
    df: pd.DataFrame, train_ratio: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    train_parts = []
    test_parts = []
    rows = []

    for class_value, class_block in df.groupby(TARGET):
        groups = (
            pd.Series(class_block[GROUP].dropna().unique())
            .sample(frac=1, random_state=random_state)
            .tolist()
        )
        train_count = max(1, int(len(groups) * train_ratio))
        if train_count >= len(groups):
            train_count = len(groups) - 1

        train_groups = set(groups[:train_count])
        test_groups = set(groups[train_count:])
        train_block = class_block[class_block[GROUP].isin(train_groups)]
        test_block = class_block[class_block[GROUP].isin(test_groups)]

        train_parts.append(train_block)
        test_parts.append(test_block)
        rows.append(
            {
                "classe": int(class_value),
                "linhas_total": int(len(class_block)),
                "linhas_treino": int(len(train_block)),
                "linhas_teste": int(len(test_block)),
                "coletas_total": int(len(groups)),
                "coletas_treino": int(len(train_groups)),
                "coletas_teste": int(len(test_groups)),
            }
        )

    train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    return train_df, test_df, rows


def build_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight=None,
        bootstrap=False,
    )


def build_neural_network() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=512,
                    learning_rate_init=0.001,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_confusion(y_true: pd.Series, y_pred: np.ndarray, output_png: Path, output_csv: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["real_0_doente", "real_1_saudavel"],
        columns=["previsto_0_doente", "previsto_1_saudavel"],
    ).to_csv(output_csv)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0 - doente", "1 - saudavel"],
    )
    display.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def classify_feature(feature: str) -> str:
    if feature in MQ_FEATURES:
        return "sensor_mq"
    if feature in AMBIENT_FEATURES:
        return "ambiente"
    return "outro"


def save_importance_plot(df: pd.DataFrame, output: Path, title: str, value_column: str) -> None:
    plot_df = df.sort_values(value_column, ascending=True).tail(12)
    colors = ["#2d6cdf" if kind == "sensor_mq" else "#d9822b" for kind in plot_df["tipo"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df[value_column], color=colors)
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def sample_for_permutation(test_df: pd.DataFrame) -> pd.DataFrame:
    if len(test_df) <= PERMUTATION_SAMPLE_SIZE:
        return test_df
    return test_df.sample(n=PERMUTATION_SAMPLE_SIZE, random_state=RANDOM_STATE)


def run_permutation_importance(
    model: Any,
    test_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    sample_df = sample_for_permutation(test_df)
    result = permutation_importance(
        model,
        sample_df[features],
        sample_df[TARGET],
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": features,
                "tipo": [classify_feature(feature) for feature in features],
                "importancia_media": result.importances_mean,
                "importancia_desvio": result.importances_std,
                "amostra_teste_usada": len(sample_df),
            }
        )
        .sort_values("importancia_media", ascending=False)
        .reset_index(drop=True)
    )


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    dirs = ensure_model_dirs(scenario["folder"])
    features = scenario["features"]
    df = load_model_data(scenario["dataset"], features)
    train_df, test_df, split_summary = split_by_group_inside_class(
        df, TRAIN_RATIO, RANDOM_STATE
    )

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    extra_trees = build_extra_trees()
    extra_trees.fit(X_train, y_train)
    extra_pred = extra_trees.predict(X_test)
    extra_metrics = compute_metrics(y_test, extra_pred)

    neural_network = build_neural_network()
    neural_network.fit(X_train, y_train)
    neural_pred = neural_network.predict(X_test)
    neural_metrics = compute_metrics(y_test, neural_pred)

    joblib.dump(extra_trees, dirs["modelos"] / f"{scenario['id']}_extra_trees.joblib")
    joblib.dump(neural_network, dirs["modelos"] / f"{scenario['id']}_rede_neural_mlp.joblib")

    save_confusion(
        y_test,
        extra_pred,
        dirs["matrizes"] / f"{scenario['id']}_extra_trees_matriz_confusao.png",
        dirs["matrizes"] / f"{scenario['id']}_extra_trees_matriz_confusao.csv",
        f"{scenario['id']} - ExtraTrees",
    )
    save_confusion(
        y_test,
        neural_pred,
        dirs["matrizes"] / f"{scenario['id']}_rede_neural_matriz_confusao.png",
        dirs["matrizes"] / f"{scenario['id']}_rede_neural_matriz_confusao.csv",
        f"{scenario['id']} - Rede neural MLP",
    )

    split_df = pd.DataFrame(split_summary)
    split_df.to_csv(dirs["metricas"] / f"{scenario['id']}_split_por_coleta.csv", index=False)

    extra_tree_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "tipo": [classify_feature(feature) for feature in features],
                "importancia": extra_trees.feature_importances_,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    extra_tree_importance.to_csv(
        dirs["importancias"] / f"{scenario['id']}_extra_trees_importancia_nativa.csv",
        index=False,
    )
    save_importance_plot(
        extra_tree_importance,
        dirs["graficos"] / f"{scenario['id']}_extra_trees_importancia_nativa.png",
        f"{scenario['id']} - importancia ExtraTrees",
        "importancia",
    )

    extra_perm = run_permutation_importance(extra_trees, test_df, features)
    neural_perm = run_permutation_importance(neural_network, test_df, features)
    extra_perm.to_csv(
        dirs["importancias"] / f"{scenario['id']}_extra_trees_importancia_permutacao.csv",
        index=False,
    )
    neural_perm.to_csv(
        dirs["importancias"] / f"{scenario['id']}_rede_neural_importancia_permutacao.csv",
        index=False,
    )
    save_importance_plot(
        extra_perm,
        dirs["graficos"] / f"{scenario['id']}_extra_trees_importancia_permutacao.png",
        f"{scenario['id']} - ExtraTrees permutacao",
        "importancia_media",
    )
    save_importance_plot(
        neural_perm,
        dirs["graficos"] / f"{scenario['id']}_rede_neural_importancia_permutacao.png",
        f"{scenario['id']} - rede neural permutacao",
        "importancia_media",
    )

    metrics = {
        "cenario": scenario["id"],
        "descricao": scenario["descricao"],
        "dataset": str(scenario["dataset"].relative_to(BASE_06)),
        "features": features,
        "linhas_dataset": int(len(df)),
        "linhas_treino": int(len(train_df)),
        "linhas_teste": int(len(test_df)),
        "coletas_total": int(df[GROUP].nunique()),
        "split": "70/30 por grupos de Coleta dentro de cada classe",
        "extra_trees": extra_metrics,
        "rede_neural_mlp": neural_metrics,
        "top5_extra_trees_nativo": extra_tree_importance.head(5).to_dict(orient="records"),
        "top5_extra_trees_permutacao": extra_perm.head(5).to_dict(orient="records"),
        "top5_rede_neural_permutacao": neural_perm.head(5).to_dict(orient="records"),
    }
    with (dirs["metricas"] / f"{scenario['id']}_metricas.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    report = [
        f"# Modelagem - {scenario['id']}",
        "",
        scenario["descricao"],
        "",
        f"Dataset: `{scenario['dataset'].relative_to(BASE_06)}`",
        f"Linhas: `{len(df)}`",
        f"Coletas: `{df[GROUP].nunique()}`",
        f"Features: `{', '.join(features)}`",
        "",
        "## ExtraTrees",
        "",
        f"- accuracy: {extra_metrics['accuracy']:.4f}",
        f"- balanced accuracy: {extra_metrics['balanced_accuracy']:.4f}",
        f"- f1 macro: {extra_metrics['f1_macro']:.4f}",
        "",
        "## Rede neural MLP",
        "",
        f"- accuracy: {neural_metrics['accuracy']:.4f}",
        f"- balanced accuracy: {neural_metrics['balanced_accuracy']:.4f}",
        f"- f1 macro: {neural_metrics['f1_macro']:.4f}",
        "",
        "## Importancia",
        "",
        "Foram salvas tres leituras:",
        "",
        "- importancia nativa do ExtraTrees;",
        "- importancia por permutacao do ExtraTrees;",
        "- importancia por permutacao da rede neural.",
        "",
    ]
    (dirs["relatorios"] / f"{scenario['id']}_resumo_modelagem.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    return {
        "cenario": scenario["id"],
        "pasta": str(scenario["folder"].relative_to(BASE_06)),
        "dataset": str(scenario["dataset"].relative_to(BASE_06)),
        "features": ", ".join(features),
        "linhas_dataset": len(df),
        "linhas_teste": len(test_df),
        "extra_trees_accuracy": extra_metrics["accuracy"],
        "extra_trees_balanced_accuracy": extra_metrics["balanced_accuracy"],
        "extra_trees_f1_macro": extra_metrics["f1_macro"],
        "rede_neural_accuracy": neural_metrics["accuracy"],
        "rede_neural_balanced_accuracy": neural_metrics["balanced_accuracy"],
        "rede_neural_f1_macro": neural_metrics["f1_macro"],
        "top1_extra_trees": extra_tree_importance.iloc[0]["feature"],
        "top1_rede_neural_permutacao": neural_perm.iloc[0]["feature"],
    }


def save_comparison(summary_df: pd.DataFrame) -> None:
    out = BASE_06 / "modelagem_comparativa"
    out.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out / "comparativo_extra_trees_rede_neural_importancia.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.36
    ax.bar(
        x - width / 2,
        summary_df["extra_trees_accuracy"] * 100,
        width,
        label="ExtraTrees",
        color="#2d6cdf",
    )
    ax.bar(
        x + width / 2,
        summary_df["rede_neural_accuracy"] * 100,
        width,
        label="Rede neural MLP",
        color="#d9822b",
    )
    ax.set_xticks(x, summary_df["cenario"], rotation=18, ha="right")
    ax.set_ylabel("Acuracia (%)")
    ax.set_title("Comparativo de modelagem nas pastas 1, 2, 3 e 4")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "comparativo_extra_trees_rede_neural.png", dpi=180)
    plt.close(fig)

    ranking_rows = []
    for _, row in summary_df.iterrows():
        ranking_rows.append(
            {
                "cenario": row["cenario"],
                "melhor_modelo": "ExtraTrees"
                if row["extra_trees_accuracy"] >= row["rede_neural_accuracy"]
                else "Rede neural MLP",
                "melhor_accuracy": max(row["extra_trees_accuracy"], row["rede_neural_accuracy"]),
                "top1_extra_trees": row["top1_extra_trees"],
                "top1_rede_neural_permutacao": row["top1_rede_neural_permutacao"],
            }
        )
    pd.DataFrame(ranking_rows).to_csv(out / "resumo_melhores_modelos_e_sensores.csv", index=False)


def main() -> None:
    summary = []
    for scenario in SCENARIOS:
        print(f"Rodando: {scenario['id']}")
        summary.append(run_scenario(scenario))

    summary_df = pd.DataFrame(summary)
    save_comparison(summary_df)

    print("\nResumo:")
    print(summary_df.to_string(index=False))
    print(f"\nArquivos gerais em: {BASE_06 / 'modelagem_comparativa'}")


if __name__ == "__main__":
    main()
