from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
TARGET = "Classe"
FEATURES = [
    "MQ2_corrigido_env",
    "MQ3_corrigido_env",
    "MQ7_corrigido_env",
    "MQ8_corrigido_env",
    "MQ135_corrigido_env",
    "MQ138_corrigido_env",
    "Soil_indice_0_1",
    "Temp_C",
    "Pres_kPa",
]
RANDOM_STATE = 42


def classify_feature(feature: str) -> str:
    if feature.endswith("_corrigido_env"):
        return "mq_corrigido_por_datasheet_ambiente"
    return "ambiente_confirmado_por_datasheet"


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


def save_confusion(y_true: pd.Series, y_pred, output_png: Path, output_csv: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["real_com_nematoide", "real_sem_nematoide"],
        columns=["previsto_com_nematoide", "previsto_sem_nematoide"],
    ).to_csv(output_csv, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Com nematoide", "Sem nematoide"],
    )
    display.plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title("Melhor modelo - matriz de confusao")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_importance_plot(df: pd.DataFrame, output: Path, value_column: str, title: str) -> None:
    plot_df = df.sort_values(value_column, ascending=True)
    colors = [
        "#2d6cdf" if tipo == "mq_corrigido_por_datasheet_ambiente" else "#d9822b"
        for tipo in plot_df["tipo"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot_df["feature"], plot_df[value_column], color=colors)
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATASET)
    required = [TARGET, "Conjunto", *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset local: {missing}")

    train_df = df[df["Conjunto"] == "Treino"].copy()
    test_df = df[df["Conjunto"] == "Teste"].copy()

    model = build_model()
    model.fit(train_df[FEATURES], train_df[TARGET])
    prediction = model.predict(test_df[FEATURES])

    metrics = {
        "cenario": "melhor_modelo_extra_trees_06_07",
        "descricao": "ExtraTrees com MQ corrigidos e ambiente confirmado por datasheets.",
        "dataset": str(DATASET.relative_to(BASE)),
        "features": FEATURES,
        "linhas_treino": int(len(train_df)),
        "linhas_teste": int(len(test_df)),
        "split": "Mesmo split 70/30 por Coleta salvo na coluna Conjunto.",
        "extra_trees": {
            "accuracy": float(accuracy_score(test_df[TARGET], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(test_df[TARGET], prediction)),
            "f1_macro": float(f1_score(test_df[TARGET], prediction, average="macro")),
        },
    }

    (BASE / "metricas").mkdir(exist_ok=True)
    (BASE / "matriz_confusao").mkdir(exist_ok=True)
    (BASE / "importancia_sensores").mkdir(exist_ok=True)
    (BASE / "modelo").mkdir(exist_ok=True)

    with (BASE / "metricas" / "metricas_melhor_modelo_reproduzido.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    joblib.dump(model, BASE / "modelo" / "modelo_extra_trees_melhor_93_20.joblib")
    save_confusion(
        test_df[TARGET],
        prediction,
        BASE / "matriz_confusao" / "matriz_confusao_melhor_modelo.png",
        BASE / "matriz_confusao" / "matriz_confusao_melhor_modelo.csv",
    )

    native = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "tipo": [classify_feature(feature) for feature in FEATURES],
                "importancia": model.feature_importances_,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    native.to_csv(
        BASE / "importancia_sensores" / "importancia_nativa_extra_trees_melhor_modelo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_importance_plot(
        native,
        BASE / "importancia_sensores" / "grafico_importancia_nativa_melhor_modelo.png",
        "importancia",
        "Melhor modelo - importancia nativa",
    )

    sample_df = test_df.sample(n=min(8000, len(test_df)), random_state=RANDOM_STATE)
    perm_result = permutation_importance(
        model,
        sample_df[FEATURES],
        sample_df[TARGET],
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "tipo": [classify_feature(feature) for feature in FEATURES],
                "importancia_media": perm_result.importances_mean,
                "importancia_desvio": perm_result.importances_std,
                "amostra_teste_usada": len(sample_df),
            }
        )
        .sort_values("importancia_media", ascending=False)
        .reset_index(drop=True)
    )
    permutation.to_csv(
        BASE / "importancia_sensores" / "importancia_permutacao_extra_trees_melhor_modelo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_importance_plot(
        permutation,
        BASE / "importancia_sensores" / "grafico_importancia_permutacao_melhor_modelo.png",
        "importancia_media",
        "Melhor modelo - importancia por permutacao",
    )

    print(json.dumps(metrics["extra_trees"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
