"""Aplica SHAP aos tres cenarios experimentais apresentados no capitulo.

Os cenarios a e b compartilham o mesmo modelo Extra Trees e sao separados
somente depois da predicao, pela faixa operacional de umidade. O cenario c
usa o modelo Extra Trees treinado no conjunto sem pressao. Esta escolha
reproduz exatamente as matrizes de confusao atualmente publicadas.
"""

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
import shap
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "xai_todos_cenarios"
RESULTS = BASE / "resultados"
RANDOM_STATE = 42
MAX_SHAP_ROWS = 2000
TARGET = "Classe"
GROUP = "Coleta"
CLASS_NAMES = {0: "Com nematoide", 1: "Sem nematoide"}

PRESSURE_FEATURES = [
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
NO_PRESSURE_FEATURES = [
    "MQ2",
    "MQ3",
    "MQ7",
    "MQ8",
    "MQ135",
    "MQ138",
    "Soil",
    "Temp.",
    "Pres.",
]

SCENARIOS = [
    {
        "id": "a_solo_seco_com_pressao",
        "title": "a) Solo seco sob pressão forçada",
        "source": "pressure",
        "filter": "dry",
        "expected_matrix": [[5864, 643], [0, 10185]],
    },
    {
        "id": "b_solo_umido_com_pressao",
        "title": "b) Solo úmido sob pressão forçada",
        "source": "pressure",
        "filter": "wet",
        "expected_matrix": [[3513, 639], [746, 525]],
    },
    {
        "id": "c_sem_pressao_sem_estratificacao",
        "title": "c) Sem pressão forçada, sem estratificação por umidade",
        "source": "no_pressure",
        "filter": None,
        "expected_matrix": [[2968, 1456], [0, 4929]],
    },
]


def split_by_collection_inside_class(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduz o split 70/30 usado para treinar o modelo do cenario c."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, block in df.groupby(TARGET):
        groups = (
            pd.Series(block[GROUP].dropna().unique())
            .sample(frac=1, random_state=RANDOM_STATE)
            .tolist()
        )
        train_count = max(1, int(len(groups) * 0.70))
        if train_count >= len(groups):
            train_count = len(groups) - 1
        train_groups = set(groups[:train_count])
        train_parts.append(block[block[GROUP].isin(train_groups)])
        test_parts.append(block[~block[GROUP].isin(train_groups)])
    train = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE)
    test = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE)
    return train, test


def load_pressure_source() -> tuple[Any, pd.DataFrame, list[str]]:
    dataset = (
        ROOT
        / "06_07_melhor_modelo"
        / "dados"
        / "dataset_melhor_modelo_sensores_corrigidos.csv"
    )
    model_path = (
        ROOT
        / "06_07_melhor_modelo"
        / "modelo"
        / "modelo_extra_trees_melhor_93_20.joblib"
    )
    df = pd.read_csv(dataset)
    test = df.loc[df["Conjunto"] == "Teste"].copy()
    return joblib.load(model_path), test, PRESSURE_FEATURES


def load_no_pressure_source() -> tuple[Any, pd.DataFrame, list[str]]:
    dataset = (
        ROOT
        / "15_07_dia_20_mais_completo"
        / "dados"
        / "dataset_dia_20_mais_com_ambiente.csv"
    )
    model_path = (
        ROOT
        / "15_07_dia_20_mais_completo"
        / "1_investigacao_hardware_banco"
        / "modelagem"
        / "modelos"
        / "01_baseline_dia_20_mais_mq_ambiente_extra_trees.joblib"
    )
    raw = pd.read_csv(dataset)
    identifiers = [c for c in ["Coleta", "Dia", "Vaso", "Tempo"] if c in raw]
    df = raw[[*identifiers, *NO_PRESSURE_FEATURES, TARGET]].copy()
    for column in [*NO_PRESSURE_FEATURES, TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[GROUP, *NO_PRESSURE_FEATURES, TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    _, test = split_by_collection_inside_class(df.reset_index(drop=True))
    return joblib.load(model_path), test, NO_PRESSURE_FEATURES


def balanced_collection_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Evita que uma coleta longa domine a explicacao global."""
    if len(df) <= limit:
        return df.copy()
    keyed = list(df.groupby([GROUP, TARGET], sort=True, dropna=False))
    quota = max(1, limit // len(keyed))
    parts = [
        block.sample(n=min(quota, len(block)), random_state=RANDOM_STATE)
        for _, block in keyed
    ]
    sampled = pd.concat(parts)
    if len(sampled) < limit:
        remaining = df.drop(index=sampled.index)
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    n=min(limit - len(sampled), len(remaining)),
                    random_state=RANDOM_STATE,
                ),
            ]
        )
    return sampled.sample(frac=1, random_state=RANDOM_STATE).head(limit)


def class_explanation(explanation: shap.Explanation, class_index: int) -> shap.Explanation:
    """Normaliza a saida SHAP binaria para uma unica classe."""
    if explanation.values.ndim == 2:
        return explanation
    if explanation.values.ndim != 3:
        raise ValueError(f"Formato SHAP inesperado: {explanation.values.shape}")
    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 2:
        base_values = base_values[:, class_index]
    elif base_values.ndim == 1 and len(base_values) > 1:
        base_values = np.repeat(base_values[class_index], explanation.values.shape[0])
    return shap.Explanation(
        values=explanation.values[:, :, class_index],
        base_values=base_values,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def canonical_feature(feature: str) -> str:
    if feature.startswith("MQ"):
        return feature.split("_")[0]
    return {
        "Soil_indice_0_1": "Umidade do solo",
        "Soil": "Umidade do solo",
        "Temp_C": "Temperatura",
        "Temp.": "Temperatura",
        "Pres_kPa": "Pressão",
        "Pres.": "Pressão",
    }.get(feature, feature)


def select_local_examples(test: pd.DataFrame) -> pd.DataFrame:
    """Seleciona um caso mediano de cada celula nao vazia da matriz."""
    descriptions = {
        (0, 0): "Com nematoide classificado corretamente",
        (0, 1): "Com nematoide classificado incorretamente como sem nematoide",
        (1, 0): "Sem nematoide classificado incorretamente como com nematoide",
        (1, 1): "Sem nematoide classificado corretamente",
    }
    rows = []
    for actual in [0, 1]:
        for predicted in [0, 1]:
            block = test.loc[
                (test[TARGET] == actual) & (test["Predicao"] == predicted)
            ].copy()
            if block.empty:
                continue
            median_probability = block["Prob_com_nematoide"].median()
            chosen_index = (
                block["Prob_com_nematoide"] - median_probability
            ).abs().idxmin()
            row = test.loc[[chosen_index]].copy()
            row["Tipo_resultado_id"] = (
                f"real_{actual}_{'correto' if actual == predicted else 'erro'}"
                f"_previsto_{predicted}"
            )
            row["Tipo_resultado"] = descriptions[(actual, predicted)]
            rows.append(row)
    return pd.concat(rows, ignore_index=False)


def save_global_plots(
    explanation: shap.Explanation,
    importance: pd.DataFrame,
    scenario_dir: Path,
    title: str,
) -> None:
    plot_df = importance.sort_values("media_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(plot_df["feature"], plot_df["media_abs_shap"], color="#2d6cdf")
    ax.set_xlabel("Média de |SHAP| para a classe com nematoide")
    ax.set_title(f"Importância global SHAP - {title}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(scenario_dir / "01_importancia_global_shap.png", dpi=220)
    plt.close(fig)

    shap.summary_plot(explanation, show=False, max_display=len(plot_df))
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.title(f"Direção e magnitude dos efeitos SHAP - {title}")
    plt.xlabel("SHAP para a classe com nematoide")
    plt.tight_layout()
    fig.savefig(scenario_dir / "02_resumo_shap_beeswarm.png", dpi=220)
    plt.close(fig)


def explain_scenario(
    scenario: dict[str, Any], model: Any, test: pd.DataFrame, features: list[str]
) -> dict[str, Any]:
    scenario_dir = RESULTS / scenario["id"]
    local_dir = scenario_dir / "explicacoes_locais"
    local_dir.mkdir(parents=True, exist_ok=True)

    if scenario["filter"] == "dry":
        test = test.loc[test["Soil_indice_0_1"] <= 0.4].copy()
    elif scenario["filter"] == "wet":
        test = test.loc[test["Soil_indice_0_1"] > 0.4].copy()
    else:
        test = test.copy()

    test["Predicao"] = model.predict(test[features]).astype(int)
    probabilities = model.predict_proba(test[features])
    class_zero_column = int(np.where(model.classes_ == 0)[0][0])
    test["Prob_com_nematoide"] = probabilities[:, class_zero_column]
    matrix = confusion_matrix(test[TARGET], test["Predicao"], labels=[0, 1])
    if matrix.tolist() != scenario["expected_matrix"]:
        raise RuntimeError(
            f"{scenario['id']}: matriz {matrix.tolist()} difere da esperada "
            f"{scenario['expected_matrix']}"
        )

    sample = balanced_collection_sample(test, MAX_SHAP_ROWS)
    explainer = shap.TreeExplainer(model)
    global_explanation = class_explanation(explainer(sample[features]), class_zero_column)
    mean_abs = np.abs(global_explanation.values).mean(axis=0)
    signed_mean = global_explanation.values.mean(axis=0)
    correlations = [
        pd.Series(sample[feature].to_numpy()).corr(
            pd.Series(global_explanation.values[:, index]), method="spearman"
        )
        for index, feature in enumerate(features)
    ]
    importance = pd.DataFrame(
        {
            "feature": features,
            "feature_canonica": [canonical_feature(f) for f in features],
            "media_abs_shap": mean_abs,
            "percentual_importancia": 100 * mean_abs / mean_abs.sum(),
            "media_shap_assinada": signed_mean,
            "correlacao_spearman_valor_shap": correlations,
            "amostra_shap": len(sample),
        }
    ).sort_values("media_abs_shap", ascending=False)
    importance.to_csv(
        scenario_dir / "importancia_global_shap.csv", index=False, encoding="utf-8-sig"
    )
    save_global_plots(global_explanation, importance, scenario_dir, scenario["title"])

    examples = select_local_examples(test)
    local_explanation = class_explanation(explainer(examples[features]), class_zero_column)
    local_rows: list[dict[str, Any]] = []
    for position, (index, row) in enumerate(examples.iterrows()):
        result_type = row["Tipo_resultado"]
        safe_type = row["Tipo_resultado_id"]
        one = shap.Explanation(
            values=local_explanation.values[position],
            base_values=local_explanation.base_values[position],
            data=local_explanation.data[position],
            feature_names=features,
        )
        shap.plots.waterfall(one, max_display=len(features), show=False)
        fig = plt.gcf()
        fig.set_size_inches(9, 5.8)
        plt.title(f"Explicação local - {scenario['title']}\n{result_type}")
        plt.tight_layout()
        fig.savefig(local_dir / f"{safe_type}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        reconstructed = float(one.base_values + one.values.sum())
        for feature_position, feature in enumerate(features):
            local_rows.append(
                {
                    "indice_original": index,
                    "coleta": row.get(GROUP, ""),
                    "tipo_resultado": result_type,
                    "tipo_resultado_id": row["Tipo_resultado_id"],
                    "classe_real": int(row[TARGET]),
                    "classe_prevista": int(row["Predicao"]),
                    "probabilidade_com_nematoide": float(row["Prob_com_nematoide"]),
                    "valor_base_shap": float(one.base_values),
                    "saida_reconstruida_shap": reconstructed,
                    "erro_aditividade": abs(
                        reconstructed - float(row["Prob_com_nematoide"])
                    ),
                    "feature": feature,
                    "valor_feature": float(row[feature]),
                    "valor_shap": float(one.values[feature_position]),
                }
            )
    pd.DataFrame(local_rows).to_csv(
        scenario_dir / "explicacoes_locais.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "cenario": scenario["id"],
        "titulo": scenario["title"],
        "modelo_real": type(model).__name__,
        "linhas_teste": int(len(test)),
        "coletas_teste": int(test[GROUP].nunique()),
        "matriz_confusao": matrix.astype(int).tolist(),
        "amostra_shap_balanceada_por_coleta_classe": int(len(sample)),
        "classe_explicada": "0 = com nematoide",
        "top_3": importance.head(3)[
            ["feature", "percentual_importancia"]
        ].to_dict(orient="records"),
        "maior_erro_aditividade_local": float(
            pd.DataFrame(local_rows)["erro_aditividade"].max()
        ),
    }


def save_comparison(summaries: list[dict[str, Any]]) -> None:
    frames = []
    for scenario in SCENARIOS:
        path = RESULTS / scenario["id"] / "importancia_global_shap.csv"
        frame = pd.read_csv(path)
        frame["cenario"] = scenario["id"]
        frames.append(frame)
    all_importance = pd.concat(frames, ignore_index=True)
    comparison = (
        all_importance.pivot_table(
            index="feature_canonica",
            columns="cenario",
            values="percentual_importancia",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    comparison.to_csv(
        RESULTS / "comparacao_importancia_shap_percentual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_df = comparison.set_index("feature_canonica")
    plot_df = plot_df.loc[plot_df.mean(axis=1).sort_values().index]
    ax = plot_df.plot.barh(figsize=(10, 7), width=0.8)
    ax.set_xlabel("Participação na importância SHAP do cenário (%)")
    ax.set_ylabel("")
    ax.set_title("Comparação da importância global SHAP entre cenários")
    ax.legend([s["title"] for s in SCENARIOS], fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS / "comparacao_importancia_shap_percentual.png", dpi=220)
    plt.close()

    with (RESULTS / "resumo_xai.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "metodo": "SHAP TreeExplainer",
                "classe_explicada": "0 = com nematoide",
                "criterio_amostragem": (
                    "Ate 2000 linhas de teste por cenario, balanceadas por coleta e classe."
                ),
                "cenarios": summaries,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sources = {
        "pressure": load_pressure_source(),
        "no_pressure": load_no_pressure_source(),
    }
    summaries = []
    for scenario in SCENARIOS:
        print(f"Aplicando XAI: {scenario['title']}")
        model, test, features = sources[scenario["source"]]
        summaries.append(explain_scenario(scenario, model, test, features))
    save_comparison(summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
