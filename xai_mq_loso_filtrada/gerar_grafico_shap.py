"""SHAP de Extra Trees com apenas MQ na base filtrada e LOSO por vaso."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analise_combinatoria_sensores"))
from rodar_combinacoes import DATASET, model
from validacao_por_vaso import vessel_id


OUT = Path(__file__).resolve().parent
EXCLUDED = ROOT / "analise_combinatoria_sensores" / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
CONFIGS = {
    "cinco": {
        "features": ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"],
        "expected_correct": 53015,
        "stem": "importancia_shap_mq_loso",
        "sample": "amostra_shap.csv",
        "method": "metodo.json",
    },
    "seis": {
        "features": ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"],
        "expected_correct": 53388,
        "stem": "importancia_shap_mq_com_mq7_loso",
        "sample": "amostra_shap_com_mq7.csv",
        "method": "metodo_com_mq7.json",
    },
}
SAMPLE_PER_VESSEL = 50
SEED = 42


def shap_class_zero(explainer: shap.TreeExplainer, values: pd.DataFrame) -> np.ndarray:
    result = np.asarray(explainer.shap_values(values, check_additivity=False))
    if result.shape == (len(values), values.shape[1], 2):
        return result[:, :, 0]
    if result.shape == (2, len(values), values.shape[1]):
        return result[0]
    raise ValueError(f"Formato SHAP inesperado: {result.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, default="seis")
    config = CONFIGS[parser.parse_args().config]
    features = config["features"]
    excluded = set(pd.read_csv(EXCLUDED).Coleta)
    data = pd.read_csv(DATASET)
    data = data.loc[~data.Coleta.isin(excluded)].copy()
    data["Vaso_id"] = vessel_id(data)
    if (len(data), data.Coleta.nunique(), data.Vaso_id.nunique()) != (55946, 31, 10):
        raise AssertionError("Base diferente de 55.946 leituras, 31 coletas e 10 vasos")

    all_shap = []
    all_samples = []
    correct = 0
    for fold, (train_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id), 1
    ):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        fitted = model().fit(train[features], train.Classe)
        predicted = fitted.predict(test[features])
        correct += int((predicted == test.Classe.to_numpy()).sum())

        sample = test.sample(n=min(SAMPLE_PER_VESSEL, len(test)), random_state=SEED + fold)
        explainer = shap.TreeExplainer(fitted)
        values = shap_class_zero(explainer, sample[features])
        if values.shape != (len(sample), len(features)):
            raise AssertionError("Dimensoes SHAP incorretas")
        reconstructed = float(explainer.expected_value[0]) + values.sum(axis=1)
        probability_zero = fitted.predict_proba(sample[features])[:, 0]
        if not np.allclose(reconstructed, probability_zero, atol=1e-8):
            raise AssertionError("Valores SHAP nao recompõem a probabilidade da classe 0")
        all_shap.append(values)
        all_samples.append(sample[["Coleta", "Vaso_id", "Classe"]].assign(fold=fold))
        print(f"fold {fold}/10: {len(sample)} leituras explicadas", flush=True)

    if correct != config["expected_correct"]:
        raise AssertionError(f"O modelo MQ nao reproduziu os {config['expected_correct']} acertos: {correct}")
    shap_values = np.vstack(all_shap)
    samples = pd.concat(all_samples, ignore_index=True)
    if len(samples) != 500 or samples.Vaso_id.nunique() != 10:
        raise AssertionError("A amostra SHAP nao contem 50 leituras por vaso")

    mean_abs = np.abs(shap_values).mean(axis=0)
    percent = 100 * mean_abs / mean_abs.sum()
    ranking = pd.DataFrame({
        "sensor": [name.replace("MQ", "MQ-") for name in features],
        "media_abs_shap": mean_abs,
        "importancia_percentual_entre_mq": percent,
    }).sort_values("importancia_percentual_entre_mq", ascending=False)
    OUT.mkdir(exist_ok=True)
    ranking.to_csv(OUT / f"{config['stem']}.csv", index=False)
    samples.to_csv(OUT / config["sample"], index=False)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    plot = ranking.iloc[::-1]
    bars = ax.barh(plot.sensor, plot.importancia_percentual_entre_mq, color="#246b9e")
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=11)
    ax.set_xlim(0, max(plot.importancia_percentual_entre_mq) * 1.18)
    ax.set_xlabel("Importância SHAP relativa entre os sensores MQ (%)")
    ax.set_title("Importância dos sensores MQ — Extra Trees, LOSO por vaso")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / f"{config['stem']}.png", dpi=240)
    fig.savefig(OUT / f"{config['stem']}.pdf")
    plt.close(fig)

    method = {
        "base": "55.946 leituras, 31 coletas, 10 vasos, apos exclusao de cinco coletas",
        "modelo": "ExtraTreesClassifier; 900 arvores, max_features=sqrt, min_samples_leaf=10, seed=42",
        "entradas": features,
        "variaveis_ambientais_no_modelo": False,
        "validacao": "LOSO por vaso; modelo refeito em cada um dos dez folds",
        "amostra": "50 leituras de teste por vaso, total 500, sorteio deterministico",
        "classe_explicada": "0 = inoculado",
        "importancia": "Media do valor SHAP absoluto, normalizada para 100% entre os MQ do modelo",
        "acertos_modelo_em_todas_as_leituras": correct,
        "limite": "Selecao de entradas e exclusoes exploratorias nos mesmos vasos; SHAP nao demonstra causalidade biologica",
    }
    (OUT / config["method"]).write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
