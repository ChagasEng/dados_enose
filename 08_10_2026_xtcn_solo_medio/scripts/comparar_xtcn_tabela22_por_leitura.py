"""Converte janelas X-TCN em leituras no mesmo teste da Tabela 22."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from avaliar_xtcn_original_por_umidade import DATASET, OUTPUT, ROOT, STRIDE, WINDOW


REFERENCE = ROOT / "06_07_melhor_modelo/dados/dataset_melhor_modelo_sensores_corrigidos.csv"
WINDOW_PREDICTIONS = OUTPUT / "predicoes_por_janela.csv"
TARGET_ACCURACY = 0.9614785526000479


def windows_to_readings(test: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if set(test.Coleta) != set(windows.Coleta):
        raise AssertionError("As coletas de teste nao correspondem")
    rows = []
    for coleta, block in test.sort_values(["Coleta", "Tempo"]).groupby("Coleta", sort=False):
        block = block.reset_index(drop=False)
        probs = windows.loc[windows.Coleta == coleta, "probabilidade_saudavel"].to_numpy()
        starts = list(range(0, max(1, len(block) - WINDOW + 1), STRIDE))
        if len(block) >= WINDOW and starts[-1] != len(block) - WINDOW:
            starts.append(len(block) - WINDOW)
        if len(probs) != len(starts):
            raise AssertionError(f"Numero de janelas diferente em {coleta}")
        score_sum = np.zeros(len(block), dtype=float)
        coverage = np.zeros(len(block), dtype=int)
        for start, probability in zip(starts, probs):
            end = min(len(block), start + WINDOW)
            score_sum[start:end] += probability
            coverage[start:end] += 1
        if (coverage == 0).any():
            raise AssertionError(f"Leitura sem janela em {coleta}")
        block["probabilidade_saudavel_xtcn"] = score_sum / coverage
        block["classe_prevista_xtcn"] = (block.probabilidade_saudavel_xtcn >= .5).astype(int)
        block["janelas_cobrindo_leitura"] = coverage
        rows.append(block)
    readings = pd.concat(rows, ignore_index=True)
    if len(readings) != len(test):
        raise AssertionError("Nem todas as leituras foram avaliadas")
    if "Soil_indice_0_1" in readings:
        readings["faixa"] = np.where(readings["Soil_indice_0_1"] <= .4, "seco", "umido")
    return readings


def main():
    reference = pd.read_csv(REFERENCE)
    test = reference[reference.Conjunto == "Teste"].copy()
    windows = pd.read_csv(WINDOW_PREDICTIONS)
    readings = windows_to_readings(test, windows)
    result = []
    for faixa in ("seco", "umido", "total"):
        subset = readings if faixa == "total" else readings[readings.faixa == faixa]
        result.append({"faixa": faixa, "n_leituras": len(subset),
                       "n_coletas": subset.Coleta.nunique(),
                       "accuracy": float(accuracy_score(subset.Classe, subset.classe_prevista_xtcn)),
                       "balanced_accuracy": float(balanced_accuracy_score(subset.Classe, subset.classe_prevista_xtcn)),
                       "f1_macro": float(f1_score(subset.Classe, subset.classe_prevista_xtcn, average="macro"))})
    result = pd.DataFrame(result)
    if int(result.loc[result.faixa == "seco", "n_leituras"].iloc[0]) != 16692:
        raise AssertionError("O subconjunto seco nao corresponde as 16692 leituras da referencia")
    result.to_csv(OUTPUT / "comparacao_tabela22_mesmo_teste_por_leitura.csv", index=False)
    readings[["Coleta", "Dia", "Tempo", "Classe", "Soil_indice_0_1", "faixa",
              "probabilidade_saudavel_xtcn", "classe_prevista_xtcn",
              "janelas_cobrindo_leitura"]].to_csv(OUTPUT / "predicoes_xtcn_por_leitura.csv", index=False)
    (OUTPUT / "metodo_comparacao_tabela22.json").write_text(json.dumps({
        "referencia_dataset": str(REFERENCE.relative_to(ROOT)),
        "modelo_xtcn": "MQ + Soil bruto, 60 epocas; previsoes por janela de predicoes_por_janela.csv",
        "mesmas_coletas_teste": True, "leituras_secas_referencia": 16692,
        "agregacao": "media das probabilidades de todas as janelas que cobrem cada leitura",
        "limiar_classe": 0.5, "limiar_solo_seco": "Soil_indice_0_1 <= 0.4",
        "accuracy_referencia": TARGET_ACCURACY,
        "observacao": "comparacao pontual no mesmo teste; validacao em 4 folds registrada separadamente",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
