"""X-TCN nas mesmas entradas, split e leituras da Tabela 22."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from avaliar_xtcn_original_por_umidade import ROOT, train_and_predict
from comparar_xtcn_tabela22_por_leitura import REFERENCE, TARGET_ACCURACY, windows_to_readings


OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features"
FEATURES = [
    "MQ2_corrigido_env", "MQ3_corrigido_env", "MQ7_corrigido_env",
    "MQ8_corrigido_env", "MQ135_corrigido_env", "MQ138_corrigido_env",
    "Soil_indice_0_1", "Temp_C", "Pres_kPa",
]


def metrics(frame: pd.DataFrame, unit: str) -> dict[str, float | int | str]:
    y = frame.classe_real if unit == "janela" else frame.Classe
    p = frame.classe_prevista if unit == "janela" else frame.classe_prevista_xtcn
    return {"unidade": unit, "n": len(frame), "n_coletas": frame.Coleta.nunique(),
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "recall_com_nematoide": float(recall_score(y, p, pos_label=0)),
            "recall_sem_nematoide": float(recall_score(y, p, pos_label=1))}


def main():
    dataset = pd.read_csv(REFERENCE)
    train = dataset[dataset.Conjunto == "Treino"].copy()
    test = dataset[dataset.Conjunto == "Teste"].copy()
    if train[FEATURES].isna().any().any() or test[FEATURES].isna().any().any():
        raise ValueError("Ha valores ausentes nas entradas do modelo")
    if set(train.Coleta) & set(test.Coleta):
        raise AssertionError("Ha coletas nos dois conjuntos")
    windows = train_and_predict(train, test, FEATURES)
    windows["faixa"] = windows.Coleta.map(
        test.groupby("Coleta").Soil_indice_0_1.mean().map(lambda x: "seco" if x <= .4 else "umido"))
    readings = windows_to_readings(test, windows)
    rows = []
    for faixa in ("seco", "umido", "total"):
        for unit, frame in (("janela", windows), ("leitura", readings)):
            sub = frame if faixa == "total" else frame[frame.faixa == faixa]
            rows.append({"faixa": faixa, **metrics(sub, unit)})
    summary = pd.DataFrame(rows)
    if int(summary[(summary.faixa == "seco") & (summary.unidade == "leitura")].n.iloc[0]) != 16692:
        raise AssertionError("O teste seco nao coincide com a Tabela 22")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "metricas_xtcn_tabela22.csv", index=False)
    windows.to_csv(OUTPUT / "predicoes_por_janela.csv", index=False)
    readings[["Coleta", "Dia", "Tempo", "Classe", "Soil_indice_0_1", "faixa",
              "probabilidade_saudavel_xtcn", "classe_prevista_xtcn",
              "janelas_cobrindo_leitura"]].to_csv(OUTPUT / "predicoes_por_leitura.csv", index=False)
    (OUTPUT / "metodo.json").write_text(json.dumps({
        "dataset": str(REFERENCE.relative_to(ROOT)),
        "split": "coluna Conjunto preexistente: 41377 leituras no treino, 22115 no teste; 24/12 coletas",
        "features": FEATURES, "epocas_xtcn": 60, "janela": 128, "stride": 64,
        "por_leitura": "media das probabilidades das janelas que cobrem cada leitura",
        "faixa_seca": "Soil_indice_0_1 <= 0.4",
        "accuracy_tabela22": TARGET_ACCURACY,
        "comparacao": "mesmas entradas, coletas, leituras e faixa; classificador trocado por X-TCN",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
