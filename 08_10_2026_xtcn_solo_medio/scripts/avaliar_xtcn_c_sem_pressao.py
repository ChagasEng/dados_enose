"""X-TCN no cenario c da tabela: sem pressao, sem estratificar umidade."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from avaliar_xtcn_mesmo_protocolo_tabela22 import metrics
from avaliar_xtcn_original_por_umidade import ROOT, train_and_predict
from comparar_rf_xtcn_medias_ambiente import split_groups
from comparar_xtcn_tabela22_por_leitura import windows_to_readings


DATASET = ROOT / "15_07_dia_20_mais_completo/dados/dataset_dia_20_mais_com_ambiente.csv"
OUTPUT = ROOT / "08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features"
FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138", "Soil", "Temp.", "Pres."]
REFERENCE = {"accuracy": 0.8443280230941944,
             "balanced_accuracy": 0.8354430379746836,
             "f1_macro": 0.8371700922964844}


def main():
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Tempo", "Classe", *FEATURES]).copy()
    df.Classe = df.Classe.astype(int)
    train, test = split_groups(df)
    if (len(train), len(test), train.Coleta.nunique(), test.Coleta.nunique()) != (17842, 9353, 11, 6):
        raise AssertionError("O split nao coincide com a referencia do cenario c")
    windows = train_and_predict(train, test, FEATURES)
    readings = windows_to_readings(test, windows)
    by_collection = readings.groupby("Coleta", as_index=False).agg(
        Classe=("Classe", "first"), probabilidade_saudavel_xtcn=("probabilidade_saudavel_xtcn", "mean"))
    by_collection["classe_prevista_xtcn"] = (by_collection.probabilidade_saudavel_xtcn >= .5).astype(int)
    result = pd.DataFrame([metrics(windows, "janela"), metrics(readings, "leitura")])
    result.insert(0, "cenario", "c_sem_pressao_sem_estratificacao")
    collection_metrics = {
        "n_coletas": len(by_collection),
        "accuracy": float(accuracy_score(by_collection.Classe, by_collection.classe_prevista_xtcn)),
        "balanced_accuracy": float(balanced_accuracy_score(by_collection.Classe, by_collection.classe_prevista_xtcn)),
        "f1_macro": float(f1_score(by_collection.Classe, by_collection.classe_prevista_xtcn, average="macro")),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "metricas_xtcn_c_sem_pressao.csv", index=False)
    windows.to_csv(OUTPUT / "predicoes_janelas_c_sem_pressao.csv", index=False)
    readings[["Coleta", "Dia", "Tempo", "Classe", "probabilidade_saudavel_xtcn",
              "classe_prevista_xtcn", "janelas_cobrindo_leitura"]].to_csv(
                  OUTPUT / "predicoes_leituras_c_sem_pressao.csv", index=False)
    by_collection.to_csv(OUTPUT / "predicoes_coletas_c_sem_pressao.csv", index=False)
    (OUTPUT / "metodo_c_sem_pressao.json").write_text(json.dumps({
        "dataset": str(DATASET.relative_to(ROOT)),
        "split": "70/30 por Coleta dentro de cada Classe, seed 42; 17842/9353 leituras; 11/6 coletas",
        "features": FEATURES, "epocas_xtcn": 60, "janela": 128, "stride": 64,
        "por_leitura": "media das probabilidades de janelas que cobrem cada leitura",
        "sem_estratificacao": True, "referencia": REFERENCE,
        "metricas_por_coleta": collection_metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print("Por coleta:", collection_metrics)


if __name__ == "__main__":
    main()
