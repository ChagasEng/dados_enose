"""Sensibilidade temporal: deixa uma das cinco datas inteiramente fora.

Reajusta solo minimo-maximo e mediana de pressao no treino de cada data.
Os mesmos vasos podem ocorrer em datas de treino e teste; esta analise
complementa, mas nao substitui, o LOSO por vaso.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/validar_data_fora.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from validar_preprocessamento_por_vaso import ROOT, SOURCE, MQ, ENV, prepare_fold


OUT = Path(__file__).resolve().parent / "validacao_data_fora"
FEATURES = {
    "nove_entradas": (*MQ, *ENV),
    "somente_ambientais": ENV,
    "somente_mq": MQ,
}
SCENARIOS = ("faixa_estrita_treino", "sem_faixa_estrita")


def summary(frame: pd.DataFrame, scenario: str, config: str, unit: str,
            day: int | str, label: int | str) -> dict:
    truth, predicted = frame.Classe.to_numpy(), frame.predicao.to_numpy()
    result = {"cenario": scenario, "configuracao": config, "unidade": unit,
              "dia_teste": day, "classe": label, "n": len(frame),
              "acertos": int(np.sum(truth == predicted)),
              "acuracia": float(accuracy_score(truth, predicted))}
    if len(np.unique(truth)) == 2:
        result["acuracia_balanceada"] = float(balanced_accuracy_score(truth, predicted))
        result["f1_macro"] = float(f1_score(truth, predicted, average="macro"))
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(SOURCE)
    if len(data) != 66081 or data.Coleta.nunique() != 36:
        raise AssertionError("Base de origem inesperada")
    data["Dia"] = data.Dia.astype(int)
    days = sorted(data.Dia.unique())
    if days != [11, 12, 13, 18, 19]:
        raise AssertionError(f"Datas inesperadas: {days}")

    rows, parameters, all_collections = [], [], []
    for scenario in SCENARIOS:
        predictions = {config: [] for config in FEATURES}
        for day in days:
            fit_raw = data[data.Dia.ne(day)]
            test_raw = data[data.Dia.eq(day)]
            if set(fit_raw.Coleta) & set(test_raw.Coleta):
                raise AssertionError("Coleta presente em treino e teste")
            train, test, info = prepare_fold(fit_raw, test_raw, scenario)
            if train.Classe.nunique() != 2 or test.empty:
                raise AssertionError("Treino sem duas classes ou teste vazio")
            parameters.append({"cenario": scenario, "dia_teste": day,
                               "leituras_treino_antes_filtro": len(fit_raw),
                               "leituras_teste_antes_filtro": len(test_raw), **info})
            for config, features in FEATURES.items():
                model = ExtraTreesClassifier(
                    n_estimators=900, max_features="sqrt", min_samples_leaf=10,
                    bootstrap=False, random_state=42, n_jobs=6,
                ).fit(train.loc[:, features], train.Classe)
                part = test[["Coleta", "Dia", "Vaso", "Classe", "Tempo"]].copy()
                part["probabilidade_classe_1"] = model.predict_proba(test.loc[:, features])[:, 1]
                part["predicao"] = (part.probabilidade_classe_1 >= 0.5).astype(int)
                predictions[config].append(part)
                print(f"{scenario}: dia {day}, {config}, teste {len(test)} leituras", flush=True)

        for config, parts in predictions.items():
            reading = pd.concat(parts, ignore_index=True)
            collection = reading.groupby(["Coleta", "Dia", "Vaso", "Classe"], as_index=False).agg(
                probabilidade_classe_1=("probabilidade_classe_1", "mean"),
                n_leituras=("Classe", "size"),
            )
            collection["predicao"] = (collection.probabilidade_classe_1 >= 0.5).astype(int)
            if reading.Coleta.nunique() != 36 or collection.Coleta.nunique() != 36:
                raise AssertionError("Alguma coleta ficou sem previsao")
            collection.insert(0, "configuracao", config)
            collection.insert(0, "cenario", scenario)
            all_collections.append(collection)
            for unit, frame in (("leitura", reading), ("coleta", collection)):
                rows.append(summary(frame, scenario, config, unit, "todos", "ambas"))
                for day in days:
                    day_frame = frame[frame.Dia.eq(day)]
                    rows.append(summary(day_frame, scenario, config, unit, day, "ambas"))
                    for label in (0, 1):
                        subset = day_frame[day_frame.Classe.eq(label)]
                        if not subset.empty:
                            rows.append(summary(subset, scenario, config, unit, day, label))
            reading.to_csv(OUT / f"predicoes_{scenario}_{config}.csv.gz", index=False,
                           compression="gzip")

    pd.DataFrame(rows).to_csv(OUT / "metricas_por_data.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "parametros_por_data.csv", index=False)
    pd.concat(all_collections, ignore_index=True).to_csv(OUT / "previsoes_por_coleta.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "validacao": "Leave-one-date-out, datas 11, 12, 13, 18 e 19 de maio de 2026",
        "fonte": str(SOURCE.relative_to(ROOT)),
        "preprocessamento": "Minimo-maximo de solo e mediana de pressao estimados apenas nas quatro datas de treino",
        "faixa_estrita": "Mediana de pressao do treino +/- 0,50 kPa; aplicada a treino e teste",
        "sem_faixa_estrita": "Mantem filtro fixo de saltos de pressao, sem banda adicional",
        "modelo": "ExtraTreesClassifier 900 arvores, max_features=sqrt, min_samples_leaf=10, bootstrap=False, seed=42",
        "limite": "Os mesmos vasos podem participar do treino em outras datas; a data 13 tem somente coletas controle",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).query("dia_teste == 'todos' and classe == 'ambas'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
