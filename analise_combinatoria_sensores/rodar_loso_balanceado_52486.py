"""Refaz a tabela LOSO em 52.486 leituras, 26.243 por classe.

Mantem as 31 coletas e dez vasos da analise filtrada. Seleciona leituras
uniformemente ao longo do tempo dentro de cada coleta, com cotas proporcionais
ao tamanho da coleta em cada classe. Nenhum resultado anterior e reutilizado.

Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/rodar_loso_balanceado_52486.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from avaliar_base_completa_parecer import bootstrap_por_vaso
from rodar_combinacoes import DATASET, model as extra_trees_model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "loso_balanceado_52486_solo_binario"
EXCLUDED = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
MQ = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
ENV = ("Condicao_solo_binaria", "Temp_C", "Pres_kPa")
SCENARIOS = {"mq": MQ, "mq_ambiente": [*MQ, *ENV], "ambiente": list(ENV)}
TARGET_PER_CLASS = 26_243
MEMBERS = ("et", "rf", "tcn", "mlp", "knn")


def quotas(frame: pd.DataFrame, target: int) -> pd.DataFrame:
    counts = frame.groupby(["Classe", "Coleta"], as_index=False).size().rename(columns={"size": "original"})
    counts["ideal"] = counts.original * target / counts.groupby("Classe").original.transform("sum")
    counts["selected"] = np.floor(counts.ideal).astype(int)
    counts["remainder"] = counts.ideal - counts.selected
    for label, group in counts.groupby("Classe"):
        missing = target - int(group.selected.sum())
        order = group.sort_values(["remainder", "Coleta"], ascending=[False, True]).index[:missing]
        counts.loc[order, "selected"] += 1
    assert counts.groupby("Classe").selected.sum().to_dict() == {0: target, 1: target}
    assert counts.selected.ge(1).all()
    return counts


def balanced_data() -> pd.DataFrame:
    raw = pd.read_csv(DATASET)
    excluded = set(pd.read_csv(EXCLUDED).Coleta)
    raw = raw.loc[~raw.Coleta.isin(excluded)].copy()
    raw["Vaso_id"] = vessel_id(raw)
    assert (len(raw), raw.Coleta.nunique(), raw.Vaso_id.nunique()) == (55_946, 31, 10)
    allocation = quotas(raw, TARGET_PER_CLASS)
    parts = []
    for row in allocation.itertuples():
        block = raw.loc[raw.Coleta.eq(row.Coleta)].sort_values("Tempo")
        n, selected = len(block), int(row.selected)
        positions = np.floor((np.arange(selected) + .5) * n / selected).astype(int)
        assert len(np.unique(positions)) == selected
        parts.append(block.iloc[positions])
    data = pd.concat(parts).sort_index().reset_index(drop=True)
    data.insert(0, "linha_balanceada", np.arange(len(data)))
    data["Condicao_solo_binaria"] = data.Soil_indice_0_1.le(.4).astype(np.int8)
    assert data.Classe.value_counts().to_dict() == {0: TARGET_PER_CLASS, 1: TARGET_PER_CLASS}
    assert (len(data), data.Coleta.nunique(), data.Vaso_id.nunique()) == (52_486, 31, 10)
    assert not data.duplicated(["Coleta", "Tempo"]).any()
    assert data.groupby("Coleta")[["Classe", "Vaso_id"]].nunique().eq(1).all().all()
    OUT.mkdir(exist_ok=True)
    allocation[["Classe", "Coleta", "original", "selected"]].to_csv(OUT / "cotas_por_coleta.csv", index=False)
    data.to_csv(OUT / "base_balanceada.csv", index=False)
    return data


def probabilities(name: str, train: pd.DataFrame, test: pd.DataFrame,
                  features: list[str]) -> np.ndarray:
    if name == "et":
        fitted = extra_trees_model().set_params(n_jobs=6).fit(train[features], train.Classe)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "rf":
        fitted = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=6, class_weight="balanced"
        ).fit(train[features], train.Classe)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "mlp":
        fitted = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=.001,
                          batch_size=2048, early_stopping=True, validation_fraction=.15,
                          n_iter_no_change=12, max_iter=250, random_state=42),
        ).fit(train[features], train.Classe)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "knn":
        fitted = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=2)
        ).fit(train[features], train.Classe)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "tcn":
        from avaliar_rf_tcn_sem_quatro_coletas import predict_tcn
        return predict_tcn(train, test, features)[0]
    raise ValueError(name)


def summarize(pred: pd.DataFrame, name: str, scenario: str) -> dict:
    y = pred.Classe
    yhat = pred.predicao
    return {
        "modelo": name, "cenario": scenario, "n": len(pred),
        "acertos": int(y.eq(yhat).sum()),
        "acuracia": float(accuracy_score(y, yhat)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, yhat)),
        "recall_classe_0": float(recall_score(y, yhat, pos_label=0)),
        "f1_macro": float(f1_score(y, yhat, average="macro")),
    }


def evaluate(data: pd.DataFrame, name: str, scenario: str,
             folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    path = OUT / f"predicoes_{scenario}_{name}.csv"
    if path.exists():
        result = pd.read_csv(path)
        assert len(result) == len(data)
        assert result.linha_balanceada.equals(data.linha_balanceada)
        assert result.Classe.equals(data.Classe)
        return result
    features = SCENARIOS[scenario]
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(folds, 1):
        train, test = data.iloc[fit_idx], data.iloc[test_idx]
        assert not (set(train.Vaso_id) & set(test.Vaso_id))
        p1 = probabilities(name, train, test, features)
        part = test[["linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo"]].copy()
        part["fold"] = fold
        part["probabilidade_classe_1"] = p1
        part["predicao"] = (p1 >= .5).astype(int)
        parts.append(part)
        print(f"{scenario} {name}: fold {fold}/10", flush=True)
    result = pd.concat(parts).sort_values("linha_balanceada").reset_index(drop=True)
    assert result.linha_balanceada.equals(data.linha_balanceada)
    result.to_csv(path, index=False)
    return result


def finish_main(pred: pd.DataFrame) -> None:
    collections = (pred.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False)
                   .probabilidade_classe_1.mean())
    collections["predicao"] = (collections.probabilidade_classe_1 >= .5).astype(int)
    vessels = (collections.groupby(["Vaso_id", "Classe"], as_index=False)
               .probabilidade_classe_1.mean())
    vessels["predicao"] = (vessels.probabilidade_classe_1 >= .5).astype(int)
    collections.to_csv(OUT / "coletas_et_mq_ambiente.csv", index=False)
    vessels.to_csv(OUT / "vasos_et_mq_ambiente.csv", index=False)
    pd.DataFrame([
        summarize(pred, "et", "leitura"),
        summarize(collections, "et", "coleta"),
        summarize(vessels, "et", "vaso"),
    ]).to_csv(OUT / "metricas_tres_unidades_et.csv", index=False)
    k = int(vessels.Classe.eq(vessels.predicao).sum())
    pd.DataFrame([{
        "acertos": k, "n": 10,
        "limite_inferior_95": 0.0 if k == 0 else float(beta.ppf(.025, k, 11 - k)),
        "limite_superior_95": 1.0 if k == 10 else float(beta.ppf(.975, k + 1, 10 - k)),
    }]).to_csv(OUT / "clopper_pearson_et.csv", index=False)
    pd.DataFrame([bootstrap_por_vaso(pred, "leitura")]).to_csv(
        OUT / "bootstrap_por_vaso_et.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=(*MEMBERS, "all"), default=["all"])
    args = parser.parse_args()
    chosen = MEMBERS if "all" in args.models else tuple(args.models)
    data = balanced_data()
    folds = list(LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id))
    assert len(folds) == 10
    rows = []
    for scenario in SCENARIOS:
        names = chosen if scenario != "ambiente" else ("et",) if "et" in chosen else ()
        for name in names:
            result = evaluate(data, name, scenario, folds)
            rows.append(summarize(result, name, scenario))
            pd.DataFrame(rows).to_csv(OUT / "metricas_parciais.csv", index=False)
            if scenario == "mq_ambiente" and name == "et":
                finish_main(result)
    if set(MEMBERS).issubset(chosen):
        all_rows = []
        for scenario in ("mq", "mq_ambiente"):
            parts = [pd.read_csv(OUT / f"predicoes_{scenario}_{name}.csv") for name in MEMBERS]
            ensemble = parts[0][["linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo", "fold"]].copy()
            for part in parts[1:]:
                for key in ("linha_balanceada", "Coleta", "Vaso_id", "Classe", "Tempo", "fold"):
                    assert part[key].equals(ensemble[key]), f"Previsoes desalinhadas: {scenario}/{key}"
            ensemble["probabilidade_classe_1"] = np.mean(
                [part.probabilidade_classe_1.to_numpy() for part in parts], axis=0
            )
            ensemble["predicao"] = (ensemble.probabilidade_classe_1 >= .5).astype(int)
            ensemble.to_csv(OUT / f"predicoes_{scenario}_ensemble.csv", index=False)
            for name, part in zip(MEMBERS, parts):
                all_rows.append(summarize(part, name, scenario))
            all_rows.append(summarize(ensemble, "ensemble", scenario))
        environment = pd.read_csv(OUT / "predicoes_ambiente_et.csv")
        all_rows.append(summarize(environment, "et", "ambiente"))
        pd.DataFrame(all_rows).to_csv(OUT / "tabela_metricas.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "base": "31 coletas, 10 vasos, depois de excluir as mesmas cinco coletas da analise anterior",
        "balanceamento": "26.243 leituras por classe; cotas proporcionais por coleta, arredondamento por maior resto; selecao sistematica uniforme no tempo dentro de cada coleta",
        "solo": "Condicao_solo_binaria=1 para Soil_indice_0_1 <= 0.4 (seco), 0 para > 0.4 (molhado)",
        "validacao": "10 folds LeaveOneGroupOut por vaso, retreinando cada classificador em nove vasos",
        "entradas": SCENARIOS,
        "seed_modelos": 42,
        "observacao": "cinco MQ e coletas excluidas selecionados nos mesmos dez vasos; resultados exploratorios",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
