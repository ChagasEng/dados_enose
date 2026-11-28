"""LOSO na base de 55.946 leituras, sem subamostragem, com pesos por classe.

Os pesos sao calculados exclusivamente nos nove vasos de treino de cada fold.
Uso: .venv/Scripts/python.exe analise_combinatoria_sensores/rodar_loso_55946_ponderado.py
"""

from __future__ import annotations

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
from sklearn.utils.class_weight import compute_sample_weight

from avaliar_base_completa_parecer import bootstrap_por_vaso
from avaliar_rf_tcn_sem_quatro_coletas import predict_tcn
from rodar_combinacoes import DATASET, model as extra_trees_model
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "loso_55946_ponderado_solo_binario"
EXCLUDED = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
MQ = ["MQ2", "MQ3", "MQ8", "MQ135", "MQ138"]
ENV = ["Condicao_solo_binaria", "Temp_C", "Pres_kPa"]
SCENARIOS = {"mq": MQ, "mq_ambiente": [*MQ, *ENV], "ambiente": ENV}
MEMBERS = ("et", "rf", "tcn", "mlp", "knn")


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATASET)
    excluded = set(pd.read_csv(EXCLUDED).Coleta)
    data = raw.loc[~raw.Coleta.isin(excluded)].copy().reset_index(drop=True)
    data.insert(0, "linha_filtrada", np.arange(len(data)))
    data["Vaso_id"] = vessel_id(data)
    data["Condicao_solo_binaria"] = data.Soil_indice_0_1.le(.4).astype(np.int8)
    assert len(data) == 55_946 and data.Coleta.nunique() == 31 and data.Vaso_id.nunique() == 10
    assert data.Classe.value_counts().to_dict() == {0: 26_342, 1: 29_604}
    assert data.groupby("Coleta")[["Classe", "Vaso_id"]].nunique().eq(1).all().all()
    assert not data[[*MQ, *ENV, "Classe", "Coleta", "Vaso_id"]].isna().any().any()
    OUT.mkdir(exist_ok=True)
    return data


def weighted_knn(train: pd.DataFrame, test: pd.DataFrame,
                 features: list[str], class_weights: dict[int, float]) -> np.ndarray:
    """Cinco vizinhos; voto ponderado por distancia e peso da classe do vizinho."""
    scaler = StandardScaler().fit(train[features])
    x_train = scaler.transform(train[features])
    x_test = scaler.transform(test[features])
    model = KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=2)
    model.fit(x_train, train.Classe)
    distances, indices = model.kneighbors(x_test)
    y_neighbors = train.Classe.to_numpy()[indices]
    zero = distances == 0
    distance_weights = np.divide(1, distances, out=np.zeros_like(distances), where=~zero)
    has_zero = zero.any(axis=1)
    distance_weights[has_zero] = zero[has_zero].astype(float)
    weights = distance_weights * np.where(y_neighbors == 0, class_weights[0], class_weights[1])
    return (weights * (y_neighbors == 1)).sum(axis=1) / weights.sum(axis=1)


def probabilities(name: str, train: pd.DataFrame, test: pd.DataFrame,
                  features: list[str]) -> np.ndarray:
    sample_weights = compute_sample_weight("balanced", train.Classe)
    counts = train.Classe.value_counts()
    class_weights = {k: len(train) / (2 * int(counts[k])) for k in (0, 1)}
    if name == "et":
        fitted = extra_trees_model().set_params(n_jobs=6)
        fitted.fit(train[features], train.Classe, sample_weight=sample_weights)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "rf":
        fitted = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=6)
        fitted.fit(train[features], train.Classe, sample_weight=sample_weights)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "mlp":
        fitted = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=.001,
                          batch_size=2048, early_stopping=True, validation_fraction=.15,
                          n_iter_no_change=12, max_iter=250, random_state=42),
        )
        fitted.fit(train[features], train.Classe, mlpclassifier__sample_weight=sample_weights)
        return fitted.predict_proba(test[features])[:, 1]
    if name == "knn":
        return weighted_knn(train, test, features, class_weights)
    if name == "tcn":
        # predict_tcn usa CrossEntropyLoss com pesos da frequencia das classes
        # nas janelas dos vasos de treino, nunca do vaso reservado.
        return predict_tcn(train, test, features)[0]
    raise ValueError(name)


def summarize(pred: pd.DataFrame, name: str, scenario: str) -> dict:
    y, yhat = pred.Classe, pred.predicao
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
        pred = pd.read_csv(path)
        assert len(pred) == len(data)
        assert pred.linha_filtrada.equals(data.linha_filtrada)
        assert pred.Classe.equals(data.Classe)
        return pred
    parts = []
    for fold, (fit_idx, test_idx) in enumerate(folds, 1):
        train, test = data.iloc[fit_idx], data.iloc[test_idx]
        assert not set(train.Vaso_id) & set(test.Vaso_id)
        p1 = probabilities(name, train, test, SCENARIOS[scenario])
        part = test[["linha_filtrada", "Coleta", "Vaso_id", "Classe", "Tempo"]].copy()
        part["fold"] = fold
        part["probabilidade_classe_1"] = p1
        part["predicao"] = (p1 >= .5).astype(int)
        parts.append(part)
        print(f"{scenario} {name}: fold {fold}/10", flush=True)
    pred = pd.concat(parts).sort_values("linha_filtrada").reset_index(drop=True)
    assert pred.linha_filtrada.equals(data.linha_filtrada)
    pred.to_csv(path, index=False)
    return pred


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
    data = load_data()
    folds = list(LeaveOneGroupOut().split(data, data.Classe, data.Vaso_id))
    assert len(folds) == 10
    rows = []
    for scenario in SCENARIOS:
        names = MEMBERS if scenario != "ambiente" else ("et",)
        for name in names:
            pred = evaluate(data, name, scenario, folds)
            rows.append(summarize(pred, name, scenario))
            pd.DataFrame(rows).to_csv(OUT / "metricas_parciais.csv", index=False)
            if scenario == "mq_ambiente" and name == "et":
                finish_main(pred)
    all_rows = []
    for scenario in ("mq", "mq_ambiente"):
        parts = [pd.read_csv(OUT / f"predicoes_{scenario}_{name}.csv") for name in MEMBERS]
        keys = ["linha_filtrada", "Coleta", "Vaso_id", "Classe", "Tempo", "fold"]
        ensemble = parts[0][keys].copy()
        for part in parts[1:]:
            for key in keys:
                assert part[key].equals(ensemble[key]), f"Previsoes desalinhadas: {scenario}/{key}"
        ensemble["probabilidade_classe_1"] = np.mean(
            [part.probabilidade_classe_1.to_numpy() for part in parts], axis=0
        )
        ensemble["predicao"] = (ensemble.probabilidade_classe_1 >= .5).astype(int)
        ensemble.to_csv(OUT / f"predicoes_{scenario}_ensemble.csv", index=False)
        for name, part in zip(MEMBERS, parts):
            all_rows.append(summarize(part, name, scenario))
        all_rows.append(summarize(ensemble, "ensemble", scenario))
    all_rows.append(summarize(pd.read_csv(OUT / "predicoes_ambiente_et.csv"), "et", "ambiente"))
    pd.DataFrame(all_rows).to_csv(OUT / "tabela_metricas.csv", index=False)
    (OUT / "metodo.json").write_text(json.dumps({
        "base": "55.946 leituras, 31 coletas, 10 vasos; mesmas cinco coletas excluidas na analise anterior; sem remocao para balancear classes",
        "classes": {"0": 26342, "1": 29604},
        "ponderacao": "pesos inversos da frequencia de classe recalculados somente nos nove vasos de treino de cada fold LOSO",
        "knn": "voto dos cinco vizinhos ponderado por distancia e peso da classe do vizinho",
        "tcn": "entropia cruzada com pesos de classe calculados nas janelas de treino",
        "solo": "Condicao_solo_binaria=1 para Soil_indice_0_1 <= 0.4 (seco), 0 para > 0.4 (molhado)",
        "validacao": "10 folds LeaveOneGroupOut por vaso",
        "entradas": SCENARIOS,
        "observacao": "selecoes de sensores e coletas feitas nos mesmos dez vasos; comparacoes exploratorias",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
