"""Compara classificadores antes e depois do controle de pressão.

Cada condição mantém a separação 70/30 por grupos de coleta. Na condição
controlada, as leituras afetadas pela pressão foram removidas e os sensores MQ
são compensados por solo, temperatura e pressão usando apenas o treino.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "resultados_comparacao_controle_pressao"
BASELINE = ROOT / "06_07" / "1_investigacao_hardware_banco" / "dados_base" / "antes_dia_20_com_ambiente_baseline.csv"
CONTROLADO = ROOT / "06_07" / "3_compensacao_umidade_temperatura" / "dados_base" / "antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv"
MQ = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV = ["Soil", "Temp.", "Pres."]
TARGET, GROUP, RANDOM_STATE = "Classe", "Coleta", 42


def carregar(caminho: Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho)[[GROUP, TARGET, *MQ, *ENV]].copy()
    for coluna in [TARGET, *MQ, *ENV]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    return dados.dropna().reset_index(drop=True)


def separar_por_coleta(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treino, teste = [], []
    for _, bloco in dados.groupby(TARGET):
        grupos = pd.Series(bloco[GROUP].unique()).sample(frac=1, random_state=RANDOM_STATE).tolist()
        grupos_treino = set(grupos[:int(len(grupos) * 0.70)])
        treino.append(bloco[bloco[GROUP].isin(grupos_treino)])
        teste.append(bloco[~bloco[GROUP].isin(grupos_treino)])
    return (
        pd.concat(treino).sample(frac=1, random_state=RANDOM_STATE),
        pd.concat(teste).sample(frac=1, random_state=RANDOM_STATE),
    )


def compensar_mq(dados: pd.DataFrame, treino: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    corrigido = dados.copy()
    solo = dados["Soil"]
    corrigido["Soil_indice_0_1"] = (solo - solo.min()) / (solo.max() - solo.min())
    corrigido["Temp_C"] = corrigido["Temp."]
    corrigido["Pres_kPa"] = corrigido["Pres."]
    ambiente = ["Soil_indice_0_1", "Temp_C", "Pres_kPa"]
    indices_treino = treino.index
    for sensor in MQ:
        ajuste = Pipeline([
            ("escala", StandardScaler()),
            ("huber", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=400)),
        ])
        ajuste.fit(corrigido.loc[indices_treino, ambiente], corrigido.loc[indices_treino, sensor])
        previsao = ajuste.predict(corrigido[ambiente])
        nivel_treino = ajuste.predict(corrigido.loc[indices_treino, ambiente]).mean()
        corrigido[f"{sensor}_corrigido_env"] = corrigido[sensor] - (previsao - nivel_treino)
    return corrigido, [f"{sensor}_corrigido_env" for sensor in MQ] + ambiente


def modelos(y_treino: pd.Series) -> dict[str, object]:
    peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()
    return {
        "Extra Trees": ExtraTreesClassifier(n_estimators=900, random_state=RANDOM_STATE, n_jobs=-1, max_features="sqrt", min_samples_leaf=10),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", max_features="sqrt", min_samples_leaf=10),
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_lambda=2.0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=350, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.1, random_state=RANDOM_STATE, class_weight="balanced"),
        "Regressão logística": Pipeline([("escala", StandardScaler()), ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Naive Bayes Gaussiano": Pipeline([("transformacao", PowerTransformer()), ("modelo", GaussianNB())]),
        "Naive Bayes Multinomial": Pipeline([("escala", MinMaxScaler()), ("modelo", MultinomialNB())]),
        "Naive Bayes Complementar": Pipeline([("escala", MinMaxScaler()), ("modelo", ComplementNB())]),
        "SVM linear": Pipeline([("escala", StandardScaler()), ("modelo", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=20000))]),
        "KNN": Pipeline([("escala", StandardScaler()), ("modelo", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))]),
        "MLP": Pipeline([("escala", StandardScaler()), ("modelo", MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", alpha=0.001, learning_rate_init=0.001, max_iter=250, early_stopping=True, validation_fraction=0.15, n_iter_no_change=12, random_state=RANDOM_STATE, batch_size=2048))]),
    }


def avaliar(nome_condicao: str, caminho: Path, com_compensacao: bool) -> list[dict[str, object]]:
    dados = carregar(caminho)
    treino, teste = separar_por_coleta(dados)
    if com_compensacao:
        dados, atributos = compensar_mq(dados, treino)
        treino, teste = dados.loc[treino.index], dados.loc[teste.index]
    else:
        atributos = [*MQ, *ENV]

    resultado = []
    for nome, modelo in modelos(treino[TARGET]).items():
        print(f"{nome_condicao}: {nome}", flush=True)
        if nome == "MLP":
            pesos = compute_sample_weight(class_weight="balanced", y=treino[TARGET])
            modelo.fit(treino[atributos], treino[TARGET], modelo__sample_weight=pesos)
        else:
            modelo.fit(treino[atributos], treino[TARGET])
        previsto = modelo.predict(teste[atributos])
        resultado.append({
            "condicao": nome_condicao,
            "modelo": nome,
            "leituras_teste": len(teste),
            "accuracy": accuracy_score(teste[TARGET], previsto),
            "balanced_accuracy": balanced_accuracy_score(teste[TARGET], previsto),
            "f1_macro": f1_score(teste[TARGET], previsto, average="macro"),
        })
    return resultado


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    resultados = avaliar("Sem controle de pressão", BASELINE, False)
    resultados += avaliar("Com controle de pressão", CONTROLADO, True)
    tabela = pd.DataFrame(resultados)
    tabela.to_csv(OUTPUT / "comparacao_classificadores_controle_pressao.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "metodo.json").write_text(json.dumps({
        "split": "70/30 por grupos de Coleta dentro de cada classe",
        "sem_controle": "MQ crus + Soil + Temp. + Pres., antes do corte por pressão",
        "com_controle": "corte estrito por pressão; MQ compensados com HuberRegressor ajustado somente no treino; Soil, Temp. e Pres. mantidos como contexto",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tabela.to_string(index=False))


if __name__ == "__main__":
    main()
