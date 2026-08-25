"""Compara classificadores no mesmo split do melhor modelo.

Os hiperparametros dos ensembles sao escolhidos apenas no conjunto de treino,
por validacao cruzada estratificada e agrupada por coleta. O teste salvo na
coluna ``Conjunto`` permanece intocado ate a avaliacao final.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
OUTPUT = BASE / "comparacao_classificadores"
TARGET = "Classe"
GROUP = "Coleta"
FEATURES = [
    "MQ2_corrigido_env", "MQ3_corrigido_env", "MQ7_corrigido_env",
    "MQ8_corrigido_env", "MQ135_corrigido_env", "MQ138_corrigido_env",
    "Soil_indice_0_1", "Temp_C", "Pres_kPa",
]
RANDOM_STATE = 42


def ensembles_candidatos() -> dict[str, list[object]]:
    comuns = [
        {"max_features": "sqrt", "min_samples_leaf": 1, "class_weight": None},
        {"max_features": "sqrt", "min_samples_leaf": 3, "class_weight": "balanced"},
        {"max_features": "sqrt", "min_samples_leaf": 10, "class_weight": None},
        {"max_features": 0.7, "min_samples_leaf": 3, "class_weight": "balanced"},
    ]
    return {
        "Extra Trees": [
            ExtraTreesClassifier(
                n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
                bootstrap=False, **parametros,
            )
            for parametros in comuns
        ],
        "Random Forest": [
            RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
                bootstrap=True, **parametros,
            )
            for parametros in comuns
        ],
    }


def selecionar_ensemble(nome: str, candidatos: list[object], treino: pd.DataFrame):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    registros = []
    melhor_modelo = None
    melhor_media = float("-inf")
    for indice, modelo in enumerate(candidatos, start=1):
        escores = cross_val_score(
            modelo, treino[FEATURES], treino[TARGET], groups=treino[GROUP],
            cv=cv, scoring="balanced_accuracy", n_jobs=1,
        )
        registro = {
            "modelo": nome,
            "candidato": indice,
            "balanced_accuracy_cv_media": float(escores.mean()),
            "balanced_accuracy_cv_desvio": float(escores.std()),
            "parametros": json.dumps(modelo.get_params(), default=str, ensure_ascii=False),
        }
        registros.append(registro)
        if escores.mean() > melhor_media:
            melhor_media = float(escores.mean())
            melhor_modelo = modelo
    return melhor_modelo, registros


def modelos_escalonados() -> dict[str, object]:
    return {
        "SVM linear": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", LinearSVC(
                C=1.0, class_weight="balanced", dual="auto",
                max_iter=30000, random_state=RANDOM_STATE,
            )),
        ]),
        "MLP": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001,
                batch_size=512, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, max_iter=400, random_state=RANDOM_STATE,
            )),
        ]),
    }


def main() -> None:
    dados = pd.read_csv(DATASET)
    treino = dados[dados["Conjunto"] == "Treino"].copy()
    teste = dados[dados["Conjunto"] == "Teste"].copy()
    OUTPUT.mkdir(exist_ok=True)

    modelos = {}
    validacao = []
    for nome, candidatos in ensembles_candidatos().items():
        print(f"Selecionando {nome} no treino...", flush=True)
        modelos[nome], registros = selecionar_ensemble(nome, candidatos, treino)
        validacao.extend(registros)
    modelos.update(modelos_escalonados())

    resultados = []
    for nome, modelo in modelos.items():
        print(f"Avaliando {nome} no teste...", flush=True)
        # A busca usa 300 arvores para ser rapida; o ajuste final preserva as
        # 900 arvores do protocolo do modelo principal.
        if nome in {"Extra Trees", "Random Forest"}:
            modelo = clone(modelo).set_params(n_estimators=900)
        modelo.fit(treino[FEATURES], treino[TARGET])
        previsto = modelo.predict(teste[FEATURES])
        resultados.append({
            "modelo": nome,
            "linhas_treino": len(treino),
            "linhas_teste": len(teste),
            "coletas_treino": treino[GROUP].nunique(),
            "coletas_teste": teste[GROUP].nunique(),
            "accuracy": accuracy_score(teste[TARGET], previsto),
            "balanced_accuracy": balanced_accuracy_score(teste[TARGET], previsto),
            "f1_macro": f1_score(teste[TARGET], previsto, average="macro"),
        })

    pd.DataFrame(validacao).to_csv(
        OUTPUT / "validacao_ensembles_no_treino.csv", index=False, encoding="utf-8-sig"
    )
    resultado = pd.DataFrame(resultados).sort_values(
        ["balanced_accuracy", "f1_macro"], ascending=False
    )
    resultado.to_csv(
        OUTPUT / "comparacao_classificadores_classicos.csv", index=False,
        encoding="utf-8-sig",
    )
    metodo = {
        "dataset": str(DATASET.relative_to(BASE)),
        "atributos": FEATURES,
        "split_externo": "70/30 por coleta, previamente salvo na coluna Conjunto",
        "selecao_ensembles": "4-fold StratifiedGroupKFold somente no treino; metrica balanced_accuracy",
        "observacao": "SVM e MLP foram ajustados em pipelines com StandardScaler.",
    }
    (OUTPUT / "metodo.json").write_text(
        json.dumps(metodo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
