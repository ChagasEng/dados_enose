"""Reproduz a modelagem da pasta 06_07 para o dataset completo dia_20_mais.

O dataset de origem nao contem Soil, Temp. ou Pres.; por isso o unico cenario
possivel e o de sensores MQ. A separacao continua sendo por grupos de Coleta,
evitando que linhas da mesma coleta vazem entre treino e teste.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
INPUT = ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv"
LOCAL_DATASET = BASE / "dados" / "dataset_dia_20_mais.csv"
BASE_SCRIPT = ROOT / "06_07" / "rodar_modelagem_extra_trees_rede_neural_importancia.py"
SCENARIO_ID = "01_dataset_completo_dia_20_mais_mq"


def carregar_modelagem_base():
    spec = importlib.util.spec_from_file_location("modelagem_base_06_07", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def escrever_readme(resultado: dict) -> None:
    extra = resultado["extra_trees_accuracy"]
    mlp = resultado["rede_neural_accuracy"]
    melhor_nome = "ExtraTrees" if extra >= mlp else "Rede neural MLP"
    melhor_valor = max(extra, mlp)
    dados = pd.read_csv(LOCAL_DATASET, usecols=["Coleta"])
    coletas_total = int(dados["Coleta"].nunique())

    texto = f"""# Dia 20+ - modelagem com dataset completo

## Base usada

- Origem: `dia_20_mais/dia_20_mais/dataset_dia_20_mais.csv`
- Linhas: {resultado['linhas_dataset']}
- Coletas: {coletas_total}
- Variaveis: `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`
- Classes: `0 = com nematoide`; `1 = sem nematoide`

O arquivo de origem nao possui `Soil`, `Temp.` nem `Pres.`. Portanto, esta
reproducao nao aplica corte por pressao nem compensacao ambiental da `06_07`;
ela executa o cenario comparavel que usa somente os sensores MQ.

## Validacao

O split e 70/30 por grupos de `Coleta`, separadamente dentro de cada classe.
Assim, uma mesma coleta nao aparece ao mesmo tempo no treino e no teste.

## Resultado

- Melhor modelo: {melhor_nome}
- Melhor accuracy: {melhor_valor:.4f}
- ExtraTrees accuracy: {extra:.4f}
- Rede neural MLP accuracy: {mlp:.4f}

Os artefatos reproduziveis estao em `modelagem/`: metricas, matrizes de
confusao, importancias por modelo e modelos serializados.
"""
    (BASE / "README.md").write_text(texto, encoding="utf-8")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset de entrada nao encontrado: {INPUT}")

    LOCAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT, LOCAL_DATASET)

    modelagem = carregar_modelagem_base()
    modelagem.BASE_06 = BASE
    scenario = {
        "id": SCENARIO_ID,
        "folder": BASE,
        "dataset": LOCAL_DATASET,
        "features": modelagem.MQ_FEATURES,
        "descricao": (
            "Dataset completo dia_20_mais, usando somente sensores MQ e "
            "validacao 70/30 por grupos de Coleta."
        ),
    }

    resultado = modelagem.run_scenario(scenario)
    pd.DataFrame([resultado]).to_csv(
        BASE / "comparativo_modelos_dia_20_mais.csv", index=False
    )
    with (BASE / "resumo_execucao.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=2, ensure_ascii=False)
    escrever_readme(resultado)

    print(pd.Series(resultado).to_string())
    print(f"\nResultados em: {BASE}")


if __name__ == "__main__":
    main()
