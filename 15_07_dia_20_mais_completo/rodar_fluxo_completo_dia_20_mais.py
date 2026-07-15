"""Replica o fluxo da pasta 06_07 para o dataset completo dia_20_mais."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RAW_SOURCE = ROOT / "dia_20_mais" / "dia_20_mais" / "dataset_dia_20_mais.csv"
ENV_SOURCE = ROOT / "comparacao" / "datasets_com_ambiente" / "dia_20_mais_com_ambiente.csv"
FILTER_SOURCE = ROOT / "06_07" / "2_filtragem_ruidos_anomalias" / "scripts" / "filtrar_por_variacao_pressao.py"
MODEL_SOURCE = ROOT / "06_07" / "rodar_modelagem_extra_trees_rede_neural_importancia.py"
MQ_FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV_FEATURES = ["Soil", "Temp.", "Pres."]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_input(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def export_class_splits(dataset: Path) -> None:
    """Salva uma copia por classe ao lado de cada dataset usado no fluxo."""
    df = pd.read_csv(dataset)
    if "Classe" not in df.columns:
        raise ValueError(f"Dataset sem coluna Classe: {dataset}")
    for classe in sorted(df["Classe"].dropna().unique()):
        output = dataset.with_name(f"{dataset.stem}_classe_{int(classe)}{dataset.suffix}")
        df.loc[df["Classe"] == classe].to_csv(output, index=False)


def copy_input_with_class_splits(source: Path, target: Path) -> None:
    copy_input(source, target)
    export_class_splits(target)


def validate_sources() -> None:
    raw = pd.read_csv(RAW_SOURCE)
    env = pd.read_csv(ENV_SOURCE)
    shared = ["Coleta", "Dia", "Vaso", *MQ_FEATURES, "Classe"]
    if len(raw) != len(env):
        raise ValueError("As bases sem ambiente e com ambiente possuem tamanhos diferentes.")
    for column in shared:
        raw_values = raw[column].astype(str).fillna("<NA>")
        env_values = env[column].astype(str).fillna("<NA>")
        # Dia e numericamente igual, mas aparece como inteiro em uma base e float na outra.
        if column == "Dia":
            if not (pd.to_numeric(raw[column]).to_numpy() == pd.to_numeric(env[column]).to_numpy()).all():
                raise ValueError("A coluna Dia nao esta alinhada entre as duas bases.")
        elif not raw_values.equals(env_values):
            raise ValueError(f"A coluna {column} nao esta alinhada entre as duas bases.")


def run_filter(env_dataset: Path) -> Path:
    filtro = load_module("filtro_pressao_dia_20_mais", FILTER_SOURCE)
    filtro.ROOT = BASE
    filtro.OUT = BASE / "comparacao" / "pressao_filtrada"
    filtro.OUT.mkdir(parents=True, exist_ok=True)
    filtro.INPUTS = [("dia_20_mais", env_dataset)]
    filtro.main()
    return filtro.OUT / "dia_20_mais_pressao_filtrada_estrito.csv"


def run_models(env_baseline: Path, strict_filtered: Path) -> pd.DataFrame:
    modelagem = load_module("modelagem_dia_20_mais", MODEL_SOURCE)
    modelagem.BASE_06 = BASE

    scenarios = [
        {
            "id": "01_baseline_dia_20_mais_mq_ambiente",
            "folder": BASE / "1_investigacao_hardware_banco",
            "dataset": env_baseline,
            "features": [*MQ_FEATURES, *ENV_FEATURES],
            "descricao": "Baseline do dia_20_mais completo, com MQ + ambiente.",
        },
        {
            "id": "02_filtrado_pressao_dia_20_mais_mq",
            "folder": BASE / "2_filtragem_ruidos_anomalias",
            "dataset": strict_filtered,
            "features": MQ_FEATURES,
            "descricao": "Dia_20_mais apos corte estrito por pressao, somente MQ.",
        },
        {
            "id": "03_filtrado_pressao_dia_20_mais_mq_ambiente",
            "folder": BASE / "3_compensacao_umidade_temperatura",
            "dataset": strict_filtered,
            "features": [*MQ_FEATURES, *ENV_FEATURES],
            "descricao": "Dia_20_mais apos corte estrito por pressao, com MQ + ambiente.",
        },
        {
            "id": "04_polido_final_dia_20_mais_mq",
            "folder": BASE / "4_polimento_inicial_modelagem",
            "dataset": strict_filtered,
            "features": MQ_FEATURES,
            "descricao": "Base polida final do dia_20_mais, somente MQ.",
        },
    ]
    results = []
    for scenario in scenarios:
        print(f"Rodando: {scenario['id']}")
        results.append(modelagem.run_scenario(scenario))
    summary = pd.DataFrame(results)
    modelagem.save_comparison(summary)
    return summary


def write_readme(summary: pd.DataFrame, strict_filtered: Path) -> None:
    source_rows = len(pd.read_csv(RAW_SOURCE))
    strict_rows = len(pd.read_csv(strict_filtered))
    best = summary.loc[summary[["extra_trees_accuracy", "rede_neural_accuracy"]].max(axis=1).idxmax()]
    best_model = (
        "ExtraTrees"
        if best["extra_trees_accuracy"] >= best["rede_neural_accuracy"]
        else "Rede neural MLP"
    )
    best_accuracy = max(best["extra_trees_accuracy"], best["rede_neural_accuracy"])
    (BASE / "README.md").write_text(
        f"""# Dia 20+ - reproducao completa do fluxo 06_07

## Bases

- Base sem ambiente: `dados/dataset_dia_20_mais.csv` ({source_rows} linhas)
- Base com ambiente: `dados/dataset_dia_20_mais_com_ambiente.csv` ({source_rows} linhas)
- Base apos corte estrito por pressao: {strict_rows} linhas

As duas bases de entrada foram validadas: possuem as mesmas linhas, coletas,
classes e leituras MQ. A segunda apenas acrescenta `Tempo`, `Soil`, `Temp.` e
`Pres.`.

## Cenarios

1. baseline MQ + ambiente;
2. corte estrito por pressao, somente MQ;
3. corte estrito por pressao, MQ + ambiente;
4. base polida final, somente MQ.

Todos usam split 70/30 por grupos de `Coleta`, dentro de cada classe.

## Melhor resultado

- Cenario: `{best['cenario']}`
- Modelo: {best_model}
- Accuracy: {best_accuracy:.4f}

Os resultados de cada cenario estao nas pastas numeradas; o comparativo geral
esta em `modelagem_comparativa/`.
""",
        encoding="utf-8",
    )


def main() -> None:
    for source in [RAW_SOURCE, ENV_SOURCE, FILTER_SOURCE, MODEL_SOURCE]:
        if not source.exists():
            raise FileNotFoundError(f"Arquivo necessario nao encontrado: {source}")
    validate_sources()

    raw_local = BASE / "dados" / "dataset_dia_20_mais.csv"
    env_local = BASE / "dados" / "dataset_dia_20_mais_com_ambiente.csv"
    copy_input_with_class_splits(RAW_SOURCE, raw_local)
    copy_input_with_class_splits(ENV_SOURCE, env_local)

    strict_filtered = run_filter(env_local)
    export_class_splits(strict_filtered)
    copies = [
        (env_local, BASE / "1_investigacao_hardware_banco" / "dados_base" / env_local.name),
        (strict_filtered, BASE / "2_filtragem_ruidos_anomalias" / "datasets_filtrados" / strict_filtered.name),
        (strict_filtered, BASE / "3_compensacao_umidade_temperatura" / "dados_base" / strict_filtered.name),
        (strict_filtered, BASE / "4_polimento_inicial_modelagem" / "datasets_limpos" / strict_filtered.name),
    ]
    for source, target in copies:
        copy_input_with_class_splits(source, target)

    summary = run_models(env_local, strict_filtered)
    write_readme(summary, strict_filtered)
    print("\nResumo:")
    print(summary.to_string(index=False))
    print(f"\nResultados em: {BASE}")


if __name__ == "__main__":
    main()
