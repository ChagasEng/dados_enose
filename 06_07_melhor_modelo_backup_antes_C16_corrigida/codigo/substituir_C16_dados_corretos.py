from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BEST = ROOT / "06_07_melhor_modelo"
SOURCE_5 = ROOT / "06_07" / "5_extratrees_sensores_ambiente_confirmados"
ATTACHMENT = Path(
    r"C:\Users\Matheus Bastos Chaga\.codex\attachments\d872cfdc-b60c-44d6-b131-b379f42ce014\pasted-text.txt"
)
COLLECTION = "dia 19 - Soja Heterodera Vaso 9"
SOURCE_DATASET = ROOT / "comparacao" / "datasets_com_ambiente" / "antes_dia_20_com_ambiente.csv"
FILTER_SCRIPT = ROOT / "comparacao" / "pressao_filtrada" / "filtrar_por_variacao_pressao.py"
FILTERED = ROOT / "comparacao" / "pressao_filtrada" / "antes_dia_20_pressao_filtrada_estrito.csv"
MODEL_5_SCRIPT = SOURCE_5 / "scripts" / "rodar_extratrees_sensores_confirmados.py"
BACKUP = ROOT / "06_07_melhor_modelo_backup_antes_C16_corrigida"


def backup_once() -> None:
    if not BACKUP.exists():
        shutil.copytree(BEST, BACKUP)
    backup_data = BACKUP / "dados_origem"
    backup_data.mkdir(parents=True, exist_ok=True)
    old_source = pd.read_csv(SOURCE_DATASET)
    old_source.loc[old_source["Coleta"].eq(COLLECTION)].to_csv(
        backup_data / "C16_antiga_antes_filtro.csv", index=False, encoding="utf-8-sig"
    )
    old_filtered = pd.read_csv(FILTERED)
    old_filtered.loc[old_filtered["Coleta"].eq(COLLECTION)].to_csv(
        backup_data / "C16_antiga_depois_filtro.csv", index=False, encoding="utf-8-sig"
    )


def load_replacement() -> pd.DataFrame:
    replacement = pd.read_csv(ATTACHMENT, sep="\t").dropna(how="all").copy()
    required = ["Tempo", "Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138", "Classe"]
    missing = [column for column in required if column not in replacement.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo recebido: {missing}")
    replacement = replacement.dropna(subset=required[:-1]).reset_index(drop=True)
    replacement["Coleta"] = COLLECTION
    replacement["Dia"] = 19
    replacement["Vaso"] = 9
    # A coleta correta substitui a C16 Heterodera; o 1 no texto recebido era rotulo incorreto.
    replacement["Classe"] = 0
    columns = ["Coleta", "Dia", "Vaso", *required[:-1], "Classe"]
    replacement = replacement[columns]
    if len(replacement) != 1921:
        raise ValueError(f"Quantidade inesperada de linhas validas: {len(replacement)}")
    return replacement


def replace_source(replacement: pd.DataFrame) -> None:
    source = pd.read_csv(SOURCE_DATASET)
    positions = source.index[source["Coleta"].eq(COLLECTION)]
    if len(positions) == 0:
        raise ValueError("C16 atual nao foi encontrada na base de origem.")
    start, end = int(positions.min()), int(positions.max())
    updated = pd.concat(
        [source.iloc[:start], replacement, source.iloc[end + 1 :]], ignore_index=True
    )
    updated.to_csv(SOURCE_DATASET, index=False, encoding="utf-8-sig")
    for classe in [0, 1]:
        updated.loc[updated["Classe"].eq(classe)].to_csv(
            SOURCE_DATASET.with_name(f"antes_dia_20_com_ambiente_classe_{classe}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    replacement.to_csv(
        BEST / "dados" / "C16_dados_corretos_recebidos_classe_0.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def update_filtered_inputs() -> None:
    copies = [
        ROOT / "06_07" / "2_filtragem_ruidos_anomalias" / "datasets_filtrados" / FILTERED.name,
        ROOT / "06_07" / "3_compensacao_umidade_temperatura" / "dados_base" / "antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv",
        ROOT / "06_07" / "4_polimento_inicial_modelagem" / "datasets_limpos" / FILTERED.name,
    ]
    for target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FILTERED, target)


def sync_best_outputs() -> None:
    mappings = [
        (SOURCE_5 / "dados_processados" / "dataset_sensores_confirmados_com_correcoes.csv", BEST / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"),
        (SOURCE_5 / "dados_processados" / "coeficientes_compensacao_ambiente_por_sensor.csv", BEST / "dados" / "coeficientes_compensacao_ambiente_por_sensor.csv"),
        (SOURCE_5 / "modelagem" / "metricas" / "resumo_extratrees_sensores_confirmados.csv", BEST / "metricas" / "comparacao_cenarios_extratrees.csv"),
        (SOURCE_5 / "modelagem" / "metricas" / "04_mq_corrigido_ambiente_com_contexto_metricas.json", BEST / "metricas" / "metricas_melhor_modelo_93_20.json"),
        (SOURCE_5 / "modelagem" / "metricas" / "04_mq_corrigido_ambiente_com_contexto_predicoes_teste.csv", BEST / "metricas" / "predicoes_teste_melhor_modelo.csv"),
        (SOURCE_5 / "modelagem" / "metricas" / "split_70_30_por_coleta.csv", BEST / "metricas" / "split_70_30_por_coleta.csv"),
        (SOURCE_5 / "modelagem" / "matrizes" / "04_mq_corrigido_ambiente_com_contexto_matriz_confusao.csv", BEST / "matriz_confusao" / "matriz_confusao_melhor_modelo.csv"),
        (SOURCE_5 / "modelagem" / "matrizes" / "04_mq_corrigido_ambiente_com_contexto_matriz_confusao.png", BEST / "matriz_confusao" / "matriz_confusao_melhor_modelo.png"),
        (SOURCE_5 / "modelagem" / "importancias" / "04_mq_corrigido_ambiente_com_contexto_importancia_nativa.csv", BEST / "importancia_sensores" / "importancia_nativa_extra_trees_melhor_modelo.csv"),
        (SOURCE_5 / "modelagem" / "importancias" / "04_mq_corrigido_ambiente_com_contexto_importancia_permutacao.csv", BEST / "importancia_sensores" / "importancia_permutacao_extra_trees_melhor_modelo.csv"),
        (SOURCE_5 / "graficos" / "04_mq_corrigido_ambiente_com_contexto_importancia_nativa.png", BEST / "importancia_sensores" / "grafico_importancia_nativa_melhor_modelo.png"),
        (SOURCE_5 / "graficos" / "04_mq_corrigido_ambiente_com_contexto_importancia_permutacao.png", BEST / "importancia_sensores" / "grafico_importancia_permutacao_melhor_modelo.png"),
        (SOURCE_5 / "modelagem" / "modelos" / "04_mq_corrigido_ambiente_com_contexto_extra_trees.joblib", BEST / "modelo" / "modelo_extra_trees_melhor_93_20.joblib"),
        (SOURCE_5 / "graficos" / "05_comparacao_metricas_extratrees.png", BEST / "graficos" / "comparacao_metricas_extratrees.png"),
        (SOURCE_5 / "graficos" / "04_correlacao_ambiente_antes_depois_correcao.png", BEST / "graficos" / "correlacao_ambiente_antes_depois_correcao.png"),
        (SOURCE_5 / "graficos" / "03_coletas_por_nematoide_mq_corrigidos_zscore.png", BEST / "graficos" / "mq_corrigidos_zscore_por_coleta.png"),
        (SOURCE_5 / "graficos" / "02_coletas_por_nematoide_sinais_corrigidos_overlay.png", BEST / "graficos" / "coletas_por_nematoide_sinais_corrigidos.png"),
        (SOURCE_5 / "graficos" / "mapa_coletas_nematoide_sensores_confirmados.csv", BEST / "graficos" / "mapa_coletas_nematoide_sensores_confirmados.csv"),
    ]
    for source, target in mappings:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def create_summary_panel() -> None:
    comparison = pd.read_csv(BEST / "metricas" / "comparacao_cenarios_extratrees.csv")
    metrics = json.loads((BEST / "metricas" / "metricas_melhor_modelo_93_20.json").read_text(encoding="utf-8-sig"))
    importance = pd.read_csv(BEST / "importancia_sensores" / "importancia_nativa_extra_trees_melhor_modelo.csv")
    matrix = pd.read_csv(BEST / "matriz_confusao" / "matriz_confusao_melhor_modelo.csv", index_col=0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].barh(comparison["titulo"], comparison["accuracy"] * 100, color="#356a8a")
    axes[0, 0].set_title("Comparacao dos cenarios")
    axes[0, 0].set_xlabel("Accuracy (%)")
    axes[0, 0].grid(axis="x", alpha=0.2)
    im = axes[0, 1].imshow(matrix.to_numpy(), cmap="Greens")
    axes[0, 1].set_title("Matriz de confusao")
    axes[0, 1].set_xticks([0, 1], ["Prev. doente", "Prev. saudavel"])
    axes[0, 1].set_yticks([0, 1], ["Real doente", "Real saudavel"])
    for (i, j), value in __import__("numpy").ndenumerate(matrix.to_numpy()):
        axes[0, 1].text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    top = importance.head(9).sort_values("importancia")
    axes[1, 0].barh(top["feature"], top["importancia"], color="#2f7d56")
    axes[1, 0].set_title("Importancia nativa")
    extra = metrics["extra_trees"]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.9,
        "C16 corrigida\n\n"
        f"Accuracy: {extra['accuracy'] * 100:.2f}%\n"
        f"Balanced accuracy: {extra['balanced_accuracy'] * 100:.2f}%\n"
        f"F1 macro: {extra['f1_macro'] * 100:.2f}%\n\n"
        "C15 e C16 agora possuem dados diferentes.",
        va="top",
        fontsize=15,
    )
    fig.suptitle("Resumo do modelo apos correcao da C16", fontsize=16)
    fig.tight_layout()
    fig.savefig(BEST / "graficos" / "painel_resumo_melhor_modelo.png", dpi=180)
    plt.close(fig)


def main() -> None:
    backup_once()
    replacement = load_replacement()
    replace_source(replacement)
    run([sys.executable, str(FILTER_SCRIPT)])
    update_filtered_inputs()
    run([sys.executable, str(MODEL_5_SCRIPT)])
    sync_best_outputs()
    run([sys.executable, str(BEST / "codigo" / "rodar_extratrees_melhor_modelo.py")])
    run([sys.executable, str(BEST / "codigo" / "gerar_grafico_e_diagnostico_coletas.py")])
    create_summary_panel()
    print("C16 substituida e artefatos principais regenerados.")


if __name__ == "__main__":
    main()
