from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dados" / "dataset_melhor_modelo_sensores_corrigidos.csv"
GRAFICOS = BASE / "graficos"
ANALISE = BASE / "analise_C13_C17_C28"

MQ_RAW = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
MQ_CORR = [f"{sensor}_corrigido_env" for sensor in MQ_RAW]
PANELS_UPDATED = ["Soil_indice_0_1", "Temp_C", "Pres_kPa", *MQ_CORR]
PANELS_DIAG = ["Pres_kPa", *MQ_RAW]
SELECTED = ["C13", "C14", "C15", "C16", "C17", "C28"]


def ensure_dirs() -> None:
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    ANALISE.mkdir(parents=True, exist_ok=True)


def load_ordered() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATASET)
    parts: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    cursor = 0
    count = 1

    for classe, class_block in df.groupby("Classe", sort=True):
        for coleta, coleta_df in class_block.groupby("Coleta", sort=False):
            label = f"C{count}"
            block = coleta_df.copy()
            block["Coleta_plot"] = label
            start = cursor
            end = cursor + len(block) - 1
            rows.append(
                {
                    "Coleta_plot": label,
                    "Coleta": coleta,
                    "Classe": int(classe),
                    "Nematoide": "Com nematoide" if int(classe) == 0 else "Sem nematoide",
                    "inicio": start,
                    "fim": end,
                    "linhas": int(len(block)),
                }
            )
            cursor = end + 1
            count += 1
            parts.append(block)

    ordered = pd.concat(parts, ignore_index=True)
    blocks = pd.DataFrame(rows)
    return ordered, blocks


def smooth(series: pd.Series, window: int = 35) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1, center=True).median()


def shade_collections(axes: list[plt.Axes], blocks: pd.DataFrame) -> None:
    fill = {0: "#f7d8d5", 1: "#dcefe3"}
    line = {0: "#e57373", 1: "#6ab187"}
    text = {0: "#b33a3a", 1: "#2f7d51"}
    for _, row in blocks.iterrows():
        cls = int(row["Classe"])
        for ax in axes:
            ax.axvspan(row["inicio"], row["fim"], color=fill[cls], alpha=0.62, linewidth=0)
            ax.axvline(row["inicio"], color=line[cls], linewidth=0.8, alpha=0.72)
        axes[0].text(
            (row["inicio"] + row["fim"]) / 2,
            1.012,
            row["Coleta_plot"],
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=text[cls],
            fontweight="bold",
        )


def plot_updated_style(ordered: pd.DataFrame, blocks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(PANELS_UPDATED), 1, figsize=(18, 20), sharex=True)
    axes_list = list(axes)
    shade_collections(axes_list, blocks)
    x = np.arange(len(ordered))

    labels = {
        "Soil_indice_0_1": "Soil\nindice 0-1",
        "Temp_C": "Temp.\nBMP280",
        "Pres_kPa": "Pres.\nkPa",
        "MQ2_corrigido_env": "MQ2\ncorr.",
        "MQ3_corrigido_env": "MQ3\ncorr.",
        "MQ7_corrigido_env": "MQ7\ncorr.",
        "MQ8_corrigido_env": "MQ8\ncorr.",
        "MQ135_corrigido_env": "MQ135\ncorr.",
        "MQ138_corrigido_env": "MQ138\ncorr.",
    }

    for ax, feature in zip(axes_list, PANELS_UPDATED):
        ax.plot(x, smooth(ordered[feature]), color="#34495e", linewidth=0.85)
        ax.set_ylabel(labels.get(feature, feature), fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(ordered))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f7d8d5", alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color="#dcefe3", alpha=0.8, label="Sem nematoide"),
    ]
    axes_list[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes_list[-1].set_xlabel("Indice da linha no dataset")
    fig.suptitle(
        "Melhor modelo - coletas demarcadas por nematoide com sensores corrigidos",
        y=0.996,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(GRAFICOS / "coletas_por_nematoide_atualizado_estilo_original.png", dpi=180)
    plt.close(fig)


def global_diff_thresholds(ordered: pd.DataFrame, features: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for feature in features:
        diffs = []
        for _, block in ordered.groupby("Coleta_plot", sort=False):
            diffs.append(block[feature].astype(float).diff().abs().dropna())
        diff_series = pd.concat(diffs)
        thresholds[feature] = float(diff_series.quantile(0.995))
    return thresholds


def stats_for_collections(
    ordered: pd.DataFrame, blocks: pd.DataFrame, thresholds: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ["Soil_indice_0_1", "Temp_C", "Pres_kPa", *MQ_RAW, *MQ_CORR]
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for _, block_row in blocks.iterrows():
        label = str(block_row["Coleta_plot"])
        data = ordered[ordered["Coleta_plot"] == label].copy()
        row: dict[str, object] = {
            "Coleta_plot": label,
            "Coleta": block_row["Coleta"],
            "Classe": int(block_row["Classe"]),
            "Nematoide": block_row["Nematoide"],
            "linhas": int(len(data)),
        }
        for feature in features:
            series = data[feature].astype(float)
            diff = series.diff().abs()
            threshold = thresholds.get(feature, float(diff.quantile(0.995)))
            row[f"{feature}_media"] = float(series.mean())
            row[f"{feature}_desvio"] = float(series.std())
            row[f"{feature}_amplitude"] = float(series.max() - series.min())
            row[f"{feature}_max_salto"] = float(diff.max())
            row[f"{feature}_spikes"] = int((diff > threshold).sum())

            if label in SELECTED and feature in ["Pres_kPa", *MQ_RAW]:
                events = data.loc[diff > threshold, ["Coleta_plot", "Coleta", "Tempo", feature]].copy()
                for idx, event in events.iterrows():
                    event_rows.append(
                        {
                            "Coleta_plot": label,
                            "Coleta": block_row["Coleta"],
                            "feature": feature,
                            "linha_no_dataset_ordenado": int(idx),
                            "Tempo": event.get("Tempo", np.nan),
                            "valor": event[feature],
                            "salto_abs": float(diff.loc[idx]),
                            "limiar_global_p995": threshold,
                        }
                    )
        row["spikes_total_pres_mq"] = int(
            row["Pres_kPa_spikes"] + sum(row[f"{sensor}_spikes"] for sensor in MQ_RAW)
        )
        row["spikes_total_mq"] = int(sum(row[f"{sensor}_spikes"] for sensor in MQ_RAW))
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def check_duplicate_pairs(ordered: pd.DataFrame) -> pd.DataFrame:
    features = ["Soil", "Temp.", "Pres.", *MQ_RAW]
    rows: list[dict[str, object]] = []
    labels = ordered["Coleta_plot"].drop_duplicates().tolist()
    blocks = {label: ordered[ordered["Coleta_plot"] == label].reset_index(drop=True) for label in labels}
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            a = blocks[label_a]
            b = blocks[label_b]
            if len(a) != len(b):
                continue
            same_cells = True
            equal_percentages = []
            for feature in features:
                equal = a[feature].to_numpy() == b[feature].to_numpy()
                equal_percentages.append(float(equal.mean()))
                if not bool(equal.all()):
                    same_cells = False
                    break
            if same_cells:
                rows.append(
                    {
                        "Coleta_A": label_a,
                        "Nome_A": a["Coleta"].iloc[0],
                        "Coleta_B": label_b,
                        "Nome_B": b["Coleta"].iloc[0],
                        "linhas": int(len(a)),
                        "percentual_igual": 100.0,
                        "colunas_comparadas": ", ".join(features),
                    }
                )
    return pd.DataFrame(rows)


def plot_diagnostic_selected(ordered: pd.DataFrame, blocks: pd.DataFrame) -> None:
    selected_blocks = blocks[blocks["Coleta_plot"].isin(SELECTED)].copy()
    parts = []
    cursor = 0
    remap_rows = []
    for _, row in selected_blocks.iterrows():
        block = ordered[ordered["Coleta_plot"] == row["Coleta_plot"]].copy()
        start = cursor
        end = cursor + len(block) - 1
        block["diag_index"] = np.arange(start, end + 1)
        remap_rows.append({**row.to_dict(), "inicio": start, "fim": end})
        parts.append(block)
        cursor = end + 1

    diag = pd.concat(parts, ignore_index=True)
    diag_blocks = pd.DataFrame(remap_rows)
    fig, axes = plt.subplots(len(PANELS_DIAG), 1, figsize=(18, 14), sharex=True)
    axes_list = list(axes)
    shade_collections(axes_list, diag_blocks)

    x = diag["diag_index"].to_numpy()
    for ax, feature in zip(axes_list, PANELS_DIAG):
        ax.plot(x, smooth(diag[feature], window=15), color="#34495e", linewidth=0.9)
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_xlim(0, len(diag))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f7d8d5", alpha=0.8, label="Com nematoide"),
        plt.Rectangle((0, 0), 1, 1, color="#dcefe3", alpha=0.8, label="Sem nematoide"),
    ]
    axes_list[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes_list[-1].set_xlabel("Indice local no recorte C13-C17 + C28")
    fig.suptitle("Diagnostico visual das coletas C13-C17 e C28", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(GRAFICOS / "diagnostico_visual_C13_C17_C28.png", dpi=180)
    plt.close(fig)


def plot_noise_ranking(stats: pd.DataFrame) -> None:
    ranking = stats.sort_values("spikes_total_pres_mq", ascending=False).head(12).copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#b33a3a" if c in SELECTED else "#2d6cdf" for c in ranking["Coleta_plot"]]
    ax.bar(ranking["Coleta_plot"], ranking["spikes_total_pres_mq"], color=colors)
    ax.set_ylabel("Eventos acima do p99,5 global")
    ax.set_title("Ranking de ruido por saltos abruptos em Pres. + MQ")
    ax.grid(axis="y", alpha=0.2)
    for x, value in enumerate(ranking["spikes_total_pres_mq"]):
        ax.text(x, value + max(ranking["spikes_total_pres_mq"]) * 0.01, str(int(value)), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "ranking_ruido_por_coleta.png", dpi=180)
    plt.close(fig)


def write_markdown(stats: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    selected = stats[stats["Coleta_plot"].isin(SELECTED)].copy()
    c28 = selected[selected["Coleta_plot"] == "C28"].iloc[0]
    c17 = selected[selected["Coleta_plot"] == "C17"].iloc[0]

    duplicate_text = "Nenhuma duplicata exata foi detectada."
    if not duplicates.empty:
        duplicate_text = "\n".join(
            f"- `{row.Coleta_A}` (`{row.Nome_A}`) e `{row.Coleta_B}` (`{row.Nome_B}`) sao 100% identicas em {row.linhas} linhas."
            for row in duplicates.itertuples()
        )
    duplicate_action = (
        "Conferir e remover uma das duplicatas detectadas antes da modelagem."
        if not duplicates.empty
        else "C15 e C16 agora possuem sinais diferentes; manter ambas e auditar cada uma separadamente."
    )

    lines = [
        "# Diagnostico C13-C17 e C28",
        "",
        "## Arquivos gerados",
        "",
        "- `graficos/coletas_por_nematoide_atualizado_estilo_original.png`",
        "- `graficos/diagnostico_visual_C13_C17_C28.png`",
        "- `graficos/ranking_ruido_por_coleta.png`",
        "- `analise_C13_C17_C28/estatisticas_por_coleta.csv`",
        "- `analise_C13_C17_C28/eventos_ruido_C13_C17_C28.csv`",
        "- `analise_C13_C17_C28/duplicatas_exatas_coletas.csv`",
        "",
        "## Mapa",
        "",
        "- `C13`: dia 19 - Soja Heterodera Vaso 6, com nematoide.",
        "- `C14`: dia 19 - Soja Heterodera Vaso 7, com nematoide.",
        "- `C15`: dia 19 - Soja Heterodera Vaso 8, com nematoide.",
        "- `C16`: dia 19 - Soja Heterodera Vaso 9, com nematoide.",
        "- `C17`: dia 19 - Soja Heterodera Vaso 1, com nematoide.",
        "- `C28`: dia 13 - Soja Saudavel Vaso 1, sem nematoide.",
        "",
        "## C28",
        "",
        f"C28 nao foi a coleta mais ruidosa pelo criterio de saltos abruptos. Ela teve `{int(c28['spikes_total_pres_mq'])}` eventos acima do limiar global, contra `{int(c17['spikes_total_pres_mq'])}` em C17. O que chama atencao em C28 e o tamanho dos degraus: `MQ2` teve amplitude de `{c28['MQ2_amplitude']:.0f}` e maior salto de `{c28['MQ2_max_salto']:.0f}`; `MQ138` teve amplitude de `{c28['MQ138_amplitude']:.0f}` e maior salto de `{c28['MQ138_max_salto']:.0f}`. A pressao em C28 ficou relativamente estavel, com amplitude de `{c28['Pres_kPa_amplitude']:.2f} kPa`.",
        "",
        "Leitura provavel: C28 parece mais uma mudanca brusca de regime/saturacao/reacomodacao dos MQ do que ruido continuo causado por pressao. Como a pressao interna ja foi filtrada e ficou estavel, a causa mais provavel esta em transiente de gas, memoria/saturacao de sensor, fluxo interno, troca/manuseio ou efeito de contaminacao/residuo na camara.",
        "",
        "## C13-C17",
        "",
        "As coletas C13-C17 sao todas do dia 19 com nematoide, mas nao se comportam como repeticoes consistentes.",
        "",
        duplicate_text,
        "",
        f"`C17` e o ponto mais critico: ela teve `{int(c17['spikes_total_pres_mq'])}` saltos abruptos em Pres.+MQ, sendo `{int(c17['spikes_total_mq'])}` nos MQ. Isso atingiu varios canais ao mesmo tempo (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135` e `MQ138`), enquanto a pressao variou pouco (`{c17['Pres_kPa_amplitude']:.2f} kPa`).",
        "",
        "Leitura provavel: C17 tem cara de instabilidade de aquisicao/sinal dos MQ, saturacao ou transiente eletrico/ADC, nao de variacao fisica de pressao.",
        "",
        "## Acao recomendada",
        "",
        f"1. {duplicate_action}",
        "2. Registrar a origem e a substituicao da C16 para manter a rastreabilidade.",
        "3. Rodar outra versao removendo ou marcando C17 como coleta anomala.",
        "4. Para C28, revisar o log experimental: troca de vaso, abertura/fechamento, tempo de estabilizacao, fluxo/bomba e possivel saturacao dos MQ.",
        "5. Nao tratar esses pontos como resposta biologica sem validar a origem fisica/operacional.",
        "",
    ]
    (ANALISE / "diagnostico_C13_C17_C28.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ordered, blocks = load_ordered()
    blocks.to_csv(GRAFICOS / "mapa_coletas_nematoide_atualizado.csv", index=False, encoding="utf-8-sig")

    plot_updated_style(ordered, blocks)
    plot_diagnostic_selected(ordered, blocks)

    thresholds = global_diff_thresholds(ordered, ["Soil_indice_0_1", "Temp_C", "Pres_kPa", *MQ_RAW, *MQ_CORR])
    pd.DataFrame(
        [{"feature": feature, "limiar_abs_diff_p995": value} for feature, value in thresholds.items()]
    ).to_csv(ANALISE / "limiares_ruido_diff_p995.csv", index=False, encoding="utf-8-sig")

    stats, events = stats_for_collections(ordered, blocks, thresholds)
    stats.to_csv(ANALISE / "estatisticas_por_coleta.csv", index=False, encoding="utf-8-sig")
    events.to_csv(ANALISE / "eventos_ruido_C13_C17_C28.csv", index=False, encoding="utf-8-sig")
    stats[stats["Coleta_plot"].isin(SELECTED)].to_csv(
        ANALISE / "estatisticas_C13_C17_C28.csv", index=False, encoding="utf-8-sig"
    )

    ranking = stats.sort_values("spikes_total_pres_mq", ascending=False)
    ranking.to_csv(ANALISE / "ranking_ruido_por_coleta.csv", index=False, encoding="utf-8-sig")
    plot_noise_ranking(stats)

    duplicates = check_duplicate_pairs(ordered)
    duplicates.to_csv(ANALISE / "duplicatas_exatas_coletas.csv", index=False, encoding="utf-8-sig")
    write_markdown(stats, duplicates)

    print("Graficos e diagnostico gerados em:")
    print(GRAFICOS / "coletas_por_nematoide_atualizado_estilo_original.png")
    print(GRAFICOS / "diagnostico_visual_C13_C17_C28.png")
    print(ANALISE / "diagnostico_C13_C17_C28.md")


if __name__ == "__main__":
    main()
