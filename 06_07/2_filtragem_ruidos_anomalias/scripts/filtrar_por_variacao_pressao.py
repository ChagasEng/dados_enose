from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "comparacao" / "pressao_filtrada"
OUT.mkdir(parents=True, exist_ok=True)

INPUTS = [
    (
        "antes_dia_20",
        ROOT / "comparacao" / "datasets_com_ambiente" / "antes_dia_20_com_ambiente.csv",
    ),
    (
        "dia_20_mais",
        ROOT / "comparacao" / "datasets_com_ambiente" / "dia_20_mais_com_ambiente.csv",
    ),
]

PRESSURE_DELTA_THRESHOLD = 0.10
WINDOW_LINES = 30
STABLE_BAND_AROUND_MEDIAN = 0.50
FEATURES = ["Soil", "Temp.", "Pres.", "MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]


def detect_pressure_events(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    remove_mask = np.zeros(len(df), dtype=bool)
    event_rows = []

    for coleta, block in df.groupby("Coleta", sort=False):
        block = block.reset_index().rename(columns={"index": "linha_global"})
        pressure = pd.to_numeric(block["Pres."], errors="coerce")
        delta = pressure.diff().abs()
        event_positions = np.flatnonzero((delta >= PRESSURE_DELTA_THRESHOLD).fillna(False).to_numpy())

        for position in event_positions:
            start = max(0, position - WINDOW_LINES)
            end = min(len(block) - 1, position + WINDOW_LINES)
            global_start = int(block.loc[start, "linha_global"])
            global_end = int(block.loc[end, "linha_global"])
            remove_mask[global_start : global_end + 1] = True

            event_rows.append(
                {
                    "dataset": dataset_name,
                    "Coleta": coleta,
                    "Dia": block.loc[position, "Dia"],
                    "Vaso": block.loc[position, "Vaso"],
                    "Classe": block.loc[position, "Classe"],
                    "Tempo_evento": block.loc[position, "Tempo"],
                    "linha_global_evento": int(block.loc[position, "linha_global"]),
                    "pressao_evento": block.loc[position, "Pres."],
                    "delta_pressao": float(delta.iloc[position]),
                    "limiar_delta_pressao": PRESSURE_DELTA_THRESHOLD,
                    "linha_corte_inicio": global_start,
                    "linha_corte_fim": global_end,
                    "Tempo_corte_inicio": block.loc[start, "Tempo"],
                    "Tempo_corte_fim": block.loc[end, "Tempo"],
                    "janela_linhas": WINDOW_LINES,
                }
            )

    return pd.DataFrame(event_rows), remove_mask


def summarize(dataset_name: str, original: pd.DataFrame, cleaned: pd.DataFrame, events: pd.DataFrame) -> dict:
    return {
        "dataset": dataset_name,
        "linhas_originais": int(len(original)),
        "linhas_removidas": int(len(original) - len(cleaned)),
        "linhas_mantidas": int(len(cleaned)),
        "percentual_removido": float((len(original) - len(cleaned)) / len(original) * 100),
        "eventos_pressao_detectados": int(len(events)),
        "limiar_delta_pressao": PRESSURE_DELTA_THRESHOLD,
        "janela_linhas_antes_depois": WINDOW_LINES,
        "pressao_min_original": float(pd.to_numeric(original["Pres."], errors="coerce").min()),
        "pressao_max_original": float(pd.to_numeric(original["Pres."], errors="coerce").max()),
        "pressao_min_filtrado": float(pd.to_numeric(cleaned["Pres."], errors="coerce").min()),
        "pressao_max_filtrado": float(pd.to_numeric(cleaned["Pres."], errors="coerce").max()),
    }


def stable_band_mask(df: pd.DataFrame) -> tuple[np.ndarray, float, float, float]:
    pressure = pd.to_numeric(df["Pres."], errors="coerce")
    median = float(pressure.median())
    low = median - STABLE_BAND_AROUND_MEDIAN
    high = median + STABLE_BAND_AROUND_MEDIAN
    outside = ((pressure < low) | (pressure > high)).fillna(False).to_numpy()
    return outside, median, low, high


def plot_pressure_before_after(dataset_name: str, original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=False)

    for ax, label, df, color in [
        (axes[0], "Antes do corte", original, "#c7503d"),
        (axes[1], "Depois do corte por pressao", cleaned, "#2f6f73"),
    ]:
        pressure = pd.to_numeric(df["Pres."], errors="coerce").rolling(9, center=True, min_periods=1).mean()
        ax.plot(np.arange(len(df)), pressure, color=color, linewidth=0.9)
        ax.set_title(f"{dataset_name} - {label}")
        ax.set_ylabel("Pres. cru")
        ax.grid(alpha=0.2)

    axes[1].set_xlabel("Indice da linha no dataset")
    fig.tight_layout()
    fig.savefig(OUT / f"{dataset_name}_pressao_antes_depois.png", dpi=180)
    plt.close(fig)


def plot_all_curves_after(dataset_name: str, cleaned: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(20, 18), sharex=True)
    x = np.arange(len(cleaned))

    for ax, feature in zip(axes, FEATURES):
        values = pd.to_numeric(cleaned[feature], errors="coerce").rolling(
            15, center=True, min_periods=1
        ).mean()
        ax.plot(x, values, color="#26364f", linewidth=0.8)
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title(f"{dataset_name} - curvas apos remover variacoes de pressao")
    axes[-1].set_xlabel("Indice da linha no dataset filtrado")
    fig.tight_layout()
    fig.savefig(OUT / f"{dataset_name}_curvas_apos_corte_pressao.png", dpi=180)
    plt.close(fig)


def save_outputs(dataset_name: str, original: pd.DataFrame, cleaned: pd.DataFrame, removed: pd.DataFrame) -> None:
    cleaned.to_csv(OUT / f"{dataset_name}_pressao_filtrada.csv", index=False)
    removed.to_csv(OUT / f"{dataset_name}_linhas_removidas_por_pressao.csv", index=False)

    if "Classe" in cleaned.columns:
        for classe in sorted(cleaned["Classe"].dropna().unique()):
            class_df = cleaned[cleaned["Classe"] == classe]
            class_df.to_csv(
                OUT / f"{dataset_name}_pressao_filtrada_classe_{int(classe)}.csv",
                index=False,
            )


def save_strict_outputs(
    dataset_name: str,
    original: pd.DataFrame,
    strict_cleaned: pd.DataFrame,
    strict_removed: pd.DataFrame,
) -> None:
    strict_cleaned.to_csv(OUT / f"{dataset_name}_pressao_filtrada_estrito.csv", index=False)
    strict_removed.to_csv(
        OUT / f"{dataset_name}_linhas_removidas_por_pressao_estrito.csv", index=False
    )

    if "Classe" in strict_cleaned.columns:
        for classe in sorted(strict_cleaned["Classe"].dropna().unique()):
            class_df = strict_cleaned[strict_cleaned["Classe"] == classe]
            class_df.to_csv(
                OUT / f"{dataset_name}_pressao_filtrada_estrito_classe_{int(classe)}.csv",
                index=False,
            )


def main() -> None:
    all_events = []
    all_summaries = []
    all_strict_summaries = []

    for dataset_name, input_path in INPUTS:
        original = pd.read_csv(input_path)
        events, remove_mask = detect_pressure_events(original, dataset_name)
        outside_band_mask, pressure_median, pressure_low, pressure_high = stable_band_mask(original)
        cleaned = original.loc[~remove_mask].reset_index(drop=True)
        removed = original.loc[remove_mask].copy()
        removed.insert(0, "linha_original", removed.index)

        strict_mask = remove_mask | outside_band_mask
        strict_cleaned = original.loc[~strict_mask].reset_index(drop=True)
        strict_removed = original.loc[strict_mask].copy()
        strict_removed.insert(0, "linha_original", strict_removed.index)

        save_outputs(dataset_name, original, cleaned, removed)
        save_strict_outputs(dataset_name, original, strict_cleaned, strict_removed)
        plot_pressure_before_after(dataset_name, original, cleaned)
        plot_all_curves_after(dataset_name, cleaned)
        plot_pressure_before_after(f"{dataset_name}_estrito", original, strict_cleaned)
        plot_all_curves_after(f"{dataset_name}_estrito", strict_cleaned)

        all_events.append(events)
        all_summaries.append(summarize(dataset_name, original, cleaned, events))
        strict_summary = summarize(f"{dataset_name}_estrito", original, strict_cleaned, events)
        strict_summary.update(
            {
                "pressao_mediana": pressure_median,
                "faixa_estavel_min": pressure_low,
                "faixa_estavel_max": pressure_high,
                "linhas_fora_faixa_estavel": int(outside_band_mask.sum()),
            }
        )
        all_strict_summaries.append(strict_summary)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_df.to_csv(OUT / "eventos_variacao_pressao.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUT / "resumo_corte_por_pressao.csv", index=False)
    strict_summary_df = pd.DataFrame(all_strict_summaries)
    strict_summary_df.to_csv(OUT / "resumo_corte_por_pressao_estrito.csv", index=False)

    (OUT / "README.txt").write_text(
        "\n".join(
            [
                "Corte por variacao de pressao",
                "",
                f"Limiar usado: delta absoluto de Pres. >= {PRESSURE_DELTA_THRESHOLD}",
                f"Janela removida: {WINDOW_LINES} linhas antes e {WINDOW_LINES} linhas depois de cada evento.",
                f"Versao estrita: tambem remove Pres. fora de mediana +/- {STABLE_BAND_AROUND_MEDIAN}.",
                "",
                "Arquivos principais:",
                "- antes_dia_20_pressao_filtrada.csv",
                "- dia_20_mais_pressao_filtrada.csv",
                "- eventos_variacao_pressao.csv",
                "- resumo_corte_por_pressao.csv",
                "- *_pressao_antes_depois.png",
                "- *_curvas_apos_corte_pressao.png",
                "",
                "Observacao: este corte remove variacoes abruptas de pressao, preservando pequenas oscilacoes normais.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Pasta:", OUT)
    print(summary_df.to_string(index=False))
    print()
    print("Versao estrita:")
    print(strict_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
