from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    ("antes_dia_20", ROOT / "comparacao" / "datasets_com_ambiente" / "antes_dia_20_com_ambiente.csv"),
    ("dia_20_mais", ROOT / "comparacao" / "datasets_com_ambiente" / "dia_20_mais_com_ambiente.csv"),
]
OUTPUT = ROOT / "orientacaoes_02_07_26" / "analises" / "candidatos_corte_por_pressao_rerun.csv"


def pressure_cut_candidates(df: pd.DataFrame, dataset_name: str, window_lines: int = 30) -> pd.DataFrame:
    rows = []

    for coleta, block in df.groupby("Coleta", sort=False):
        block = block.reset_index().rename(columns={"index": "linha_global"})
        pres = pd.to_numeric(block["Pres."], errors="coerce")
        delta = pres.diff().abs()
        mad = (delta - delta.median()).abs().median()
        threshold = max(delta.quantile(0.995), delta.median() + 8 * mad)

        for idx in delta[delta >= threshold].index.tolist():
            start = max(0, idx - window_lines)
            end = min(len(block) - 1, idx + window_lines)
            rows.append(
                {
                    "dataset": dataset_name,
                    "Coleta": coleta,
                    "Dia": block.loc[idx, "Dia"],
                    "Vaso": block.loc[idx, "Vaso"],
                    "Classe": block.loc[idx, "Classe"],
                    "Tempo_evento": block.loc[idx, "Tempo"],
                    "linha_global_evento": int(block.loc[idx, "linha_global"]),
                    "Pres_evento": block.loc[idx, "Pres."],
                    "delta_pressao": float(delta.loc[idx]),
                    "limiar_delta_pressao": float(threshold),
                    "linha_global_corte_inicio": int(block.loc[start, "linha_global"]),
                    "linha_global_corte_fim": int(block.loc[end, "linha_global"]),
                    "Tempo_corte_inicio": block.loc[start, "Tempo"],
                    "Tempo_corte_fim": block.loc[end, "Tempo"],
                    "linhas_corte": int(end - start + 1),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for dataset_name, path in INPUTS:
        df = pd.read_csv(path)
        frames.append(pressure_cut_candidates(df, dataset_name))

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "delta_pressao"], ascending=[True, False]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Arquivo gerado: {OUTPUT}")
    print(f"Eventos candidatos: {len(result)}")


if __name__ == "__main__":
    main()
