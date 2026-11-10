"""Duração e mínimo de pressão nas abas originais das 36 coletas analisadas."""

from pathlib import Path

import openpyxl
import pandas as pd

from rodar_combinacoes import DATASET


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "confundimento_dia"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    included = pd.read_csv(DATASET)[["Coleta", "Classe", "Dia"]].drop_duplicates()
    book = openpyxl.load_workbook(ROOT / "coletas.xlsx", read_only=True, data_only=True)
    result = []
    for row in included.itertuples(index=False):
        if row.Coleta not in book.sheetnames:
            raise ValueError(f"Aba original ausente: {row.Coleta}")
        sheet = book[row.Coleta]
        values = sheet.values
        header = next(values)
        time_index, pressure_index = header.index("Tempo"), header.index("Pres.")
        times, pressures = [], []
        for item in values:
            try:
                time = float(item[time_index])
                pressure = float(item[pressure_index])
            except (TypeError, ValueError):
                continue
            times.append(time)
            pressures.append(pressure)
        if not times:
            raise ValueError(f"Sem leituras válidas: {row.Coleta}")
        smallest = min(range(len(pressures)), key=pressures.__getitem__)
        result.append({"Coleta": row.Coleta, "Classe": row.Classe,
                       "Dia": int(row.Dia), "n_leituras_originais": len(times),
                       "tempo_min": min(times), "tempo_max": max(times),
                       "duracao_original_s": max(times) - min(times),
                       "pressao_min_kpa": pressures[smallest],
                       "tempo_pressao_min_s": times[smallest],
                       "pressao_mediana_kpa": pd.Series(pressures).median()})
    frame = pd.DataFrame(result)
    frame.to_csv(OUT / "duracao_original_por_coleta.csv", index=False)
    summary = frame.groupby("Classe")[["duracao_original_s", "tempo_pressao_min_s",
                                        "pressao_min_kpa"]].agg(["min", "median", "max"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.to_csv(OUT / "duracao_original_por_classe.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
