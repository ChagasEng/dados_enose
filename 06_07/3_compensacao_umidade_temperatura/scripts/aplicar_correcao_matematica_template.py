from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "dados_base" / "antes_dia_20_pressao_filtrada_estrito_com_ambiente.csv"
FACTORS = BASE / "templates" / "fatores_correcao_datasheet_template.csv"
OUTPUT = BASE / "dados_base" / "antes_dia_20_pressao_filtrada_estrito_corrigido_template.csv"


def main() -> None:
    df = pd.read_csv(INPUT)
    factors = pd.read_csv(FACTORS)

    for _, row in factors.iterrows():
        sensor = row["sensor"]
        if sensor not in df.columns:
            continue

        temp_ref = float(row["temp_ref"])
        soil_ref = float(row["soil_ref"])
        coef_temp = float(row["coef_temp_percent_por_unidade"]) / 100.0
        coef_soil = float(row["coef_soil_percent_por_unidade"]) / 100.0

        fator = 1.0
        if "Temp." in df.columns:
            fator = fator + coef_temp * (pd.to_numeric(df["Temp."], errors="coerce") - temp_ref)
        if "Soil" in df.columns:
            fator = fator + coef_soil * (pd.to_numeric(df["Soil"], errors="coerce") - soil_ref)

        fator = fator.clip(lower=0.01)
        df[f"{sensor}_corrigido"] = pd.to_numeric(df[sensor], errors="coerce") / fator

    df.to_csv(OUTPUT, index=False)
    print(f"Arquivo gerado: {OUTPUT}")
    print("Aviso: com os coeficientes zerados do template, as colunas corrigidas ficam iguais as originais.")


if __name__ == "__main__":
    main()
