"""
Coleta de indicadores macroeconomicos do Banco Central (API SGS).

Baixa as series trimestrais usadas pelo modelo e grava
`dados/brutos/indicadores_macro_sgs.csv` no formato esperado por
`integrar_painel_final.py` (colunas com nomes longos, separador ';' e decimal
',', encoding latin1).

Series (codigos SGS):
  - 22109  PIB trimestral (indice)                 -> Crescimento_PIB_Trimestral
  - 433    IPCA mensal (% a.m.), composto p/ trim. -> IPCA_Trimestral
  - 20786  Spread medio (% a.a.), media trimestral -> Spread_Medio_Trimestral
  - 4189   Selic acumulada no mes anualizada,       -> Selic_Media_Trimestral
           media trimestral
  - 24369  Desemprego (%), media trimestral        -> Desemprego_Trimestral

NOTA: requer acesso de rede a api.bcb.gov.br. Em ambientes com egresso
restrito a chamada falha e a serie e omitida do arquivo.
"""

import json
import os
import subprocess

import pandas as pd

# output_col: (codigo_sgs, agregacao_trimestral)
SERIES_MAP = {
    "Crescimento_PIB_Trimestral": (22109, "last"),
    "IPCA_Trimestral": (433, "compound"),
    "Spread_Medio_Trimestral": (20786, "mean"),
    "Selic_Media_Trimestral": (4189, "mean"),
    "Desemprego_Trimestral": (24369, "mean"),
}


def fetch_sgs_series(series_code, start_date="01/01/2009"):
    """Baixa uma serie do SGS/BCB via curl e retorna DataFrame (data, valor)."""
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
        f"?formato=json&dataInicial={start_date}"
    )
    try:
        print(f"Buscando serie {series_code} via curl...")
        res = subprocess.run(
            ["curl", "-s", "-L", "-A", "Mozilla/5.0", url],
            capture_output=True, text=True,
        )
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df["data"] = pd.to_datetime(df["data"], dayfirst=True)
                df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
                return df[["data", "valor"]]
        print(f"  Serie {series_code}: resposta vazia ou invalida.")
    except Exception as e:
        print(f"  Erro na serie {series_code}: {e}")
    return None


def _aggregate_quarterly(serie, agg):
    """Agrega uma serie mensal/diaria para o fim de trimestre (QE)."""
    s = serie.set_index("data")["valor"]
    if agg == "compound":  # taxas (% a.m.) compostas no trimestre
        return ((1 + s / 100).resample("QE").prod() - 1) * 100
    if agg == "last":
        return s.resample("QE").last()
    return s.resample("QE").mean()  # 'mean'


def coletar_indicadores_macro():
    print("=" * 60)
    print("COLETANDO INDICADORES MACRO (PIB, IPCA, SPREAD, SELIC, DESEMPREGO)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(base_dir, "dados", "brutos", "indicadores_macro_sgs.csv")

    cols = []
    for name, (code, agg) in SERIES_MAP.items():
        raw = fetch_sgs_series(code)
        if raw is not None:
            cols.append(_aggregate_quarterly(raw, agg).to_frame(name))
            print(f"  [OK] {name} (SGS {code}).")
        else:
            print(f"  [FALHA] {name} (SGS {code}) — omitida.")

    if not cols:
        print("Nenhuma serie macro foi baixada (verifique o acesso a api.bcb.gov.br).")
        return

    df_final = pd.concat(cols, axis=1).reset_index()
    df_final.rename(columns={"data": "Data"}, inplace=True)
    df_final["Data_Key"] = df_final["Data"].apply(lambda x: f"01/{x.month:02d}/{x.year}")
    df_final.drop(columns=["Data"], inplace=True)

    df_final.to_csv(output_path, sep=";", decimal=",", index=False, encoding="latin1")
    print(f"\n[FIM] Salvo em: {output_path}")
    print(f"Colunas: {df_final.columns.tolist()}")
    if "Selic_Media_Trimestral" in df_final.columns:
        s = df_final["Selic_Media_Trimestral"]
        print(f"Selic: {s.notna().sum()} trimestres, media {s.mean():.2f}% a.a.")


if __name__ == "__main__":
    coletar_indicadores_macro()
