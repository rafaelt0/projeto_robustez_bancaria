import pandas as pd
import glob
import os
import re
from pathlib import Path

DATA_PATH = r"D:\downloads_bcb"


# ============================================================
# Extrair data do nome do arquivo (bcb_MM_AAAA.csv)
# ============================================================
def extract_date_from_filename(filename):
    match = re.search(r"bcb_(\d{2})_(\d{4})", filename)
    if match:
        month = match.group(1)
        year = match.group(2)
        return pd.to_datetime(f"{year}-{month}-01")
    return None


# ============================================================
# Flatten MultiIndex e garantir colunas únicas
# ============================================================
def flatten_columns(df):

    if isinstance(df.columns, pd.MultiIndex):

        new_cols = []

        for col_tuple in df.columns:

            parts = [
                str(c).strip()
                for c in col_tuple
                if str(c).strip() != "" and "Unnamed" not in str(c)
            ]

            col_name = "_".join(parts)
            new_cols.append(col_name)

        df.columns = new_cols

    # remover espaços
    df.columns = df.columns.str.strip()

    # garantir unicidade
    counts = {}
    unique_cols = []

    for col in df.columns:
        if col in counts:
            counts[col] += 1
            unique_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            unique_cols.append(col)

    df.columns = unique_cols

    return df


# ============================================================
# Build Panel
# ============================================================
def build_panel():

    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    print(f"Total de arquivos encontrados: {len(files)}")

    dfs = []

    for file in files:
        try:
            print(f"Lendo: {os.path.basename(file)}")

            df = pd.read_csv(
                file,
                sep=";",
                decimal=".",
                header=[0,1]
            )

            df = flatten_columns(df)

            # remover duplicadas internas
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reset_index(drop=True)

            df["Data"] = extract_date_from_filename(file)

            dfs.append(df)

        except Exception as e:
            print(f"Erro em {file}: {e}")

    if not dfs:
        raise ValueError("Nenhum arquivo válido encontrado.")

    # concat protegido
    panel_df = pd.concat(dfs, ignore_index=True, sort=False, copy=False)
    panel_df = panel_df.loc[:, ~panel_df.columns.duplicated()]

    # ========================================================
    # Selecionar apenas colunas desejadas
    # ========================================================

    colunas_interesse = []

    for col in panel_df.columns:

        if (
            "Instituição" in col
            or "Data" in col
            or "RWA para Risco de Crédito" in col
            or "RWA para Risco de Mercado" in col
            or "RWA para Risco Operacional" in col
            or "Índice de Capital Principal" in col
            or "Razão de Alavancagem" in col
        ):
            colunas_interesse.append(col)

    # adicionar Data
    if "Data" not in colunas_interesse:
        colunas_interesse.append("Data")

    panel_df = panel_df[colunas_interesse]

    # ========================================================
    # Filtrar bancos desejados
    # ========================================================

    codigos_filtrados = [
        1000080099, 1000080329, 1000080738, 1000080075,
        1000080185, 1000081847, 1000080336, 1000082475,
        1000084693, 1000080109, 1000080745, 1000080879,
        1000080192, 1000080154, 1000080484, 1000084844
    ]

    # Tentamos encontrar a coluna Código ou similar
    codigo_col = next((c for c in panel_df.columns if "Código" in c or "Cdigo" in c), None)

    if codigo_col:
        panel_df = panel_df[panel_df[codigo_col].isin(codigos_filtrados)]
        panel_df = panel_df.sort_values(by=[codigo_col, "Data"]).reset_index(drop=True)

    return panel_df


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    panel_df = build_panel()

    print("\nPainel criado com sucesso!")
    print("Shape:", panel_df.shape)
    print(panel_df.head())

    output_path = Path(DATA_PATH) / "painel_bcb_capital.csv"
    panel_df.to_csv(output_path, index=False)

    print(f"\nArquivo salvo em: {output_path}")
