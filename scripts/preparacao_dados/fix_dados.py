import os
import pandas as pd
import glob
import re

# Configuration
DOWNLOADS_DIR = "c:/Users/WIN 10/.gemini/antigravity/scratch/downloads"
OUTPUT_FILE = "c:/Users/WIN 10/.gemini/antigravity/scratch/downloads/dados_com_data.csv"

# Account Codes - Resolution 2682 (2015-2024)
RISK_LEVELS = {
    'AA': '31100003',
    'A': '31200006',
    'B': '31300009',
    'C': '31400002',
    'D': '31500005',
    'E': '31600008',
    'F': '31700001',
    'G': '31800004',
    'H': '31900007'
}

# Account Codes - Resolution 4966 / IFRS 9 (2025+)
STAGES = {
    'S1': '3311000002',
    'S2': '3312000001',
    'S3': '3313000000'
}

def parse_saldo(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
        try:
            return float(val)
        except ValueError:
            pass
    return 0.0

def process_file(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r'(\d{4})(\d{2})', filename)
    if not match: return None
    
    year = int(match.group(1))
    month = match.group(2)
    # Format date as DD/MM/YYYY to match painel_final
    formatted_date = f"01/{month}/{year}"
    
    if year < 2025:
        target_dict = RISK_LEVELS
        cols_to_sum = ['E', 'F', 'G', 'H']
        total_cols = list(RISK_LEVELS.keys())
    else:
        target_dict = STAGES
        cols_to_sum = ['S3']
        total_cols = list(STAGES.keys())
    
    target_accounts = list(target_dict.values())

    try:
        df = pd.read_csv(filepath, sep=';', skiprows=3, encoding='latin1', on_bad_lines='skip', low_memory=False)
    except: return None
        
    df.columns = [c.strip().replace('#', '') for c in df.columns]
    
    if 'CONTA' not in df.columns:
        return None

    df['CONTA_STR'] = df['CONTA'].astype(str).str.strip()
    df_filtered = df[df['CONTA_STR'].isin(target_accounts)].copy()
    
    if df_filtered.empty:
        return None

    if 'SALDO' not in df_filtered.columns:
         return None
         
    df_filtered['SALDO_CLEAN'] = df_filtered['SALDO'].apply(parse_saldo)
    inv_map = {v: k for k, v in target_dict.items()}
    df_filtered['RISK_CAT'] = df_filtered['CONTA_STR'].map(inv_map)

    index_cols = ['CNPJ', 'NOME_INSTITUICAO', 'NOME_CONGL']
    available_index = [c for c in index_cols if c in df_filtered.columns]
    
    df_pivot = df_filtered.pivot_table(
        index=available_index,
        columns='RISK_CAT',
        values='SALDO_CLEAN',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Add the date column at the beginning
    df_pivot.insert(0, 'Data', formatted_date)

    # Ensure all columns exist
    all_cat_cols = list(RISK_LEVELS.keys()) + list(STAGES.keys())
    for col in all_cat_cols:
        if col not in df_pivot.columns: df_pivot[col] = 0.0

    df_pivot['Bad_Debt'] = df_pivot[[c for c in cols_to_sum if c in df_pivot.columns]].sum(axis=1)
    df_pivot['Total_Portfolio'] = df_pivot[[c for c in total_cols if c in df_pivot.columns]].sum(axis=1)
    df_pivot['NPL'] = 0.0
    mask = df_pivot['Total_Portfolio'] > 0
    df_pivot.loc[mask, 'NPL'] = df_pivot.loc[mask, 'Bad_Debt'] / df_pivot.loc[mask, 'Total_Portfolio']
    
    return df_pivot

def main():
    csv_files = sorted(list(set(glob.glob(os.path.join(DOWNLOADS_DIR, "*BLOPRUDENCIAL.CSV")) + glob.glob(os.path.join(DOWNLOADS_DIR, "*BLOPRUDENCIAL.csv")))))
    all_dfs = [df for f in csv_files if (df := process_file(f)) is not None]
             
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Reorder columns to be more intuitive: Date, Identity, then data
        cols = ['Data', 'CNPJ', 'NOME_INSTITUICAO', 'NOME_CONGL'] + \
               [c for c in final_df.columns if c not in ['Data', 'CNPJ', 'NOME_INSTITUICAO', 'NOME_CONGL', 'Bad_Debt', 'Total_Portfolio', 'NPL']] + \
               ['Bad_Debt', 'Total_Portfolio', 'NPL']
        
        final_df = final_df[cols]
        final_df.to_csv(OUTPUT_FILE, index=False, sep=';', decimal=',')
        print(f"Processed {len(all_dfs)} files. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
