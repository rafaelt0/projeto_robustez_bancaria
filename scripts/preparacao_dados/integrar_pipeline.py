import pandas as pd
import os
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_DIR = WORKSPACE_DIR / "data"
CAPITAL_FILE = Path(r"D:\downloads_bcb\painel_bcb_capital.csv")
NPL_FILE = DATA_DIR / "processed" / "dados.csv"
MACRO_FILE = DATA_DIR / "raw" / "indicadores_macro_sgs.csv"
OUTPUT_FILE = DATA_DIR / "raw" / "painel_final.csv"

def parse_percent(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        val = val.replace('%', '').replace('.', '').replace(',', '.')
        try:
            return float(val) / 100.0
        except:
            return 0.0
    return 0.0

def parse_number(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
        try:
            return float(val)
        except:
            return 0.0
    return 0.0

def integrate():
    print("Iniciando integracao do painel final...")
    
    # 1. Carregar Capital e RWA
    print("Lendo dados de Capital...")
    df_cap = pd.read_csv(CAPITAL_FILE, low_memory=False)
    
    # Remover colunas duplicadas (comum em arquivos do BCB achatados)
    df_cap = df_cap.loc[:, ~df_cap.columns.duplicated()]
    
    # Identificar colunas candidatas (pode haver várias devido a mudança de fórmulas e MultiIndex)
    cap_candidates = [c for c in df_cap.columns if "capital principal" in c.lower() and "(" in c and "/" in c]
    alav_candidates = [c for c in df_cap.columns if "alavancagem" in c.lower()]
    
    # RWA e outros
    rwa_cred_col = next((c for c in df_cap.columns if "risco de cr" in c.lower()), None)
    rwa_merc_col = next((c for c in df_cap.columns if "risco de mercado" in c.lower()), None)
    rwa_oper_col = next((c for c in df_cap.columns if "risco operacional" in c.lower()), None)
    inst_col = next((c for c in df_cap.columns if "institui" in c.lower()), None)
    data_col = next((c for c in df_cap.columns if "data" in c.lower()), None)

    print(f"Colunas Capital detectadas: {cap_candidates}")
    
    # Criar DataFrame limpo
    df_clean_cap = pd.DataFrame()
    df_clean_cap['Instituicao'] = df_cap[inst_col].astype(str).str.strip().str.upper()
    df_clean_cap['Data'] = pd.to_datetime(df_cap[data_col], errors='coerce')
    
    # Coalesce Capital Principal
    def get_best_value(row, candidates, parser):
        for cand in candidates:
            val = parser(row[cand])
            if val != 0: return val
        return 0.0

    print("Processando Capital Principal (Coalesce)...")
    df_clean_cap['Capital_Principal'] = df_cap.apply(lambda r: get_best_value(r, cap_candidates, parse_percent), axis=1)
    
    print("Processando Alavancagem (Coalesce)...")
    df_clean_cap['Alavancagem'] = df_cap.apply(lambda r: get_best_value(r, alav_candidates, parse_percent), axis=1)
    
    print("Processando RWA...")
    if rwa_cred_col: df_clean_cap['RWA_Credito'] = df_cap[rwa_cred_col].apply(parse_number)
    else: df_clean_cap['RWA_Credito'] = 0.0
    
    if rwa_merc_col: df_clean_cap['RWA_Mercado'] = df_cap[rwa_merc_col].apply(parse_number)
    else: df_clean_cap['RWA_Mercado'] = 0.0
    
    if rwa_oper_col: df_clean_cap['RWA_Operacional'] = df_cap[rwa_oper_col].apply(parse_number)
    else: df_clean_cap['RWA_Operacional'] = 0.0

    df_cap = df_clean_cap.dropna(subset=['Data'])

    # 2. Carregar NPL
    print("Lendo dados de NPL...")
    df_npl = pd.read_csv(NPL_FILE, sep=';', decimal=',', low_memory=False)
    df_npl['Data'] = pd.to_datetime(df_npl['Data'], format='%d/%m/%Y', errors='coerce')
    df_npl = df_npl.rename(columns={'NOME_CONGL': 'Instituicao'})
    df_npl['Instituicao'] = df_npl['Instituicao'].astype(str).str.strip().str.upper()
    
    # 3. Merge Bancário
    print("Realizando merge de dados bancarios...")
    # Padronizar nomes para merge (remover sufixos, etc se necessario, mas aqui usaremos join aproximado se falhar)
    # Por enquanto, merge exato por Instituicao e Data
    df_bank = pd.merge(df_cap, df_npl[['Instituicao', 'Data', 'NPL']], on=['Instituicao', 'Data'], how='inner')
    
    # 4. Carregar Macro
    print("Lendo indicadores macroeconomicos...")
    df_macro = pd.read_csv(MACRO_FILE, sep=';', decimal=',', low_memory=False)
    df_macro['Data'] = pd.to_datetime(df_macro['Data_Key'], format='%d/%m/%Y', errors='coerce')
    df_macro = df_macro.rename(columns={
        'Crescimento_PIB_Trimestral': 'PIB',
        'IPCA_Trimestral': 'IPCA',
        'Spread_Medio_Trimestral': 'Spread'
    })
    
    # 5. Merge Final
    # O df_bank é mensal, o macro é trimestral. Precisamos expandir o macro para todos os meses
    df_macro = df_macro.set_index('Data').resample('MS').ffill().reset_index()
    
    print("Realizando merge final com indicadores macro...")
    df_final = pd.merge(df_bank, df_macro[['Data', 'PIB', 'IPCA', 'Spread']], on='Data', how='left')
    
    # 6. Salvar
    cols_to_keep = ['Instituicao', 'Data', 'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 
                    'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'NPL']
    
    # Garantir que todas as colunas existem
    for c in cols_to_keep:
        if c not in df_final.columns: df_final[c] = 0.0
        
    df_final = df_final[cols_to_keep].dropna(subset=['NPL'])
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Sucesso! Painel final salvo em {OUTPUT_FILE}")
    print(f"Shape final: {df_final.shape}")

    # 7. Stress Testing
    try:
        from scripts.analysis.stress_testing import run_stress_test
        run_stress_test()
    except Exception as e:
        print(f"Erro ao executar Stress Test: {e}")

if __name__ == "__main__":
    integrate()
