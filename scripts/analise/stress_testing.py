import pandas as pd
import numpy as np
import os

def calculate_risk(row, coefs, X_mean, X_std, pib_shock=0.0, spread_shock=0.0, capital_shock=1.0):
    v = {
        'RWA_Credito_lag4': row.get('RWA_Credito', 0),
        'RWA_Mercado_lag4': row.get('RWA_Mercado', 0),
        'RWA_Operacional_lag4': row.get('RWA_Operacional', 0),
        'Capital_Principal_lag4': row.get('Capital_Principal', 0) * capital_shock,
        'Alavancagem_lag4': row.get('Alavancagem', 0),
        'PIB_lag4': row.get('PIB', 0) + pib_shock,
        'Spread_lag4': row.get('Spread', 0) + spread_shock
    }
    v['RWA_Operacional_lag4_x_Alavancagem_lag4'] = v['RWA_Operacional_lag4'] * v['Alavancagem_lag4']

    v_scaled = {}
    for k in v:
        val = v[k] if (pd.notnull(v[k]) and not np.isnan(v[k])) else X_mean[k]
        v_scaled[k] = (val - X_mean[k]) / (X_std[k] if X_std[k] != 0 else 1.0)

    log_odds = coefs['const']
    for k in v_scaled:
        log_odds += coefs.get(k, 0) * v_scaled[k]
    
    return 1 / (1 + np.exp(-np.clip(log_odds, -20, 20)))

def run_stress_testing():
    print("\n" + "="*80)
    print("INICIANDO ANALISE DE STRESS TESTING (CENARIOS 2025)")
    print("="*80)

    # 1. Carregar Dados
    raw_path = 'dados/brutos/painel_final.csv'
    if not os.path.exists(raw_path):
        print(f"Erro: Arquivo {raw_path} nao encontrado.")
        return

    df = pd.read_csv(raw_path)
    cols_num = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread']
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values(['Instituicao', 'Data'], inplace=True)
    df['PIB'] = df['PIB'].ffill()
    df['Spread'] = df['Spread'].ffill()
    
    last_date = df['Data'].max()
    df_base = df[df['Data'] == last_date].copy().dropna(subset=['Instituicao', 'RWA_Credito'])
    
    print(f"Usando dados de {last_date.date()} como base para simulação ({len(df_base)} instituições).")

    # 2. Parametros do Modelo
    coefs = {
        'const': -21.4909, 'RWA_Credito_lag4': -79.1330, 'RWA_Mercado_lag4': 1.1895,
        'RWA_Operacional_lag4': 0.8966, 'Capital_Principal_lag4': -0.0770,
        'Alavancagem_lag4': -0.5944, 'PIB_lag4': 0.0981, 'Spread_lag4': 0.4314,
        'RWA_Operacional_lag4_x_Alavancagem_lag4': 2.6068
    }
    X_mean = {
        'RWA_Credito_lag4': 47893019.866, 'RWA_Mercado_lag4': 1062413.580, 
        'RWA_Operacional_lag4': 5042233.430, 'Capital_Principal_lag4': 0.1910,
        'Alavancagem_lag4': 0.1148, 'PIB_lag4': 166.577, 'Spread_lag4': 31.919, 
        'RWA_Operacional_lag4_x_Alavancagem_lag4': 460144.633
    }
    X_std = {
        'RWA_Credito_lag4': 147903230.94, 'RWA_Mercado_lag4': 4762447.64, 
        'RWA_Operacional_lag4': 14801409.28, 'Capital_Principal_lag4': 0.1417,
        'Alavancagem_lag4': 0.1307, 'PIB_lag4': 4.111, 'Spread_lag4': 5.629, 
        'RWA_Operacional_lag4_x_Alavancagem_lag4': 1258094.49
    }

    # 3. Simular Cenários
    print("Simulando Cenários...")
    
    # Baseline
    df_base['Prob_Baseline'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std), axis=1)
    
    # Severo (-3% PIB, +2% Spread, -15% Capital)
    df_base['Prob_Stress_Severo'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std, pib_shock=-3.0, spread_shock=2.0, capital_shock=0.85), axis=1)
    
    # Intermediario/Crise Sistêmica (-6% PIB, +5% Spread, -25% Capital)
    df_base['Prob_Stress_Interm'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std, pib_shock=-6.0, spread_shock=5.0, capital_shock=0.75), axis=1)
    
    # Cálculo de Queda de Resiliência (Impacto no Score no cenário mais forte)
    df_base['Score_Baseline'] = -np.log(df_base['Prob_Baseline'] / (1 - df_base['Prob_Baseline'] + 1e-10))
    df_base['Score_Stress'] = -np.log(df_base['Prob_Stress_Interm'] / (1 - df_base['Prob_Stress_Interm'] + 1e-10))
    df_base['Queda_Resiliencia'] = df_base['Score_Stress'] - df_base['Score_Baseline']

    # 4. Salvar Resultados
    output_dir = 'resultados/relatorios'
    os.makedirs(output_dir, exist_ok=True)
    df_base.to_csv(f'{output_dir}/stress_test_results.csv', index=False)
    
    # 5. Gerar Tabela LaTeX
    generate_latex_table(df_base)

def generate_latex_table(df):
    df_sorted = df.sort_values('Queda_Resiliencia')
    top_affected = df_sorted.head(15) 
    least_affected = df_sorted.tail(10).iloc[::-1]
    
    latex = """
% ==========================================================
% TABELA DE STRESS TESTING (CENARIOS ADVERSOS)
% ==========================================================
\\begin{table}[htbp]
  \\centering
  \\caption{Analise de Sensibilidade (Cenarios: Severo [-3\\% PIB] vs Intermed. [-6\\% PIB])}
  \\label{tab:stress_test}
  \\begin{tabular}{lcccc}
    \\hline
    \\textbf{Instituição} & \\textbf{Baseline} & \\textbf{Severo} & \\textbf{Intermed.} & \\textbf{Impacto} \\\\
    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel A: 15 Instituições Mais Vulneráveis (Queda no Score)}} \\\\
    \\hline
"""
    for _, row in top_affected.iterrows():
        inst = row['Instituicao'].replace('&', '\\&').replace('_', '\\_')[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"
    
    latex += """    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel B: 10 Instituições Mais Resilientes (Menor Impacto)}} \\\\
    \\hline
"""
    for _, row in least_affected.iterrows():
        inst = row['Instituicao'].replace('&', '\\&').replace('_', '\\_')[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"

    latex += """    \\hline
  \\end{tabular}
\\end{table}
"""
    with open('resultados/relatorios/tabela_stress_test.tex', 'w', encoding='utf-8') as f:
        f.write(latex)
    print("Tabela LaTeX de Stress Test gerada com sucesso.")

if __name__ == "__main__":
    run_stress_testing()
