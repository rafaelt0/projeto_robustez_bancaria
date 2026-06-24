import pandas as pd
import numpy as np
import os

def calculate_risk(row, coefs, X_mean, X_std, pib_shock=0.0, spread_shock=0.0, capital_shock=1.0, desemp_shock=0.0):
    v = {
        'RWA_Credito_lag4': row.get('RWA_Credito', 0),
        'RWA_Mercado_lag4': row.get('RWA_Mercado', 0),
        'RWA_Operacional_lag4': row.get('RWA_Operacional', 0),
        'Capital_Principal_lag4': row.get('Capital_Principal', 0) * capital_shock,
        'Alavancagem_lag4': row.get('Alavancagem', 0),
        'PIB_lag4': row.get('PIB', 0) + pib_shock,
        'Spread_lag4': row.get('Spread', 0) + spread_shock,
        'Desemprego_lag4': row.get('Desemprego', 0) + desemp_shock
    }
    v['RWA_Operacional_lag4_x_Alavancagem_lag4'] = v['RWA_Operacional_lag4'] * v['Alavancagem_lag4']

    v_scaled = {}
    for k in v:
        mean_val = X_mean.get(k, 0)
        std_val = X_std.get(k, 1.0)
        val = v[k] if (pd.notnull(v[k]) and not np.isnan(v[k])) else mean_val
        v_scaled[k] = (val - mean_val) / (std_val if std_val != 0 else 1.0)

    log_odds = coefs['const']
    for k in v_scaled:
        log_odds += coefs.get(k, 0) * v_scaled[k]
    
    return 1 / (1 + np.exp(-np.clip(log_odds, -20, 20)))

def run_stress_testing():
    print("\n" + "="*80)
    print("ANALISE DE STRESS TESTING (PIB, SPREAD, CAPITAL, DESEMPREGO)")
    print("="*80)

    raw_path = 'dados/brutos/painel_final.csv'
    if not os.path.exists(raw_path): return

    df = pd.read_csv(raw_path)
    cols_num = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego']
    for c in cols_num: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values(['Instituicao', 'Data'], inplace=True)
    for c in ['PIB', 'Spread', 'Desemprego']: df[c] = df[c].ffill()
    
    last_date = df['Data'].max()
    df_base = df[df['Data'] == last_date].copy().dropna(subset=['Instituicao', 'RWA_Credito'])
    
    # 2. Parametros do Modelo (Atualizados com Desemprego)
    coefs = {
        'const': -21.4659, 'RWA_Credito_lag4': -79.0757, 'RWA_Mercado_lag4': 1.2326,
        'RWA_Operacional_lag4': 0.8887, 'Capital_Principal_lag4': -0.0783,
        'Alavancagem_lag4': -0.5946, 'PIB_lag4': 0.0917, 'Spread_lag4': 0.4041,
        'Desemprego_lag4': -0.0738, 'RWA_Operacional_lag4_x_Alavancagem_lag4': 2.6154
    }
    X_mean = {'RWA_Credito_lag4': 54705087.69, 'RWA_Mercado_lag4': 960173.80, 'RWA_Operacional_lag4': 6392813.71, 'Capital_Principal_lag4': 0.1935, 'Alavancagem_lag4': 0.1048, 'PIB_lag4': 173.45, 'Spread_lag4': 29.88, 'Desemprego_lag4': 11.13, 'RWA_Operacional_lag4_x_Alavancagem_lag4': 568912.40}
    X_std = {'RWA_Credito_lag4': 170617596.64, 'RWA_Mercado_lag4': 4352295.50, 'RWA_Operacional_lag4': 19697359.39, 'Capital_Principal_lag4': 0.1701, 'Alavancagem_lag4': 0.1241, 'PIB_lag4': 9.68, 'Spread_lag4': 5.30, 'Desemprego_lag4': 2.46, 'RWA_Operacional_lag4_x_Alavancagem_lag4': 1687865.77}

    # 3. Simular Cenários
    # Baseline
    df_base['Prob_Baseline'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std), axis=1)
    # Severo (-3% PIB, +2% Spread, -15% Capital, +2% Desemprego)
    df_base['Prob_Stress_Severo'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std, pib_shock=-3.0, spread_shock=2.0, capital_shock=0.85, desemp_shock=2.0), axis=1)
    # Crise Sistêmica (-6% PIB, +5% Spread, -25% Capital, +5% Desemprego)
    df_base['Prob_Stress_Interm'] = df_base.apply(lambda r: calculate_risk(r, coefs, X_mean, X_std, pib_shock=-6.0, spread_shock=5.0, capital_shock=0.75, desemp_shock=5.0), axis=1)
    
    df_base['Score_Baseline'] = -np.log(df_base['Prob_Baseline'] / (1 - df_base['Prob_Baseline'] + 1e-10))
    df_base['Score_Stress'] = -np.log(df_base['Prob_Stress_Interm'] / (1 - df_base['Prob_Stress_Interm'] + 1e-10))
    df_base['Queda_Resiliencia'] = df_base['Score_Stress'] - df_base['Score_Baseline']

    os.makedirs('resultados/relatorios', exist_ok=True)
    df_base.to_csv('resultados/relatorios/stress_test_results.csv', index=False)
    generate_latex_table(df_base)

def generate_latex_table(df):
    df_sorted = df.sort_values('Queda_Resiliencia')
    top_affected = df_sorted.head(15) 
    least_affected = df_sorted.tail(10).iloc[::-1]
    
    latex = """
% ==========================================================
% TABELA DE STRESS TESTING (CENARIOS COM DESEMPREGO)
% ==========================================================
\\begin{table}[htbp]
  \\centering
  \\caption{Analise de Sensibilidade (Cenarios: Severo vs Crise Sistêmica [+5\\% Desemprego])}
  \\label{tab:stress_test}
  \\begin{tabular}{lcccc}
    \\hline
    \\textbf{Instituição} & \\textbf{Baseline} & \\textbf{Severo} & \\textbf{Sistêmica} & \\textbf{Impacto} \\\\
    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel A: 15 Instituições Mais Vulneráveis (Menor Resiliência)}} \\\\
    \\hline
"""
    for _, row in top_affected.iterrows():
        inst = row['Instituicao'].replace('&', '\\&').replace('_', '\\_')[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"
    
    latex += """    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel B: 10 Instituições Mais Resilientes (Maior Estabilidade)}} \\\\
    \\hline
"""
    for _, row in least_affected.iterrows():
        inst = row['Instituicao'].replace('&', '\\&').replace('_', '\\_')[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"

    latex += "    \\hline\n  \\end{tabular}\n\\end{table}\n"
    with open('resultados/relatorios/tabela_stress_test.tex', 'w', encoding='utf-8') as f: f.write(latex)
    print("Tabela LaTeX de Stress Test gerada.")

if __name__ == "__main__":
    run_stress_testing()
