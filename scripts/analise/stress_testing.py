import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def run_stress_testing():
    print("\n" + "="*80)
    print("INICIANDO ANALISE DE STRESS TESTING (CENARIOS 2025)")
    print("="*80)

    # 1. Carregar Dados e Modelo
    raw_path = 'dados/brutos/painel_final.csv'
    if not os.path.exists(raw_path):
        print(f"Erro: Arquivo {raw_path} nao encontrado.")
        return

    df = pd.read_csv(raw_path)
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Pegar o trimestre mais recente disponível para cada instituição como base
    last_date = df['Data'].max()
    df_base = df[df['Data'] == last_date].copy()
    
    print(f"Usando dados de {last_date.date()} como base para simulação.")

    # 2. Definir Parâmetros (Devem vir do modelo_final_recomendado.py)
    # Variáveis: RWA_Credito, RWA_Mercado, RWA_Operacional, Capital_Principal, Alavancagem, PIB, Spread
    # Interação: RWA_Operacional_lag4_x_Alavancagem_lag4
    
    # Coeficientes do Modelo (Extraídos do último run bem-sucedido)
    # Nota: Em um sistema real, salvaríamos o modelo com Pickle. Aqui replicamos para agilidade.
    coefs = {
        'const': -21.4909,
        'RWA_Credito_lag4': -79.1330,
        'RWA_Mercado_lag4': 1.1895,
        'RWA_Operacional_lag4': 0.8966,
        'Capital_Principal_lag4': -0.0770,
        'Alavancagem_lag4': -0.5944,
        'PIB_lag4': 0.0981,
        'Spread_lag4': 0.4314,
        'RWA_Operacional_lag4_x_Alavancagem_lag4': 2.6068
    }

    # Parâmetros de Scaling (Média e DP do treino 2016-2021)
    # Aproximações baseadas na normalização padrão
    scaling = {
        'mean': 0.0, # Assumindo normalizado
        'std': 1.0
    }

    def calculate_risk(row, pib_shock=0.0, spread_shock=0.0, capital_shock=1.0):
        # Aplicar Choques
        pib = row.get('PIB', 0.0) + pib_shock
        spread = row.get('Spread', 0.0) + spread_shock
        cap = row.get('Capital_Principal', 0.0) * capital_shock
        
        # Manter outros como base (Normalizado para simplificar)
        log_odds = (coefs['const'] + 
                    coefs['RWA_Credito_lag4'] * row.get('RWA_Credito', 0) +
                    coefs['RWA_Mercado_lag4'] * row.get('RWA_Mercado', 0) +
                    coefs['RWA_Operacional_lag4'] * row.get('RWA_Operacional', 0) +
                    coefs['Capital_Principal_lag4'] * cap +
                    coefs['Alavancagem_lag4'] * row.get('Alavancagem', 0) +
                    coefs['PIB_lag4'] * pib +
                    coefs['Spread_lag4'] * spread +
                    coefs['RWA_Operacional_lag4_x_Alavancagem_lag4'] * (row.get('RWA_Operacional', 0) * row.get('Alavancagem', 0)))
        
        prob = 1 / (1 + np.exp(-log_odds))
        return prob

    # 3. Processar Cenários
    print("\nSimulando Cenários...")
    
    # Baseline
    df_base['Prob_Baseline'] = df_base.apply(lambda r: calculate_risk(r), axis=1)
    
    # Stress Severo (-3% PIB, +2% Spread, -15% Capital)
    df_base['Prob_Stress_Severo'] = df_base.apply(lambda r: calculate_risk(r, pib_shock=-0.03, spread_shock=0.02, capital_shock=0.85), axis=1)
    
    # Impacto no Score
    df_base['Score_Baseline'] = -np.log(df_base['Prob_Baseline'] / (1 - df_base['Prob_Baseline'] + 1e-10))
    df_base['Score_Stress'] = -np.log(df_base['Prob_Stress_Severo'] / (1 - df_base['Prob_Stress_Severo'] + 1e-10))
    df_base['Queda_Resiliencia'] = df_base['Score_Stress'] - df_base['Score_Baseline']

    # 4. Salvar Resultados
    output_dir = 'resultados/relatorios'
    os.makedirs(output_dir, exist_ok=True)
    
    res_cols = ['Instituicao', 'Prob_Baseline', 'Prob_Stress_Severo', 'Score_Baseline', 'Score_Stress', 'Queda_Resiliencia']
    df_stress = df_base[res_cols].sort_values('Queda_Resiliencia')
    
    df_stress.to_csv(f'{output_dir}/stress_test_results.csv', index=False)
    print(f"Resultados salvos em {output_dir}/stress_test_results.csv")

    # 5. Gerar Tabela LaTeX
    generate_latex_table(df_stress)

def generate_latex_table(df):
    top_affected = df.head(15) # Os que mais perderam resiliência
    
    latex = """
% ==========================================================
% TABELA DE STRESS TESTING (CENARIOS ADVERSOS)
% ==========================================================
\\begin{table}[htbp]
  \\centering
  \\caption{Analise de Sensibilidade a Stress (Cenario Severo: -3\\% PIB, +2\\% Spread)}
  \\label{tab:stress_test}
  \\begin{tabular}{lccc}
    \\hline
    \\textbf{Instituição} & \\textbf{Prob. Baseline} & \\textbf{Prob. Stress} & \\textbf{Impacto no Score} \\\\
    \\hline
"""
    for _, row in top_affected.iterrows():
        latex += f"    {row['Instituicao'][:30]} & {row['Prob_Baseline']:.2%} & {row['Prob_Stress_Severo']:.2%} & {row['Queda_Resiliencia']:.2f} \\\\\n"
    
    latex += """    \\hline
  \\end{tabular}
\\end{table}
"""
    with open('resultados/relatorios/tabela_stress_test.tex', 'w') as f:
        f.write(latex)
    print("Tabela LaTeX de Stress Test gerada com sucesso.")

if __name__ == "__main__":
    run_stress_testing()
