import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import os

# Paths
WORKSPACE_DIR = Path(".")
DATA_PATH = WORKSPACE_DIR / "dados" / "brutos" / "painel_final.csv"

def generate_econometrica_table():
    print("="*60)
    print("GERANDO TABELA ECONOMETRICA (FE LOGIT) COM DESEMPREGO")
    print("="*60)

    if not DATA_PATH.exists():
        print(f"Erro: {DATA_PATH} nao encontrado.")
        return

    # 1. Carregar Dados
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'])
    
    # 2. Preenchimento de Faltantes (Macro)
    # No painel_final, as colunas ja sao PIB, IPCA, Spread, Desemprego
    for col in ['Desemprego', 'PIB', 'Spread']:
        if col in df.columns:
            df[col] = df.groupby('Instituicao')[col].ffill().bfill()
            
    # Volatilidade do NPL (Proxy de Risco dinamico)
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(8, 4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())
    
    # 3. Lags
    LAG = 4
    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'NPL_Volatility_8Q']
    df_model = df.copy()
    features = []
    for f in core_features:
        name = f'{f}_lag{LAG}'
        df_model[name] = df_model.groupby('Instituicao')[f].shift(LAG)
        features.append(name)
    
    # Termo de Interação
    df_model['RWA_Operacional_lag4_x_Alavancagem_lag4'] = df_model['RWA_Operacional_lag4'] * df_model['Alavancagem_lag4']
    features_ext = features + ['RWA_Operacional_lag4_x_Alavancagem_lag4']
    
    # Target
    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Target'] = (df_model['NPL'] > threshold_p90).astype(int)
    
    # 4. Preparar para Efeitos Fixos (Instituições com variação no Target)
    inst_variance = df_model.groupby('Instituicao')['Target'].std()
    eligible = inst_variance[inst_variance > 0].index
    df_fe = df_model[df_model['Instituicao'].isin(eligible)].dropna(subset=features_ext + ['Target']).copy()
    
    if df_fe.empty:
        print("Erro: Nenhum dado elegivel para Efeitos Fixos.")
        return

    # Regression com Pesos (Balanceamento)
    dummies = pd.get_dummies(df_fe['Instituicao'], prefix='FE', drop_first=True)
    X_main = df_fe[features_ext]
    X_scaled = (X_main - X_main.mean()) / X_main.std()
    X = pd.concat([X_scaled, dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y = df_fe['Target'].astype(float)
    
    counts = y.value_counts()
    weight_stress = counts[0] / counts[1]
    weights = y.apply(lambda x: weight_stress if x == 1 else 1.0)
    
    res = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights).fit()
    
    null_res = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), var_weights=weights).fit()
    pseudo_r2 = 1 - (res.llf / null_res.llf)
    
    # Extract only main variables for the table
    summary_data = []
    for f in ['const'] + features_ext:
        coef = res.params[f]
        std_err = res.bse[f]
        p_val = res.pvalues[f]
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        display_name = f.replace('_lag4', '').replace('_', ' ')
        summary_data.append([display_name, f"{coef:.4f}{stars}", f"({std_err:.4f})"])

    # 5. Gerar LaTeX
    latex_out = r"""\begin{table}[htbp]
\centering
\caption{Determinantes do Estresse Bancário (Logit com Efeitos Fixos e Desemprego)}
\label{tab:fe_logit_v2}
\begin{tabular}{lc}
\toprule
\textbf{Variável} & \textbf{Coeficiente} \\
                  & \textbf{(Erro Padrão)} \\
\midrule
"""
    for row in summary_data:
        var_name = row[0].replace('_', '\\_')
        latex_out += f"{var_name} & {row[1]} \\\\\n & {row[2]} \\\\\n\\addlinespace\n"

    latex_out += "\\midrule\n"
    latex_out += f"Observações & {int(res.nobs)} \\\\\n"
    latex_out += f"Pseudo $R^2$ & {pseudo_r2:.4f} \\\\\n"
    latex_out += f"Número de Inst. & {len(eligible)} \\\\\n"
    latex_out += "Efeitos Fixos & SIM \\\\\n"
    latex_out += "\\bottomrule\n"
    latex_out += "\\multicolumn{2}{l}{\\small Notas: *** p<0.01, ** p<0.05, * p<0.1.} \\\\\n"
    latex_out += "\\end{tabular}\n\\end{table}\n"

    output_path = WORKSPACE_DIR / "resultados" / "relatorios" / "tabela_fe_econometrica.tex"
    with open(output_path, "w", encoding="utf-8") as f: f.write(latex_out)
    print(f"✅ Tabela Econométrica gerada: {output_path}")

if __name__ == "__main__":
    generate_econometrica_table()
