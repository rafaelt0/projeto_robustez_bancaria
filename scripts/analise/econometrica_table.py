import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utilitarios"))
import config

# Paths
DATA_PATH = config.RAW_PANEL

def generate_econometrica_table():
    print("="*60)
    print("GERANDO TABELA ECONOMETRICA (FE LOGIT) - MESMA ESPECIFICACAO DO MODELO FINAL")
    print("="*60)

    if not DATA_PATH.exists():
        print(f"Erro: {DATA_PATH} nao encontrado.")
        return

    # 1. Carregar Dados
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    # 2. Preenchimento de Faltantes (Macro)
    for col in ['Desemprego', 'PIB', 'Spread']:
        if col in df.columns:
            df[col] = df.groupby('Instituicao')[col].ffill().bfill()

    # Volatilidade do NPL (proxy de risco dinamico) via config
    df = config.add_npl_volatility(df)

    # 3. Lags e interacao (mesma especificacao do modelo final)
    df_model, features_ext = config.build_lagged_features(df.copy())

    # Target P90
    threshold_p90 = df_model['NPL'].quantile(config.P90_QUANTILE)
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
        display_name = f.replace(f'_lag{config.LAG}', '').replace('_', ' ')
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

    output_path = config.REPORTS_DIR / "tabela_fe_econometrica.tex"
    with open(output_path, "w", encoding="utf-8") as f: f.write(latex_out)
    print(f"✅ Tabela Econométrica gerada: {output_path}")

if __name__ == "__main__":
    generate_econometrica_table()
