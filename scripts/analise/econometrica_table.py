import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
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

    # Features derivadas (volatilidade do NPL + participacoes/porte de RWA)
    df = config.add_npl_volatility(df)
    df = config.add_rwa_features(df)

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

    # Padronizacao das variaveis principais e dummies de efeito fixo.
    # O logit FE com muitas dummies + evento raro separa perfeitamente, entao
    # usa-se regularizacao L2 (ridge), como no modelo principal. A penalidade
    # nao se aplica ao intercepto (primeira coluna).
    mean, std = df_fe[features_ext].mean(), df_fe[features_ext].std()

    def build_design(sub):
        Xs = (sub[features_ext] - mean) / std
        dums = pd.get_dummies(sub['Instituicao'], prefix='FE', drop_first=True)
        return sm.add_constant(pd.concat([Xs, dums], axis=1).astype(float), has_constant='add')

    def fit_fe_ridge(X, y):
        alpha = np.array([0.0] + [config.L2_ALPHA] * (X.shape[1] - 1))
        w = y.map({0: 1.0, 1: (y == 0).sum() / (y == 1).sum()})
        return sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w).fit_regularized(
            alpha=alpha, L1_wt=0.0
        )

    X = build_design(df_fe)
    y = df_fe['Target'].astype(float)
    res = fit_fe_ridge(X, y)

    # Pseudo R2 de McFadden com log-verossimilhanca ponderada
    w_full = y.map({0: 1.0, 1: (y == 0).sum() / (y == 1).sum()})
    p_hat = np.clip(np.asarray(res.predict(X)), 1e-12, 1 - 1e-12)
    p0 = np.average(y, weights=w_full)
    ll_model = np.sum(w_full * (y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))
    ll_null = np.sum(w_full * (y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    pseudo_r2 = 1 - ll_model / ll_null

    # Erros-padrao/p-valores das variaveis principais via bootstrap por instituicao.
    # As dummies mudam a cada reamostragem (nuisance); extraem-se apenas os
    # coeficientes das variaveis de interesse, sempre presentes.
    keep = ['const'] + features_ext
    groups = {i: g for i, g in df_fe.groupby('Instituicao')}
    insts = np.array(list(groups.keys()))
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(config.BOOTSTRAP_N):
        chosen = rng.choice(len(insts), size=len(insts), replace=True)
        sub = pd.concat([groups[insts[i]] for i in chosen], ignore_index=True)
        yb = sub['Target'].astype(float)
        if yb.nunique() < 2:
            continue
        try:
            rb = fit_fe_ridge(build_design(sub), yb)
            boot.append(rb.params.reindex(keep).values)
        except Exception:
            continue
    boot = np.array(boot, dtype=float)
    se_map = dict(zip(keep, np.nanstd(boot, axis=0, ddof=1)))

    # Extract only main variables for the table
    summary_data = []
    for f in keep:
        coef = res.params[f]
        std_err = se_map.get(f, np.nan)
        z = coef / std_err if std_err and not np.isnan(std_err) else 0.0
        p_val = 2 * (1 - norm.cdf(abs(z)))
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
    latex_out += f"Observações & {len(y)} \\\\\n"
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
