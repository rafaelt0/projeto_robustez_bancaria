import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_PATH = WORKSPACE_DIR / "data" / "raw" / "painel_final.csv"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "stress_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT_PATH = OUTPUT_DIR / "hausman_results.tex"

# Macro series (fallback if needed, but should be in data)
macro_data = {
    'Data': ['2015-12-01', '2016-03-01', '2016-06-01', '2016-09-01', '2016-12-01',
             '2017-03-01', '2017-06-01', '2017-09-01', '2017-12-01',
             '2018-03-01', '2018-06-01', '2018-09-01', '2018-12-01',
             '2019-03-01', '2019-06-01', '2019-09-01', '2019-12-01',
             '2020-03-01', '2020-06-01', '2020-09-01', '2020-12-01',
             '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01',
             '2022-03-01', '2022-06-01', '2022-09-01', '2022-12-01',
             '2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01',
             '2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01',
             '2025-03-01', '2025-06-01', '2025-09-01', '2025-12-01'],
    'Desemprego': [9.0, 10.9, 11.3, 11.8, 12.0, 13.7, 13.0, 12.4, 11.8, 13.1, 12.4, 11.9, 11.6, 12.7, 12.0, 11.8, 11.0, 12.2, 13.3, 14.6, 13.9, 14.7, 14.1, 12.6, 11.1, 11.1, 9.3, 8.7, 7.9, 8.8, 8.0, 7.7, 7.4, 7.9, 6.9, 6.4, 6.2, 6.6, 5.6, 5.6, 5.1],
    'Selic': [14.25, 14.25, 14.25, 14.25, 13.75, 12.25, 10.25, 8.25, 7.00, 6.50, 6.50, 6.50, 6.50, 6.50, 6.50, 5.50, 4.50, 3.75, 2.25, 2.00, 2.00, 2.75, 4.25, 5.75, 9.25, 11.75, 13.25, 13.75, 13.75, 13.75, 13.75, 12.75, 11.75, 10.75, 10.50, 10.50, 11.25, 13.25, 14.25, 15.00, 15.00]
}

def load_and_prep_data():
    print("Loading and preparing data...")
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Merge Macro and Fill
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    
    # Merge if not present or just to be safe
    # Check if 'Desemprego' is in df, if not merge
    if 'Desemprego' not in df.columns:
        df = df.merge(df_macro, on='Data', how='left')
    
    # Fill macro vars per institution
    for col in ['Desemprego', 'Selic', 'PIB', 'Spread']:
        if col in df.columns:
            df[col] = df.groupby('Instituicao')[col].ffill().bfill()
            
    # Add Volatility
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    # Setup Regression Components
    LAG = 1
    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    
    df_model = df.copy()
    lagged_features = []
    
    # Create Panel structure
    df_model['Time'] = df_model.groupby('Instituicao').cumcount()
    
    for f in core_features:
        col_name = f'{f}_lag{LAG}'
        df_model[col_name] = df_model.groupby('Instituicao')[f].shift(LAG)
        lagged_features.append(col_name)
    
    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Target'] = (df_model['NPL'] > threshold_p90).astype(int)
    
    inter_col = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'
    df_model[inter_col] = df_model[f'RWA_Operacional_lag{LAG}'] * df_model[f'Alavancagem_lag{LAG}']
    
    features = lagged_features + [inter_col]
    
    # Drop NaNs
    df_clean = df_model.dropna(subset=features + ['Target', 'Instituicao']).copy()
    
    # Filter for FE feasibility
    inst_variance = df_clean.groupby('Instituicao')['Target'].std()
    eligible_inst = inst_variance[inst_variance > 0].index.tolist()
    df_fe = df_clean[df_clean['Instituicao'].isin(eligible_inst)].copy()
    
    return df_fe, features, 'Target'

def hausman_test(beta_fe, cov_fe, beta_re, cov_re):
    """
    Compute Hausman Test Statistic
    H = (b_fe - b_re)' * inv(V_fe - V_re) * (b_fe - b_re)
    df = rank(V_fe - V_re)
    """
    # Ensure alignment
    common_vars = [v for v in beta_fe.index if v in beta_re.index and v != 'const'] # Hausman usually excludes intercept
    
    b_diff = beta_fe[common_vars] - beta_re[common_vars]
    v_diff = cov_fe.loc[common_vars, common_vars] - cov_re.loc[common_vars, common_vars]
    
    # Compute statistic
    try:
        chi2_stat = b_diff.T @ np.linalg.inv(v_diff) @ b_diff
        df = len(common_vars)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return chi2_stat, df, p_value
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix in Hausman test. Using pseudo-inverse.")
        chi2_stat = b_diff.T @ np.linalg.pinv(v_diff) @ b_diff
        df = len(common_vars)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return chi2_stat, df, p_value

def generate_latex_table(fe_model, hausman_results, feature_names):
    """
    Generate LaTeX table with Fixed Effects results and Hausman statistic.
    """
    chi2, df, p_val = hausman_results
    
    # Mapping for nice names
    name_map = {
        'RWA_Credito_lag1': 'RWA Credit (t-1)',
        'RWA_Mercado_lag1': 'RWA Market (t-1)',
        'RWA_Operacional_lag1': 'RWA Operational (t-1)',
        'Capital_Principal_lag1': 'Tier 1 Capital (t-1)',
        'Alavancagem_lag1': 'Leverage (t-1)',
        'PIB_lag1': 'GDP Growth (t-1)',
        'Spread_lag1': 'Spread (t-1)',
        'Desemprego_lag1': 'Unemployment (t-1)',
        'Selic_lag1': 'Selic Rate (t-1)',
        'NPL_Volatility_8Q_lag1': 'NPL Volatility (8Q lag)',
        'RWA_Operacional_lag1_x_Alavancagem_lag1': 'Op. Risk x Leverage'
    }
    
    latex_str = r"""
\begin{table}[ht]
\centering
\caption{Fixed Effects Logit Model Results and Hausman Specification Test}
\label{tab:fe_logit_hausman}
\begin{tabular}{l c c}
\hline
\textbf{Variable} & \textbf{Coefficient} & \textbf{Std. Error} \\
\hline
"""
    
    params = fe_model.params
    bse = fe_model.bse
    pvalues = fe_model.pvalues
    
    for var in feature_names:
        if var in params:
            nice_name = name_map.get(var, var.replace('_', ' '))
            coef = params[var]
            std_err = bse[var]
            pval = pvalues[var]
            
            stars = ""
            if pval < 0.01: stars = "***"
            elif pval < 0.05: stars = "**"
            elif pval < 0.1: stars = "*"
            
            latex_str += f"{nice_name} & {coef:.4f}{stars} & ({std_err:.4f}) \\\\\n"
            
    latex_str += r"""\hline
\multicolumn{3}{c}{\textbf{Hausman Test}} \\
\hline
"""
    latex_str += f"Chi-squared Statistic & {chi2:.4f} & \\\\\n"
    latex_str += f"Degrees of Freedom & {df} & \\\\\n"
    latex_str += f"P-value & {p_val:.4f} & \\\\\n"
    
    latex_str += r"""\hline
\multicolumn{3}{l}{\footnotesize Significance: *** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\
\multicolumn{3}{l}{\footnotesize Note: H0: Random Effects estimator is efficient and consistent.} \\ 
\multicolumn{3}{l}{\footnotesize H1: Fixed Effects estimator is consistent.}
\end{tabular}
\end{table}
"""
    
    return latex_str
    
    return latex_str
    
    return latex_str


def main():
    df, features, target = load_and_prep_data()
    
    # Standardize features for convergence
    print("Standardizing features...")
    df_scaled = df.copy()
    for f in features:
        df_scaled[f] = (df[f] - df[f].mean()) / df[f].std()
    
    # 1. Fixed Effects (LSDV)
    print("Estimating Fixed Effects (LSDV)...")
    # Identify institutions
    # We use dummies
    fe_dummies = pd.get_dummies(df_scaled['Instituicao'], prefix='FE', drop_first=True)
    X_fe = pd.concat([df_scaled[features], fe_dummies], axis=1)
    X_fe = sm.add_constant(X_fe).astype(float)
    y = df_scaled[target].astype(float)
    
    # Try different solver if BFGS fails
    try:
        fe_model = sm.Logit(y, X_fe).fit(disp=0, method='bfgs', maxiter=5000)
    except Exception:
        print("BFGS failed, trying limited-memory BFGS...")
        fe_model = sm.Logit(y, X_fe).fit(disp=0, method='lbfgs', maxiter=5000)
    
    # 2. Random Effects Proxy (GEE Exchangeable)
    print("Estimating Random Effects Proxy (GEE)...")
    # Use scaled features here too for valid comparison? 
    # Yes, coefficients must be comparable.
    X_re = sm.add_constant(df_scaled[features]).astype(float)
    groups = df_scaled['Instituicao']
    
    # Use Binomial Family with Exchangeable Covariance
    re_model = sm.GEE(y, X_re, groups=groups, 
                      family=sm.families.Binomial(), 
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    
    # 3. Hausman Test
    print("Performing Hausman Test...")
    res = hausman_test(fe_model.params, fe_model.cov_params(), 
                       re_model.params, re_model.cov_params())
    
    print(f"Hausman Chi2: {res[0]:.4f}, p-value: {res[2]:.4f}")
    
    # 4. Generate LaTeX
    latex_table = generate_latex_table(fe_model, res, features)
    
    with open(TABLE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(latex_table)
        
    print(f"LaTeX table saved to {TABLE_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
